from __future__ import annotations

import hashlib
import json
import os
import re
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel, ConfigDict, Field

from traceanchor.agents.schemas import AgentRole
from traceanchor.config import FORBIDDEN_AGENT_COLUMNS, LLMConfig, LLMProviderConfig
from traceanchor.ingest.common import atomic_write_json


_PRIVATE_OUTPUT_PATTERN = re.compile(
    r"(?:\bCVE-\d{4}-\d+\b|exploit[_ -]?anchor|gold[_ -]?answer|"
    r"dataset[_ -]?family|container[_ -]?role)",
    re.IGNORECASE,
)
_HTTP_ERROR_PATTERN = re.compile(r"^provider HTTP (?P<status>[1-5][0-9]{2})$")


def canonical_hash(value: object) -> str:
    canonical = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class ProviderModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProviderMessage(ProviderModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)


class ProviderRequest(ProviderModel):
    agent_role: AgentRole
    task_type: Literal["investigator_action", "correlation_action", "format_repair"]
    messages: list[ProviderMessage] = Field(min_length=1)
    response_schema: dict[str, Any]
    response_name: str = Field(min_length=1, max_length=64)
    max_output_tokens: int = Field(gt=0, le=3000)
    temperature: float = Field(ge=0.0, le=2.0)


class ProviderUsage(ProviderModel):
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)


class ProviderResponse(ProviderModel):
    content: str
    usage: ProviderUsage = Field(default_factory=ProviderUsage)
    provider_request_id: str | None = None


class CompletionRecord(ProviderModel):
    cache_key: str
    input_sha256: str
    response_sha256: str
    provider: str
    model: str
    agent_role: AgentRole
    task_type: str
    cache_hit: bool
    attempts: int = Field(gt=0)
    usage: ProviderUsage
    cost_rmb: float = Field(ge=0.0)
    provider_request_id: str | None = None


class ProviderError(RuntimeError):
    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


class ProviderBudgetExceeded(ProviderError):
    pass


def provider_error_code(error: ProviderError) -> str:
    """Return a non-sensitive diagnostic code for a provider failure.

    Provider errors are intentionally not persisted verbatim because adapters
    outside this module may include request details. The built-in adapter
    messages are reduced to status/category codes for audit diagnostics.
    """
    message = str(error)
    http_match = _HTTP_ERROR_PATTERN.fullmatch(message)
    if http_match:
        return f"provider_http_{http_match.group('status')}"
    if message in {
        "TimeoutException",
        "ConnectTimeout",
        "ReadTimeout",
        "WriteTimeout",
        "PoolTimeout",
        "ConnectError",
        "ReadError",
        "WriteError",
        "NetworkError",
    }:
        return f"provider_network_{message.lower()}"
    categories = {
        "provider model/API environment still contains CHANGE_ME": "provider_placeholder",
        "provider input/output RMB token costs are not configured": "provider_cost_unset",
        "external LLM payloads are disabled by project configuration": "provider_external_payload_disabled",
        "provider returned non-JSON HTTP response": "provider_response_non_json",
        "provider returned an invalid response envelope": "provider_response_invalid_envelope",
        "provider response is missing message content": "provider_response_missing_message",
        "provider response is missing text content": "provider_response_missing_text",
        "provider response is missing output text": "provider_response_missing_output_text",
        "provider response is missing candidate content": "provider_response_missing_candidate",
        "provider response is incomplete": "provider_response_incomplete",
        "provider refused structured output": "provider_response_refusal",
        "provider output contains a forbidden private-label reference": "provider_output_private_label",
        "provider output contains forbidden private-field keys": "provider_output_private_field",
    }
    if message in categories:
        return categories[message]
    if message.startswith("configured API-key environment variable is unset:"):
        return "provider_api_key_env_unset"
    if message.startswith("configured base-URL environment variable is unset:"):
        return "provider_base_url_env_unset"
    if message.startswith("unsupported provider:"):
        return "provider_unsupported"
    return "provider_error"


class LLMProvider(ABC):
    provider_name: str
    model: str

    @abstractmethod
    def complete_once(self, request: ProviderRequest) -> ProviderResponse:
        """Make exactly one completion attempt without content retries."""


def _response_content(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=True, sort_keys=True)


def strict_response_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Convert a Pydantic schema to the strict Responses wire convention."""

    def convert(value: Any) -> Any:
        if isinstance(value, dict):
            converted = {
                key: convert(nested)
                for key, nested in value.items()
                if key != "default"
            }
            properties = converted.get("properties")
            if isinstance(properties, dict):
                converted["required"] = list(properties)
                converted["additionalProperties"] = False
            return converted
        if isinstance(value, list):
            return [convert(item) for item in value]
        return value

    return convert(schema)


class _HTTPProvider(LLMProvider):
    def __init__(
        self,
        *,
        provider_name: str,
        model: str,
        api_key: str,
        base_url: str,
        client: httpx.Client | None = None,
    ) -> None:
        self.provider_name = provider_name
        self.model = model
        self._api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.client = client or httpx.Client(timeout=httpx.Timeout(60.0, connect=15.0))

    def _post(
        self,
        path: str,
        *,
        headers: dict[str, str],
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        try:
            response = self.client.post(f"{self.base_url}{path}", headers=headers, json=payload)
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            raise ProviderError(type(exc).__name__, retryable=True) from None
        if response.status_code == 429 or response.status_code >= 500:
            raise ProviderError(
                f"provider HTTP {response.status_code}", retryable=True
            )
        if response.status_code >= 400:
            raise ProviderError(f"provider HTTP {response.status_code}", retryable=False)
        try:
            value = response.json()
        except ValueError:
            raise ProviderError("provider returned non-JSON HTTP response") from None
        if not isinstance(value, dict):
            raise ProviderError("provider returned an invalid response envelope")
        return value


class OpenAICompatibleProvider(_HTTPProvider):
    def __init__(
        self,
        *,
        response_format: Literal["json_schema", "json_object"] = "json_schema",
        **kwargs: Any,
    ) -> None:
        super().__init__(provider_name="openai_compatible", **kwargs)
        self.response_format = response_format

    def complete_once(self, request: ProviderRequest) -> ProviderResponse:
        messages = [item.model_dump() for item in request.messages]
        if self.response_format == "json_object":
            response_format: dict[str, Any] = {"type": "json_object"}
            schema_instruction = (
                "\n\nOUTPUT JSON SCHEMA\n"
                "Return exactly one complete JSON object matching this JSON Schema, "
                "with no Markdown, truncation, or surrounding prose. Use property "
                "names and enum values exactly, include all required nested fields, "
                "preserve exactly one mutually exclusive action branch, and do not "
                "add properties that the schema does not allow:\n"
                + json.dumps(
                    request.response_schema,
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            for message in messages:
                if message["role"] == "system":
                    message["content"] += schema_instruction
                    break
            else:
                messages.insert(
                    0,
                    {
                        "role": "system",
                        "content": schema_instruction.lstrip(),
                    },
                )
        else:
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": request.response_name,
                    "strict": True,
                    "schema": request.response_schema,
                },
            }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": request.temperature,
            "max_tokens": request.max_output_tokens,
            "response_format": response_format,
        }
        value = self._post(
            "/chat/completions",
            headers={"Authorization": f"Bearer {self._api_key}"},
            payload=payload,
        )
        try:
            content = value["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise ProviderError("provider response is missing message content") from None
        usage = value.get("usage") or {}
        return ProviderResponse(
            content=_response_content(content),
            usage=ProviderUsage(
                input_tokens=int(usage.get("prompt_tokens", 0)),
                output_tokens=int(usage.get("completion_tokens", 0)),
            ),
            provider_request_id=value.get("id"),
        )


class OpenAIResponsesProvider(_HTTPProvider):
    def __init__(
        self,
        *,
        response_format: Literal["json_schema", "json_object"] = "json_schema",
        **kwargs: Any,
    ) -> None:
        super().__init__(provider_name="openai_responses", **kwargs)
        self.response_format = response_format

    def complete_once(self, request: ProviderRequest) -> ProviderResponse:
        messages = [item.model_dump() for item in request.messages]
        if self.response_format == "json_object":
            response_format: dict[str, Any] = {"type": "json_object"}
            schema_instruction = (
                "\n\nOUTPUT JSON SCHEMA\n"
                "Return exactly one complete JSON object matching this JSON Schema, "
                "with no Markdown, truncation, or surrounding prose. Use property "
                "names and enum values exactly, include all required nested fields, "
                "preserve exactly one mutually exclusive action branch, and do not "
                "add properties that the schema does not allow:\n"
                + json.dumps(
                    request.response_schema,
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            for message in messages:
                if message["role"] == "system":
                    message["content"] += schema_instruction
                    break
            else:
                messages.insert(
                    0,
                    {
                        "role": "system",
                        "content": schema_instruction.lstrip(),
                    },
                )
        else:
            response_format = {
                "type": "json_schema",
                "name": request.response_name,
                "strict": True,
                "schema": strict_response_schema(request.response_schema),
            }
        value = self._post(
            "/v1/responses",
            headers={"Authorization": f"Bearer {self._api_key}"},
            payload={
                "model": self.model,
                "input": messages,
                "temperature": request.temperature,
                "max_output_tokens": request.max_output_tokens,
                "store": False,
                "text": {"format": response_format},
            },
        )
        if value.get("status") == "incomplete":
            raise ProviderError("provider response is incomplete")
        output = value.get("output") or []
        texts = []
        refused = False
        for item in output:
            if not isinstance(item, dict) or item.get("type") != "message":
                continue
            for content in item.get("content") or []:
                if not isinstance(content, dict):
                    continue
                if content.get("type") == "output_text":
                    texts.append(str(content.get("text", "")))
                elif content.get("type") == "refusal":
                    refused = True
        if refused:
            raise ProviderError("provider refused structured output")
        text = "".join(texts)
        if not text:
            candidate = value.get("output_text")
            text = candidate if isinstance(candidate, str) else ""
        if not text:
            raise ProviderError("provider response is missing output text")
        usage = value.get("usage") or {}
        return ProviderResponse(
            content=text,
            usage=ProviderUsage(
                input_tokens=int(usage.get("input_tokens", 0)),
                output_tokens=int(usage.get("output_tokens", 0)),
            ),
            provider_request_id=value.get("id"),
        )


class AnthropicProvider(_HTTPProvider):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(provider_name="anthropic", **kwargs)

    def complete_once(self, request: ProviderRequest) -> ProviderResponse:
        system = "\n\n".join(
            item.content for item in request.messages if item.role == "system"
        )
        messages = [
            item.model_dump() for item in request.messages if item.role != "system"
        ]
        value = self._post(
            "/v1/messages",
            headers={
                "x-api-key": self._api_key,
                "anthropic-version": "2023-06-01",
            },
            payload={
                "model": self.model,
                "system": system,
                "messages": messages,
                "temperature": request.temperature,
                "max_tokens": request.max_output_tokens,
                "tools": [
                    {
                        "name": "emit_agent_action",
                        "description": "Return the next TraceAnchor AgentAction.",
                        "input_schema": request.response_schema,
                    }
                ],
                "tool_choice": {"type": "tool", "name": "emit_agent_action"},
            },
        )
        blocks = value.get("content") or []
        tool_inputs = [
            item.get("input")
            for item in blocks
            if isinstance(item, dict)
            and item.get("type") == "tool_use"
            and item.get("name") == "emit_agent_action"
            and isinstance(item.get("input"), dict)
        ]
        text = (
            json.dumps(tool_inputs[0], ensure_ascii=True, sort_keys=True)
            if len(tool_inputs) == 1
            else ""
        )
        fallback_text = "".join(
            str(item.get("text", ""))
            for item in blocks
            if isinstance(item, dict) and item.get("type") == "text"
        )
        if not text:
            text = fallback_text
        if not text:
            raise ProviderError("provider response is missing text content")
        usage = value.get("usage") or {}
        return ProviderResponse(
            content=text,
            usage=ProviderUsage(
                input_tokens=int(usage.get("input_tokens", 0)),
                output_tokens=int(usage.get("output_tokens", 0)),
            ),
            provider_request_id=value.get("id"),
        )


class GeminiProvider(_HTTPProvider):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(provider_name="gemini", **kwargs)

    def complete_once(self, request: ProviderRequest) -> ProviderResponse:
        contents = []
        system_parts = []
        for item in request.messages:
            if item.role == "system":
                system_parts.append({"text": item.content})
            else:
                contents.append(
                    {
                        "role": "model" if item.role == "assistant" else "user",
                        "parts": [{"text": item.content}],
                    }
                )
        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": request.temperature,
                "maxOutputTokens": request.max_output_tokens,
                "responseMimeType": "application/json",
                "responseJsonSchema": request.response_schema,
            },
        }
        if system_parts:
            payload["systemInstruction"] = {"parts": system_parts}
        value = self._post(
            f"/v1beta/models/{self.model}:generateContent",
            headers={"x-goog-api-key": self._api_key},
            payload=payload,
        )
        try:
            parts = value["candidates"][0]["content"]["parts"]
            content = "".join(str(item.get("text", "")) for item in parts)
        except (KeyError, IndexError, TypeError):
            raise ProviderError("provider response is missing candidate content") from None
        usage = value.get("usageMetadata") or {}
        return ProviderResponse(
            content=content,
            usage=ProviderUsage(
                input_tokens=int(usage.get("promptTokenCount", 0)),
                output_tokens=int(usage.get("candidatesTokenCount", 0)),
            ),
            provider_request_id=value.get("responseId"),
        )


def _assert_blind_response(content: str) -> None:
    if _PRIVATE_OUTPUT_PATTERN.search(content):
        raise ProviderError("provider output contains a forbidden private-label reference")
    try:
        value = json.loads(content)
    except json.JSONDecodeError:
        return

    def inspect(item: Any) -> None:
        if isinstance(item, dict):
            forbidden = {str(key).lower() for key in item}.intersection(FORBIDDEN_AGENT_COLUMNS)
            if forbidden:
                raise ProviderError("provider output contains forbidden private-field keys")
            for nested in item.values():
                inspect(nested)
        elif isinstance(item, list):
            for nested in item:
                inspect(nested)

    inspect(value)


class ProviderRuntime:
    def __init__(
        self,
        provider: LLMProvider,
        llm_config: LLMConfig,
        provider_config: LLMProviderConfig | None,
        cache_dir: Path,
        *,
        budget_rmb: float | None = None,
    ) -> None:
        self.provider = provider
        self.llm_config = llm_config
        self.provider_config = provider_config
        self.cache_dir = cache_dir
        self.records: list[CompletionRecord] = []
        self.spent_rmb = 0.0
        self.budget_rmb = (
            float(llm_config.total_budget_rmb) if budget_rmb is None else float(budget_rmb)
        )
        if self.budget_rmb <= 0:
            raise ProviderBudgetExceeded("no LLM RMB budget remains")

    def _cost(self, usage: ProviderUsage) -> float:
        if self.provider_config is None:
            return 0.0
        input_rate = self.provider_config.cost_rmb_per_million_input_tokens or 0.0
        output_rate = self.provider_config.cost_rmb_per_million_output_tokens or 0.0
        return (
            usage.input_tokens * input_rate + usage.output_tokens * output_rate
        ) / 1_000_000.0

    def complete(
        self,
        request: ProviderRequest,
        *,
        prompt_sha256: str,
        tool_schema_sha256: str,
        ledger_version: str,
        run_nonce: str,
    ) -> ProviderResponse:
        input_hash = canonical_hash([item.model_dump() for item in request.messages])
        key_fields = {
            "provider": self.provider.provider_name,
            "model": self.provider.model,
            "prompt_sha256": prompt_sha256,
            "tool_schema_sha256": tool_schema_sha256,
            "response_schema_sha256": canonical_hash(request.response_schema),
            "input_sha256": input_hash,
            "ledger_version": ledger_version,
            "run_nonce": run_nonce,
        }
        cache_key = canonical_hash(key_fields)
        cache_path = self.cache_dir / f"{cache_key}.json"
        if self.llm_config.cache_enabled and cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            response = ProviderResponse.model_validate(cached["response"])
            _assert_blind_response(response.content)
            record = CompletionRecord(
                cache_key=cache_key,
                input_sha256=input_hash,
                response_sha256=canonical_hash(response.model_dump(mode="json")),
                provider=self.provider.provider_name,
                model=self.provider.model,
                agent_role=request.agent_role,
                task_type=request.task_type,
                cache_hit=True,
                attempts=1,
                usage=response.usage,
                cost_rmb=0.0,
                provider_request_id=response.provider_request_id,
            )
            self.records.append(record)
            return response

        attempts = 0
        while True:
            attempts += 1
            try:
                response = self.provider.complete_once(request)
                break
            except ProviderError as exc:
                if not exc.retryable or attempts > self.llm_config.retry_provider_errors:
                    raise
                time.sleep(float(2 ** (attempts - 1)))

        _assert_blind_response(response.content)
        cost = self._cost(response.usage)
        self.spent_rmb += cost
        response_hash = canonical_hash(response.model_dump(mode="json"))
        record = CompletionRecord(
            cache_key=cache_key,
            input_sha256=input_hash,
            response_sha256=response_hash,
            provider=self.provider.provider_name,
            model=self.provider.model,
            agent_role=request.agent_role,
            task_type=request.task_type,
            cache_hit=False,
            attempts=attempts,
            usage=response.usage,
            cost_rmb=cost,
            provider_request_id=response.provider_request_id,
        )
        self.records.append(record)
        if self.llm_config.cache_enabled:
            atomic_write_json(
                cache_path,
                {
                    "schema_version": 1,
                    "cache_key": cache_key,
                    "key_fields": key_fields,
                    "response": response.model_dump(mode="json"),
                    "response_sha256": response_hash,
                },
            )
        if self.spent_rmb > self.budget_rmb:
            raise ProviderBudgetExceeded("LLM run exceeded the configured RMB budget")
        return response


def _configured_secret(spec: LLMProviderConfig) -> str:
    if "CHANGE_ME" in spec.api_key_env or "CHANGE_ME" in spec.model:
        raise ProviderError("provider model/API environment still contains CHANGE_ME")
    value = os.environ.get(spec.api_key_env)
    if not value:
        raise ProviderError(f"configured API-key environment variable is unset: {spec.api_key_env}")
    return value


def _base_url(spec: LLMProviderConfig, default: str) -> str:
    if spec.base_url_env is None:
        return default
    value = os.environ.get(spec.base_url_env)
    if not value:
        raise ProviderError(f"configured base-URL environment variable is unset: {spec.base_url_env}")
    return value


def is_loopback_url(value: str) -> bool:
    hostname = (urlparse(value).hostname or "").lower()
    return hostname in {"localhost", "127.0.0.1", "::1"}


def build_http_provider(
    spec: LLMProviderConfig,
    *,
    external_payload_allowed: bool,
    client: httpx.Client | None = None,
) -> LLMProvider:
    if (
        spec.cost_rmb_per_million_input_tokens is None
        or spec.cost_rmb_per_million_output_tokens is None
    ):
        raise ProviderError("provider input/output RMB token costs are not configured")
    key = _configured_secret(spec)
    if spec.provider == "openai_compatible":
        base_url = _base_url(spec, "https://api.openai.com/v1")
        if not external_payload_allowed and not is_loopback_url(base_url):
            raise ProviderError("external LLM payloads are disabled by project configuration")
        return OpenAICompatibleProvider(
            model=spec.model,
            api_key=key,
            base_url=base_url,
            client=client,
            response_format=spec.response_format,
        )
    if spec.provider == "openai_responses":
        base_url = _base_url(spec, "https://api.openai.com")
        if not external_payload_allowed and not is_loopback_url(base_url):
            raise ProviderError("external LLM payloads are disabled by project configuration")
        return OpenAIResponsesProvider(
            model=spec.model,
            api_key=key,
            base_url=base_url,
            client=client,
            response_format=spec.response_format,
        )
    if spec.provider == "anthropic":
        base_url = _base_url(spec, "https://api.anthropic.com")
        if not external_payload_allowed and not is_loopback_url(base_url):
            raise ProviderError("external LLM payloads are disabled by project configuration")
        return AnthropicProvider(
            model=spec.model, api_key=key, base_url=base_url, client=client
        )
    if spec.provider == "gemini":
        base_url = _base_url(spec, "https://generativelanguage.googleapis.com")
        if not external_payload_allowed and not is_loopback_url(base_url):
            raise ProviderError("external LLM payloads are disabled by project configuration")
        return GeminiProvider(
            model=spec.model, api_key=key, base_url=base_url, client=client
        )
    raise ProviderError(f"unsupported provider: {spec.provider}")


__all__ = [
    "AnthropicProvider",
    "CompletionRecord",
    "GeminiProvider",
    "LLMProvider",
    "OpenAICompatibleProvider",
    "OpenAIResponsesProvider",
    "ProviderBudgetExceeded",
    "ProviderError",
    "provider_error_code",
    "ProviderMessage",
    "ProviderRequest",
    "ProviderResponse",
    "ProviderRuntime",
    "ProviderUsage",
    "build_http_provider",
    "canonical_hash",
    "is_loopback_url",
    "strict_response_schema",
]
