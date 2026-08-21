from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import ValidationError

from traceanchor.agents.broker import (
    BrokerBudgetExceeded,
    BrokerError,
    BrokerRecoverableError,
    BrokerSafetyError,
    ToolBroker,
    tool_schema_hash,
)
from traceanchor.agents.provider import (
    LLMProvider,
    ProviderBudgetExceeded,
    ProviderError,
    ProviderMessage,
    ProviderRequest,
    ProviderRuntime,
    canonical_hash,
    provider_error_code,
)
from traceanchor.agents.prompts import load_system_prompt, prompt_hashes
from traceanchor.agents.schemas import (
    AgentAction,
    AgentRole,
    BlindAlert,
    CorrelationAction,
    FinalInvestigation,
    InvestigatorAction,
    InvestigatorFindings,
)
from traceanchor.agents.verifier import verify_investigation
from traceanchor.config import LLMProviderConfig, TraceAnchorConfig, config_hash
from traceanchor.ingest.common import atomic_write_json


class InvestigationState(str, Enum):
    ALERTED = "ALERTED"
    PRIMARY_TRIAGE = "PRIMARY_TRIAGE"
    CROSS_MODAL_QUERY = "CROSS_MODAL_QUERY"
    CORRELATION = "CORRELATION"
    VERIFICATION = "VERIFICATION"
    FINAL = "FINAL"
    ABSTAIN = "ABSTAIN"
    INFRASTRUCTURE_ERROR = "INFRASTRUCTURE_ERROR"


class StructuredOutputError(RuntimeError):
    pass


def _action_model(
    role: AgentRole,
) -> type[InvestigatorAction] | type[CorrelationAction] | type[AgentAction]:
    if role in {"network_investigator", "host_investigator"}:
        return InvestigatorAction
    if role == "correlation_agent":
        return CorrelationAction
    return AgentAction


def _validate_action(
    role: AgentRole,
    content: str,
) -> AgentAction:
    validated = _action_model(role).model_validate_json(content)
    if isinstance(validated, AgentAction):
        return validated
    return validated.to_agent_action()


def _root_candidate_indexes_to_prune(
    error: ValidationError,
) -> list[int] | None:
    indexes: list[int] = []
    for item in error.errors(include_url=False, include_context=False, include_input=False):
        path = tuple(item["loc"])
        if (
            item["type"] != "string_pattern_mismatch"
            or len(path) != 4
            or path[0] != "final_output"
            or path[1] != "root_cause_candidates"
            or not isinstance(path[2], int)
            or path[3] != "entity_id"
        ):
            return None
        indexes.append(path[2])
    return sorted(set(indexes), reverse=True) if indexes else None


def _structural_action(
    role: AgentRole,
    content: str,
) -> tuple[AgentAction | None, dict[str, Any], ValidationError | ValueError | None]:
    """Validate one response with only the v7 removal-only structural recovery."""
    metadata: dict[str, Any] = {
        "first_object_extracted": False,
        "trailing_content_discarded": False,
        "root_cause_candidates_removed": 0,
    }
    try:
        value, end = json.JSONDecoder().raw_decode(content.lstrip())
    except json.JSONDecodeError:
        try:
            return _validate_action(role, content), metadata, None
        except (ValidationError, ValueError) as exc:
            return None, metadata, exc
    if not isinstance(value, dict):
        try:
            return _validate_action(role, content), metadata, None
        except (ValidationError, ValueError) as exc:
            return None, metadata, exc

    trailing = bool(content.lstrip()[end:].strip())
    candidate = json.dumps(value, ensure_ascii=True, separators=(",", ":"))
    try:
        action = _validate_action(role, candidate)
    except ValidationError as exc:
        indexes = _root_candidate_indexes_to_prune(exc)
        final_output = value.get("final_output")
        roots = final_output.get("root_cause_candidates") if isinstance(final_output, dict) else None
        if indexes is None or not isinstance(roots, list) or any(
            index >= len(roots) for index in indexes
        ):
            return None, metadata, exc
        for index in indexes:
            roots.pop(index)
        try:
            action = _validate_action(
                role, json.dumps(value, ensure_ascii=True, separators=(",", ":"))
            )
        except (ValidationError, ValueError) as pruned_error:
            return None, metadata, pruned_error
        metadata["root_cause_candidates_removed"] = len(indexes)
    except ValueError as exc:
        return None, metadata, exc

    if trailing:
        metadata["first_object_extracted"] = True
        metadata["trailing_content_discarded"] = True
    return action, metadata, None


def _validation_diagnostics(error: ValidationError) -> list[dict[str, Any]]:
    return [
        {
            "type": str(item["type"]),
            "path": [str(value) for value in item["loc"]],
        }
        for item in error.errors(
            include_url=False,
            include_context=False,
            include_input=False,
        )
    ]


def _repair_instruction(role: AgentRole) -> str:
    branch_instruction = (
        "For a finish action, include final_output; findings is not permitted."
        if role == "correlation_agent"
        else "For a finish action, include findings; final_output is not permitted."
    )
    return (
        "Repair JSON/schema only. Return a shorter complete JSON object with no "
        "Markdown, truncation, or surrounding prose. If the input was truncated, "
        "retain only complete higher-confidence existing items and remove incomplete "
        "or redundant trailing items; do not reconstruct missing text. Preserve "
        "exactly one mutually exclusive action branch and include all required nested "
        f"fields from the supplied schema. {branch_instruction} Shorten reason to at "
        "most 500 characters without adding meaning. Remove malformed Evidence IDs. "
        "If a fact, root-cause candidate, step, ATT&CK mapping, or response "
        "consideration then has no valid Evidence ID, remove the entire dependent "
        "item. Evidence IDs are immutable: do not invent, normalize, guess, or copy "
        "a different identifier. If the invalid output contains multiple top-level "
        "JSON values, retain exactly one intended action object and discard the "
        "others; do not merge factual content between them. If an Investigator "
        "finish has no valid complete fact or hypothesis, return the top-level "
        "abstain action without findings. Remove the entire root-cause candidate "
        "when its entity_id is invalid; never invent or normalize an Entity ID. "
        "Remove fields not declared by the supplied schema, including root_cause. "
        "Do not fabricate required fields to preserve an invalid item. Do not add "
        "facts, evidence, claims, or identifiers."
    )


def _output_constraints() -> str:
    return (
        "Return one syntactically complete JSON object and finish JSON before the "
        "output limit; omit lower-confidence optional items rather than truncating. "
        "Keep reason at most 500 characters. Emit only Evidence IDs matching the "
        "frozen schema pattern; never invent, normalize, guess, or copy identifiers."
    )


def _investigation_guidance(role: AgentRole) -> dict[str, Any]:
    if role in {"network_investigator", "host_investigator"}:
        return {
            "initial_max_records": 50,
            "broadening_rule": (
                "Start with the narrowest query supported by observed fields. "
                "Page or broaden only to validate a named candidate or answer an "
                "explicit cross-modal question."
            ),
        }
    if role == "correlation_agent":
        return {
            "root_cause_rule": (
                "A root-cause candidate requires an approved find_socket_owner or "
                "join_host_network_evidence chain that returned its entity and "
                "same-scene cited Evidence IDs; otherwise leave roots empty and "
                "state an evidence gap."
            ),
            "step_rule": (
                "Every final step must cite Evidence IDs from a successful "
                "build_timeline result and use the inclusive minimum and maximum "
                "returned timestamps. Omit a step when no usable timeline exists."
            ),
            "attack_rule": (
                "An ATT&CK item requires an exact technique returned by the "
                "approved knowledge retrieval tool."
            ),
        }
    return {}


def _insufficient_output(gap: str) -> FinalInvestigation:
    return FinalInvestigation.model_validate(
        {
            "incident_summary": "insufficient_evidence",
            "root_cause_candidates": [],
            "steps": [],
            "attack_techniques": [],
            "evidence_gaps": [gap],
            "response_considerations": [],
            "verifier": {
                "decision": "insufficient_evidence",
                "removed_claims": [],
                "unsupported_claim_rate": 0.0,
            },
        }
    )


def _compact_state(value: dict[str, Any], max_bytes: int) -> str:
    state = copy.deepcopy(value)

    def serialize() -> str:
        return json.dumps(state, ensure_ascii=True, sort_keys=True, separators=(",", ":"))

    payload = serialize()
    while len(payload.encode("utf-8")) > max_bytes:
        candidates = []
        for item in state.get("tool_transcript", []):
            if not isinstance(item, dict):
                continue
            result = item.get("result")
            records = result.get("records") if isinstance(result, dict) else None
            if isinstance(records, list) and records:
                candidates.append(records)
        if not candidates:
            raise BrokerBudgetExceeded("Agent context exceeds the configured 64 KiB limit")
        largest = max(candidates, key=len)
        largest.pop()
        state["context_truncated"] = True
        payload = serialize()
    return payload


def _request(
    config: TraceAnchorConfig,
    config_path: Path,
    role: AgentRole,
    task_type: str,
    state: dict[str, Any],
) -> ProviderRequest:
    max_kib = int(
        getattr(config.privacy_and_blinding, "external_llm_max_context_kib", 64)
    )
    system_prompt = load_system_prompt(config_path.parent, role)
    action_model = _action_model(role)
    response_schema = action_model.model_json_schema()
    # Reserve for the system prompt, response schema, and provider envelope.
    # Count the schema twice because some adapters must also state it in text.
    overhead = len(system_prompt.encode("utf-8")) + 2 * len(
        json.dumps(response_schema, ensure_ascii=True).encode("utf-8")
    ) + 4096
    available = max_kib * 1024 - overhead
    if available <= 0:
        raise BrokerBudgetExceeded("Agent protocol overhead exceeds the context budget")
    request_state = dict(state)
    request_state["output_constraints"] = _output_constraints()
    content = _compact_state(request_state, available)
    return ProviderRequest(
        agent_role=role,
        task_type=task_type,
        messages=[
            ProviderMessage(
                role="system",
                content=system_prompt,
            ),
            ProviderMessage(role="user", content=content),
        ],
        response_schema=response_schema,
        response_name=(
            "traceanchor_correlation_action"
            if action_model is CorrelationAction
            else "traceanchor_investigator_action"
            if action_model is InvestigatorAction
            else "traceanchor_agent_action"
        ),
        max_output_tokens=config.agents.max_output_tokens,
        temperature=config.agents.temperature,
    )


def _complete_action(
    config: TraceAnchorConfig,
    config_path: Path,
    runtime: ProviderRuntime,
    broker: ToolBroker,
    role: AgentRole,
    task_type: str,
    state: dict[str, Any],
    run_nonce: str,
    structural_audit: list[dict[str, Any]],
) -> AgentAction:
    prompts = prompt_hashes(config_path.parent)
    request = _request(config, config_path, role, task_type, state)
    response = runtime.complete(
        request,
        prompt_sha256=prompts[role],
        tool_schema_sha256=tool_schema_hash(),
        ledger_version=broker.ledger_version,
        run_nonce=run_nonce,
    )
    action, metadata, error = _structural_action(role, response.content)
    structural_audit.append(
        {
            "source": "original",
            "response_sha256": canonical_hash(response.model_dump(mode="json")),
            **metadata,
        }
    )
    if action is not None:
        return action
    assert error is not None
    if isinstance(error, (ValidationError, ValueError)):
        exc = error
        validation_errors = (
            _validation_diagnostics(exc)
            if isinstance(exc, ValidationError)
            else [{"type": "value_error", "path": []}]
        )
        repair_state = {
            "invalid_output": response.content,
            "validation_errors": validation_errors,
            "instruction": _repair_instruction(role),
        }
        repair = _request(config, config_path, role, "format_repair", repair_state)
        repaired = runtime.complete(
            repair,
            prompt_sha256=prompts[role],
            tool_schema_sha256=tool_schema_hash(),
            ledger_version=broker.ledger_version,
            run_nonce=run_nonce,
        )
        action, metadata, _repair_error = _structural_action(role, repaired.content)
        structural_audit.append(
            {
                "source": "repair",
                "response_sha256": canonical_hash(repaired.model_dump(mode="json")),
                **metadata,
            }
        )
        if action is not None:
            return action
    raise StructuredOutputError("structured Agent output failed schema validation") from None


def _run_investigator(
    config: TraceAnchorConfig,
    config_path: Path,
    runtime: ProviderRuntime,
    broker: ToolBroker,
    role: AgentRole,
    questions: list[str],
    run_nonce: str,
    *,
    structural_audit: list[dict[str, Any]] | None = None,
) -> tuple[InvestigatorFindings, list[dict[str, Any]]]:
    transcript: list[dict[str, Any]] = []
    audit = structural_audit if structural_audit is not None else []
    for turn in range(config.agents.max_tool_calls_per_role + 1):
        remaining = broker.remaining_tool_calls(role)
        must_terminate = remaining == 0
        state = {
            "scenario_token": broker.alert.scenario_token,
            "allowed_start_ts_ns": broker.window_start_ns,
            "allowed_end_ts_ns": broker.window_end_ns,
            "turn_index": turn,
            "remaining_tool_calls": remaining,
            "must_terminate": must_terminate,
            "available_tools": {} if must_terminate else broker.available_schemas(role),
            "cross_modal_questions": questions,
            "investigation_guidance": _investigation_guidance(role),
            "tool_transcript": transcript,
        }
        action = _complete_action(
            config,
            config_path,
            runtime,
            broker,
            role,
            "investigator_action",
            state,
            run_nonce,
            audit,
        )
        if action.action == "abstain":
            return InvestigatorFindings(abstain=True), transcript
        if action.action == "finish":
            if action.findings is None:
                raise StructuredOutputError("investigator finish omitted findings")
            return action.findings, transcript
        assert action.tool_call is not None
        if must_terminate:
            return InvestigatorFindings(abstain=True), transcript
        try:
            result = broker.call(role, action.tool_call.name, action.tool_call.arguments)
        except BrokerRecoverableError as exc:
            feedback = broker.rejection_feedback(role, action.tool_call.name, exc)
            transcript.append({"tool": action.tool_call.name, **feedback})
            continue
        transcript.append(
            {
                "tool": action.tool_call.name,
                "result": result,
            }
        )
    raise BrokerBudgetExceeded(f"{role} did not finish within its tool budget")


def _run_correlation(
    config: TraceAnchorConfig,
    config_path: Path,
    runtime: ProviderRuntime,
    broker: ToolBroker,
    findings: list[InvestigatorFindings],
    prior_transcript: list[dict[str, Any]],
    run_nonce: str,
    *,
    structural_audit: list[dict[str, Any]] | None = None,
) -> tuple[FinalInvestigation, list[dict[str, Any]]]:
    transcript = list(prior_transcript)
    correlation_only: list[dict[str, Any]] = []
    audit = structural_audit if structural_audit is not None else []
    for turn in range(config.agents.max_tool_calls_per_role + 1):
        remaining = broker.remaining_tool_calls("correlation_agent")
        must_terminate = remaining == 0
        state = {
            "scenario_token": broker.alert.scenario_token,
            "allowed_start_ts_ns": broker.window_start_ns,
            "allowed_end_ts_ns": broker.window_end_ns,
            "turn_index": turn,
            "remaining_tool_calls": remaining,
            "must_terminate": must_terminate,
            "available_tools": (
                {} if must_terminate else broker.available_schemas("correlation_agent")
            ),
            "investigator_findings": [item.model_dump(mode="json") for item in findings],
            "investigation_guidance": _investigation_guidance("correlation_agent"),
            "tool_transcript": transcript,
        }
        action = _complete_action(
            config,
            config_path,
            runtime,
            broker,
            "correlation_agent",
            "correlation_action",
            state,
            run_nonce,
            audit,
        )
        if action.action == "abstain":
            return _insufficient_output(action.reason or "Correlation Agent abstained."), transcript
        if action.action == "finish":
            if action.final_output is None:
                raise StructuredOutputError("correlation finish omitted final output")
            return action.final_output, transcript
        assert action.tool_call is not None
        if must_terminate:
            return _insufficient_output(
                "Correlation Agent requested a tool after its budget was exhausted."
            ), transcript
        try:
            result = broker.call(
                "correlation_agent", action.tool_call.name, action.tool_call.arguments
            )
        except BrokerRecoverableError as exc:
            feedback = broker.rejection_feedback(
                "correlation_agent", action.tool_call.name, exc
            )
            item = {"tool": action.tool_call.name, **feedback}
            transcript.append(item)
            correlation_only.append(item)
            continue
        item = {"tool": action.tool_call.name, "result": result}
        transcript.append(item)
        correlation_only.append(item)
    raise BrokerBudgetExceeded("Correlation Agent did not finish within its tool budget")


def _route(alert: BlindAlert) -> tuple[AgentRole, AgentRole, str]:
    primary = alert.channels[0]
    if len(alert.channels) == 2:
        scores = {
            "network": float(alert.peak_score_network or 0.0),
            "host": float(alert.peak_score_host or 0.0),
        }
        primary = sorted(alert.channels, key=lambda value: (-scores[value], alert.channels.index(value)))[0]
    if primary == "network":
        return "network_investigator", "host_investigator", "network_first"
    return "host_investigator", "network_investigator", "host_first"


def run_investigation(
    config: TraceAnchorConfig,
    config_path: Path,
    alert: BlindAlert,
    provider: LLMProvider,
    provider_config: LLMProviderConfig | None,
    *,
    run_id: str,
    run_nonce: str,
    persist: bool = True,
    cost_budget_rmb: float | None = None,
    run_partition: Literal["development", "test"] = "development",
) -> dict[str, Any]:
    if not run_id or any(value not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for value in run_id):
        raise ValueError("run_id contains unsupported characters")
    resolved = config.resolved_dict(config_path)
    run_root = Path(resolved["paths"]["agent_runs_dir"]) / run_partition / run_id
    if persist and run_root.exists():
        raise FileExistsError("Agent run exists; refusing to overwrite audit history")
    started = datetime.now(timezone.utc)
    broker = ToolBroker(config, config_path, alert, run_id=run_id)
    runtime = ProviderRuntime(
        provider,
        config.llm,
        provider_config,
        Path(resolved["paths"]["artifacts_dir"]) / "agent_cache",
        budget_rmb=cost_budget_rmb,
    )
    transitions = [InvestigationState.ALERTED]
    failure_class: str | None = None
    error: str | None = None
    provider_error: str | None = None
    output = _insufficient_output("Investigation did not start.")
    route = "unselected"
    structural_audit: list[dict[str, Any]] = []
    try:
        primary, secondary, route = _route(alert)
        transitions.append(InvestigationState.PRIMARY_TRIAGE)
        primary_findings, primary_tools = _run_investigator(
            config,
            config_path,
            runtime,
            broker,
            primary,
            [],
            run_nonce,
            structural_audit=structural_audit,
        )
        transitions.append(InvestigationState.CROSS_MODAL_QUERY)
        questions = primary_findings.cross_modal_questions
        secondary_findings, secondary_tools = _run_investigator(
            config,
            config_path,
            runtime,
            broker,
            secondary,
            questions,
            run_nonce,
            structural_audit=structural_audit,
        )
        transitions.append(InvestigationState.CORRELATION)
        draft, transcript = _run_correlation(
            config,
            config_path,
            runtime,
            broker,
            [primary_findings, secondary_findings],
            [*primary_tools, *secondary_tools],
            run_nonce,
            structural_audit=structural_audit,
        )
        transitions.append(InvestigationState.VERIFICATION)
        output = verify_investigation(draft, broker, transcript)
        transitions.append(
            InvestigationState.ABSTAIN
            if output.verifier.decision == "insufficient_evidence"
            else InvestigationState.FINAL
        )
    except BrokerSafetyError:
        failure_class = "SAFETY_BLOCK"
        error = "A requested action crossed an Agent safety boundary."
        output = _insufficient_output(error)
        transitions.append(InvestigationState.ABSTAIN)
    except (BrokerBudgetExceeded, ProviderBudgetExceeded):
        failure_class = "BUDGET_EXCEEDED"
        error = "The investigation exhausted a configured budget."
        output = _insufficient_output(error)
        transitions.append(InvestigationState.ABSTAIN)
    except StructuredOutputError:
        failure_class = "SCHEMA_ERROR"
        error = "The provider failed the structured-output contract."
        output = _insufficient_output(error)
        transitions.append(InvestigationState.ABSTAIN)
    except BrokerError:
        failure_class = "TOOL_ERROR"
        error = "An approved evidence tool failed."
        output = _insufficient_output(error)
        transitions.append(InvestigationState.INFRASTRUCTURE_ERROR)
    except ProviderError as exc:
        failure_class = "PROVIDER_ERROR"
        error = "The configured LLM provider failed."
        provider_error = provider_error_code(exc)
        output = _insufficient_output(error)
        transitions.append(InvestigationState.INFRASTRUCTURE_ERROR)

    finished = datetime.now(timezone.utc)
    output_value = output.model_dump(mode="json")
    output_hash = canonical_hash(output_value)
    schema_path = config_path.parent / config.agents.output_schema
    manifest = {
        "schema_version": 1,
        "classification": "BLINDED AGENT OUTPUT",
        "run_id": run_id,
        "run_nonce": run_nonce,
        "run_partition": run_partition,
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "config_sha256": config_hash(config, config_path),
        "provider": provider.provider_name,
        "model": provider.model,
        "route": route,
        "alert": alert.model_dump(mode="json"),
        "state_transitions": [value.value for value in transitions],
        "outcome": transitions[-1].value,
        "failure_class": failure_class,
        "error": error,
        "provider_error_code": provider_error,
        "prompt_sha256": prompt_hashes(config_path.parent),
        "tool_schema_sha256": tool_schema_hash(),
        "output_schema_sha256": canonical_hash(json.loads(schema_path.read_text(encoding="utf-8"))),
        "ledger_version": broker.ledger_version,
        "tool_calls": [item.model_dump(mode="json") for item in broker.audit],
        "tool_calls_total": broker.total_calls,
        "structural_handling": structural_audit,
        "provider_calls": [item.model_dump(mode="json") for item in runtime.records],
        "usage": {
            "input_tokens": sum(item.usage.input_tokens for item in runtime.records if not item.cache_hit),
            "output_tokens": sum(item.usage.output_tokens for item in runtime.records if not item.cache_hit),
            "cost_rmb": runtime.spent_rmb,
        },
        "output_sha256": output_hash,
    }
    if persist:
        atomic_write_json(run_root / "output.json", output_value)
        atomic_write_json(run_root / "manifest.json", manifest)
    return {"manifest": manifest, "output": output_value}


__all__ = ["InvestigationState", "run_investigation"]
