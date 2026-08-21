from __future__ import annotations

from collections import Counter
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from traceanchor.agents.provider import canonical_hash
from traceanchor.agents.schemas import AgentRole, BlindAlert, ToolName
from traceanchor.config import traceanchorConfig
from traceanchor.evidence.schemas import (
    AlertContextRequest,
    AttackKnowledgeRequest,
    ConnectionRequest,
    FileActivityRequest,
    JoinEvidenceRequest,
    PacketMetadataRequest,
    ProcessTreeRequest,
    SocketOwnerRequest,
    SyscallRequest,
    TimelineRequest,
    ToolModel,
    ValidateEvidenceRequest,
)
from traceanchor.evidence.tools import EvidenceToolError, EvidenceTools


_TOOL_MODELS: dict[str, type[ToolModel]] = {
    "get_alert_context": AlertContextRequest,
    "list_connections": ConnectionRequest,
    "get_packet_metadata": PacketMetadataRequest,
    "find_socket_owner": SocketOwnerRequest,
    "get_process_tree": ProcessTreeRequest,
    "list_syscalls": SyscallRequest,
    "get_file_activity": FileActivityRequest,
    "join_host_network_evidence": JoinEvidenceRequest,
    "build_timeline": TimelineRequest,
    "retrieve_attack_knowledge": AttackKnowledgeRequest,
    "validate_evidence_ids": ValidateEvidenceRequest,
}
_ROLE_TOOLS: dict[AgentRole, tuple[ToolName, ...]] = {
    "orchestrator": ("get_alert_context",),
    "network_investigator": (
        "list_connections",
        "get_packet_metadata",
        "find_socket_owner",
        "expand_window",
    ),
    "host_investigator": (
        "list_syscalls",
        "get_file_activity",
        "get_process_tree",
        "find_socket_owner",
        "expand_window",
    ),
    "correlation_agent": (
        "join_host_network_evidence",
        "build_timeline",
        "retrieve_attack_knowledge",
        "expand_window",
    ),
    "evidence_verifier": ("validate_evidence_ids",),
}
_KNOWN_TOOL_NAMES = frozenset((*_TOOL_MODELS.keys(), "expand_window"))


class BrokerModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ToolAuditRecord(BrokerModel):
    sequence: int = Field(gt=0)
    agent_role: AgentRole
    tool: ToolName
    parameter_sha256: str
    output_sha256: str
    result_count: int = Field(ge=0)
    truncated: bool
    error: str | None = None


class BrokerError(RuntimeError):
    pass


class BrokerSafetyError(BrokerError):
    pass


class BrokerRecoverableError(BrokerError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class BrokerBudgetExceeded(BrokerError):
    pass


def _agent_tool_schema(model: type[ToolModel]) -> dict[str, Any]:
    schema = model.model_json_schema()
    properties = schema.get("properties")
    if isinstance(properties, dict):
        properties.pop("scenario_token", None)
    required = schema.get("required")
    if isinstance(required, list):
        schema["required"] = [value for value in required if value != "scenario_token"]
    return schema


def all_tool_schemas() -> dict[str, dict[str, Any]]:
    result = {name: _agent_tool_schema(model) for name, model in _TOOL_MODELS.items()}
    result["expand_window"] = {
        "type": "object",
        "additionalProperties": False,
        "properties": {},
    }
    return result


def tool_schema_hash() -> str:
    return canonical_hash(
        {
            "schemas": all_tool_schemas(),
            "role_allowlists": _ROLE_TOOLS,
        }
    )


class ToolBroker:
    def __init__(
        self,
        config: traceanchorConfig,
        config_path: Any,
        alert: BlindAlert,
        *,
        run_id: str,
    ) -> None:
        self.config = config
        self.alert = alert
        self.tools = EvidenceTools(config, config_path, run_id=run_id, agent="agent_broker")
        self.window_start_ns = alert.allowed_start_ts_ns
        self.window_end_ns = alert.allowed_end_ts_ns
        self.expanded = False
        self.counts: Counter[str] = Counter()
        self.audit: list[ToolAuditRecord] = []

    @property
    def ledger_version(self) -> str:
        return self.tools.ledger_version

    @property
    def total_calls(self) -> int:
        return sum(self.counts.values())

    def available_schemas(self, role: AgentRole) -> dict[str, dict[str, Any]]:
        schemas = all_tool_schemas()
        return {name: schemas[name] for name in _ROLE_TOOLS[role]}

    def remaining_tool_calls(self, role: AgentRole) -> int:
        role_remaining = (
            self.config.agents.max_tool_calls_per_role - self.counts[role]
        )
        total_remaining = self.config.agents.max_total_tool_calls - self.total_calls
        return max(0, min(role_remaining, total_remaining))

    def rejection_feedback(
        self,
        role: AgentRole,
        name: ToolName,
        error: BrokerRecoverableError,
    ) -> dict[str, Any]:
        schemas = self.available_schemas(role)
        feedback: dict[str, Any] = {
            "ok": False,
            "error": {
                "code": error.code,
                "message": "Tool request rejected without execution; correct the request or terminate.",
            },
            "allowed_start_ts_ns": self.window_start_ns,
            "allowed_end_ts_ns": self.window_end_ns,
            "allowed_tools": schemas,
        }
        if error.code == "ROLE_TOOL_NOT_ALLOWED":
            feedback["requested_tool_schema"] = all_tool_schemas()[name]
        else:
            feedback["arguments_schema"] = schemas[name]
        return feedback

    def _reserve(self, role: AgentRole) -> None:
        if self.total_calls >= self.config.agents.max_total_tool_calls:
            raise BrokerBudgetExceeded("total tool-call budget exhausted")
        if self.counts[role] >= self.config.agents.max_tool_calls_per_role:
            raise BrokerBudgetExceeded(f"tool-call budget exhausted for {role}")
        self.counts[role] += 1

    def _validate_scope(self, arguments: dict[str, Any]) -> dict[str, Any]:
        values = dict(arguments)
        self._validate_scenario_token(values)
        values.pop("scenario_token", None)
        values["scenario_token"] = self.alert.scenario_token
        start = values.get("start_ts_ns")
        end = values.get("end_ts_ns")
        ts = values.get("ts_ns")
        if start is not None and int(start) < self.window_start_ns:
            raise BrokerRecoverableError("OUT_OF_WINDOW")
        if end is not None and int(end) > self.window_end_ns:
            raise BrokerRecoverableError("OUT_OF_WINDOW")
        if ts is not None and not self.window_start_ns <= int(ts) <= self.window_end_ns:
            raise BrokerRecoverableError("OUT_OF_WINDOW")
        return values

    def _validate_scenario_token(self, arguments: dict[str, Any]) -> None:
        supplied_token = arguments.get("scenario_token", self.alert.scenario_token)
        if supplied_token != self.alert.scenario_token:
            raise BrokerSafetyError("cross-scene tool request blocked")

    def _record(
        self,
        role: AgentRole,
        name: ToolName,
        arguments: dict[str, Any],
        result: dict[str, Any] | None,
        error: str | None,
    ) -> None:
        result = result or {}
        self.audit.append(
            ToolAuditRecord(
                sequence=len(self.audit) + 1,
                agent_role=role,
                tool=name,
                parameter_sha256=canonical_hash(arguments),
                output_sha256=canonical_hash(result) if result else "",
                result_count=int(result.get("result_count", 0)),
                truncated=bool(result.get("truncated", False)),
                error=error,
            )
        )

    def _expand_window(self) -> dict[str, Any]:
        if self.expanded:
            raise BrokerRecoverableError("WINDOW_EXPANSION_ALREADY_USED")
        if not self.config.agents.allow_one_window_expansion:
            raise BrokerSafetyError("window expansion is disabled")
        amount = self.config.agents.expansion_seconds_each_side * 1_000_000_000
        self.window_start_ns = max(0, self.window_start_ns - amount)
        self.window_end_ns += amount
        self.expanded = True
        return {
            "tool": "expand_window",
            "scenario_token": self.alert.scenario_token,
            "records": [
                {
                    "allowed_start_ts_ns": self.window_start_ns,
                    "allowed_end_ts_ns": self.window_end_ns,
                }
            ],
            "result_count": 1,
            "truncated": False,
            "next_cursor": None,
            "ledger_version": self.ledger_version,
        }

    def call(
        self,
        role: AgentRole,
        name: ToolName,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        if name not in _KNOWN_TOOL_NAMES:
            raise BrokerSafetyError(f"tool {name} is not allowed for {role}")
        self._reserve(role)
        sanitized: dict[str, Any] = dict(arguments)
        try:
            self._validate_scenario_token(sanitized)
            if name not in _ROLE_TOOLS[role]:
                raise BrokerRecoverableError("ROLE_TOOL_NOT_ALLOWED")
            if name == "expand_window":
                if sanitized:
                    raise BrokerRecoverableError("INVALID_ARGUMENTS")
                result = self._expand_window()
            else:
                try:
                    sanitized = self._validate_scope(sanitized)
                    model = _TOOL_MODELS[name].model_validate(sanitized)
                except (ValidationError, ValueError):
                    raise BrokerRecoverableError("INVALID_ARGUMENTS") from None
                method = getattr(self.tools, name)
                result = method(model).model_dump(mode="json")
        except BrokerRecoverableError as exc:
            self._record(role, name, sanitized, None, exc.code)
            raise
        except (BrokerError, EvidenceToolError, ValidationError, ValueError) as exc:
            self._record(role, name, sanitized, None, str(exc))
            if isinstance(exc, BrokerError):
                raise
            raise BrokerError(str(exc)) from None
        self._record(role, name, sanitized, result, None)
        return result


__all__ = [
    "BrokerBudgetExceeded",
    "BrokerError",
    "BrokerRecoverableError",
    "BrokerSafetyError",
    "ToolAuditRecord",
    "ToolBroker",
    "all_tool_schemas",
    "tool_schema_hash",
]
