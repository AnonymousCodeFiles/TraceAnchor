from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator


EVIDENCE_ID_PATTERN = r"^(?:sc|pcap|res):tw_[0-9a-f]{24}:[^:]+(?::[0-9]+)?$"
ENTITY_ID_PATTERN = r"^ent_[0-9a-f]{24}$"
EvidenceId = Annotated[str, Field(pattern=EVIDENCE_ID_PATTERN)]


def _unique_evidence_ids(values: list[str]) -> list[str]:
    if len(values) != len(set(values)):
        raise ValueError("Evidence IDs must be unique")
    return values


EvidenceIds = Annotated[
    list[EvidenceId],
    Field(min_length=1, max_length=200, json_schema_extra={"uniqueItems": True}),
    AfterValidator(_unique_evidence_ids),
]
OptionalEvidenceIds = Annotated[
    list[EvidenceId],
    Field(max_length=200, json_schema_extra={"uniqueItems": True}),
    AfterValidator(_unique_evidence_ids),
]

AgentRole = Literal[
    "orchestrator",
    "network_investigator",
    "host_investigator",
    "correlation_agent",
    "evidence_verifier",
]
ToolName = Literal[
    "get_alert_context",
    "list_connections",
    "get_packet_metadata",
    "find_socket_owner",
    "get_process_tree",
    "list_syscalls",
    "get_file_activity",
    "join_host_network_evidence",
    "build_timeline",
    "retrieve_attack_knowledge",
    "validate_evidence_ids",
    "expand_window",
]


class AgentModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BlindAlert(AgentModel):
    alert_id: str = Field(pattern=r"^al_[0-9a-f]{24}$")
    scenario_token: str = Field(pattern=r"^tw_[0-9a-f]{24}$")
    channels: list[Literal["network", "host"]] = Field(min_length=1, max_length=2)
    peak_score_network: float | None = Field(default=None, ge=0.0, le=1.0)
    peak_score_host: float | None = Field(default=None, ge=0.0, le=1.0)
    allowed_start_ts_ns: int = Field(ge=0)
    allowed_end_ts_ns: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_window(self) -> "BlindAlert":
        if self.allowed_end_ts_ns < self.allowed_start_ts_ns:
            raise ValueError("alert window is reversed")
        if len(self.channels) != len(set(self.channels)):
            raise ValueError("alert channels must be unique")
        if "network" in self.channels and self.peak_score_network is None and len(self.channels) > 1:
            raise ValueError("dual-channel alert is missing the network peak score")
        if "host" in self.channels and self.peak_score_host is None and len(self.channels) > 1:
            raise ValueError("dual-channel alert is missing the host peak score")
        return self


class InvestigatorFact(AgentModel):
    statement: str = Field(min_length=1, max_length=1000)
    evidence_ids: EvidenceIds
    confidence: float = Field(ge=0.0, le=1.0)
    relation_type: Literal["observed", "correlated"]
    untrusted_text_present: bool

    @model_validator(mode="after")
    def validate_evidence(self) -> "InvestigatorFact":
        if len(self.evidence_ids) != len(set(self.evidence_ids)):
            raise ValueError("fact repeats Evidence IDs")
        return self


class InvestigatorHypothesis(AgentModel):
    statement: str = Field(min_length=1, max_length=1000)
    supporting_evidence_ids: OptionalEvidenceIds = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list, max_length=50)
    confidence: float = Field(ge=0.0, le=1.0)


class InvestigatorFindings(AgentModel):
    facts: list[InvestigatorFact] = Field(default_factory=list, max_length=100)
    hypotheses: list[InvestigatorHypothesis] = Field(default_factory=list, max_length=50)
    cross_modal_questions: list[str] = Field(default_factory=list, max_length=20)
    abstain: bool = False

    @model_validator(mode="after")
    def require_content_or_abstention(self) -> "InvestigatorFindings":
        if not self.abstain and not (self.facts or self.hypotheses):
            raise ValueError("findings require content or abstention")
        return self


class RootCauseCandidate(AgentModel):
    entity_id: str = Field(pattern=ENTITY_ID_PATTERN)
    rank: int = Field(gt=0)
    evidence_ids: EvidenceIds
    confidence: float = Field(ge=0.0, le=1.0)


class IncidentStep(AgentModel):
    step_id: str = Field(pattern=r"^S[1-9][0-9]*$")
    start_ts: int = Field(ge=0)
    end_ts: int = Field(ge=0)
    claim: str = Field(min_length=1, max_length=1500)
    relation_type: Literal["observed", "correlated", "possibly_causal"]
    evidence_ids: EvidenceIds
    confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_step(self) -> "IncidentStep":
        if self.end_ts < self.start_ts:
            raise ValueError("step end precedes start")
        if len(self.evidence_ids) != len(set(self.evidence_ids)):
            raise ValueError("step repeats Evidence IDs")
        return self


class AttackTechnique(AgentModel):
    technique_id: str = Field(pattern=r"^T[0-9]{4}(?:\.[0-9]{3})?$")
    evidence_ids: EvidenceIds
    confidence: float = Field(ge=0.0, le=1.0)


class ResponseConsideration(AgentModel):
    suggestion: str = Field(min_length=1, max_length=1000)
    evidence_ids: EvidenceIds
    risk: str = Field(min_length=1, max_length=500)
    execute: Literal[False]


class VerifierSummary(AgentModel):
    decision: Literal["supported", "partially_supported", "insufficient_evidence"]
    removed_claims: list[str] = Field(default_factory=list, max_length=200)
    unsupported_claim_rate: float = Field(ge=0.0, le=1.0)


class FinalInvestigation(AgentModel):
    incident_summary: str = Field(min_length=1, max_length=2000)
    root_cause_candidates: list[RootCauseCandidate] = Field(default_factory=list, max_length=20)
    steps: list[IncidentStep] = Field(default_factory=list, max_length=100)
    attack_techniques: list[AttackTechnique] = Field(default_factory=list, max_length=50)
    evidence_gaps: list[str] = Field(default_factory=list, max_length=100)
    response_considerations: list[ResponseConsideration] = Field(
        default_factory=list, max_length=50
    )
    verifier: VerifierSummary

    @model_validator(mode="after")
    def validate_claim_graph(self) -> "FinalInvestigation":
        step_ids = [item.step_id for item in self.steps]
        if len(step_ids) != len(set(step_ids)):
            raise ValueError("duplicate step IDs")
        ranks = [item.rank for item in self.root_cause_candidates]
        if len(ranks) != len(set(ranks)):
            raise ValueError("duplicate root-cause ranks")
        if ranks and sorted(ranks) != list(range(1, len(ranks) + 1)):
            raise ValueError("root-cause ranks must be contiguous from one")
        if self.verifier.decision == "supported" and not self.steps:
            raise ValueError("supported output requires at least one evidence-backed step")
        return self


class ToolCall(AgentModel):
    name: ToolName
    arguments: dict[str, Any] = Field(default_factory=dict)


class AgentAction(AgentModel):
    action: Literal["tool", "finish", "abstain"]
    tool_call: ToolCall | None = None
    findings: InvestigatorFindings | None = None
    final_output: FinalInvestigation | None = None
    reason: str | None = Field(default=None, max_length=500)

    @model_validator(mode="after")
    def validate_action(self) -> "AgentAction":
        if self.action == "tool":
            if self.tool_call is None or self.findings is not None or self.final_output is not None:
                raise ValueError("tool action requires only tool_call")
        elif self.action == "finish":
            if self.tool_call is not None or (self.findings is None) == (self.final_output is None):
                raise ValueError("finish requires exactly one structured output")
        elif self.tool_call is not None or self.findings is not None or self.final_output is not None:
            raise ValueError("abstain action cannot include tool or output")
        return self


class InvestigatorAction(AgentModel):
    action: Literal["tool", "finish", "abstain"]
    tool_call: ToolCall | None = None
    findings: InvestigatorFindings | None = None
    reason: str | None = Field(default=None, max_length=500)

    @model_validator(mode="after")
    def validate_action(self) -> "InvestigatorAction":
        if self.action == "tool":
            if self.tool_call is None or self.findings is not None:
                raise ValueError("tool action requires only tool_call")
        elif self.action == "finish":
            if self.tool_call is not None or self.findings is None:
                raise ValueError("investigator finish requires findings")
        elif self.tool_call is not None or self.findings is not None:
            raise ValueError("abstain action cannot include tool or output")
        return self

    def to_agent_action(self) -> AgentAction:
        return AgentAction(
            action=self.action,
            tool_call=self.tool_call,
            findings=self.findings,
            reason=self.reason,
        )


class CorrelationAction(AgentModel):
    action: Literal["tool", "finish", "abstain"]
    tool_call: ToolCall | None = None
    final_output: FinalInvestigation | None = None
    reason: str | None = Field(default=None, max_length=500)

    @model_validator(mode="after")
    def validate_action(self) -> "CorrelationAction":
        if self.action == "tool":
            if self.tool_call is None or self.final_output is not None:
                raise ValueError("tool action requires only tool_call")
        elif self.action == "finish":
            if self.tool_call is not None or self.final_output is None:
                raise ValueError("correlation finish requires final_output")
        elif self.tool_call is not None or self.final_output is not None:
            raise ValueError("abstain action cannot include tool or output")
        return self

    def to_agent_action(self) -> AgentAction:
        return AgentAction(
            action=self.action,
            tool_call=self.tool_call,
            final_output=self.final_output,
            reason=self.reason,
        )


__all__ = [
    "AgentAction",
    "AgentRole",
    "BlindAlert",
    "CorrelationAction",
    "FinalInvestigation",
    "InvestigatorAction",
    "InvestigatorFindings",
    "ToolCall",
    "ToolName",
]
