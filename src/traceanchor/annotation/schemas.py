from __future__ import annotations

import re
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


EvidenceId = str
EntityId = str

_EVIDENCE_ID = re.compile(
    r"^(?:sc|pcap|res):(?P<token>tw_[0-9a-f]{24}):[^:]+(?::[0-9]+)?$"
)


class AnnotationModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AutomationProvenance(AnnotationModel):
    candidates_only: Literal[True] = True
    human_selection_required: Literal[True] = True
    automatically_selected_gold: Literal[False] = False


class AnnotationMetadata(AnnotationModel):
    annotator_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{1,63}$")
    annotation_mode: Literal["independent", "adjudicated"]
    started_at: datetime
    completed_at: datetime | None = None
    human_verified: bool = False
    source_annotation_ids: list[str] = Field(default_factory=list, max_length=10)
    notes: str | None = Field(default=None, max_length=4000)


class RootCauseEntity(AnnotationModel):
    entity_id: EntityId = Field(pattern=r"^ent_[0-9a-f]{24}$")
    entity_type: Literal["connection", "socket", "process", "thread", "file", "other"]
    rationale: str = Field(min_length=3, max_length=2000)
    evidence_ids: list[EvidenceId] = Field(min_length=1, max_length=50)
    confidence: float = Field(ge=0.0, le=1.0)


class IncidentStep(AnnotationModel):
    step_id: str = Field(pattern=r"^step_[0-9]{2,3}$")
    step_type: Literal[
        "external_connection",
        "socket_accept",
        "request_read",
        "process_spawn",
        "execution",
        "file_read",
        "file_write",
        "outbound_connection",
        "discovery",
        "privilege_change",
        "persistence",
        "other_observed",
    ]
    summary: str = Field(min_length=3, max_length=2000)
    start_ts_ns: int = Field(ge=0)
    end_ts_ns: int = Field(ge=0)
    evidence_ids: list[EvidenceId] = Field(min_length=1, max_length=100)
    confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def ordered_times(self) -> "IncidentStep":
        if self.end_ts_ns < self.start_ts_ns:
            raise ValueError("step end precedes start")
        return self


class ProvenanceEdge(AnnotationModel):
    source_evidence_id: EvidenceId
    target_evidence_id: EvidenceId
    relation_type: Literal["provenance", "precedes", "correlates_with"]
    basis_evidence_ids: list[EvidenceId] = Field(min_length=1, max_length=50)
    rationale: str = Field(min_length=3, max_length=2000)
    confidence: float = Field(ge=0.0, le=1.0)


class PossiblyCausalEdge(AnnotationModel):
    source_evidence_id: EvidenceId
    target_evidence_id: EvidenceId
    relation_type: Literal["possibly_causes"] = "possibly_causes"
    basis_evidence_ids: list[EvidenceId] = Field(min_length=1, max_length=50)
    rationale: str = Field(min_length=3, max_length=2000)
    confidence: float = Field(ge=0.0, le=1.0)


class AttackTechnique(AnnotationModel):
    technique_id: str = Field(pattern=r"^T[0-9]{4}(?:\.[0-9]{3})?$")
    name: str = Field(min_length=2, max_length=200)
    step_ids: list[str] = Field(min_length=1, max_length=50)
    evidence_ids: list[EvidenceId] = Field(min_length=1, max_length=100)
    mapping_reason: str = Field(min_length=3, max_length=2000)
    confidence: float = Field(ge=0.0, le=1.0)
    candidate_technique_ids: list[str] = Field(default_factory=list, max_length=20)


class EvidenceGap(AnnotationModel):
    gap_id: str = Field(pattern=r"^gap_[0-9]{2,3}$")
    description: str = Field(min_length=3, max_length=2000)
    impact: Literal["root_cause", "step", "edge", "technique", "other"]


class GoldAnnotation(AnnotationModel):
    schema_version: Literal[1] = 1
    status: Literal["draft", "completed"]
    annotation_id: str = Field(pattern=r"^ann:INC-[0-9]{3}:[A-Za-z0-9_.-]+$")
    incident_id: str = Field(pattern=r"^INC-[0-9]{3}$")
    private_family: str = Field(min_length=3, max_length=200)
    scenario_token: str = Field(pattern=r"^tw_[0-9a-f]{24}$")
    agent_split: Literal["development", "test"]
    investigation_start_ts_ns: int = Field(ge=0)
    investigation_end_ts_ns: int = Field(ge=0)
    anchor_times_ns: list[int] = Field(min_length=1, max_length=100)
    root_cause_ambiguous: bool = False
    root_cause_entities: list[RootCauseEntity] = Field(default_factory=list, max_length=50)
    core_evidence_ids: list[EvidenceId] = Field(default_factory=list, max_length=200)
    supporting_evidence_ids: list[EvidenceId] = Field(default_factory=list, max_length=500)
    steps: list[IncidentStep] = Field(default_factory=list, max_length=100)
    provenance_edges: list[ProvenanceEdge] = Field(default_factory=list, max_length=500)
    possibly_causal_edges: list[PossiblyCausalEdge] = Field(
        default_factory=list, max_length=200
    )
    attack_techniques: list[AttackTechnique] = Field(default_factory=list, max_length=100)
    known_evidence_gaps: list[EvidenceGap] = Field(default_factory=list, max_length=100)
    annotation_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    automation_provenance: AutomationProvenance = Field(
        default_factory=AutomationProvenance
    )
    metadata: AnnotationMetadata

    @model_validator(mode="after")
    def validate_annotation_graph(self) -> "GoldAnnotation":
        if (
            self.metadata.completed_at is not None
            and self.metadata.completed_at < self.metadata.started_at
        ):
            raise ValueError("annotation completion precedes start")
        if self.investigation_end_ts_ns < self.investigation_start_ts_ns:
            raise ValueError("investigation end precedes start")
        if any(
            value < self.investigation_start_ts_ns
            or value > self.investigation_end_ts_ns
            for value in self.anchor_times_ns
        ):
            raise ValueError("anchor lies outside investigation range")
        if len(self.anchor_times_ns) != len(set(self.anchor_times_ns)):
            raise ValueError("duplicate anchor time")

        core = set(self.core_evidence_ids)
        supporting = set(self.supporting_evidence_ids)
        if len(core) != len(self.core_evidence_ids):
            raise ValueError("duplicate core Evidence ID")
        if len(supporting) != len(self.supporting_evidence_ids):
            raise ValueError("duplicate supporting Evidence ID")
        if core.intersection(supporting):
            raise ValueError("core and supporting evidence must be disjoint")
        evidence = core.union(supporting)
        for evidence_id in evidence:
            match = _EVIDENCE_ID.match(evidence_id)
            if match is None:
                raise ValueError(f"malformed Evidence ID: {evidence_id}")
            if match.group("token") != self.scenario_token:
                raise ValueError(f"cross-scene Evidence ID: {evidence_id}")

        step_ids = [step.step_id for step in self.steps]
        if len(step_ids) != len(set(step_ids)):
            raise ValueError("duplicate step_id")
        for step in self.steps:
            if len(step.evidence_ids) != len(set(step.evidence_ids)):
                raise ValueError(f"step {step.step_id} repeats Evidence IDs")
            if not set(step.evidence_ids).issubset(evidence):
                raise ValueError(f"step {step.step_id} references unclassified evidence")
            if step.start_ts_ns < self.investigation_start_ts_ns or step.end_ts_ns > self.investigation_end_ts_ns:
                raise ValueError(f"step {step.step_id} lies outside investigation range")
        for entity in self.root_cause_entities:
            if not set(entity.evidence_ids).issubset(evidence):
                raise ValueError("root cause references unclassified evidence")
        for edge in [*self.provenance_edges, *self.possibly_causal_edges]:
            referenced = {
                edge.source_evidence_id,
                edge.target_evidence_id,
                *edge.basis_evidence_ids,
            }
            if not referenced.issubset(evidence):
                raise ValueError("edge references unclassified evidence")
        known_steps = set(step_ids)
        for technique in self.attack_techniques:
            if not set(technique.step_ids).issubset(known_steps):
                raise ValueError("ATT&CK mapping references unknown step")
            if not set(technique.evidence_ids).issubset(evidence):
                raise ValueError("ATT&CK mapping references unclassified evidence")

        if self.status == "completed":
            if not self.metadata.human_verified:
                raise ValueError("completed annotation must be human verified")
            if self.metadata.completed_at is None:
                raise ValueError("completed annotation requires completed_at")
            if not self.root_cause_entities or not core or not self.steps:
                raise ValueError("completed annotation lacks root cause, core evidence, or steps")
            if self.annotation_confidence is None:
                raise ValueError("completed annotation requires confidence")
            if self.metadata.annotation_mode == "independent" and (
                self.metadata.source_annotation_ids
            ):
                raise ValueError("independent annotation cannot declare source annotations")
            if self.metadata.annotation_mode == "adjudicated":
                sources = self.metadata.source_annotation_ids
                if len(sources) != 2 or len(set(sources)) != 2:
                    raise ValueError(
                        "adjudication requires exactly two source annotations with distinct IDs"
                    )
        return self


__all__ = ["GoldAnnotation"]
