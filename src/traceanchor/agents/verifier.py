from __future__ import annotations

import re
from typing import Any

from traceanchor.agents.broker import BrokerError, ToolBroker
from traceanchor.agents.schemas import FinalInvestigation


_INJECTION_OR_EXECUTION = re.compile(
    r"(?:ignore (?:all |the )?(?:previous|prior) instructions|system prompt|"
    r"execute (?:this|the following)|run (?:this )?command)",
    re.IGNORECASE,
)
_DEFINITIVE_CAUSAL = re.compile(
    r"\b(?:caused|proves?|definitely|certainly|responsible for)\b", re.IGNORECASE
)


def _walk_rows(value: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if isinstance(value, dict):
        rows.append(value)
        for nested in value.values():
            rows.extend(_walk_rows(nested))
    elif isinstance(value, list):
        for nested in value:
            rows.extend(_walk_rows(nested))
    return rows


def _transcript_entities(transcript: list[dict[str, Any]]) -> set[str]:
    values = set()
    for row in _walk_rows(transcript):
        for key in ("entity_id", "source_entity_id", "target_entity_id"):
            value = row.get(key)
            if isinstance(value, str) and value.startswith("ent_"):
                values.add(value)
    return values


def _retrieved_techniques(transcript: list[dict[str, Any]]) -> set[str]:
    values = set()
    for item in transcript:
        if item.get("tool") != "retrieve_attack_knowledge":
            continue
        for row in _walk_rows(item.get("result")):
            value = row.get("technique_id")
            if isinstance(value, str):
                values.add(value)
    return values


def _insufficient(
    draft: FinalInvestigation,
    removed: list[str],
    gap: str,
    unsupported_rate: float,
) -> FinalInvestigation:
    gaps = list(dict.fromkeys([*draft.evidence_gaps, gap]))
    return FinalInvestigation.model_validate(
        {
            "incident_summary": "insufficient_evidence",
            "root_cause_candidates": [],
            "steps": [],
            "attack_techniques": [],
            "evidence_gaps": gaps,
            "response_considerations": [],
            "verifier": {
                "decision": "insufficient_evidence",
                "removed_claims": removed,
                "unsupported_claim_rate": unsupported_rate,
            },
        }
    )


def verify_investigation(
    draft: FinalInvestigation,
    broker: ToolBroker,
    tool_transcript: list[dict[str, Any]],
) -> FinalInvestigation:
    claims_before = (
        len(draft.root_cause_candidates)
        + len(draft.steps)
        + len(draft.attack_techniques)
        + len(draft.response_considerations)
    )
    evidence_ids = list(
        dict.fromkeys(
            evidence_id
            for group in (
                draft.root_cause_candidates,
                draft.steps,
                draft.attack_techniques,
                draft.response_considerations,
            )
            for item in group
            for evidence_id in item.evidence_ids
        )
    )
    if not evidence_ids:
        return _insufficient(
            draft,
            [],
            "The verifier received no cited Evidence IDs.",
            1.0 if claims_before else 0.0,
        )
    try:
        validated = broker.call(
            "evidence_verifier",
            "validate_evidence_ids",
            {"evidence_ids": evidence_ids},
        )
    except BrokerError:
        return _insufficient(
            draft,
            ["All claims: Evidence ID validation failed."],
            "Evidence ID validation failed.",
            1.0,
        )
    valid_rows = {
        str(row["evidence_id"]): row
        for row in validated["records"]
        if row.get("valid") is True
    }
    observed_entities = _transcript_entities(tool_transcript)
    retrieved_techniques = _retrieved_techniques(tool_transcript)
    removed: list[str] = []

    roots = []
    for item in draft.root_cause_candidates:
        if item.entity_id not in observed_entities:
            removed.append(f"Root cause {item.entity_id}: entity was not returned by a tool.")
        elif not all(value in valid_rows for value in item.evidence_ids):
            removed.append(f"Root cause {item.entity_id}: invalid Evidence ID.")
        else:
            roots.append(item)
    roots = [item.model_copy(update={"rank": index + 1}) for index, item in enumerate(roots)]

    steps = []
    for item in draft.steps:
        rows = [valid_rows.get(value) for value in item.evidence_ids]
        if any(row is None for row in rows):
            removed.append(f"Step {item.step_id}: invalid Evidence ID.")
            continue
        timestamps = [int(row["ts_ns"]) for row in rows if row.get("ts_ns") is not None]
        if not timestamps or any(not item.start_ts <= value <= item.end_ts for value in timestamps):
            removed.append(f"Step {item.step_id}: evidence lies outside the claimed time range.")
            continue
        if _INJECTION_OR_EXECUTION.search(item.claim):
            removed.append(f"Step {item.step_id}: instruction-like evidence text was rejected.")
            continue
        if item.relation_type == "possibly_causal" and _DEFINITIVE_CAUSAL.search(item.claim):
            removed.append(f"Step {item.step_id}: definitive causal wording lacked support.")
            continue
        steps.append(item)

    techniques = []
    for item in draft.attack_techniques:
        if item.technique_id not in retrieved_techniques:
            removed.append(f"ATT&CK {item.technique_id}: technique was not retrieved.")
        elif not all(value in valid_rows for value in item.evidence_ids):
            removed.append(f"ATT&CK {item.technique_id}: invalid Evidence ID.")
        else:
            techniques.append(item)

    responses = []
    for index, item in enumerate(draft.response_considerations, start=1):
        if _INJECTION_OR_EXECUTION.search(item.suggestion):
            removed.append(f"Response consideration {index}: instruction-like text was rejected.")
        elif not all(value in valid_rows for value in item.evidence_ids):
            removed.append(f"Response consideration {index}: invalid Evidence ID.")
        else:
            responses.append(item)

    unsupported_rate = len(removed) / max(claims_before, 1)
    if not steps:
        return _insufficient(
            draft,
            removed,
            "No evidence-backed incident steps remained after verification.",
            unsupported_rate,
        )
    decision = "supported" if not removed and roots else "partially_supported"
    gaps = list(draft.evidence_gaps)
    if removed:
        gaps.append("The verifier removed one or more unsupported claims.")
    return FinalInvestigation.model_validate(
        {
            **draft.model_dump(mode="json"),
            "root_cause_candidates": [item.model_dump(mode="json") for item in roots],
            "steps": [item.model_dump(mode="json") for item in steps],
            "attack_techniques": [item.model_dump(mode="json") for item in techniques],
            "evidence_gaps": list(dict.fromkeys(gaps)),
            "response_considerations": [item.model_dump(mode="json") for item in responses],
            "verifier": {
                "decision": decision,
                "removed_claims": removed,
                "unsupported_claim_rate": unsupported_rate,
            },
        }
    )


__all__ = ["verify_investigation"]
