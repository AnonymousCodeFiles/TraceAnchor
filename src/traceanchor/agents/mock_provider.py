from __future__ import annotations

import json
import hashlib
import re
from typing import Any

from traceanchor.agents.provider import (
    LLMProvider,
    ProviderRequest,
    ProviderResponse,
    ProviderUsage,
)


_EVIDENCE_ID = re.compile(r"^(?:sc|pcap|res):tw_[0-9a-f]{24}:[^:]+(?::[0-9]+)?$")


def _state(request: ProviderRequest) -> dict[str, Any]:
    try:
        value = json.loads(request.messages[-1].content)
    except (json.JSONDecodeError, IndexError):
        return {}
    return value if isinstance(value, dict) else {}


def _tool_names(state: dict[str, Any]) -> set[str]:
    names = set()
    for item in state.get("tool_transcript", []):
        if isinstance(item, dict) and isinstance(item.get("tool"), str):
            names.add(item["tool"])
    return names


def _walk_evidence(value: Any) -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(item, str) and "evidence_id" in str(key) and _EVIDENCE_ID.match(item):
                found.append(item)
            else:
                found.extend(_walk_evidence(item))
    elif isinstance(value, list):
        for item in value:
            found.extend(_walk_evidence(item))
    return list(dict.fromkeys(found))


def _records(state: dict[str, Any], tool: str) -> list[dict[str, Any]]:
    for item in reversed(state.get("tool_transcript", [])):
        if isinstance(item, dict) and item.get("tool") == tool:
            result = item.get("result") or {}
            rows = result.get("records") if isinstance(result, dict) else None
            if isinstance(rows, list):
                return [row for row in rows if isinstance(row, dict)]
    return []


def _investigator_action(request: ProviderRequest, state: dict[str, Any]) -> dict[str, Any]:
    names = _tool_names(state)
    start = int(state.get("allowed_start_ts_ns", 0))
    end = int(state.get("allowed_end_ts_ns", start))
    common = {"start_ts_ns": start, "end_ts_ns": end, "max_records": 40}
    if request.agent_role == "network_investigator":
        if "list_connections" not in names:
            return {"action": "tool", "tool_call": {"name": "list_connections", "arguments": common}}
        if "get_packet_metadata" not in names:
            return {"action": "tool", "tool_call": {"name": "get_packet_metadata", "arguments": common}}
        evidence = _walk_evidence(state.get("tool_transcript", []))[:20]
        if not evidence:
            return {"action": "abstain", "reason": "No network evidence in the allowed window."}
        return {
            "action": "finish",
            "findings": {
                "facts": [{
                    "statement": "Network activity was observed in the allowed alert window.",
                    "evidence_ids": evidence,
                    "confidence": 0.8,
                    "relation_type": "observed",
                    "untrusted_text_present": False,
                }],
                "hypotheses": [],
                "cross_modal_questions": [
                    "Which host process or syscall activity aligns with the observed network evidence?"
                ],
                "abstain": False,
            },
        }
    if request.agent_role == "host_investigator":
        if "list_syscalls" not in names:
            return {"action": "tool", "tool_call": {"name": "list_syscalls", "arguments": common}}
        if "get_file_activity" not in names:
            return {"action": "tool", "tool_call": {"name": "get_file_activity", "arguments": common}}
        evidence = _walk_evidence(state.get("tool_transcript", []))[:20]
        if not evidence:
            return {"action": "abstain", "reason": "No host evidence in the allowed window."}
        return {
            "action": "finish",
            "findings": {
                "facts": [{
                    "statement": "Host activity was observed in the allowed alert window.",
                    "evidence_ids": evidence,
                    "confidence": 0.8,
                    "relation_type": "observed",
                    "untrusted_text_present": False,
                }],
                "hypotheses": [],
                "cross_modal_questions": [
                    "Which network evidence aligns with the observed host activity?"
                ],
                "abstain": False,
            },
        }
    return {"action": "abstain", "reason": "Mock investigator received an unsupported role."}


def _event_rows(state: dict[str, Any]) -> list[dict[str, Any]]:
    timeline = _records(state, "build_timeline")
    if timeline:
        return timeline
    rows = []
    for tool in ("get_packet_metadata", "list_syscalls", "get_file_activity"):
        rows.extend(_records(state, tool))
    return rows


def _correlation_action(state: dict[str, Any]) -> dict[str, Any]:
    names = _tool_names(state)
    start = int(state.get("allowed_start_ts_ns", 0))
    end = int(state.get("allowed_end_ts_ns", start))
    common = {"start_ts_ns": start, "end_ts_ns": end, "max_records": 40}
    if "join_host_network_evidence" not in names:
        return {
            "action": "tool",
            "tool_call": {"name": "join_host_network_evidence", "arguments": common},
        }
    evidence = _walk_evidence(
        [state.get("investigator_findings", []), state.get("tool_transcript", [])]
    )[:40]
    if evidence and "build_timeline" not in names:
        return {
            "action": "tool",
            "tool_call": {"name": "build_timeline", "arguments": {"evidence_ids": evidence}},
        }
    if "retrieve_attack_knowledge" not in names:
        return {
            "action": "tool",
            "tool_call": {
                "name": "retrieve_attack_knowledge",
                "arguments": {"observed_behavior": "network connection host process file activity", "max_records": 5},
            },
        }
    rows = _event_rows(state)
    steps = []
    for row in rows[:5]:
        evidence_id = row.get("evidence_id")
        if not isinstance(evidence_id, str) or not _EVIDENCE_ID.match(evidence_id):
            continue
        ts = int(row.get("ts_ns", start))
        event_type = str(row.get("event_type") or row.get("syscall") or "telemetry")
        steps.append({
            "step_id": f"S{len(steps) + 1}",
            "start_ts": ts,
            "end_ts": ts,
            "claim": f"Observed {event_type} activity in the allowed investigation window.",
            "relation_type": "observed",
            "evidence_ids": [evidence_id],
            "confidence": 0.8,
        })
    if not steps and evidence:
        steps = [{
            "step_id": "S1",
            "start_ts": start,
            "end_ts": end,
            "claim": "Observed telemetry activity in the allowed investigation window.",
            "relation_type": "observed",
            "evidence_ids": [evidence[0]],
            "confidence": 0.6,
        }]
    if not steps:
        output = {
            "incident_summary": "insufficient_evidence",
            "root_cause_candidates": [],
            "steps": [],
            "attack_techniques": [],
            "evidence_gaps": ["No evidence-backed timeline could be constructed."],
            "response_considerations": [],
            "verifier": {
                "decision": "insufficient_evidence",
                "removed_claims": [],
                "unsupported_claim_rate": 1.0,
            },
        }
    else:
        output = {
            "incident_summary": "Evidence-backed host or network activity was observed.",
            "root_cause_candidates": [],
            "steps": steps,
            "attack_techniques": [],
            "evidence_gaps": ["A root-cause entity was not established by the available evidence."],
            "response_considerations": [],
            "verifier": {
                "decision": "partially_supported",
                "removed_claims": [],
                "unsupported_claim_rate": 0.0,
            },
        }
    return {"action": "finish", "final_output": output}


class MockProvider(LLMProvider):
    provider_name = "mock"
    model = "traceanchor-deterministic-mock-v1"

    def complete_once(self, request: ProviderRequest) -> ProviderResponse:
        state = _state(request)
        if request.task_type == "investigator_action":
            value = _investigator_action(request, state)
        elif request.task_type == "correlation_action":
            value = _correlation_action(state)
        else:
            raw = state.get("invalid_output")
            value = raw if isinstance(raw, dict) else {
                "action": "abstain",
                "reason": "The mock format repair had no structured content to preserve.",
            }
        content = json.dumps(value, ensure_ascii=True, sort_keys=True)
        input_tokens = sum(len(item.content.split()) for item in request.messages)
        return ProviderResponse(
            content=content,
            usage=ProviderUsage(
                input_tokens=input_tokens,
                output_tokens=len(content.split()),
            ),
            provider_request_id="mock_" + hashlib.sha256(content.encode("utf-8")).hexdigest()[:24],
        )


__all__ = ["MockProvider"]
