from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from traceanchor.ingest.common import seconds_to_ns


def _absolute_ns(value: Any) -> int | None:
    if not isinstance(value, dict) or value.get("absolute") is None:
        return None
    return seconds_to_ns(value["absolute"])


def parse_json_metadata(
    path: Path,
    uid: str,
    token: str,
    family_private: str,
) -> tuple[dict[str, object], dict[str, object], list[dict[str, object]]]:
    with path.open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    time = document.get("time") or {}
    ready_ns = _absolute_ns(time.get("container_ready"))
    warmup_ns = _absolute_ns(time.get("warmup_end"))
    recording_seconds = float(document.get("recording_time") or 0.0)
    # LID-DS recording_time starts after warmup; container_ready is setup time.
    start_ns = warmup_ns if warmup_ns is not None else ready_ns
    if start_ns is None:
        anchors = time.get("exploit") or []
        start_ns = min(
            (seconds_to_ns(item["absolute"]) for item in anchors if "absolute" in item),
            default=0,
        )
    end_ns = start_ns + seconds_to_ns(recording_seconds)
    public = {
        "scenario_token": token,
        "start_ts_ns": start_ns,
        "end_ts_ns": end_ns,
        "recording_time_seconds": recording_seconds,
        "network_available": True,
        "host_available": True,
        "resource_available": True,
        "quality_status": "ok",
    }
    private = {
        "scenario_uid": uid,
        "scenario_token": token,
        "family_private": family_private,
        "exploit": bool(document.get("exploit", False)),
        "exploit_name": document.get("exploit_name"),
        "image": document.get("image"),
        "container_ready_ts_ns": ready_ns,
        "warmup_end_ts_ns": warmup_ns,
        "recording_time_seconds": recording_seconds,
        "containers_json": json.dumps(
            document.get("container") or [], sort_keys=True, separators=(",", ":")
        ),
    }
    anchors_out: list[dict[str, object]] = []
    for index, anchor in enumerate(time.get("exploit") or []):
        if anchor.get("absolute") is None:
            continue
        anchors_out.append(
            {
                "anchor_id": f"anchor:{token}:{index + 1}",
                "scenario_uid": uid,
                "scenario_token": token,
                "ts_ns": seconds_to_ns(anchor["absolute"]),
                "raw_timestamp": str(anchor["absolute"]),
                "name_private": anchor.get("name"),
                "source_private": anchor.get("source"),
            }
        )
    return public, private, anchors_out
