from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterator

from traceanchor.ingest.common import seconds_to_ns


RESOURCE_FIELDS = (
    "cpu_usage",
    "memory_usage",
    "network_received",
    "network_send",
    "storage_read",
    "storage_written",
)


def _nullable_float(value: str | None) -> float | None:
    if value is None or value.strip().upper() in {"", "NULL", "NAN"}:
        return None
    return float(value)


def iter_resources(
    path: Path,
    token: str,
    bucket_seconds: int = 60,
) -> Iterator[dict[str, object]]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        for row_no, raw in enumerate(reader, start=1):
            timestamp = raw.get("timestamp")
            try:
                if timestamp is None:
                    raise ValueError("missing timestamp")
                ts_ns = seconds_to_ns(timestamp)
                values = {name: _nullable_float(raw.get(name)) for name in RESOURCE_FIELDS}
                missing_mask = sum(
                    (1 << index) for index, name in enumerate(RESOURCE_FIELDS) if values[name] is None
                )
                status = "ok"
            except ValueError:
                ts_ns = 0
                values = {name: None for name in RESOURCE_FIELDS}
                missing_mask = (1 << len(RESOURCE_FIELDS)) - 1
                status = "malformed"
            yield {
                "evidence_id": f"res:{token}:{row_no}",
                "scenario_token": token,
                "row_no": row_no,
                "ts_ns": ts_ns,
                "raw_timestamp": timestamp,
                **values,
                "missing_mask": missing_mask,
                "time_bucket": ts_ns // (bucket_seconds * 1_000_000_000),
                "parse_status": status,
            }

