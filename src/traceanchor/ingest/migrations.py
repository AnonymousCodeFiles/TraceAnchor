from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq

from traceanchor.config import TraceAnchorConfig
from traceanchor.ingest.common import atomic_write_json, atomic_write_parquet, sha256_file
from traceanchor.ingest.schemas import SCENARIO_PUBLIC_SCHEMA


PUBLIC_INTERVAL_VERSION = 2


def migrate_public_intervals(
    config: TraceAnchorConfig,
    config_path: Path,
) -> dict[str, object]:
    resolved = config.resolved_dict(config_path)
    artifacts = Path(resolved["paths"]["artifacts_dir"])
    markers_root = Path(resolved["paths"]["completion_markers_dir"]) / "ingest"
    migrated = []
    skipped = []
    for marker_path in sorted(markers_root.glob("*.done")):
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        token = str(marker["scenario_token"])
        if marker.get("public_interval_version") == PUBLIC_INTERVAL_VERSION:
            skipped.append(token)
            continue
        private_path = Path(resolved["paths"]["evaluator_dir"]) / token / "scenario_private.parquet"
        public_path = Path(resolved["paths"]["parquet_dir"]) / token / "scenario_public.parquet"
        private = pq.read_table(private_path).to_pylist()[0]
        old_public = pq.read_table(public_path).to_pylist()[0]
        start_ns = private["warmup_end_ts_ns"] or private["container_ready_ts_ns"] or 0
        end_ns = start_ns + round(float(private["recording_time_seconds"]) * 1_000_000_000)
        old_public["start_ts_ns"] = start_ns
        old_public["end_ts_ns"] = end_ns
        atomic_write_parquet(public_path, SCENARIO_PUBLIC_SCHEMA, [old_public])
        relative_key = str(public_path.relative_to(artifacts))
        marker["outputs"][relative_key] = sha256_file(public_path)
        marker["public_interval_version"] = PUBLIC_INTERVAL_VERSION
        marker.setdefault("migrations", []).append(
            {
                "name": "warmup_start_interval_v2",
                "applied_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        atomic_write_json(marker_path, marker)
        report_path = artifacts / "reports" / "ingestion" / f"{token}.json"
        if report_path.exists():
            atomic_write_json(report_path, marker)
        migrated.append(token)
    return {"public_interval_version": PUBLIC_INTERVAL_VERSION, "migrated": migrated, "skipped": skipped}
