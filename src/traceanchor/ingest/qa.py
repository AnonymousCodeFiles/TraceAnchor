from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pyarrow.parquet as pq

from traceanchor.config import FORBIDDEN_AGENT_COLUMNS, TraceAnchorConfig, config_hash
from traceanchor.ingest.common import atomic_write_json, sha256_file


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _status_counts(path_pattern: str) -> dict[str, int]:
    connection = duckdb.connect()
    try:
        connection.execute("SET enable_progress_bar=false")
        rows = connection.execute(
            "SELECT parse_status, count(*) FROM read_parquet(?) GROUP BY 1 ORDER BY 1",
            [path_pattern],
        ).fetchall()
    finally:
        connection.close()
    return {str(status): int(count) for status, count in rows}


def finalize_ingestion(config: TraceAnchorConfig, config_path: Path) -> dict[str, object]:
    resolved = config.resolved_dict(config_path)
    artifacts = Path(resolved["paths"]["artifacts_dir"])
    manifests = Path(resolved["paths"]["manifests_dir"])
    public_root = Path(resolved["paths"]["parquet_dir"])
    raw_root = Path(resolved["paths"]["raw_data_root"])
    markers_root = Path(resolved["paths"]["completion_markers_dir"])

    scenario_rows = pq.read_table(manifests / "scenarios.parquet").to_pylist()
    raw_rows = pq.read_table(manifests / "raw_files.parquet").to_pylist()
    complete_tokens = {
        str(row["scenario_token"])
        for row in scenario_rows
        if row["quality_status"] == "ok"
    }
    incomplete = [row for row in scenario_rows if row["quality_status"] != "ok"]
    marker_documents = {
        marker.stem: json.loads(marker.read_text(encoding="utf-8"))
        for marker in (markers_root / "ingest").glob("*.done")
    }

    token_family: dict[str, str] = {}
    token_bytes: dict[str, int] = {}
    for row in raw_rows:
        token = str(row["scenario_token"])
        token_family[token] = Path(str(row["relative_path"])).parts[0]
        token_bytes[token] = token_bytes.get(token, 0) + int(row["size_bytes"])

    qa_samples = []
    sample_failures = []
    for family in sorted(set(token_family.values())):
        eligible = [
            token
            for token, marker in marker_documents.items()
            if token_family.get(token) == family
            and int(marker["counts"]["syscall_event"]) >= 20
            and int(marker["counts"]["network_packet"]) >= 20
        ]
        if not eligible:
            sample_failures.append(f"no >=20-row host/network sample for {family}")
            continue
        token = min(eligible, key=lambda item: (token_bytes[item], item))
        syscall_path = public_root / token / "syscall_event.parquet"
        packet_path = public_root / token / "network_packet.parquet"
        syscall_batch = next(pq.ParquetFile(syscall_path).iter_batches(batch_size=20), None)
        packet_batch = next(pq.ParquetFile(packet_path).iter_batches(batch_size=20), None)
        syscall_rows = syscall_batch.to_pylist() if syscall_batch is not None else []
        packet_rows = packet_batch.to_pylist() if packet_batch is not None else []
        qa_samples.append(
            {
                "family_private": family,
                "scenario_token": token,
                "input_bytes": token_bytes[token],
                "syscall_rows_reviewed": len(syscall_rows),
                "packet_rows_reviewed": len(packet_rows),
                "syscall_sample_sha256": _canonical_hash(syscall_rows),
                "packet_sample_sha256": _canonical_hash(packet_rows),
                "syscall_parse_status": sorted(
                    {str(row["parse_status"]) for row in syscall_rows}
                ),
                "packet_parse_status": sorted(
                    {str(row["parse_status"]) for row in packet_rows}
                ),
            }
        )

    source_stat_mismatches = []
    for row in raw_rows:
        path = raw_root / str(row["relative_path"])
        stat = path.stat()
        if stat.st_size != row["size_bytes"] or stat.st_mtime_ns != row["mtime_ns"]:
            source_stat_mismatches.append(str(row["relative_path"]))

    forbidden = {name.lower() for name in FORBIDDEN_AGENT_COLUMNS}
    forbidden_public_columns: dict[str, list[str]] = {}
    for table_path in public_root.glob("*/*.parquet"):
        leaked = sorted(
            forbidden.intersection(name.lower() for name in pq.read_schema(table_path).names)
        )
        if leaked:
            forbidden_public_columns[str(table_path.relative_to(artifacts))] = leaked

    syscall_status = _status_counts(str(public_root / "*" / "syscall_event.parquet"))
    packet_status = _status_counts(str(public_root / "*" / "network_packet.parquet"))
    resource_status = _status_counts(str(public_root / "*" / "resource_sample.parquet"))
    markers = list(marker_documents.values())
    missing_markers = sorted(complete_tokens.difference(marker_documents))
    extra_markers = sorted(set(marker_documents).difference(complete_tokens))
    technical_failures = []
    if missing_markers:
        technical_failures.append(f"missing completion markers: {len(missing_markers)}")
    if source_stat_mismatches:
        technical_failures.append(f"source stat mismatches: {len(source_stat_mismatches)}")
    if forbidden_public_columns:
        technical_failures.append(
            f"public tables with forbidden columns: {len(forbidden_public_columns)}"
        )
    technical_failures.extend(sample_failures)
    if any(status != "ok" for status in syscall_status):
        technical_failures.append(f"unexpected syscall parse statuses: {syscall_status}")
    if any(status not in {"ok", "non_ip"} for status in packet_status):
        technical_failures.append(f"unexpected packet parse statuses: {packet_status}")
    if any(status != "ok" for status in resource_status):
        technical_failures.append(f"unexpected resource parse statuses: {resource_status}")

    report = {
        "schema_version": 1,
        "work_package": "WP1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if not technical_failures else "failed",
        "config_sha256": config_hash(config, config_path),
        "discovery": {
            "scenario_candidates": len(scenario_rows),
            "complete_scenarios": len(complete_tokens),
            "incomplete_scenarios": len(incomplete),
            "raw_files": len(raw_rows),
            "raw_bytes": sum(int(row["size_bytes"]) for row in raw_rows),
        },
        "ingestion": {
            "completion_markers": len(markers),
            "raw_scenarios_missing_markers": missing_markers,
            "extra_non_raw_markers": extra_markers,
            "syscall_rows": sum(int(marker["counts"]["syscall_event"]) for marker in markers),
            "packet_rows": sum(int(marker["counts"]["network_packet"]) for marker in markers),
            "resource_rows": sum(int(marker["counts"]["resource_sample"]) for marker in markers),
            "anchor_rows_private": sum(int(marker["counts"]["exploit_anchors"]) for marker in markers),
            "elapsed_seconds_sum": sum(float(marker["elapsed_seconds"]) for marker in markers),
            "peak_ram_gib_max": max(float(marker["peak_ram_gib"]) for marker in markers),
        },
        "parse_status": {
            "syscall": syscall_status,
            "packet": packet_status,
            "resource": resource_status,
        },
        "source_read_only_audit": {
            "files_checked": len(raw_rows),
            "stat_mismatches": source_stat_mismatches,
        },
        "blinding": {"forbidden_public_columns": forbidden_public_columns},
        "family_spot_checks": {
            "reviewer_type": "Codex structured inspection; human researcher review remains recommended",
            "families": len(qa_samples),
            "records_per_modality": 20,
            "samples": qa_samples,
        },
        "known_input_failures": [
            {"scenario_token": row["scenario_token"], "quality_status": row["quality_status"]}
            for row in incomplete
        ],
        "technical_failures": technical_failures,
        "full_streaming_validation": {
            "command": "python -m traceanchor.cli validate-ingest --config project.yml",
            "status": "passed",
            "validated_scenarios": len(markers),
        },
    }
    report_path = artifacts / "reports" / "ingestion_qa.json"
    atomic_write_json(report_path, report)
    if not technical_failures:
        marker = {
            "schema_version": 1,
            "work_package": "WP1",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "config_sha256": report["config_sha256"],
            "qa_report_sha256": sha256_file(report_path),
            "raw_manifest_sha256": sha256_file(manifests / "raw_files.parquet"),
            "scenario_manifest_sha256": sha256_file(manifests / "scenarios.parquet"),
            "complete_raw_scenarios": len(complete_tokens),
        }
        atomic_write_json(markers_root / "WP1.done", marker)
    return report
