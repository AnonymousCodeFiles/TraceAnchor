from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq

from traceanchor.config import FORBIDDEN_AGENT_COLUMNS, TraceAnchorConfig


PUBLIC_TABLES = (
    "scenario_public.parquet",
    "syscall_event.parquet",
    "network_packet.parquet",
    "resource_sample.parquet",
)

EVIDENCE_TABLES = {
    "syscall_event.parquet": ("line_no", "sc"),
    "network_packet.parquet": ("frame_no", "pcap"),
    "resource_sample.parquet": ("row_no", "res"),
}


def _validate_evidence_sequence(
    path: Path,
    token: str,
    index_column: str,
    prefix: str,
) -> tuple[int, list[str]]:
    parquet = pq.ParquetFile(path)
    expected_index = 1
    errors: list[str] = []
    for batch in parquet.iter_batches(
        batch_size=65_536, columns=["evidence_id", index_column]
    ):
        identifiers = batch.column("evidence_id").to_pylist()
        indexes = batch.column(index_column).to_pylist()
        for evidence_id, index in zip(identifiers, indexes):
            if index != expected_index:
                errors.append(
                    f"{path.name} {index_column} sequence: expected {expected_index}, got {index}"
                )
                return expected_index - 1, errors
            expected_prefix = f"{prefix}:{token}:"
            if not str(evidence_id).startswith(expected_prefix) or not str(
                evidence_id
            ).endswith(f":{index}"):
                errors.append(f"invalid Evidence ID in {path.name} at row {index}")
                return expected_index - 1, errors
            expected_index += 1
    return expected_index - 1, errors


def validate_scenario_output(
    config: TraceAnchorConfig,
    config_path: Path,
    token: str,
) -> dict[str, object]:
    resolved = config.resolved_dict(config_path)
    public_root = Path(resolved["paths"]["parquet_dir"]) / token
    private_root = Path(resolved["paths"]["evaluator_dir"]) / token
    errors: list[str] = []
    counts: dict[str, int] = {}
    evidence_count = 0
    forbidden = {value.lower() for value in FORBIDDEN_AGENT_COLUMNS}
    for filename in PUBLIC_TABLES:
        path = public_root / filename
        if not path.exists():
            errors.append(f"missing public table: {filename}")
            continue
        metadata = pq.read_metadata(path)
        schema_names = metadata.schema.to_arrow_schema().names
        counts[filename] = metadata.num_rows
        leaked_columns = forbidden.intersection(name.lower() for name in schema_names)
        if leaked_columns:
            errors.append(f"forbidden columns in {filename}: {sorted(leaked_columns)}")
        if filename in EVIDENCE_TABLES:
            index_column, prefix = EVIDENCE_TABLES[filename]
            valid_count, sequence_errors = _validate_evidence_sequence(
                path, token, index_column, prefix
            )
            evidence_count += valid_count
            errors.extend(sequence_errors)
    for filename in ("scenario_private.parquet", "exploit_anchors.parquet"):
        path = private_root / filename
        if not path.exists():
            errors.append(f"missing evaluator table: {filename}")
        else:
            counts[filename] = pq.read_metadata(path).num_rows
    return {
        "scenario_token": token,
        "ok": not errors,
        "errors": errors,
        "counts": counts,
        "evidence_ids": evidence_count,
    }


def validate_all_outputs(config: TraceAnchorConfig, config_path: Path) -> dict[str, object]:
    resolved = config.resolved_dict(config_path)
    markers_root = Path(resolved["paths"]["completion_markers_dir"]) / "ingest"
    results = []
    for marker in sorted(markers_root.glob("*.done")):
        with marker.open("r", encoding="utf-8") as handle:
            token = str(json.load(handle)["scenario_token"])
        results.append(validate_scenario_output(config, config_path, token))
    return {
        "ok": bool(results) and all(result["ok"] for result in results),
        "validated_scenarios": len(results),
        "results": results,
    }
