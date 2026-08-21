from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any

from traceanchor.agents.provider import canonical_hash
from traceanchor.config import TraceAnchorConfig
from traceanchor.ingest.common import atomic_write_json


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize_runs(config: TraceAnchorConfig, config_path: Path) -> dict[str, Any]:
    resolved = config.resolved_dict(config_path)
    run_root = Path(resolved["paths"]["agent_runs_dir"])
    rows = []
    errors = []
    manifest_run_ids = set()
    for path in sorted(run_root.glob("*/*/manifest.json")):
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
            output_path = path.parent / "output.json"
            output = json.loads(output_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            errors.append(f"invalid run files: {path.parent.name}")
            continue
        if canonical_hash(output) != manifest.get("output_sha256"):
            errors.append(f"output hash mismatch: {manifest.get('run_id', path.parent.name)}")
        if int(manifest.get("tool_calls_total", 0)) > config.agents.max_total_tool_calls:
            errors.append(f"tool budget exceeded: {manifest.get('run_id', path.parent.name)}")
        rows.append(
            {
                "run_id": manifest.get("run_id"),
                "partition": manifest.get("run_partition"),
                "provider": manifest.get("provider"),
                "model": manifest.get("model"),
                "outcome": manifest.get("outcome"),
                "failure_class": manifest.get("failure_class"),
                "tool_calls": manifest.get("tool_calls_total", 0),
                "input_tokens": manifest.get("usage", {}).get("input_tokens", 0),
                "output_tokens": manifest.get("usage", {}).get("output_tokens", 0),
                "cost_rmb": manifest.get("usage", {}).get("cost_rmb", 0.0),
            }
        )
        manifest_run_ids.add(str(manifest.get("run_id", path.parent.name)))
    interrupted_attempts = []
    audit_dir = Path(resolved["paths"]["artifacts_dir"]) / "tool_audit"
    for path in sorted(audit_dir.glob("*.jsonl")):
        if path.stem in manifest_run_ids or not path.stem.startswith(("development_", "test_")):
            continue
        interrupted_attempts.append(
            {
                "run_id": path.stem,
                "failure_class": "TOOL_ERROR",
                "tool_audit": str(path.relative_to(config_path.parent)),
                "tool_audit_sha256": _file_sha256(path),
                "audit_records": sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line),
            }
        )
    result = {
        "schema_version": 1,
        "ok": not errors,
        "errors": errors,
        "runs": len(rows),
        "final": sum(item["outcome"] == "FINAL" for item in rows),
        "abstain": sum(item["outcome"] == "ABSTAIN" for item in rows),
        "infrastructure_errors": sum(
            item["outcome"] == "INFRASTRUCTURE_ERROR" for item in rows
        ),
        "interrupted_attempts": interrupted_attempts,
        "tool_calls": sum(int(item["tool_calls"]) for item in rows),
        "input_tokens": sum(int(item["input_tokens"]) for item in rows),
        "output_tokens": sum(int(item["output_tokens"]) for item in rows),
        "cost_rmb": sum(float(item["cost_rmb"]) for item in rows),
        "by_run": rows,
    }
    report_path = Path(resolved["paths"]["artifacts_dir"]) / "reports" / "WP06_run_summary.json"
    atomic_write_json(report_path, result)
    result["report"] = str(report_path.relative_to(config_path.parent))
    return result


__all__ = ["summarize_runs"]
