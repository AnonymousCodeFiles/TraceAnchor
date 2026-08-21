from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from traceanchor.agents.broker import tool_schema_hash
from traceanchor.agents.provider import canonical_hash, is_loopback_url
from traceanchor.agents.prompts import prompt_hashes
from traceanchor.config import TraceAnchorConfig, config_hash
from traceanchor.ingest.common import atomic_write_json


MIN_GROUNDED_CLAIM_PRECISION = 0.95

DEVIATION_MARKER_NAME = "AGENT_PROTOCOL_FROZEN_WITH_DEVIATION"
DEVIATION_REPORT_RELATIVE = (
    "artifacts/reports/WP06_agent_development_c2960b3c6c654175.json"
)
DEVIATION_REPORT_SHA256 = (
    "2c2547e28fa9891f1ece95c91e04afb7a78f674357e489f2abe2d6e447afd6cf"
)
DEVIATION_RUN_NONCE = "primary-minimax-v7-deviation-v1"
DEVIATION_CONFIG_SHA256 = (
    "14301a2a96d8b1f53d756257c26ea8412922b6b589fc6f7eaee2649ec37ab78b"
)
DEVIATION_OBSERVED_GCP = 0.9070346320346321
DEVIATION_MANIFEST_INVENTORY_SHA256 = (
    "321eda58bb7605c394dccd9bb0575505b34c2c07fdb38ed296c17610b10b9022"
)
DEVIATION_TOOL_AUDIT_INVENTORY_SHA256 = (
    "848a57e82456f2639fffb6e69706019d7fa22861495c6a71f43fcc812560217c"
)
DEVIATION_DECISION_RELATIVE = "artifacts/decisions/WP06_best_available_deviation.md"
DEVIATION_DECISION_SHA256 = (
    "2f4e4742bac212583114eaf6c3f193c0632a0dffbad528b3d8a24c8c54957f6c"
)
DEVIATION_RESTORED_PROTOCOL_SHA256 = (
    "e688e3dcffb9a571b199f192a1749c398b795ea1638ed4cf653991eae4c6d2cd"
)
DEVIATION_RESTORED_TOOL_SCHEMA_SHA256 = (
    "7292b9df89c0e63aada6854f2c1b1e05a9739002560abfd31d99b3b14cebdb1f"
)
DEVIATION_RESTORED_OUTPUT_SCHEMA_SHA256 = (
    "9bd7f9214e3d2736a27a29c297c6cea370c90b819498f1429da245a19ced8240"
)
DEVIATION_RESTORED_PROMPT_SHA256 = {
    "correlation_agent": "7deb14f31f639d783857d56bfa5aaab17b7feea5f4c545b84f8884bf493a1faf",
    "evidence_verifier": "9fe496a5353d76f5d1f74270079e5df056ad7e1367fecf04b40bd67e732799e8",
    "host_investigator": "00fd9c527b33300d0fce6f19a9c2d97d720403fc4f58fbfea1a2a354b73aa6fd",
    "network_investigator": "4c33490bf3b66143e875a65c817eac152bb3472f53fbe497a07c6edf630dcc4c",
    "orchestrator": "9638077aef75a224088156000e8af9f83b344357246364944ee019dfeb931a55",
    "safety_prefix": "2ebe34c40c99adb2ed1ef9bb1a5575859f9edb743876dcb07b409add27360f90",
}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _protocol_files(root: Path) -> list[Path]:
    files = [
        root / "src" / "traceanchor" / "config.py",
        root / "src" / "traceanchor" / "cli.py",
        root / "src" / "traceanchor" / "evidence" / "schemas.py",
        root / "src" / "traceanchor" / "evidence" / "tools.py",
        root / "src" / "traceanchor" / "evaluation" / "agent_runner.py",
        root / "schemas" / "claim_evidence.json",
    ]
    files.extend(sorted((root / "src" / "traceanchor" / "agents").glob("*.py")))
    files.extend(sorted((root / "prompts" / "agents").glob("*.txt")))
    return sorted(set(files))


def protocol_code_hash(root: Path) -> str:
    values = []
    for path in _protocol_files(root):
        if not path.is_file():
            raise FileNotFoundError(f"protocol file is missing: {path.relative_to(root)}")
        values.append(
            {
                "path": str(path.relative_to(root)),
                "sha256": _file_sha256(path),
            }
        )
    return canonical_hash(values)


def _git_commit(root: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    value = result.stdout.strip()
    return value if result.returncode == 0 and len(value) == 40 else None


def _primary_errors(config: TraceAnchorConfig) -> list[str]:
    spec = config.llm.primary
    errors = []
    if spec.provider not in {
        "openai_compatible",
        "openai_responses",
        "anthropic",
        "gemini",
    }:
        errors.append("primary provider is not a supported real provider")
    if "CHANGE_ME" in spec.provider or "CHANGE_ME" in spec.model or "CHANGE_ME" in spec.api_key_env:
        errors.append("primary provider/model/API environment contains CHANGE_ME")
    if spec.cost_rmb_per_million_input_tokens is None:
        errors.append("primary input-token RMB cost is not configured")
    if spec.cost_rmb_per_million_output_tokens is None:
        errors.append("primary output-token RMB cost is not configured")
    if "CHANGE_ME" not in spec.api_key_env and not os.environ.get(spec.api_key_env):
        errors.append("primary API-key environment variable is unset")
    if not config.privacy_and_blinding.external_llm_payload_allowed:
        if spec.provider == "openai_compatible" and spec.base_url_env:
            base_url = os.environ.get(spec.base_url_env)
            if not base_url:
                errors.append("primary base-URL environment variable is unset")
            elif not is_loopback_url(base_url):
                errors.append("external payloads are disabled and primary base URL is not loopback")
        else:
            errors.append("external payloads are disabled for the configured primary provider")
    return errors


def _development_reports(
    root: Path,
    config: TraceAnchorConfig,
) -> tuple[list[dict[str, Any]], list[str]]:
    reports = []
    errors = []
    for path in sorted((root / "artifacts" / "reports").glob("WP06_agent_development_*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            errors.append(f"invalid Development report: {path.name}")
            continue
        if (
            value.get("complete") is True
            and value.get("provider") == config.llm.primary.provider
            and value.get("model") == config.llm.primary.model
            and value.get("config_sha256") == config_hash(config, root / "project.yml")
        ):
            reports.append({"path": path, "value": value})
    if not reports:
        errors.append("no complete 15-case Development report matches the primary protocol")
    elif max(float(item["value"].get("mean_grounded_claim_precision", 0.0)) for item in reports) < MIN_GROUNDED_CLAIM_PRECISION:
        errors.append(
            f"Development grounded-claim precision is below {MIN_GROUNDED_CLAIM_PRECISION}"
        )
    return reports, errors


def _marker_fields(config: TraceAnchorConfig, config_path: Path) -> dict[str, Any]:
    root = config_path.parent
    schema_path = root / config.agents.output_schema
    return {
        "config_sha256": config_hash(config, config_path),
        "provider": config.llm.primary.provider,
        "model": config.llm.primary.model,
        "prompt_sha256": prompt_hashes(root),
        "tool_schema_sha256": tool_schema_hash(),
        "output_schema_sha256": canonical_hash(
            json.loads(schema_path.read_text(encoding="utf-8"))
        ),
        "protocol_code_sha256": protocol_code_hash(root),
    }


def _inventory_sha256(root: Path, paths: list[Path]) -> str:
    """Hash a sorted path/hash inventory without including file contents."""
    lines = []
    for path in sorted(paths, key=lambda item: str(item)):
        if not path.is_file():
            raise FileNotFoundError(path)
        relative = path.relative_to(root) if path.is_relative_to(root) else path
        lines.append(f"{_file_sha256(path)}  {relative}\n")
    return hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()


def _deviation_report_bindings(
    config: TraceAnchorConfig, config_path: Path
) -> tuple[dict[str, Any], list[str]]:
    """Validate the one human-reviewed run and return hash-only bindings."""
    root = config_path.parent
    resolved = config.resolved_dict(config_path)
    artifact_root = Path(resolved["paths"]["artifacts_dir"])
    errors: list[str] = []
    report_path = root / DEVIATION_REPORT_RELATIVE
    if not report_path.is_file():
        errors.append("reviewed 15-case Development report is missing")
        return {}, errors
    report_sha256 = _file_sha256(report_path)
    if report_sha256 != DEVIATION_REPORT_SHA256:
        errors.append("reviewed Development report hash does not match decision")
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        errors.append("reviewed Development report is unreadable")
        return {}, errors

    expected_report = {
        "complete": True,
        "cases": 15,
        "expected_cases": 15,
        "failures": 0,
        "abstentions": 0,
        "leakage_errors": 0,
        "run_nonce": DEVIATION_RUN_NONCE,
        "provider": "openai_compatible",
        "model": "MiniMax-M2.5",
        "config_sha256": DEVIATION_CONFIG_SHA256,
    }
    for key, expected in expected_report.items():
        if report.get(key) != expected:
            errors.append(f"reviewed Development report field mismatch: {key}")
    if float(report.get("mean_grounded_claim_precision", -1.0)) != DEVIATION_OBSERVED_GCP:
        errors.append("reviewed Development GCP does not match decision")
    if not float(report.get("mean_grounded_claim_precision", 1.0)) < MIN_GROUNDED_CLAIM_PRECISION:
        errors.append("reviewed Development report unexpectedly passes ordinary gate")

    case_results = report.get("case_results")
    manifest_paths: list[Path] = []
    if not isinstance(case_results, list) or len(case_results) != 15:
        errors.append("reviewed Development report does not contain 15 case results")
        case_results = []
    for item in case_results:
        if not isinstance(item, dict) or not isinstance(item.get("manifest"), str):
            errors.append("reviewed Development case is missing a manifest path")
            continue
        manifest_path = root / item["manifest"]
        manifest_paths.append(manifest_path)
        if not manifest_path.is_file():
            errors.append("reviewed Development manifest is missing")
        elif item.get("manifest_sha256") != _file_sha256(manifest_path):
            errors.append("reviewed Development manifest hash mismatch")
    if len(manifest_paths) == 15:
        try:
            manifest_inventory_sha256 = _inventory_sha256(root, manifest_paths)
        except FileNotFoundError:
            manifest_inventory_sha256 = ""
        if manifest_inventory_sha256 != DEVIATION_MANIFEST_INVENTORY_SHA256:
            errors.append("reviewed manifest inventory hash mismatch")
    else:
        manifest_inventory_sha256 = ""

    tool_audit_paths: list[Path] = []
    cache_paths: dict[str, Path] = {}
    for manifest_path in manifest_paths:
        if not manifest_path.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            errors.append("reviewed manifest is unreadable")
            continue
        run_id = manifest.get("run_id")
        if not isinstance(run_id, str):
            errors.append("reviewed manifest is missing run_id")
        else:
            tool_audit_paths.append(artifact_root / "tool_audit" / f"{run_id}.jsonl")
        for call in manifest.get("provider_calls", []):
            cache_key = call.get("cache_key") if isinstance(call, dict) else None
            if isinstance(cache_key, str):
                cache_paths[cache_key] = artifact_root / "agent_cache" / f"{cache_key}.json"
    if len(tool_audit_paths) == 15:
        if any(not path.is_file() for path in tool_audit_paths):
            errors.append("reviewed tool audit is missing")
        try:
            tool_audit_inventory_sha256 = _inventory_sha256(root, tool_audit_paths)
        except FileNotFoundError:
            tool_audit_inventory_sha256 = ""
        if tool_audit_inventory_sha256 != DEVIATION_TOOL_AUDIT_INVENTORY_SHA256:
            errors.append("reviewed tool audit inventory hash mismatch")
    else:
        tool_audit_inventory_sha256 = ""
    cache_inventory_paths = sorted(cache_paths.values(), key=lambda item: str(item))
    if any(not path.is_file() for path in cache_inventory_paths):
        errors.append("reviewed provider cache entry is missing")
    try:
        cache_inventory_sha256 = _inventory_sha256(root, cache_inventory_paths)
    except FileNotFoundError:
        cache_inventory_sha256 = ""

    decision_path = root / DEVIATION_DECISION_RELATIVE
    if not decision_path.is_file() or _file_sha256(decision_path) != DEVIATION_DECISION_SHA256:
        errors.append("deviation decision record hash mismatch")
    environment_path = root / "artifacts" / "manifests" / "environment.json"
    ledger_index_path = root / "artifacts" / "evidence" / "ledger_index.json"
    evidence_db_path = Path(resolved["paths"]["evidence_db"])
    for path, label in (
        (environment_path, "environment manifest"),
        (ledger_index_path, "Evidence Ledger index"),
        (evidence_db_path, "Agent Evidence Ledger"),
    ):
        if not path.is_file():
            errors.append(f"{label} is missing")

    active_fields = _marker_fields(config, config_path)
    restored = {
        "protocol_code_sha256": DEVIATION_RESTORED_PROTOCOL_SHA256,
        "tool_schema_sha256": DEVIATION_RESTORED_TOOL_SCHEMA_SHA256,
        "output_schema_sha256": DEVIATION_RESTORED_OUTPUT_SCHEMA_SHA256,
        "prompt_sha256": DEVIATION_RESTORED_PROMPT_SHA256,
    }
    if active_fields["config_sha256"] != DEVIATION_CONFIG_SHA256:
        errors.append("active configuration hash does not match reviewed run")
    if active_fields["tool_schema_sha256"] != DEVIATION_RESTORED_TOOL_SCHEMA_SHA256:
        errors.append("active tool schema is not exact v7")
    if active_fields["output_schema_sha256"] != DEVIATION_RESTORED_OUTPUT_SCHEMA_SHA256:
        errors.append("active output schema is not exact v7")
    if active_fields["prompt_sha256"] != DEVIATION_RESTORED_PROMPT_SHA256:
        errors.append("active prompts are not exact v7")
    bindings = {
        "active_protocol": active_fields,
        "restored_protocol": restored,
        "decision_record": DEVIATION_DECISION_RELATIVE,
        "decision_record_sha256": DEVIATION_DECISION_SHA256,
        "development_report": DEVIATION_REPORT_RELATIVE,
        "development_report_sha256": report_sha256,
        "manifest_inventory_sha256": manifest_inventory_sha256,
        "manifest_count": len(manifest_paths),
        "tool_audit_inventory_sha256": tool_audit_inventory_sha256,
        "tool_audit_count": len(tool_audit_paths),
        "cache_inventory_sha256": cache_inventory_sha256,
        "cache_count": len(cache_inventory_paths),
        "environment_manifest_sha256": (
            _file_sha256(environment_path) if environment_path.is_file() else None
        ),
        "ledger_index_sha256": (
            _file_sha256(ledger_index_path) if ledger_index_path.is_file() else None
        ),
        "evidence_ledger_sha256": (
            _file_sha256(evidence_db_path) if evidence_db_path.is_file() else None
        ),
        "minimum_grounded_claim_precision": MIN_GROUNDED_CLAIM_PRECISION,
        "observed_grounded_claim_precision": DEVIATION_OBSERVED_GCP,
        "gate_passed": False,
        "run_nonce": DEVIATION_RUN_NONCE,
        "report_session_id": report.get("session_id"),
    }
    return bindings, errors


def freeze_agent_protocol_with_deviation(
    config: TraceAnchorConfig,
    config_path: Path,
) -> dict[str, Any]:
    """Create the reviewed best-available marker without changing ordinary freeze semantics."""
    root = config_path.parent
    resolved = config.resolved_dict(config_path)
    marker_dir = Path(resolved["paths"]["completion_markers_dir"])
    marker_path = marker_dir / DEVIATION_MARKER_NAME
    ordinary_marker = marker_dir / "AGENT_PROTOCOL_FROZEN"
    wp6_path = marker_dir / "WP6.done"
    errors: list[str] = []
    if marker_path.exists() or ordinary_marker.exists() or wp6_path.exists():
        errors.append("protocol marker already exists; refusing to overwrite frozen state")
    for required in (marker_dir / "WP4.done", marker_dir / "WP5.done"):
        if not required.is_file():
            errors.append(f"required completion marker is missing: {required.name}")
    bindings, binding_errors = _deviation_report_bindings(config, config_path)
    errors.extend(binding_errors)
    report_path = Path(resolved["paths"]["artifacts_dir"]) / "reports" / "WP06_protocol_deviation_freeze.json"
    report = {
        "schema_version": 1,
        "ok": not errors,
        "status": "frozen_with_deviation" if not errors else "blocked",
        "classification": "BEST-AVAILABLE PROTOCOL DEVIATION AUDIT",
        "errors": errors,
        **bindings,
    }
    atomic_write_json(report_path, report)
    report_sha256 = _file_sha256(report_path)
    if errors:
        return {
            **report,
            "report": str(report_path.relative_to(root)),
            "report_sha256": report_sha256,
        }
    marker = {
        "schema_version": 1,
        "status": "frozen_with_deviation",
        "classification": "BEST-AVAILABLE AGENT PROTOCOL; ORDINARY GATE NOT PASSED",
        **bindings,
        "deviation_freeze_report": str(report_path.relative_to(root)),
        "deviation_freeze_report_sha256": report_sha256,
        "downstream_test_authorized": False,
        "wp7_authorized": False,
    }
    atomic_write_json(marker_path, marker)
    atomic_write_json(
        wp6_path,
        {
            "schema_version": 1,
            "status": "complete_with_deviation",
            "gate_passed": False,
            "agent_protocol_marker": str(marker_path.relative_to(root)),
            "agent_protocol_marker_sha256": _file_sha256(marker_path),
            "config_sha256": bindings["active_protocol"]["config_sha256"],
            "downstream_test_authorized": False,
            "wp7_authorized": False,
        },
    )
    return {
        **report,
        "marker": str(marker_path.relative_to(root)),
        "marker_sha256": _file_sha256(marker_path),
        "report": str(report_path.relative_to(root)),
        "report_sha256": report_sha256,
    }


def freeze_agent_protocol(
    config: TraceAnchorConfig,
    config_path: Path,
) -> dict[str, Any]:
    root = config_path.parent
    resolved = config.resolved_dict(config_path)
    marker_dir = Path(resolved["paths"]["completion_markers_dir"])
    marker_path = marker_dir / "AGENT_PROTOCOL_FROZEN"
    wp6_path = marker_dir / "WP6.done"
    errors = _primary_errors(config)
    for required in (marker_dir / "WP4.done", marker_dir / "WP5.done"):
        if not required.is_file():
            errors.append(f"required completion marker is missing: {required.name}")
    reports, report_errors = _development_reports(root, config)
    errors.extend(report_errors)
    try:
        fields = _marker_fields(config, config_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        fields = {}
        errors.append(f"protocol hash construction failed: {type(exc).__name__}")
    if marker_path.exists() or wp6_path.exists():
        errors.append("protocol marker already exists; refusing to overwrite frozen state")
    report = {
        "schema_version": 1,
        "ok": not errors,
        "errors": errors,
        "minimum_grounded_claim_precision": MIN_GROUNDED_CLAIM_PRECISION,
        "git_commit": _git_commit(root),
        "git_repository_available": _git_commit(root) is not None,
        "protocol": fields,
        "development_reports": [
            {
                "path": str(item["path"].relative_to(root)),
                "sha256": _file_sha256(item["path"]),
                "session_id": item["value"].get("session_id"),
                "mean_evidence_f1": item["value"].get("mean_evidence_f1"),
                "mean_grounded_claim_precision": item["value"].get(
                    "mean_grounded_claim_precision"
                ),
                "total_cost_rmb": item["value"].get("total_cost_rmb"),
            }
            for item in reports
        ],
    }
    report_path = Path(resolved["paths"]["artifacts_dir"]) / "reports" / "WP06_protocol_freeze.json"
    atomic_write_json(report_path, report)
    if errors:
        return {**report, "report": str(report_path.relative_to(root))}
    marker = {
        "schema_version": 1,
        "status": "frozen",
        "classification": "IMMUTABLE AGENT TEST PROTOCOL",
        **fields,
        "git_commit": report["git_commit"],
        "development_report_sha256": [
            item["sha256"] for item in report["development_reports"]
        ],
        "freeze_report": str(report_path.relative_to(root)),
        "freeze_report_sha256": _file_sha256(report_path),
    }
    atomic_write_json(marker_path, marker)
    atomic_write_json(
        wp6_path,
        {
            "schema_version": 1,
            "status": "complete",
            "agent_protocol_marker": str(marker_path.relative_to(root)),
            "agent_protocol_marker_sha256": _file_sha256(marker_path),
            "config_sha256": fields["config_sha256"],
        },
    )
    return {
        **report,
        "marker": str(marker_path.relative_to(root)),
        "marker_sha256": _file_sha256(marker_path),
    }


def assert_protocol_frozen(config: TraceAnchorConfig, config_path: Path) -> dict[str, Any]:
    resolved = config.resolved_dict(config_path)
    marker_path = Path(resolved["paths"]["completion_markers_dir"]) / "AGENT_PROTOCOL_FROZEN"
    if not marker_path.is_file():
        raise RuntimeError("Agent Test is locked: AGENT_PROTOCOL_FROZEN is absent")
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise RuntimeError("Agent Test is locked: protocol marker is unreadable") from None
    expected = _marker_fields(config, config_path)
    if marker.get("status") != "frozen" or any(
        marker.get(key) != value for key, value in expected.items()
    ):
        raise RuntimeError("Agent Test is locked: protocol marker hashes no longer match")
    return marker


def assert_protocol_frozen_with_deviation(
    config: TraceAnchorConfig, config_path: Path
) -> dict[str, Any]:
    """Validate the reviewed deviation marker; ordinary freeze remains separate."""
    resolved = config.resolved_dict(config_path)
    marker_path = Path(resolved["paths"]["completion_markers_dir"]) / DEVIATION_MARKER_NAME
    if not marker_path.is_file():
        raise RuntimeError("Agent Test is locked: deviation marker is absent")
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise RuntimeError("Agent Test is locked: deviation marker is unreadable") from None
    bindings, errors = _deviation_report_bindings(config, config_path)
    if errors:
        raise RuntimeError("Agent Test is locked: deviation marker bindings are invalid")
    for key, value in bindings.items():
        if marker.get(key) != value:
            raise RuntimeError("Agent Test is locked: deviation marker hashes no longer match")
    if marker.get("status") != "frozen_with_deviation" or marker.get("gate_passed") is not False:
        raise RuntimeError("Agent Test is locked: deviation marker status is invalid")
    if marker.get("downstream_test_authorized") is not False or marker.get("wp7_authorized") is not False:
        raise RuntimeError("Agent Test is locked: deviation downstream flags are invalid")
    report_path = Path(resolved["paths"]["artifacts_dir"]) / "reports" / "WP06_protocol_deviation_freeze.json"
    if marker.get("deviation_freeze_report") != str(report_path.relative_to(config_path.parent)):
        raise RuntimeError("Agent Test is locked: deviation report binding is invalid")
    if marker.get("deviation_freeze_report_sha256") != _file_sha256(report_path):
        raise RuntimeError("Agent Test is locked: deviation report hash no longer matches")
    wp6_path = Path(resolved["paths"]["completion_markers_dir"]) / "WP6.done"
    if not wp6_path.is_file():
        raise RuntimeError("Agent Test is locked: deviation WP6 marker is absent")
    try:
        wp6 = json.loads(wp6_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise RuntimeError("Agent Test is locked: deviation WP6 marker is unreadable") from None
    if (
        wp6.get("status") != "complete_with_deviation"
        or wp6.get("gate_passed") is not False
        or wp6.get("agent_protocol_marker_sha256") != _file_sha256(marker_path)
    ):
        raise RuntimeError("Agent Test is locked: deviation WP6 marker hashes no longer match")
    return marker


__all__ = [
    "DEVIATION_MARKER_NAME",
    "MIN_GROUNDED_CLAIM_PRECISION",
    "assert_protocol_frozen",
    "assert_protocol_frozen_with_deviation",
    "freeze_agent_protocol",
    "freeze_agent_protocol_with_deviation",
    "protocol_code_hash",
]
