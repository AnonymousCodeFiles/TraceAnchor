from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pydantic import ValidationError

from traceanchor.annotation.workflow import (
    adjudicate_annotations,
    compute_agreement,
    create_annotation_draft,
    sample_gold,
    validate_gold,
)
from traceanchor.annotation.csv_workflow import (
    WORKSHEET_SELECTIONS,
    export_annotation_csv,
    import_annotation_csv,
)
from traceanchor.agents.protocol import (
    freeze_agent_protocol,
    freeze_agent_protocol_with_deviation,
)
from traceanchor.agents.reporting import summarize_runs
from traceanchor.config import config_hash, load_config
from traceanchor.build.splits import make_splits, validate_splits
from traceanchor.build.features import build_features_for_split, validate_datasets
from traceanchor.detector.alerting import calibrate_alerts, evaluate_alerts
from traceanchor.detector.scoring import score_all_detectors, score_detector
from traceanchor.detector.training import train_detector
from traceanchor.evidence.store import build_evidence_store
from traceanchor.evidence.validation import validate_blind_view, run_tool_smoke
from traceanchor.evaluation.agent_runner import run_agent_cases
from traceanchor.environment import collect_checks, write_environment_manifest
from traceanchor.ingest.manifest import (
    build_raw_manifest,
    candidate_from_example,
    discover_scenarios,
    load_candidate_from_manifest,
    select_family_sample_from_manifest,
)
from traceanchor.ingest.migrations import migrate_public_intervals
from traceanchor.ingest.qa import finalize_ingestion
from traceanchor.ingest.runner import ingest_candidate
from traceanchor.ingest.validate import validate_all_outputs, validate_scenario_output


def _env_check(args: argparse.Namespace) -> int:
    config_path = Path(args.config).resolve()
    try:
        config = load_config(config_path)
        resolved = config.resolved_dict(config_path)
    except (OSError, ValueError, ValidationError) as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2

    checks, metadata = collect_checks(config, config_path)
    manifest_path = write_environment_manifest(config, config_path, checks, metadata)
    result = {
        "config_sha256": config_hash(config, config_path),
        "hard_requirements_met": all(item.ok for item in checks if item.required),
        "manifest": str(manifest_path),
        "checks": [
            {
                "name": item.name,
                "ok": item.ok,
                "expected": item.expected,
                "actual": item.actual,
            }
            for item in checks
        ],
        "paths": {
            "raw_data_root": resolved["paths"]["raw_data_root"],
            "artifacts_dir": resolved["paths"]["artifacts_dir"],
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["hard_requirements_met"] else 1


def _load_cli_config(path_value: str):
    path = Path(path_value).resolve()
    return path, load_config(path)


def _manifest(args: argparse.Namespace) -> int:
    try:
        config_path, config = _load_cli_config(args.config)
        report = build_raw_manifest(config, config_path)
    except (OSError, ValueError, ValidationError) as exc:
        print(f"manifest error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _ingest(args: argparse.Namespace) -> int:
    try:
        config_path, config = _load_cli_config(args.config)
        if args.example:
            candidates = [candidate_from_example(config, config_path)]
        elif args.scenario:
            candidates = [load_candidate_from_manifest(config, config_path, args.scenario)]
        elif args.family_sample:
            candidates = select_family_sample_from_manifest(config, config_path)
        else:
            resolved = config.resolved_dict(config_path)
            root = Path(resolved["paths"]["raw_data_root"])
            candidates = [
                candidate
                for candidate in discover_scenarios(root, config.project.global_seed)
                if candidate.complete
            ]
        results = [
            ingest_candidate(config, config_path, candidate, resume=args.resume)
            for candidate in candidates
        ]
    except (OSError, KeyError, ValueError, ValidationError) as exc:
        print(f"ingest error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"results": results}, indent=2, sort_keys=True))
    return 0


def _validate_ingest(args: argparse.Namespace) -> int:
    try:
        config_path, config = _load_cli_config(args.config)
        result = (
            validate_scenario_output(config, config_path, args.scenario)
            if args.scenario
            else validate_all_outputs(config, config_path)
        )
    except (OSError, ValueError, ValidationError) as exc:
        print(f"validation error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


def _migrate_public_intervals(args: argparse.Namespace) -> int:
    try:
        config_path, config = _load_cli_config(args.config)
        result = migrate_public_intervals(config, config_path)
    except (OSError, ValueError, ValidationError) as exc:
        print(f"migration error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _finalize_ingest(args: argparse.Namespace) -> int:
    try:
        config_path, config = _load_cli_config(args.config)
        result = finalize_ingestion(config, config_path)
    except (OSError, ValueError, ValidationError) as exc:
        print(f"finalization error: {exc}", file=sys.stderr)
        return 2
    summary = {
        "status": result["status"],
        "discovery": result["discovery"],
        "ingestion": result["ingestion"],
        "parse_status": result["parse_status"],
        "spot_check_families": result["family_spot_checks"]["families"],
        "technical_failures": result["technical_failures"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if result["status"] == "passed" else 1


def _make_splits(args: argparse.Namespace) -> int:
    try:
        config_path, config = _load_cli_config(args.config)
        report = make_splits(config, config_path)
        validation = validate_splits(config, config_path)
    except (OSError, ValueError, ValidationError) as exc:
        print(f"split error: {exc}", file=sys.stderr)
        return 2
    result = {"report": report, "validation": validation}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if validation["ok"] else 1


def _build_features(args: argparse.Namespace) -> int:
    try:
        config_path, config = _load_cli_config(args.config)
        result = build_features_for_split(
            config, config_path, split=args.split, resume=args.resume
        )
    except (OSError, ValueError, ValidationError) as exc:
        print(f"feature build error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _validate_datasets(args: argparse.Namespace) -> int:
    try:
        config_path, config = _load_cli_config(args.config)
        result = validate_datasets(config, config_path)
    except (OSError, ValueError, ValidationError) as exc:
        print(f"dataset validation error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


def _train_detector(args: argparse.Namespace) -> int:
    try:
        config_path, config = _load_cli_config(args.config)
        result = train_detector(
            config,
            config_path,
            channel=args.channel,
            seed=args.seed,
            resume=args.resume,
            smoke_test=args.smoke_test,
        )
    except (OSError, RuntimeError, ValueError, ValidationError) as exc:
        print(f"detector training error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _score_detector(args: argparse.Namespace) -> int:
    try:
        config_path, config = _load_cli_config(args.config)
        if args.all_seeds:
            result = score_all_detectors(config, config_path, args.split)
        else:
            if args.channel is None or args.seed is None:
                raise ValueError("--channel and --seed are required without --all-seeds")
            result = score_detector(
                config, config_path, args.channel, args.seed, args.split
            )
    except (OSError, RuntimeError, ValueError, ValidationError) as exc:
        print(f"detector scoring error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _calibrate_alerts(args: argparse.Namespace) -> int:
    try:
        config_path, config = _load_cli_config(args.config)
        result = calibrate_alerts(config, config_path)
    except (OSError, RuntimeError, ValueError, ValidationError) as exc:
        print(f"alert calibration error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _evaluate_alerts(args: argparse.Namespace) -> int:
    try:
        config_path, config = _load_cli_config(args.config)
        result = evaluate_alerts(config, config_path)
    except (OSError, RuntimeError, ValueError, ValidationError) as exc:
        print(f"alert evaluation error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _build_evidence_store(args: argparse.Namespace) -> int:
    try:
        config_path, config = _load_cli_config(args.config)
        result = build_evidence_store(
            config,
            config_path,
            selection=args.selection,
            resume=args.resume,
        )
    except (OSError, RuntimeError, ValueError, ValidationError) as exc:
        print(f"evidence-store error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if args.selection != "all" or result.get("completed_all_scenarios", False) else 1


def _validate_blind_view(args: argparse.Namespace) -> int:
    try:
        config_path, config = _load_cli_config(args.config)
        result = validate_blind_view(config, config_path)
    except (OSError, RuntimeError, ValueError, ValidationError) as exc:
        print(f"blind-view validation error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


def _tool_smoke(args: argparse.Namespace) -> int:
    try:
        config_path, config = _load_cli_config(args.config)
        result = run_tool_smoke(config, config_path, all_families=args.all_families)
    except (OSError, RuntimeError, ValueError, ValidationError) as exc:
        print(f"tool smoke error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


def _sample_gold(args: argparse.Namespace) -> int:
    try:
        config_path, config = _load_cli_config(args.config)
        result = sample_gold(
            config,
            config_path,
            split=args.split,
            incident_ids=args.incident,
        )
    except (OSError, RuntimeError, ValueError, ValidationError) as exc:
        print(f"gold sampling error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


def _annotate(args: argparse.Namespace) -> int:
    try:
        config_path, config = _load_cli_config(args.config)
        result = create_annotation_draft(
            config,
            config_path,
            incident_id=args.incident,
            annotator_id=args.annotator,
        )
    except (OSError, RuntimeError, ValueError, ValidationError) as exc:
        print(f"annotation draft error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _export_pilot_csv(args: argparse.Namespace) -> int:
    try:
        config_path, config = _load_cli_config(args.config)
        result = export_annotation_csv(
            config,
            config_path,
            Path(args.output).resolve(),
            selection=args.selection,
        )
    except (OSError, RuntimeError, ValueError, ValidationError) as exc:
        print(f"pilot CSV export error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _import_pilot_csv(args: argparse.Namespace) -> int:
    try:
        config_path, config = _load_cli_config(args.config)
        result = import_annotation_csv(
            config,
            config_path,
            Path(args.input).resolve(),
            annotator_id=args.annotator,
            selection=args.selection,
            allow_incomplete=args.allow_incomplete,
        )
    except (OSError, RuntimeError, ValueError, ValidationError) as exc:
        print(f"pilot CSV import error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _validate_gold(args: argparse.Namespace) -> int:
    try:
        config_path, config = _load_cli_config(args.config)
        result = validate_gold(config, config_path)
    except (OSError, RuntimeError, ValueError, ValidationError) as exc:
        print(f"gold validation error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


def _agreement(args: argparse.Namespace) -> int:
    try:
        config_path, config = _load_cli_config(args.config)
        result = compute_agreement(config, config_path)
    except (OSError, RuntimeError, ValueError, ValidationError) as exc:
        print(f"agreement error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


def _adjudicate(args: argparse.Namespace) -> int:
    try:
        config_path, config = _load_cli_config(args.config)
        result = adjudicate_annotations(
            config, config_path, incident_id=args.incident
        )
    except (OSError, RuntimeError, ValueError, ValidationError) as exc:
        print(f"adjudication error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _run_agent_dev(args: argparse.Namespace) -> int:
    try:
        config_path, config = _load_cli_config(args.config)
        result = run_agent_cases(
            config,
            config_path,
            selection="development",
            provider_mode=args.provider,
            candidate_index=args.candidate_index,
            limit=args.limit,
            run_nonce=args.run_nonce,
            resume=args.resume,
        )
    except (OSError, RuntimeError, ValueError, ValidationError) as exc:
        print(f"Agent Development error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["failures"] == 0 else 1


def _freeze_agent_protocol(args: argparse.Namespace) -> int:
    try:
        config_path, config = _load_cli_config(args.config)
        result = freeze_agent_protocol(config, config_path)
    except (OSError, RuntimeError, ValueError, ValidationError) as exc:
        print(f"Agent protocol freeze error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


def _freeze_agent_protocol_with_deviation(args: argparse.Namespace) -> int:
    try:
        config_path, config = _load_cli_config(args.config)
        result = freeze_agent_protocol_with_deviation(config, config_path)
    except (OSError, RuntimeError, ValueError, ValidationError) as exc:
        print(f"Agent deviation freeze error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


def _summarize_agent_runs(args: argparse.Namespace) -> int:
    try:
        config_path, config = _load_cli_config(args.config)
        result = summarize_runs(config, config_path)
    except (OSError, RuntimeError, ValueError, ValidationError) as exc:
        print(f"Agent run summary error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="traceanchor")
    subparsers = parser.add_subparsers(dest="command", required=True)
    env_parser = subparsers.add_parser(
        "env-check", help="validate configuration and freeze the local environment manifest"
    )
    env_parser.add_argument("--config", required=True)
    env_parser.set_defaults(handler=_env_check)

    manifest_parser = subparsers.add_parser(
        "manifest", help="discover and hash raw LID-DS scenario files"
    )
    manifest_parser.add_argument("--config", required=True)
    manifest_parser.set_defaults(handler=_manifest)

    ingest_parser = subparsers.add_parser(
        "ingest", help="stream one or more scenarios to immutable Parquet"
    )
    ingest_parser.add_argument("--config", required=True)
    selection = ingest_parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--scenario")
    selection.add_argument("--example", action="store_true")
    selection.add_argument("--family-sample", action="store_true")
    selection.add_argument("--all", action="store_true")
    ingest_parser.add_argument("--resume", action="store_true")
    ingest_parser.set_defaults(handler=_ingest)

    validate_parser = subparsers.add_parser(
        "validate-ingest", help="validate ingested schemas, counts, and Evidence IDs"
    )
    validate_parser.add_argument("--config", required=True)
    validate_parser.add_argument("--scenario")
    validate_parser.set_defaults(handler=_validate_ingest)

    migration_parser = subparsers.add_parser(
        "migrate-public-intervals", help="migrate pre-freeze QA public intervals to v2"
    )
    migration_parser.add_argument("--config", required=True)
    migration_parser.set_defaults(handler=_migrate_public_intervals)

    finalize_parser = subparsers.add_parser(
        "finalize-ingest", help="write the WP1 QA report and completion marker"
    )
    finalize_parser.add_argument("--config", required=True)
    finalize_parser.set_defaults(handler=_finalize_ingest)

    split_parser = subparsers.add_parser(
        "make-splits", help="freeze gold, trigger, and Agent scenario splits"
    )
    split_parser.add_argument("--config", required=True)
    split_parser.set_defaults(handler=_make_splits)

    features_parser = subparsers.add_parser(
        "build-features", help="build causal per-second network and host features"
    )
    features_parser.add_argument("--config", required=True)
    features_parser.add_argument(
        "--split", required=True, choices=("train", "validation", "test")
    )
    features_parser.add_argument("--resume", action="store_true")
    features_parser.set_defaults(handler=_build_features)

    datasets_parser = subparsers.add_parser(
        "validate-datasets", help="validate WP2 split, provenance, and causality gates"
    )
    datasets_parser.add_argument("--config", required=True)
    datasets_parser.set_defaults(handler=_validate_datasets)

    train_detector_parser = subparsers.add_parser(
        "train-detector", help="train one independent causal detector branch"
    )
    train_detector_parser.add_argument("--config", required=True)
    train_detector_parser.add_argument(
        "--channel", required=True, choices=("network", "host")
    )
    train_detector_parser.add_argument(
        "--seed", required=True, type=int, choices=(42, 123, 456)
    )
    train_detector_parser.add_argument("--resume", action="store_true")
    train_detector_parser.add_argument("--smoke-test", action="store_true")
    train_detector_parser.set_defaults(handler=_train_detector)

    score_detector_parser = subparsers.add_parser(
        "score-detector", help="write per-second detector scores"
    )
    score_detector_parser.add_argument("--config", required=True)
    score_detector_parser.add_argument(
        "--split", required=True, choices=("train", "validation", "test")
    )
    score_detector_parser.add_argument("--all-seeds", action="store_true")
    score_detector_parser.add_argument("--channel", choices=("network", "host"))
    score_detector_parser.add_argument("--seed", type=int, choices=(42, 123, 456))
    score_detector_parser.set_defaults(handler=_score_detector)

    calibrate_parser = subparsers.add_parser(
        "calibrate-alerts", help="freeze Validation-only alert parameters"
    )
    calibrate_parser.add_argument("--config", required=True)
    calibrate_parser.set_defaults(handler=_calibrate_alerts)

    evaluate_parser = subparsers.add_parser(
        "evaluate-alerts", help="materialize and evaluate frozen Test alerts"
    )
    evaluate_parser.add_argument("--config", required=True)
    evaluate_parser.set_defaults(handler=_evaluate_alerts)

    evidence_parser = subparsers.add_parser(
        "build-evidence-store", help="build isolated Agent/Evaluator Evidence Ledgers"
    )
    evidence_parser.add_argument("--config", required=True)
    evidence_parser.add_argument(
        "--selection", choices=("example", "family_sample", "all"), default="all"
    )
    evidence_parser.add_argument("--resume", action="store_true")
    evidence_parser.set_defaults(handler=_build_evidence_store)

    blind_parser = subparsers.add_parser(
        "validate-blind-view", help="run Agent catalog leakage and isolation gates"
    )
    blind_parser.add_argument("--config", required=True)
    blind_parser.set_defaults(handler=_validate_blind_view)

    smoke_parser = subparsers.add_parser(
        "tool-smoke", help="run deterministic positive and negative tool chains"
    )
    smoke_parser.add_argument("--config", required=True)
    smoke_parser.add_argument("--all-families", action="store_true")
    smoke_parser.set_defaults(handler=_tool_smoke)

    gold_sample_parser = subparsers.add_parser(
        "sample-gold", help="verify frozen gold cases and build evaluator-only candidate tasks"
    )
    gold_sample_parser.add_argument("--config", required=True)
    gold_sample_parser.add_argument(
        "--split", choices=("development", "test", "all"), default="development"
    )
    gold_sample_parser.add_argument(
        "--incident",
        action="append",
        help=(
            "rebuild only this frozen incident; repeat for multiple incidents and "
            "write repair-specific audit outputs"
        ),
    )
    gold_sample_parser.set_defaults(handler=_sample_gold)

    annotate_parser = subparsers.add_parser(
        "annotate", help="create a non-overwriting human annotation draft"
    )
    annotate_parser.add_argument("--config", required=True)
    annotate_parser.add_argument("--incident", required=True)
    annotate_parser.add_argument("--annotator", required=True)
    annotate_parser.set_defaults(handler=_annotate)

    export_pilot_parser = subparsers.add_parser(
        "export-pilot-csv",
        help="export one evaluator-only Development pilot worksheet",
    )
    export_pilot_parser.add_argument("--config", required=True)
    export_pilot_parser.add_argument("--output", required=True)
    export_pilot_parser.add_argument(
        "--selection", choices=WORKSHEET_SELECTIONS, default="pilot"
    )
    export_pilot_parser.set_defaults(handler=_export_pilot_csv)

    import_pilot_parser = subparsers.add_parser(
        "import-pilot-csv",
        help="validate and import one completed human pilot worksheet",
    )
    import_pilot_parser.add_argument("--config", required=True)
    import_pilot_parser.add_argument("--input", required=True)
    import_pilot_parser.add_argument("--annotator", required=True)
    import_pilot_parser.add_argument(
        "--selection", choices=WORKSHEET_SELECTIONS, default="pilot"
    )
    import_pilot_parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="import completed incidents only when every skipped incident is entirely blank",
    )
    import_pilot_parser.set_defaults(handler=_import_pilot_csv)

    gold_validate_parser = subparsers.add_parser(
        "validate-gold", help="validate annotation schemas and Evidence IDs"
    )
    gold_validate_parser.add_argument("--config", required=True)
    gold_validate_parser.set_defaults(handler=_validate_gold)

    agreement_parser = subparsers.add_parser(
        "agreement", help="compute pre-adjudication human agreement"
    )
    agreement_parser.add_argument("--config", required=True)
    agreement_parser.set_defaults(handler=_agreement)

    adjudicate_parser = subparsers.add_parser(
        "adjudicate", help="create human reconciliation packets and empty drafts"
    )
    adjudicate_parser.add_argument("--config", required=True)
    adjudicate_parser.add_argument("--incident")
    adjudicate_parser.set_defaults(handler=_adjudicate)

    agent_dev_parser = subparsers.add_parser(
        "run-agent-dev", help="run blinded Agent Development cases"
    )
    agent_dev_parser.add_argument("--config", required=True)
    agent_dev_parser.add_argument(
        "--provider", choices=("mock", "primary", "candidate"), default="mock"
    )
    agent_dev_parser.add_argument("--candidate-index", type=int)
    agent_dev_parser.add_argument("--limit", type=int)
    agent_dev_parser.add_argument("--run-nonce", default="development-v1")
    agent_dev_parser.add_argument("--resume", action="store_true")
    agent_dev_parser.set_defaults(handler=_run_agent_dev)

    freeze_agent_parser = subparsers.add_parser(
        "freeze-agent-protocol", help="freeze the reviewed Agent protocol and unlock Test"
    )
    freeze_agent_parser.add_argument("--config", required=True)
    freeze_agent_parser.set_defaults(handler=_freeze_agent_protocol)

    deviation_freeze_parser = subparsers.add_parser(
        "freeze-agent-protocol-with-deviation",
        help="record the reviewed best-available protocol deviation",
    )
    deviation_freeze_parser.add_argument("--config", required=True)
    deviation_freeze_parser.set_defaults(handler=_freeze_agent_protocol_with_deviation)

    summarize_agent_parser = subparsers.add_parser(
        "summarize-runs", help="validate and summarize blinded Agent run manifests"
    )
    summarize_agent_parser.add_argument("--config", required=True)
    summarize_agent_parser.set_defaults(handler=_summarize_agent_runs)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
