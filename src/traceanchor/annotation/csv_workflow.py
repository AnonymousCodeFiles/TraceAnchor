from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb

from traceanchor.annotation.schemas import GoldAnnotation
from traceanchor.annotation.workflow import (
    _load_frozen_cases,
    _validate_against_ledger,
    _validate_core_claim_contract,
)
from traceanchor.config import TraceAnchorConfig, config_hash
from traceanchor.ingest.common import atomic_write_json, sha256_file


WORKSHEET_VERSION = "traceanchor-pilot-label-v1"
CLASSIFICATION = "PRIVATE GROUND TRUTH - EVALUATOR ONLY"
WORKSHEET_SELECTIONS = (
    "pilot",
    "remaining-primary",
    "remaining-secondary",
    "double-overlap",
    "entity-repair",
    "round4-calibration",
    "round4-remaining",
)
ENTITY_REPAIR_INCIDENTS = {"INC-031", "INC-033"}
FIELDNAMES = [
    "worksheet_version",
    "classification",
    "row_id",
    "row_sha256",
    "incident_id",
    "private_family",
    "scenario_token",
    "row_type",
    "candidate_id",
    "timestamp_ns",
    "anchor_delta_ms",
    "content_json",
    "label_help",
    "label",
]


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def immutable_row_hash(row: dict[str, str]) -> str:
    immutable = {
        key: row[key]
        for key in FIELDNAMES
        if key not in {"row_sha256", "label"}
    }
    return hashlib.sha256(_canonical_json(immutable).encode("utf-8")).hexdigest()


def _row(
    *,
    incident_id: str,
    private_family: str,
    scenario_token: str,
    row_type: str,
    candidate_id: str,
    content: Any,
    label_help: str,
    timestamp_ns: int | None = None,
    anchor_delta_ms: float | None = None,
) -> dict[str, str]:
    row = {
        "worksheet_version": WORKSHEET_VERSION,
        "classification": CLASSIFICATION,
        "row_id": f"{incident_id}:{row_type}:{candidate_id}",
        "row_sha256": "",
        "incident_id": incident_id,
        "private_family": private_family,
        "scenario_token": scenario_token,
        "row_type": row_type,
        "candidate_id": candidate_id,
        "timestamp_ns": "" if timestamp_ns is None else str(timestamp_ns),
        "anchor_delta_ms": (
            "" if anchor_delta_ms is None else f"{anchor_delta_ms:.6f}"
        ),
        "content_json": _canonical_json(content),
        "label_help": label_help,
        "label": "",
    }
    row["row_sha256"] = immutable_row_hash(row)
    return row


def _slot_rows(
    task: dict[str, Any], row_type: str, count: int, label_help: str
) -> list[dict[str, str]]:
    return [
        _row(
            incident_id=str(task["incident_id"]),
            private_family=str(task["private_family"]),
            scenario_token=str(task["scenario_token"]),
            row_type=row_type,
            candidate_id=f"{row_type.lower()}_{index:03d}",
            content={"slot": index, "blank_means": "unused"},
            label_help=label_help,
        )
        for index in range(1, count + 1)
    ]


def _task_rows(task: dict[str, Any]) -> list[dict[str, str]]:
    incident_id = str(task["incident_id"])
    private_family = str(task["private_family"])
    scenario_token = str(task["scenario_token"])
    anchor_times = [int(item["ts_ns"]) for item in task["private_anchors"]]
    primary_anchor = min(anchor_times)
    rows = [
        _row(
            incident_id=incident_id,
            private_family=private_family,
            scenario_token=scenario_token,
            row_type="CASE",
            candidate_id="case_review",
            content={
                "investigation_start_ts_ns": task["investigation_start_ts_ns"],
                "investigation_end_ts_ns": task["investigation_end_ts_ns"],
                "private_anchors": task["private_anchors"],
                "private_scenario": task["private_scenario"],
                "candidate_counts": task["candidate_counts"],
                "candidate_truncated": task["candidate_truncated"],
                "source_shards": task["source_shards"],
            },
            label_help=(
                '必填 JSON，例如 {"review_complete":true,'
                '"annotation_confidence":0.9,"root_cause_ambiguous":false,'
                '"notes":""}；只有完成证据反查后才能设 review_complete=true'
            ),
        )
    ]
    for event in task["candidate_events"]:
        timestamp = int(event["ts_ns"])
        rows.append(
            _row(
                incident_id=incident_id,
                private_family=private_family,
                scenario_token=scenario_token,
                row_type="EVIDENCE",
                candidate_id=str(event["evidence_id"]),
                timestamp_ns=timestamp,
                anchor_delta_ms=(timestamp - primary_anchor) / 1_000_000,
                content=event,
                label_help="留空=不选；入选时只填 CORE 或 SUPPORTING",
            )
        )
    for entity in task["candidate_entities"]:
        rows.append(
            _row(
                incident_id=incident_id,
                private_family=private_family,
                scenario_token=scenario_token,
                row_type="ENTITY",
                candidate_id=str(entity["entity_id"]),
                content=entity,
                label_help=(
                    '留空=非根因；选中时填 JSON：{"select":true,"confidence":0.9,'
                    '"rationale":"依据","evidence_ids":["Evidence ID"]}'
                ),
            )
        )
    rows.extend(
        _slot_rows(
            task,
            "ADDITIONAL_EVIDENCE",
            50,
            (
                '留空=不用；补充候选集外证据时填 JSON：{"evidence_id":"...",'
                '"class":"CORE"}，class 只能为 CORE 或 SUPPORTING'
            ),
        )
    )
    rows.extend(
        _slot_rows(
            task,
            "STEP",
            20,
            (
                '留空=不用；使用时填 JSON：{"step_type":"execution",'
                '"summary":"可观察步骤","start_ts_ns":0,"end_ts_ns":0,'
                '"evidence_ids":["..."],"confidence":0.9}'
            ),
        )
    )
    rows.extend(
        _slot_rows(
            task,
            "PROVENANCE_EDGE",
            30,
            (
                '留空=不用；使用时填 JSON：{"source_evidence_id":"...",'
                '"target_evidence_id":"...","relation_type":"provenance",'
                '"basis_evidence_ids":["..."],"rationale":"依据",'
                '"confidence":0.9}；relation_type 可为 provenance/precedes/'
                "correlates_with"
            ),
        )
    )
    rows.extend(
        _slot_rows(
            task,
            "POSSIBLY_CAUSAL_EDGE",
            20,
            (
                '留空=不用；使用时填 JSON：{"source_evidence_id":"...",'
                '"target_evidence_id":"...","basis_evidence_ids":["..."],'
                '"rationale":"可能因果依据","confidence":0.8}'
            ),
        )
    )
    rows.extend(
        _slot_rows(
            task,
            "ATTACK",
            20,
            (
                '留空=不用；使用时填 JSON：{"technique_id":"T0000",'
                '"name":"名称","step_ids":["step_01"],'
                '"evidence_ids":["..."],"mapping_reason":"可观察行为",'
                '"confidence":0.8,"candidate_technique_ids":[]}'
            ),
        )
    )
    rows.extend(
        _slot_rows(
            task,
            "GAP",
            20,
            (
                '留空=不用；使用时填 JSON：{"description":"缺失内容",'
                '"impact":"root_cause"}；impact 可为 root_cause/step/edge/'
                "technique/other"
            ),
        )
    )
    return rows


def _selected_cases(
    config: TraceAnchorConfig,
    config_path: Path,
    selection: str,
) -> list[dict[str, Any]]:
    if selection not in WORKSHEET_SELECTIONS:
        raise ValueError(
            f"selection must be one of: {', '.join(WORKSHEET_SELECTIONS)}"
        )
    cases = _load_frozen_cases(config, config_path)
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        by_family[str(case["family_private"])].append(case)
    development = {
        family: family_cases
        for family, family_cases in by_family.items()
        if all(case["agent_split"] == "development" for case in family_cases)
    }
    test = {
        family: family_cases
        for family, family_cases in by_family.items()
        if all(case["agent_split"] == "test" for case in family_cases)
    }
    if len(development) != 5 or len(test) != 10:
        raise ValueError("frozen cases must contain five Development and ten Test families")

    def ranked_case(family_cases: list[dict[str, Any]], rank: int) -> dict[str, Any]:
        matches = [
            case
            for case in family_cases
            if int(case["family_selection_rank"]) == rank
        ]
        if len(matches) != 1:
            raise ValueError(f"family must contain exactly one frozen rank-{rank} case")
        return matches[0]

    pilot = [ranked_case(family_cases, 1) for family_cases in development.values()]
    pilot_ids = {str(case["incident_id"]) for case in pilot}
    if selection == "pilot":
        selected = pilot
    elif selection == "remaining-primary":
        selected = [
            case for case in cases if str(case["incident_id"]) not in pilot_ids
        ]
    elif selection == "remaining-secondary":
        selected = [
            *(ranked_case(family_cases, 2) for family_cases in development.values()),
            *(ranked_case(family_cases, 1) for family_cases in test.values()),
        ]
    elif selection == "double-overlap":
        selected = [
            *pilot,
            *(ranked_case(family_cases, 2) for family_cases in development.values()),
            *(ranked_case(family_cases, 1) for family_cases in test.values()),
        ]
    elif selection == "round4-calibration":
        selected = [
            *(ranked_case(family_cases, 1) for family_cases in development.values()),
            *(ranked_case(family_cases, 2) for family_cases in development.values()),
        ]
    elif selection == "round4-remaining":
        selected = [
            ranked_case(family_cases, 1) for family_cases in test.values()
        ]
    else:
        selected = [
            case
            for case in cases
            if str(case["incident_id"]) in ENTITY_REPAIR_INCIDENTS
        ]
    selected.sort(key=lambda item: str(item["incident_id"]))
    expected_counts = {
        "pilot": 5,
        "remaining-primary": 40,
        "remaining-secondary": 15,
        "double-overlap": 20,
        "entity-repair": 2,
        "round4-calibration": 10,
        "round4-remaining": 10,
    }
    if len(selected) != expected_counts[selection]:
        raise ValueError(f"unexpected case count for {selection}: {len(selected)}")
    excludes_pilot = selection in {"remaining-primary", "remaining-secondary"}
    if excludes_pilot and pilot_ids.intersection(
        str(case["incident_id"]) for case in selected
    ):
        raise ValueError(f"{selection} must exclude completed pilot cases")
    return selected


def _load_tasks(
    config: TraceAnchorConfig,
    config_path: Path,
    selection: str,
) -> list[dict[str, Any]]:
    resolved = config.resolved_dict(config_path)
    annotations_dir = Path(resolved["paths"]["annotations_dir"])
    tasks: list[dict[str, Any]] = []
    for case in _selected_cases(config, config_path, selection):
        task_path = (
            annotations_dir
            / "tasks"
            / str(case["agent_split"])
            / f"{case['incident_id']}.json"
        )
        if not task_path.exists():
            raise FileNotFoundError(f"candidate task is missing: {task_path}")
        task = json.loads(task_path.read_text(encoding="utf-8"))
        if (
            task.get("incident_id") != case["incident_id"]
            or task.get("scenario_token") != case["scenario_token"]
            or task.get("private_family") != case["family_private"]
            or task.get("candidate_only") is not True
            or task.get("not_gold") is not True
        ):
            raise ValueError(f"candidate task differs from frozen case: {task_path}")
        tasks.append(task)
    return tasks


def _expected_rows(
    config: TraceAnchorConfig,
    config_path: Path,
    selection: str,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    tasks = _load_tasks(config, config_path, selection)
    return tasks, [row for task in tasks for row in _task_rows(task)]


def export_development_pilot_csv(
    config: TraceAnchorConfig,
    config_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    return export_annotation_csv(
        config,
        config_path,
        output_path,
        selection="pilot",
    )


def export_annotation_csv(
    config: TraceAnchorConfig,
    config_path: Path,
    output_path: Path,
    *,
    selection: str,
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(
            f"worksheet already exists; refusing to overwrite labels: {output_path}"
        )
    tasks, rows = _expected_rows(config, config_path, selection)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    counts = Counter(row["row_type"] for row in rows)
    return {
        "schema_version": 1,
        "classification": CLASSIFICATION,
        "worksheet_version": WORKSHEET_VERSION,
        "selection": selection,
        "output": str(output_path),
        "sha256": sha256_file(output_path),
        "incidents": [str(item["incident_id"]) for item in tasks],
        "families": len({str(item["private_family"]) for item in tasks}),
        "rows": len(rows),
        "row_type_counts": dict(sorted(counts.items())),
        "label_column": FIELDNAMES[-1],
        "labels_prefilled": 0,
        "candidate_only": True,
        "automatically_selected_gold": False,
    }


def _parse_json_label(row: dict[str, str]) -> dict[str, Any]:
    try:
        value = json.loads(row["label"])
    except json.JSONDecodeError as exc:
        raise ValueError(f"{row['row_id']} label is not valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{row['row_id']} label must be a JSON object")
    return value


def _require_label_keys(
    row: dict[str, str],
    value: dict[str, Any],
    *,
    required: set[str],
    optional: set[str] | None = None,
) -> None:
    optional = optional or set()
    missing = required.difference(value)
    extra = set(value).difference(required.union(optional))
    if missing:
        raise ValueError(f"{row['row_id']} label is missing keys: {sorted(missing)}")
    if extra:
        raise ValueError(f"{row['row_id']} label has unknown keys: {sorted(extra)}")


def _read_and_verify_rows(
    config: TraceAnchorConfig,
    config_path: Path,
    input_path: Path,
    *,
    selection: str,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    tasks, expected_rows = _expected_rows(config, config_path, selection)
    with input_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != FIELDNAMES:
            raise ValueError("worksheet columns differ from the frozen CSV schema")
        actual_rows = list(reader)
    if len(actual_rows) != len(expected_rows):
        raise ValueError(
            f"worksheet row count changed: {len(actual_rows)} != {len(expected_rows)}"
        )
    immutable_fields = [
        item for item in FIELDNAMES if item not in {"row_sha256", "label"}
    ]
    for index, (actual, expected) in enumerate(
        zip(actual_rows, expected_rows, strict=True), start=2
    ):
        if any(actual[field] != expected[field] for field in immutable_fields):
            raise ValueError(f"immutable worksheet content changed at CSV line {index}")
        if actual["row_sha256"] != expected["row_sha256"]:
            raise ValueError(f"row_sha256 changed at CSV line {index}")
        if actual["row_sha256"] != immutable_row_hash(actual):
            raise ValueError(f"row integrity check failed at CSV line {index}")
    return tasks, actual_rows


def _annotation_from_rows(
    task: dict[str, Any],
    rows: list[dict[str, str]],
    annotator_id: str,
    completed_at: datetime,
) -> GoldAnnotation:
    core: list[str] = []
    supporting: list[str] = []
    roots: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    provenance_edges: list[dict[str, Any]] = []
    possibly_causal_edges: list[dict[str, Any]] = []
    attack_techniques: list[dict[str, Any]] = []
    gaps: list[dict[str, Any]] = []
    case_label: dict[str, Any] | None = None
    for row in rows:
        label = row["label"].strip()
        if not label:
            continue
        row_type = row["row_type"]
        if row_type == "EVIDENCE":
            normalized = label.upper()
            if normalized == "CORE":
                core.append(row["candidate_id"])
            elif normalized == "SUPPORTING":
                supporting.append(row["candidate_id"])
            else:
                raise ValueError(
                    f"{row['row_id']} label must be CORE, SUPPORTING, or blank"
                )
            continue
        value = _parse_json_label(row)
        if row_type == "CASE":
            _require_label_keys(
                row,
                value,
                required={
                    "review_complete",
                    "annotation_confidence",
                    "root_cause_ambiguous",
                },
                optional={"notes"},
            )
            if value["review_complete"] is not True:
                raise ValueError(f"{row['row_id']} is not marked review_complete=true")
            if case_label is not None:
                raise ValueError(f"multiple CASE labels for {task['incident_id']}")
            case_label = value
        elif row_type == "ADDITIONAL_EVIDENCE":
            _require_label_keys(
                row, value, required={"evidence_id", "class"}
            )
            classification = str(value["class"]).upper()
            if classification == "CORE":
                core.append(str(value["evidence_id"]))
            elif classification == "SUPPORTING":
                supporting.append(str(value["evidence_id"]))
            else:
                raise ValueError(f"{row['row_id']} class must be CORE or SUPPORTING")
        elif row_type == "ENTITY":
            _require_label_keys(
                row,
                value,
                required={"select", "confidence", "rationale", "evidence_ids"},
            )
            if value.pop("select") is not True:
                raise ValueError(f"{row['row_id']} selected entity must set select=true")
            content = json.loads(row["content_json"])
            roots.append(
                {
                    "entity_id": row["candidate_id"],
                    "entity_type": content["entity_type"],
                    **value,
                }
            )
        elif row_type == "STEP":
            steps.append({"step_id": row["candidate_id"], **value})
        elif row_type == "PROVENANCE_EDGE":
            provenance_edges.append(value)
        elif row_type == "POSSIBLY_CAUSAL_EDGE":
            possibly_causal_edges.append(
                {"relation_type": "possibly_causes", **value}
            )
        elif row_type == "ATTACK":
            attack_techniques.append(value)
        elif row_type == "GAP":
            gaps.append({"gap_id": row["candidate_id"], **value})
        else:
            raise ValueError(f"unsupported labeled row type: {row_type}")
    if case_label is None:
        raise ValueError(f"CASE label is required for {task['incident_id']}")
    annotation = GoldAnnotation.model_validate(
        {
            "schema_version": 1,
            "status": "completed",
            "annotation_id": f"ann:{task['incident_id']}:{annotator_id}",
            "incident_id": task["incident_id"],
            "private_family": task["private_family"],
            "scenario_token": task["scenario_token"],
            "agent_split": task["agent_split"],
            "investigation_start_ts_ns": task["investigation_start_ts_ns"],
            "investigation_end_ts_ns": task["investigation_end_ts_ns"],
            "anchor_times_ns": [int(item["ts_ns"]) for item in task["private_anchors"]],
            "root_cause_ambiguous": case_label["root_cause_ambiguous"],
            "root_cause_entities": roots,
            "core_evidence_ids": core,
            "supporting_evidence_ids": supporting,
            "steps": steps,
            "provenance_edges": provenance_edges,
            "possibly_causal_edges": possibly_causal_edges,
            "attack_techniques": attack_techniques,
            "known_evidence_gaps": gaps,
            "annotation_confidence": case_label["annotation_confidence"],
            "metadata": {
                "annotator_id": annotator_id,
                "annotation_mode": "independent",
                "started_at": completed_at,
                "completed_at": completed_at,
                "human_verified": True,
                "notes": case_label.get("notes"),
            },
        }
    )
    _validate_core_claim_contract(annotation)
    return annotation


def import_development_pilot_csv(
    config: TraceAnchorConfig,
    config_path: Path,
    input_path: Path,
    *,
    annotator_id: str,
) -> dict[str, Any]:
    return import_annotation_csv(
        config,
        config_path,
        input_path,
        annotator_id=annotator_id,
        selection="pilot",
        allow_incomplete=False,
    )


def import_annotation_csv(
    config: TraceAnchorConfig,
    config_path: Path,
    input_path: Path,
    *,
    annotator_id: str,
    selection: str,
    allow_incomplete: bool = False,
) -> dict[str, Any]:
    tasks, rows = _read_and_verify_rows(
        config,
        config_path,
        input_path,
        selection=selection,
    )
    rows_by_incident: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        rows_by_incident[row["incident_id"]].append(row)
    skipped_incidents: list[str] = []
    completed_tasks: list[dict[str, Any]] = []
    for task in tasks:
        incident_id = str(task["incident_id"])
        incident_rows = rows_by_incident[incident_id]
        case_labels = [
            row["label"].strip()
            for row in incident_rows
            if row["row_type"] == "CASE"
        ]
        if len(case_labels) != 1:
            raise ValueError(f"expected exactly one CASE row for {incident_id}")
        if case_labels[0] or not allow_incomplete:
            completed_tasks.append(task)
            continue
        if any(row["label"].strip() for row in incident_rows):
            raise ValueError(
                f"{incident_id} has labels but no completed CASE label; "
                "partial incidents cannot be imported"
            )
        skipped_incidents.append(incident_id)
    completed_at = datetime.fromtimestamp(input_path.stat().st_mtime, timezone.utc)
    annotations = [
        _annotation_from_rows(
            task,
            rows_by_incident[str(task["incident_id"])],
            annotator_id,
            completed_at,
        )
        for task in completed_tasks
    ]
    resolved = config.resolved_dict(config_path)
    frozen_cases = _load_frozen_cases(config, config_path)
    by_incident = {str(item["incident_id"]): item for item in frozen_cases}
    ledger_errors: list[str] = []
    with duckdb.connect(str(resolved["paths"]["evidence_db"]), read_only=True) as agent:
        with duckdb.connect(
            str(resolved["paths"]["evaluator_db"]), read_only=True
        ) as evaluator:
            versions = {
                str(item[0])
                for item in agent.execute(
                    "select distinct version from attack_knowledge"
                ).fetchall()
            }
            if versions != {config.gold_annotation.attack_knowledge_version}:
                ledger_errors.append(
                    "ATT&CK snapshot version differs from frozen configuration"
                )
            knowledge_ids = {
                str(item[0])
                for item in agent.execute(
                    "select technique_id from attack_knowledge"
                ).fetchall()
            }
            for annotation in annotations:
                ledger_errors.extend(
                    f"{annotation.annotation_id}: {error}"
                    for error in _validate_against_ledger(
                        annotation,
                        by_incident[annotation.incident_id],
                        agent,
                        evaluator,
                        knowledge_ids,
                    )
                )
    if ledger_errors:
        raise ValueError("; ".join(ledger_errors))

    annotations_dir = Path(resolved["paths"]["annotations_dir"])
    output_paths = [
        annotations_dir
        / ("dev" if annotation.agent_split == "development" else "gold")
        / f"{annotation.incident_id}__{annotator_id}.json"
        for annotation in annotations
    ]
    existing = [str(path) for path in output_paths if path.exists()]
    if existing:
        raise FileExistsError(
            f"refusing to overwrite existing human annotations: {existing}"
        )
    for path, annotation in zip(output_paths, annotations, strict=True):
        atomic_write_json(path, annotation.model_dump(mode="json"))
    report = {
        "schema_version": 1,
        "classification": CLASSIFICATION,
        "ok": True,
        "config_sha256": config_hash(config, config_path),
        "worksheet": str(input_path),
        "worksheet_sha256": sha256_file(input_path),
        "worksheet_version": WORKSHEET_VERSION,
        "selection": selection,
        "partial_import": bool(skipped_incidents),
        "worksheet_incidents": len(tasks),
        "annotations_imported": len(annotations),
        "skipped_incidents": skipped_incidents,
        "annotator_id": annotator_id,
        "human_verified": True,
        "automatically_selected_gold": False,
        "labels_nonempty": sum(bool(row["label"].strip()) for row in rows),
        "annotations": [
            {
                "incident_id": annotation.incident_id,
                "annotation_id": annotation.annotation_id,
                "path": str(path),
                "sha256": sha256_file(path),
                "core_evidence": len(annotation.core_evidence_ids),
                "supporting_evidence": len(annotation.supporting_evidence_ids),
                "steps": len(annotation.steps),
            }
            for path, annotation in zip(output_paths, annotations, strict=True)
        ],
    }
    report_path = (
        Path(resolved["paths"]["artifacts_dir"])
        / "reports"
        / f"WP05_csv_import_{annotator_id}.json"
    )
    atomic_write_json(report_path, report)
    return {**report, "report": str(report_path)}


__all__ = [
    "FIELDNAMES",
    "WORKSHEET_SELECTIONS",
    "export_annotation_csv",
    "export_development_pilot_csv",
    "import_annotation_csv",
    "import_development_pilot_csv",
    "immutable_row_hash",
]
