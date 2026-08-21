from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
import resource
import shutil
import statistics
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import duckdb
import pyarrow.parquet as pq
from scipy.stats import kendalltau
from sklearn.metrics import cohen_kappa_score

from traceanchor.annotation.schemas import GoldAnnotation
from traceanchor.config import TraceAnchorConfig, config_hash
from traceanchor.ingest.common import atomic_write_json, sha256_file


_EVIDENCE_ID = re.compile(
    r"^(?:sc|pcap|res):(?P<token>tw_[0-9a-f]{24}):[^:]+(?::[0-9]+)?$"
)


def _mean(values: Iterable[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(value)]
    return statistics.mean(finite) if finite else None


def _set_scores(left: set[str], right: set[str]) -> dict[str, float]:
    if not left and not right:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "jaccard": 1.0}
    overlap = len(left.intersection(right))
    precision = overlap / len(right) if right else 0.0
    recall = overlap / len(left) if left else 0.0
    f1 = 2.0 * overlap / (len(left) + len(right)) if left or right else 1.0
    union = len(left.union(right))
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "jaccard": overlap / union if union else 1.0,
    }


def _annotation_hash(annotation: GoldAnnotation) -> str:
    canonical = json.dumps(
        annotation.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _load_frozen_cases(
    config: TraceAnchorConfig, config_path: Path
) -> list[dict[str, Any]]:
    resolved = config.resolved_dict(config_path)
    splits = Path(resolved["paths"]["splits_dir"])
    gold_path = splits / "gold_reserved.parquet"
    agent_path = splits / "agent_split.parquet"
    trigger_path = splits / "trigger_split.parquet"
    raw_manifest = Path(resolved["paths"]["manifests_dir"]) / "raw_files.parquet"
    for path in (gold_path, agent_path, trigger_path, raw_manifest):
        if not path.exists():
            raise FileNotFoundError(f"frozen WP2 input missing: {path.name}")
    gold = pq.read_table(gold_path).to_pylist()
    agent = {str(row["scenario_token"]): row for row in pq.read_table(agent_path).to_pylist()}
    trigger = {
        str(row["scenario_token"]): row for row in pq.read_table(trigger_path).to_pylist()
    }
    if len(gold) != 45 or len(agent) != 45:
        raise ValueError("frozen gold/Agent manifests must contain exactly 45 cases")
    raw_hash = sha256_file(raw_manifest)
    family_counts = Counter(str(row["family_private"]) for row in gold)
    if len(family_counts) != 15 or set(family_counts.values()) != {3}:
        raise ValueError("frozen gold manifest is not three cases for each of 15 families")
    cases = []
    for row in gold:
        token = str(row["scenario_token"])
        split = agent.get(token)
        trigger_row = trigger.get(token)
        if split is None or trigger_row is None:
            raise ValueError(f"frozen split join missing for {token}")
        if str(row["raw_manifest_sha256"]) != raw_hash:
            raise ValueError("gold selection raw manifest hash changed")
        if str(split["family_private"]) != str(row["family_private"]):
            raise ValueError(f"private family mismatch for {token}")
        if str(trigger_row["trigger_split"]) != "test" or not bool(
            trigger_row["gold_reserved"]
        ):
            raise ValueError(f"gold case is not reserved in Trigger Test: {token}")
        agent_split = str(split["agent_split"])
        locked = bool(split["locked_until_protocol_freeze"])
        if (agent_split == "test") != locked:
            raise ValueError(f"Agent Test lock mismatch for {token}")
        cases.append(
            {
                **row,
                "incident_id": str(split["incident_id"]),
                "agent_split": agent_split,
                "locked_until_protocol_freeze": locked,
            }
        )
    cases.sort(key=lambda item: str(item["incident_id"]))
    if Counter(str(row["agent_split"]) for row in cases) != {
        "development": 15,
        "test": 30,
    }:
        raise ValueError("Agent Development/Test counts changed")
    return cases


def _entity_id(token: str, entity_type: str, canonical_key: str) -> str:
    value = f"{token}|{entity_type}|{canonical_key}"
    return "ent_" + hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]


def _candidate_event_entity_ids(
    token: str, events: list[dict[str, Any]]
) -> set[str]:
    entity_ids: set[str] = set()
    for event in events:
        if event["event_type"] == "packet":
            key = "|".join(
                str(event.get(name)) if event.get(name) is not None else default
                for name, default in (
                    ("ip_protocol", "unknown"),
                    ("src_ip", "?"),
                    ("src_port", "?"),
                    ("dst_ip", "?"),
                    ("dst_port", "?"),
                )
            )
            entity_ids.add(_entity_id(token, "connection", key))
            continue
        if event["event_type"] != "syscall":
            continue
        for name in ("pid", "child_pid", "parent_pid"):
            if event.get(name) is not None:
                entity_ids.add(_entity_id(token, "process", f"pid:{event[name]}"))
        if event.get("tid") is not None:
            entity_ids.add(_entity_id(token, "thread", f"tid:{event['tid']}"))
        if event.get("file_path"):
            entity_ids.add(
                _entity_id(token, "file", f"path:{event['file_path']}")
            )
    return entity_ids


def _candidate_task(
    case: dict[str, Any],
    agent: duckdb.DuckDBPyConnection,
    evaluator: duckdb.DuckDBPyConnection,
    resolved: dict[str, Any],
) -> dict[str, Any]:
    token = str(case["scenario_token"])
    public = agent.execute(
        "select start_ts_ns,end_ts_ns from scenario_public where scenario_token=?",
        [token],
    ).fetchone()
    private = evaluator.execute(
        """select family_private,exploit,exploit_name,image,containers_json
           from scenario_private where scenario_token=?""",
        [token],
    ).fetchone()
    anchors = evaluator.execute(
        """select anchor_id,ts_ns,raw_timestamp,name_private,source_private
           from exploit_anchor where scenario_token=? order by ts_ns,anchor_id""",
        [token],
    ).fetchall()
    if public is None or private is None or not anchors:
        raise ValueError(f"candidate inputs missing for {token}")
    anchor_times = [int(row[1]) for row in anchors]
    start = max(int(public[0]), min(anchor_times) - 30_000_000_000)
    end = min(int(public[1]), max(anchor_times) + 60_000_000_000)
    center = anchor_times[0]

    syscall_rows = agent.execute(
        """select evidence_id,ts_ns,pid,tid,process_name,syscall,direction,
                  result_class,fd,file_path,socket_src_ip,socket_src_port,
                  socket_dst_ip,socket_dst_port,child_pid,parent_pid
           from syscall_event where scenario_token=? and ts_ns between ? and ?
           order by abs(ts_ns-?),ts_ns,line_no limit 300""",
        [token, start, end, center],
    ).fetchall()
    packet_rows = agent.execute(
        """select evidence_id,ts_ns,frame_no,src_ip,src_port,dst_ip,dst_port,
                  ip_protocol,tcp_flags,payload_len
           from network_packet where scenario_token=? and ts_ns between ? and ?
           order by abs(ts_ns-?),ts_ns,frame_no limit 300""",
        [token, start, end, center],
    ).fetchall()
    resource_rows = agent.execute(
        """select evidence_id,ts_ns,row_no,cpu_usage,memory_usage,network_received,
                  network_send,storage_read,storage_written,missing_mask
           from resource_sample where scenario_token=? and ts_ns between ? and ?
           order by abs(ts_ns-?),ts_ns,row_no limit 100""",
        [token, start, end, center],
    ).fetchall()
    syscall_names = [
        "evidence_id",
        "ts_ns",
        "pid",
        "tid",
        "process_name",
        "syscall",
        "direction",
        "result_class",
        "fd",
        "file_path",
        "socket_src_ip",
        "socket_src_port",
        "socket_dst_ip",
        "socket_dst_port",
        "child_pid",
        "parent_pid",
    ]
    packet_names = [
        "evidence_id",
        "ts_ns",
        "frame_no",
        "src_ip",
        "src_port",
        "dst_ip",
        "dst_port",
        "ip_protocol",
        "tcp_flags",
        "payload_len",
    ]
    resource_names = [
        "evidence_id",
        "ts_ns",
        "row_no",
        "cpu_usage",
        "memory_usage",
        "network_received",
        "network_send",
        "storage_read",
        "storage_written",
        "missing_mask",
    ]
    events = [
        {"event_type": "syscall", **dict(zip(syscall_names, row))}
        for row in syscall_rows
    ] + [
        {"event_type": "packet", **dict(zip(packet_names, row))}
        for row in packet_rows
    ] + [
        {"event_type": "resource", **dict(zip(resource_names, row))}
        for row in resource_rows
    ]
    events.sort(key=lambda row: (int(row["ts_ns"]), str(row["evidence_id"])))
    counts = {
        name: int(value)
        for name, value in zip(
            ("syscall", "packet", "resource"),
            agent.execute(
                """select
                     (select count(*) from syscall_event where scenario_token=? and ts_ns between ? and ?),
                     (select count(*) from network_packet where scenario_token=? and ts_ns between ? and ?),
                     (select count(*) from resource_sample where scenario_token=? and ts_ns between ? and ?)""",
                [token, start, end, token, start, end, token, start, end],
            ).fetchone(),
        )
    }
    entity_total = int(
        agent.execute(
            """select count(*) from entity
               where scenario_token=? and last_ts_ns>=? and first_ts_ns<=?""",
            [token, start, end],
        ).fetchone()[0]
    )
    preferred_entity_ids = sorted(_candidate_event_entity_ids(token, events))
    if len(preferred_entity_ids) > 1000:
        raise ValueError(
            f"candidate events resolve to more than 1000 entities for {token}"
        )
    preferred_placeholders = ",".join("?" for _ in preferred_entity_ids)
    preferred_order = (
        f"case when entity_id in ({preferred_placeholders}) then 0 else 1 end,"
        if preferred_entity_ids
        else "0,"
    )
    entity_rows = agent.execute(
        f"""select entity_id,entity_type,canonical_key,canonical_attributes_json,
                   first_ts_ns,last_ts_ns,evidence_count
            from entity where scenario_token=? and last_ts_ns>=? and first_ts_ns<=?
            order by {preferred_order}
                     case when first_ts_ns<=? and last_ts_ns>=? then 0
                          else least(abs(first_ts_ns-?),abs(last_ts_ns-?)) end,
                     evidence_count desc,entity_type,canonical_key
            limit 1000""",
        [token, start, end, *preferred_entity_ids, center, center, center, center],
    ).fetchall()
    loaded_entity_ids = {str(row[0]) for row in entity_rows}
    existing_preferred_ids = (
        {
            str(row[0])
            for row in agent.execute(
                f"""select entity_id from entity
                     where scenario_token=? and entity_id in ({preferred_placeholders})""",
                [token, *preferred_entity_ids],
            ).fetchall()
        }
        if preferred_entity_ids
        else set()
    )
    if not existing_preferred_ids.issubset(loaded_entity_ids):
        raise ValueError(f"candidate-derived entities were truncated for {token}")
    entities = [
        {
            "entity_id": row[0],
            "entity_type": row[1],
            "canonical_key": row[2],
            "canonical_attributes_json": row[3],
            "first_ts_ns": row[4],
            "last_ts_ns": row[5],
            "evidence_count": row[6],
        }
        for row in entity_rows
    ]
    counts["entity"] = entity_total
    return {
        "schema_version": 1,
        "classification": "PRIVATE GROUND TRUTH - EVALUATOR ONLY",
        "candidate_only": True,
        "not_gold": True,
        "human_selection_required": True,
        "incident_id": case["incident_id"],
        "scenario_token": token,
        "private_family": private[0],
        "agent_split": case["agent_split"],
        "locked_until_protocol_freeze": case["locked_until_protocol_freeze"],
        "selection": {
            "family_selection_rank": case["family_selection_rank"],
            "selection_reason": case["selection_reason"],
            "raw_manifest_sha256": case["raw_manifest_sha256"],
        },
        "private_scenario": {
            "exploit": private[1],
            "exploit_name": private[2],
            "image": private[3],
            "containers_json": private[4],
        },
        "private_anchors": [
            {
                "anchor_id": row[0],
                "ts_ns": row[1],
                "raw_timestamp": row[2],
                "name_private": row[3],
                "source_private": row[4],
            }
            for row in anchors
        ],
        "investigation_start_ts_ns": start,
        "investigation_end_ts_ns": end,
        "candidate_counts": counts,
        "candidate_truncated": {
            "syscall": counts["syscall"] > 300,
            "packet": counts["packet"] > 300,
            "resource": counts["resource"] > 100,
            "entity": entity_total > len(entities),
        },
        "candidate_events": events,
        "candidate_entities": entities,
        "source_shards": {
            "public_parquet_dir": str(Path(resolved["paths"]["parquet_dir"]) / token),
            "private_evaluator_dir": str(Path(resolved["paths"]["evaluator_dir"]) / token),
        },
    }


def sample_gold(
    config: TraceAnchorConfig,
    config_path: Path,
    *,
    split: str = "development",
    incident_ids: list[str] | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    if split not in {"development", "test", "all"}:
        raise ValueError("split must be development, test, or all")
    requested_incidents = incident_ids or []
    if len(requested_incidents) != len(set(requested_incidents)):
        raise ValueError("incident IDs for repair must be unique")
    repair = bool(requested_incidents)
    resolved = config.resolved_dict(config_path)
    completion = Path(resolved["paths"]["completion_markers_dir"])
    if not (completion / "WP4.done").exists():
        raise RuntimeError("WP4 completion marker is required")
    cases = _load_frozen_cases(config, config_path)
    if split in {"test", "all"}:
        pilot_errors = _pilot_gate_errors(config, config_path, cases)
        if pilot_errors:
            raise RuntimeError(
                "development codebook pilot agreement must pass before Test tasks: "
                + "; ".join(pilot_errors)
            )
    split_cases = (
        cases
        if split == "all"
        else [row for row in cases if row["agent_split"] == split]
    )
    if repair:
        known_incidents = {str(row["incident_id"]) for row in cases}
        unknown = sorted(set(requested_incidents).difference(known_incidents))
        if unknown:
            raise ValueError(f"unknown incident IDs for repair: {unknown}")
        selected = [
            row
            for row in split_cases
            if str(row["incident_id"]) in set(requested_incidents)
        ]
        wrong_split = sorted(
            set(requested_incidents).difference(
                str(row["incident_id"]) for row in selected
            )
        )
        if wrong_split:
            raise ValueError(
                f"incident IDs are outside requested split {split}: {wrong_split}"
            )
    else:
        selected = split_cases
    annotations_dir = Path(resolved["paths"]["annotations_dir"])
    artifacts_dir = Path(resolved["paths"]["artifacts_dir"])
    schema_path = config_path.parent / "schemas" / "gold_annotation.schema.json"
    codebook_path = annotations_dir / "codebook" / "README.md"
    if not codebook_path.exists():
        raise FileNotFoundError("gold annotation codebook is missing")
    atomic_write_json(schema_path, GoldAnnotation.model_json_schema())
    manifest_path = artifacts_dir / "annotations" / "gold_case_manifest.json"
    atomic_write_json(
        manifest_path,
        {
            "schema_version": 1,
            "classification": "PRIVATE GROUND TRUTH - EVALUATOR ONLY",
            "config_sha256": config_hash(config, config_path),
            "cases": cases,
        },
    )
    task_hashes: dict[str, str] = {}
    previous_task_hashes: dict[str, str] = {}
    archived_tasks: dict[str, str] = {}
    generated_tasks: list[tuple[dict[str, Any], dict[str, Any]]] = []
    with duckdb.connect(str(resolved["paths"]["evidence_db"]), read_only=True) as agent:
        with duckdb.connect(str(resolved["paths"]["evaluator_db"]), read_only=True) as evaluator:
            knowledge_version = agent.execute(
                "select distinct version from attack_knowledge"
            ).fetchall()
            if knowledge_version != [(config.gold_annotation.attack_knowledge_version,)]:
                raise ValueError("configured ATT&CK version differs from Evidence Ledger")
            for case in selected:
                generated_tasks.append(
                    (case, _candidate_task(case, agent, evaluator, resolved))
                )
    for case, task in generated_tasks:
        incident_id = str(case["incident_id"])
        task_path = (
            annotations_dir
            / "tasks"
            / str(case["agent_split"])
            / f"{incident_id}.json"
        )
        if repair and task_path.exists():
            previous_hash = sha256_file(task_path)
            archive_path = (
                annotations_dir
                / "history"
                / "task_repairs"
                / f"{incident_id}__{previous_hash}.json"
            )
            archive_path.parent.mkdir(parents=True, exist_ok=True)
            if archive_path.exists():
                if sha256_file(archive_path) != previous_hash:
                    raise ValueError(f"task repair archive hash mismatch: {archive_path}")
            else:
                shutil.copy2(task_path, archive_path)
            previous_task_hashes[incident_id] = previous_hash
            archived_tasks[incident_id] = str(archive_path)
        atomic_write_json(task_path, task)
        task_hashes[incident_id] = sha256_file(task_path)
    report = {
        "schema_version": 1,
        "ok": True,
        "classification": "PRIVATE GROUND TRUTH - EVALUATOR ONLY",
        "frozen_cases": len(cases),
        "development_cases": sum(row["agent_split"] == "development" for row in cases),
        "test_cases": sum(row["agent_split"] == "test" for row in cases),
        "generated_split": split,
        "generated_tasks": len(selected),
        "repair": repair,
        "requested_incidents": requested_incidents,
        "candidate_only": True,
        "automatically_selected_gold": False,
        "manifest_sha256": sha256_file(manifest_path),
        "schema_sha256": sha256_file(schema_path),
        "codebook_sha256": sha256_file(codebook_path),
        "elapsed_seconds": time.monotonic() - started,
        "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        * 1024,
        "task_hashes": task_hashes,
        "previous_task_hashes": previous_task_hashes,
        "archived_tasks": archived_tasks,
    }
    report_path = artifacts_dir / "reports" / (
        "WP05_sampling_repair.json" if repair else "WP05_sampling.json"
    )
    atomic_write_json(report_path, report)
    atomic_write_json(
        completion
        / ("WP5_sampling_repair.done" if repair else f"WP5_sampling_{split}.done"),
        {
            "schema_version": 1,
            "report_sha256": sha256_file(report_path),
            "config_sha256": config_hash(config, config_path),
        },
    )
    return report


def create_annotation_draft(
    config: TraceAnchorConfig,
    config_path: Path,
    *,
    incident_id: str,
    annotator_id: str,
) -> dict[str, Any]:
    cases = _load_frozen_cases(config, config_path)
    matches = [row for row in cases if row["incident_id"] == incident_id]
    if len(matches) != 1:
        raise ValueError("unknown incident_id")
    case = matches[0]
    resolved = config.resolved_dict(config_path)
    annotations_dir = Path(resolved["paths"]["annotations_dir"])
    task_path = (
        annotations_dir / "tasks" / str(case["agent_split"]) / f"{incident_id}.json"
    )
    if not task_path.exists():
        raise FileNotFoundError(
            f"candidate task missing for {incident_id}; run sample-gold for its split"
        )
    task = json.loads(task_path.read_text(encoding="utf-8"))
    now = datetime.now(timezone.utc)
    draft = GoldAnnotation.model_validate(
        {
            "schema_version": 1,
            "status": "draft",
            "annotation_id": f"ann:{incident_id}:{annotator_id}",
            "incident_id": incident_id,
            "private_family": case["family_private"],
            "scenario_token": case["scenario_token"],
            "agent_split": case["agent_split"],
            "investigation_start_ts_ns": task["investigation_start_ts_ns"],
            "investigation_end_ts_ns": task["investigation_end_ts_ns"],
            "anchor_times_ns": [row["ts_ns"] for row in task["private_anchors"]],
            "metadata": {
                "annotator_id": annotator_id,
                "annotation_mode": "independent",
                "started_at": now,
                "human_verified": False,
            },
        }
    )
    directory = "dev" if case["agent_split"] == "development" else "gold"
    output = annotations_dir / directory / f"{incident_id}__{annotator_id}.json"
    if output.exists():
        raise FileExistsError("annotation draft exists; refusing to overwrite human work")
    atomic_write_json(output, draft.model_dump(mode="json"))
    return {
        "schema_version": 1,
        "classification": "PRIVATE GROUND TRUTH - EVALUATOR ONLY",
        "incident_id": incident_id,
        "task": str(task_path),
        "draft": str(output),
        "instructions": "Edit the draft manually; candidates are not gold. Set completed/human_verified only after reverse-checking every Evidence ID.",
    }


def _annotation_paths(annotations_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for directory in ("dev", "gold", "adjudicated"):
        paths.extend(sorted((annotations_dir / directory).glob("*.json")))
    return sorted(paths)


def _annotation_evidence(annotation: GoldAnnotation) -> set[str]:
    return set(annotation.core_evidence_ids).union(annotation.supporting_evidence_ids)


def _validate_core_claim_contract(annotation: GoldAnnotation) -> None:
    """Enforce the revision 6 minimal claim-to-core transition contract."""
    core = set(annotation.core_evidence_ids)
    claimed_core: set[str] = set()
    for entity in annotation.root_cause_entities:
        linked = core.intersection(entity.evidence_ids)
        if len(linked) != 1:
            raise ValueError(
                f"{annotation.incident_id} root entity {entity.entity_id} must "
                "reference exactly one CORE Evidence ID"
            )
        claimed_core.update(linked)
    for step in annotation.steps:
        linked = core.intersection(step.evidence_ids)
        if len(linked) != 1:
            raise ValueError(
                f"{annotation.incident_id} step {step.step_id} must reference "
                "exactly one CORE Evidence ID"
            )
        claimed_core.update(linked)
    unclaimed = sorted(core.difference(claimed_core))
    if unclaimed:
        raise ValueError(
            f"{annotation.incident_id} CORE Evidence IDs are not claimed by a "
            f"root entity or step: {unclaimed}"
        )
    step_types = [step.step_type for step in annotation.steps]
    duplicate_types = sorted(
        step_type
        for step_type, count in Counter(step_types).items()
        if count > 1
    )
    if duplicate_types:
        raise ValueError(
            f"{annotation.incident_id} step types must be unique: "
            f"{duplicate_types}"
        )
    present = set(step_types)
    precedence = (
        ("request_read", "socket_accept"),
        ("execution", "process_spawn"),
        ("file_write", "file_read"),
    )
    for preferred, redundant in precedence:
        if {preferred, redundant}.issubset(present):
            raise ValueError(
                f"{annotation.incident_id} {preferred} supersedes "
                f"{redundant} in the minimal incident chain"
            )


def _adjudication_contract_errors(annotation: GoldAnnotation) -> list[str]:
    if (
        annotation.status != "completed"
        or annotation.metadata.annotation_mode != "adjudicated"
    ):
        return []
    try:
        _validate_core_claim_contract(annotation)
    except ValueError as exc:
        return [f"revision 6 adjudication contract: {exc}"]
    return []


def _validate_against_ledger(
    annotation: GoldAnnotation,
    case: dict[str, Any],
    agent: duckdb.DuckDBPyConnection,
    evaluator: duckdb.DuckDBPyConnection,
    knowledge_ids: set[str],
) -> list[str]:
    errors: list[str] = []
    if annotation.scenario_token != case["scenario_token"]:
        errors.append("scenario_token differs from frozen case")
    if annotation.private_family != case["family_private"]:
        errors.append("private_family differs from frozen case")
    if annotation.agent_split != case["agent_split"]:
        errors.append("agent_split differs from frozen case")
    expected_anchors = [
        int(row[0])
        for row in evaluator.execute(
            "select ts_ns from exploit_anchor where scenario_token=? order by ts_ns,anchor_id",
            [annotation.scenario_token],
        ).fetchall()
    ]
    if annotation.anchor_times_ns != expected_anchors:
        errors.append("anchor_times_ns differs from frozen evaluator anchors")

    evidence = sorted(_annotation_evidence(annotation))
    for evidence_id in evidence:
        match = _EVIDENCE_ID.match(evidence_id)
        if match is None:
            errors.append(f"malformed Evidence ID: {evidence_id}")
        elif match.group("token") != annotation.scenario_token:
            errors.append(f"cross-scene Evidence ID: {evidence_id}")
    if evidence:
        placeholders = ",".join("?" for _ in evidence)
        rows = agent.execute(
            f"""select evidence_id,ts_ns from syscall_event
                  where scenario_token=? and evidence_id in ({placeholders})
                union all
                select evidence_id,ts_ns from network_packet
                  where scenario_token=? and evidence_id in ({placeholders})
                union all
                select evidence_id,ts_ns from resource_sample
                  where scenario_token=? and evidence_id in ({placeholders})""",
            [
                annotation.scenario_token,
                *evidence,
                annotation.scenario_token,
                *evidence,
                annotation.scenario_token,
                *evidence,
            ],
        ).fetchall()
        found = {str(row[0]) for row in rows}
        missing = sorted(set(evidence).difference(found))
        if missing:
            errors.append(f"Evidence IDs absent from ledger: {missing}")
        outside = sorted(
            str(row[0])
            for row in rows
            if int(row[1]) < annotation.investigation_start_ts_ns
            or int(row[1]) > annotation.investigation_end_ts_ns
        )
        if outside:
            errors.append(f"Evidence IDs outside investigation range: {outside}")

    entity_ids = sorted({item.entity_id for item in annotation.root_cause_entities})
    if entity_ids:
        placeholders = ",".join("?" for _ in entity_ids)
        found_entities = {
            str(row[0])
            for row in agent.execute(
                f"select entity_id from entity where scenario_token=? and entity_id in ({placeholders})",
                [annotation.scenario_token, *entity_ids],
            ).fetchall()
        }
        missing_entities = sorted(set(entity_ids).difference(found_entities))
        if missing_entities:
            errors.append(f"root-cause entities absent from ledger: {missing_entities}")
    unknown_techniques = sorted(
        {
            item.technique_id
            for item in annotation.attack_techniques
            if item.technique_id not in knowledge_ids
        }
    )
    if unknown_techniques:
        errors.append(f"ATT&CK IDs absent from frozen snapshot: {unknown_techniques}")
    return errors


def _load_completed_independent(annotations_dir: Path) -> list[GoldAnnotation]:
    annotations: list[GoldAnnotation] = []
    for directory in ("dev", "gold"):
        for path in sorted((annotations_dir / directory).glob("*.json")):
            annotation = GoldAnnotation.model_validate_json(path.read_text(encoding="utf-8"))
            if (
                annotation.status == "completed"
                and annotation.metadata.annotation_mode == "independent"
            ):
                annotations.append(annotation)
    return annotations


def _agreement_source_errors(
    agreement_case: dict[str, Any], annotations: list[GoldAnnotation]
) -> list[str]:
    errors: list[str] = []
    expected_ids = agreement_case.get("annotation_ids")
    expected_hashes = agreement_case.get("annotation_sha256")
    actual_ids = [item.annotation_id for item in annotations]
    if not isinstance(expected_ids, list) or sorted(expected_ids) != sorted(actual_ids):
        errors.append("agreement source annotation IDs differ from current originals")
    if not isinstance(expected_hashes, dict):
        errors.append("agreement source annotation hashes are missing")
        return errors
    for annotation in annotations:
        if expected_hashes.get(annotation.annotation_id) != _annotation_hash(annotation):
            errors.append(f"agreement record stale for {annotation.annotation_id}")
    return errors


def _pilot_gate_errors(
    config: TraceAnchorConfig,
    config_path: Path,
    cases: list[dict[str, Any]],
) -> list[str]:
    """Re-establish the pilot gate from its report and current human sources."""
    resolved = config.resolved_dict(config_path)
    artifacts_dir = Path(resolved["paths"]["artifacts_dir"])
    annotations_dir = Path(resolved["paths"]["annotations_dir"])
    completion = Path(resolved["paths"]["completion_markers_dir"])
    marker_path = completion / "WP5_pilot_gate.done"
    report_path = artifacts_dir / "reports" / "WP05_agreement.json"
    errors: list[str] = []
    if not marker_path.exists():
        return ["WP5 pilot marker is missing"]
    if not report_path.exists():
        return ["WP5 agreement report is missing"]
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"pilot gate artifact is unreadable: {exc}"]
    current_config_hash = config_hash(config, config_path)
    if marker.get("report_sha256") != sha256_file(report_path):
        errors.append("pilot marker does not bind the current agreement report")
    if marker.get("config_sha256") != current_config_hash:
        errors.append("pilot marker configuration hash is stale")
    if report.get("config_sha256") != current_config_hash:
        errors.append("agreement report configuration hash is stale")
    if not report.get("pilot_ok") or report.get("errors"):
        errors.append("agreement report does not pass the pilot gate")

    by_incident = {str(item["incident_id"]): item for item in cases}
    groups: dict[str, list[GoldAnnotation]] = defaultdict(list)
    annotations = _load_completed_independent(annotations_dir)
    for annotation in annotations:
        case = by_incident.get(annotation.incident_id)
        if case is not None and case["agent_split"] == "development":
            groups[annotation.incident_id].append(annotation)
    current_pairs: dict[str, list[GoldAnnotation]] = {}
    for incident_id, values in groups.items():
        annotators = {item.metadata.annotator_id for item in values}
        if len(values) > 2:
            errors.append(f"{incident_id} has more than two completed pilot annotations")
        elif len(values) == 2 and len(annotators) == 2:
            current_pairs[incident_id] = sorted(
                values, key=lambda item: item.annotation_id
            )
        elif len(values) >= 2:
            errors.append(f"{incident_id} pilot annotations are not independent")

    report_cases = {
        str(item.get("incident_id")): item
        for item in report.get("cases", [])
        if item.get("agent_split") == "development"
    }
    if set(report_cases) != set(current_pairs):
        errors.append("agreement report pilot cases differ from current completed pairs")
    evidence_scores: list[float] = []
    families: set[str] = set()
    with duckdb.connect(str(resolved["paths"]["evidence_db"]), read_only=True) as agent:
        with duckdb.connect(
            str(resolved["paths"]["evaluator_db"]), read_only=True
        ) as evaluator:
            versions = {
                str(row[0])
                for row in agent.execute(
                    "select distinct version from attack_knowledge"
                ).fetchall()
            }
            if versions != {config.gold_annotation.attack_knowledge_version}:
                errors.append("ATT&CK snapshot differs from the frozen configuration")
            knowledge_ids = {
                str(row[0])
                for row in agent.execute(
                    "select technique_id from attack_knowledge"
                ).fetchall()
            }
            for incident_id, values in sorted(current_pairs.items()):
                case = by_incident[incident_id]
                families.add(str(case["family_private"]))
                report_case = report_cases.get(incident_id)
                if report_case is not None:
                    errors.extend(_agreement_source_errors(report_case, values))
                    if report_case.get("private_family") != case["family_private"]:
                        errors.append(f"agreement family differs for {incident_id}")
                for annotation in values:
                    errors.extend(
                        f"{annotation.annotation_id}: {value}"
                        for value in _validate_against_ledger(
                            annotation, case, agent, evaluator, knowledge_ids
                        )
                    )
                    if annotation.annotation_id != (
                        f"ann:{annotation.incident_id}:"
                        f"{annotation.metadata.annotator_id}"
                    ):
                        errors.append(
                            f"annotation_id does not match metadata for "
                            f"{annotation.annotation_id}"
                        )
                evidence_scores.append(
                    _set_scores(
                        set(values[0].core_evidence_ids),
                        set(values[1].core_evidence_ids),
                    )["f1"]
                )

    mean_evidence_f1 = _mean(evidence_scores)
    threshold = float(config.gold_annotation.evidence_f1_agreement_gate)
    if len(current_pairs) < 5:
        errors.append("pilot has fewer than five double-annotated cases")
    if len(families) != 5:
        errors.append("pilot does not cover all five Development families")
    if mean_evidence_f1 is None or mean_evidence_f1 < threshold:
        errors.append("pilot Core Evidence F1 is below the configured gate")
    if report.get("development_pairs") != len(current_pairs):
        errors.append("reported Development pair count is stale")
    reported_pilot_f1 = report.get("aggregate", {}).get(
        "development_core_evidence_f1"
    )
    if (
        mean_evidence_f1 is None
        or not isinstance(reported_pilot_f1, (int, float))
        or not math.isclose(float(reported_pilot_f1), mean_evidence_f1)
    ):
        errors.append("reported pilot Core Evidence F1 is stale")
    if marker.get("development_pairs") != len(current_pairs):
        errors.append("pilot marker pair count is stale")
    marker_f1 = marker.get("core_evidence_f1")
    if (
        mean_evidence_f1 is None
        or not isinstance(marker_f1, (int, float))
        or not math.isclose(float(marker_f1), mean_evidence_f1)
    ):
        errors.append("pilot marker Core Evidence F1 is stale")
    return errors


def validate_gold(config: TraceAnchorConfig, config_path: Path) -> dict[str, Any]:
    resolved = config.resolved_dict(config_path)
    annotations_dir = Path(resolved["paths"]["annotations_dir"])
    artifacts_dir = Path(resolved["paths"]["artifacts_dir"])
    completion = Path(resolved["paths"]["completion_markers_dir"])
    cases = _load_frozen_cases(config, config_path)
    by_incident = {str(row["incident_id"]): row for row in cases}
    errors: list[str] = []
    file_results: list[dict[str, Any]] = []
    loaded: list[tuple[Path, GoldAnnotation]] = []
    with duckdb.connect(str(resolved["paths"]["evidence_db"]), read_only=True) as agent:
        with duckdb.connect(str(resolved["paths"]["evaluator_db"]), read_only=True) as evaluator:
            versions = {str(row[0]) for row in agent.execute("select distinct version from attack_knowledge").fetchall()}
            if versions != {config.gold_annotation.attack_knowledge_version}:
                errors.append("ATT&CK snapshot version differs from frozen configuration")
            knowledge_ids = {
                str(row[0])
                for row in agent.execute("select technique_id from attack_knowledge").fetchall()
            }
            for path in _annotation_paths(annotations_dir):
                try:
                    annotation = GoldAnnotation.model_validate_json(
                        path.read_text(encoding="utf-8")
                    )
                    case = by_incident.get(annotation.incident_id)
                    if case is None:
                        file_errors = ["incident_id absent from frozen manifest"]
                    else:
                        file_errors = _validate_against_ledger(
                            annotation, case, agent, evaluator, knowledge_ids
                        )
                    file_errors.extend(_adjudication_contract_errors(annotation))
                    if annotation.annotation_id != (
                        f"ann:{annotation.incident_id}:{annotation.metadata.annotator_id}"
                    ):
                        file_errors.append("annotation_id does not match metadata")
                    if file_errors:
                        errors.extend(f"{path.name}: {value}" for value in file_errors)
                    loaded.append((path, annotation))
                    file_results.append(
                        {
                            "path": str(path),
                            "annotation_id": annotation.annotation_id,
                            "status": annotation.status,
                            "ok": not file_errors,
                            "errors": file_errors,
                            "sha256": sha256_file(path),
                        }
                    )
                except Exception as exc:
                    message = f"{path.name}: {type(exc).__name__}: {exc}"
                    errors.append(message)
                    file_results.append(
                        {"path": str(path), "ok": False, "errors": [message]}
                    )

    independent: dict[str, list[GoldAnnotation]] = defaultdict(list)
    adjudicated: dict[str, list[GoldAnnotation]] = defaultdict(list)
    for _path, annotation in loaded:
        if annotation.status != "completed":
            continue
        target = (
            adjudicated
            if annotation.metadata.annotation_mode == "adjudicated"
            else independent
        )
        target[annotation.incident_id].append(annotation)
    for incident_id, values in independent.items():
        annotators = [item.metadata.annotator_id for item in values]
        if len(annotators) != len(set(annotators)):
            errors.append(f"duplicate independent annotator for {incident_id}")
    independent_ids = {
        item.annotation_id: item
        for values in independent.values()
        for item in values
    }
    for incident_id, values in adjudicated.items():
        for annotation in values:
            sources = set(annotation.metadata.source_annotation_ids)
            if not sources.issubset(independent_ids):
                errors.append(f"adjudication sources missing for {incident_id}")
            elif any(
                independent_ids[source].incident_id != incident_id for source in sources
            ):
                errors.append(f"adjudication source crosses incident for {incident_id}")
    final_incidents: set[str] = set()
    unresolved_double: list[str] = []
    for incident_id in by_incident:
        if len(adjudicated.get(incident_id, [])) == 1:
            final_incidents.add(incident_id)
        elif len(adjudicated.get(incident_id, [])) > 1:
            errors.append(f"multiple completed adjudications for {incident_id}")
        elif len(independent.get(incident_id, [])) == 1:
            final_incidents.add(incident_id)
        elif len(independent.get(incident_id, [])) >= 2:
            unresolved_double.append(incident_id)
    double_incidents = sorted(
        incident_id for incident_id, values in independent.items() if len(values) >= 2
    )
    agreement_path = artifacts_dir / "reports" / "WP05_agreement.json"
    agreement = (
        json.loads(agreement_path.read_text(encoding="utf-8"))
        if agreement_path.exists()
        else {"ok": False}
    )
    agreement_cases = {
        str(item["incident_id"]): item for item in agreement.get("cases", [])
    }
    for incident_id in double_incidents:
        agreement_case = agreement_cases.get(incident_id)
        if agreement_case is None:
            errors.append(f"agreement record missing for {incident_id}")
            continue
        errors.extend(
            f"{incident_id}: {value}"
            for value in _agreement_source_errors(
                agreement_case, independent[incident_id]
            )
        )
        expected_sources = set(agreement_case.get("annotation_ids", []))
        for annotation in adjudicated.get(incident_id, []):
            if set(annotation.metadata.source_annotation_ids) != expected_sources:
                errors.append(
                    f"adjudication sources differ from agreement for {incident_id}"
                )
    if agreement.get("ok"):
        agreement_marker_path = completion / "WP5_agreement.done"
        if not agreement_marker_path.exists():
            errors.append("passing agreement marker is missing")
        else:
            try:
                agreement_marker = json.loads(
                    agreement_marker_path.read_text(encoding="utf-8")
                )
                if agreement_marker.get("report_sha256") != sha256_file(
                    agreement_path
                ):
                    errors.append("agreement marker does not bind the current report")
                if agreement_marker.get("config_sha256") != config_hash(
                    config, config_path
                ):
                    errors.append("agreement marker configuration hash is stale")
                if agreement.get("config_sha256") != config_hash(
                    config, config_path
                ):
                    errors.append("agreement report configuration hash is stale")
            except (OSError, json.JSONDecodeError) as exc:
                errors.append(f"agreement marker is unreadable: {exc}")
    complete = (
        not errors
        and len(final_incidents) == 45
        and len(double_incidents) >= config.gold_annotation.double_annotated_cases_min
        and not unresolved_double
        and bool(agreement.get("ok"))
    )
    report = {
        "schema_version": 1,
        "ok": not errors,
        "complete": complete,
        "errors": errors,
        "annotation_files": len(file_results),
        "draft_files": sum(item.get("status") == "draft" for item in file_results),
        "completed_independent": sum(len(values) for values in independent.values()),
        "completed_adjudicated": sum(len(values) for values in adjudicated.values()),
        "double_annotated_incidents": len(double_incidents),
        "final_incidents": len(final_incidents),
        "missing_final_incidents": sorted(set(by_incident).difference(final_incidents)),
        "unresolved_double_incidents": unresolved_double,
        "files": file_results,
    }
    report_path = artifacts_dir / "reports" / "WP05_gold_validation.json"
    atomic_write_json(report_path, report)
    if complete:
        atomic_write_json(
            completion / "WP5.done",
            {
                "schema_version": 1,
                "validation_sha256": sha256_file(report_path),
                "agreement_sha256": sha256_file(agreement_path),
                "config_sha256": config_hash(config, config_path),
            },
        )
    else:
        (completion / "WP5.done").unlink(missing_ok=True)
    return report


def _step_order_tau(left: GoldAnnotation, right: GoldAnnotation) -> float | None:
    left_rank = {
        step_type: index
        for index, step_type in enumerate(step.step_type for step in left.steps)
        if step_type not in {
            earlier.step_type for earlier in left.steps[:index]
        }
    }
    right_rank = {
        step_type: index
        for index, step_type in enumerate(step.step_type for step in right.steps)
        if step_type not in {
            earlier.step_type for earlier in right.steps[:index]
        }
    }
    common = sorted(set(left_rank).intersection(right_rank))
    if len(common) < 2:
        return None
    value = float(kendalltau([left_rank[item] for item in common], [right_rank[item] for item in common]).statistic)
    return value if math.isfinite(value) else None


def _edge_macro_f1(left: GoldAnnotation, right: GoldAnnotation) -> float:
    left_edges = [*left.provenance_edges, *left.possibly_causal_edges]
    right_edges = [*right.provenance_edges, *right.possibly_causal_edges]
    relation_types = sorted(
        {edge.relation_type for edge in left_edges}.union(
            edge.relation_type for edge in right_edges
        )
    )
    if not relation_types:
        return 1.0
    scores = []
    for relation_type in relation_types:
        left_set = {
            f"{edge.source_evidence_id}|{edge.target_evidence_id}"
            for edge in left_edges
            if edge.relation_type == relation_type
        }
        right_set = {
            f"{edge.source_evidence_id}|{edge.target_evidence_id}"
            for edge in right_edges
            if edge.relation_type == relation_type
        }
        scores.append(_set_scores(left_set, right_set)["f1"])
    return statistics.mean(scores)


def compute_agreement(config: TraceAnchorConfig, config_path: Path) -> dict[str, Any]:
    resolved = config.resolved_dict(config_path)
    annotations_dir = Path(resolved["paths"]["annotations_dir"])
    artifacts_dir = Path(resolved["paths"]["artifacts_dir"])
    completion = Path(resolved["paths"]["completion_markers_dir"])
    annotations = _load_completed_independent(annotations_dir)
    frozen_cases = _load_frozen_cases(config, config_path)
    by_incident = {str(item["incident_id"]): item for item in frozen_cases}
    errors: list[str] = []
    valid_annotations: list[GoldAnnotation] = []
    with duckdb.connect(str(resolved["paths"]["evidence_db"]), read_only=True) as agent:
        with duckdb.connect(
            str(resolved["paths"]["evaluator_db"]), read_only=True
        ) as evaluator:
            versions = {
                str(row[0])
                for row in agent.execute(
                    "select distinct version from attack_knowledge"
                ).fetchall()
            }
            if versions != {config.gold_annotation.attack_knowledge_version}:
                errors.append("ATT&CK snapshot version differs from frozen configuration")
            technique_universe = [
                str(row[0])
                for row in agent.execute(
                    "select technique_id from attack_knowledge order by technique_id"
                ).fetchall()
            ]
            knowledge_ids = set(technique_universe)
            for annotation in annotations:
                case = by_incident.get(annotation.incident_id)
                if case is None:
                    errors.append(
                        f"{annotation.annotation_id}: incident absent from frozen manifest"
                    )
                    continue
                annotation_errors = _validate_against_ledger(
                    annotation, case, agent, evaluator, knowledge_ids
                )
                if annotation.annotation_id != (
                    f"ann:{annotation.incident_id}:"
                    f"{annotation.metadata.annotator_id}"
                ):
                    annotation_errors.append("annotation_id does not match metadata")
                if annotation_errors:
                    errors.extend(
                        f"{annotation.annotation_id}: {value}"
                        for value in annotation_errors
                    )
                    continue
                valid_annotations.append(annotation)
    groups: dict[str, list[GoldAnnotation]] = defaultdict(list)
    for annotation in valid_annotations:
        groups[annotation.incident_id].append(annotation)
    cases = []
    for incident_id in sorted(groups):
        values = sorted(groups[incident_id], key=lambda item: item.annotation_id)
        annotators = {item.metadata.annotator_id for item in values}
        if len(values) < 2:
            continue
        if len(annotators) != len(values):
            errors.append(f"{incident_id} repeats an independent annotator")
            continue
        if len(values) > 2:
            errors.append(f"{incident_id} has more than two completed independent annotations")
            continue
        left, right = values
        evidence = _set_scores(set(left.core_evidence_ids), set(right.core_evidence_ids))
        roots_left = {item.entity_id for item in left.root_cause_entities}
        roots_right = {item.entity_id for item in right.root_cause_entities}
        steps = _set_scores(
            {item.step_type for item in left.steps},
            {item.step_type for item in right.steps},
        )
        attack_left = {item.technique_id for item in left.attack_techniques}
        attack_right = {item.technique_id for item in right.attack_techniques}
        attack = _set_scores(attack_left, attack_right)
        left_binary = [int(item in attack_left) for item in technique_universe]
        right_binary = [int(item in attack_right) for item in technique_universe]
        kappa = (
            None
            if len(set(left_binary + right_binary)) < 2
            else float(cohen_kappa_score(left_binary, right_binary))
        )
        cases.append(
            {
                "incident_id": incident_id,
                "agent_split": left.agent_split,
                "private_family": left.private_family,
                "annotation_ids": [left.annotation_id, right.annotation_id],
                "annotation_sha256": {
                    left.annotation_id: _annotation_hash(left),
                    right.annotation_id: _annotation_hash(right),
                },
                "core_evidence": evidence,
                "root_cause_exact": roots_left == roots_right,
                "root_cause_top_set_agreement": bool(roots_left.intersection(roots_right)),
                "step_type": steps,
                "step_order_kendall_tau": _step_order_tau(left, right),
                "attack": {**attack, "cohen_kappa": kappa},
                "edge_type_macro_f1": _edge_macro_f1(left, right),
            }
        )
    mean_evidence_f1 = _mean(item["core_evidence"]["f1"] for item in cases)
    development_cases = [
        item for item in cases if item["agent_split"] == "development"
    ]
    development_pairs = len(development_cases)
    development_families = {
        str(item["private_family"]) for item in development_cases
    }
    development_mean_evidence_f1 = _mean(
        item["core_evidence"]["f1"] for item in development_cases
    )
    threshold = float(config.gold_annotation.evidence_f1_agreement_gate)
    pilot_ok = (
        not errors
        and development_pairs >= 5
        and len(development_families) == 5
        and development_mean_evidence_f1 is not None
        and development_mean_evidence_f1 >= threshold
    )
    ok = (
        not errors
        and len(cases) >= int(config.gold_annotation.double_annotated_cases_min)
        and mean_evidence_f1 is not None
        and mean_evidence_f1 >= threshold
    )
    report = {
        "schema_version": 1,
        "config_sha256": config_hash(config, config_path),
        "ok": ok,
        "pilot_ok": pilot_ok,
        "errors": errors,
        "double_annotated_cases": len(cases),
        "development_pairs": development_pairs,
        "development_families": sorted(development_families),
        "required_double_annotated_cases": config.gold_annotation.double_annotated_cases_min,
        "evidence_f1_gate": threshold,
        "aggregate": {
            "core_evidence_f1": mean_evidence_f1,
            "development_core_evidence_f1": development_mean_evidence_f1,
            "core_evidence_jaccard": _mean(
                item["core_evidence"]["jaccard"] for item in cases
            ),
            "root_cause_exact": _mean(
                float(item["root_cause_exact"]) for item in cases
            ),
            "root_cause_top_set_agreement": _mean(
                float(item["root_cause_top_set_agreement"]) for item in cases
            ),
            "step_type_f1": _mean(item["step_type"]["f1"] for item in cases),
            "step_order_kendall_tau": _mean(
                item["step_order_kendall_tau"] for item in cases
            ),
            "attack_f1": _mean(item["attack"]["f1"] for item in cases),
            "attack_cohen_kappa": _mean(
                item["attack"]["cohen_kappa"] for item in cases
            ),
            "edge_type_macro_f1": _mean(
                item["edge_type_macro_f1"] for item in cases
            ),
        },
        "cases": cases,
    }
    report_path = artifacts_dir / "reports" / "WP05_agreement.json"
    atomic_write_json(report_path, report)
    if pilot_ok:
        atomic_write_json(
            completion / "WP5_pilot_gate.done",
            {
                "schema_version": 1,
                "report_sha256": sha256_file(report_path),
                "config_sha256": config_hash(config, config_path),
                "development_pairs": development_pairs,
                "core_evidence_f1": development_mean_evidence_f1,
            },
        )
    else:
        (completion / "WP5_pilot_gate.done").unlink(missing_ok=True)
    if ok:
        atomic_write_json(
            completion / "WP5_agreement.done",
            {
                "schema_version": 1,
                "report_sha256": sha256_file(report_path),
                "config_sha256": config_hash(config, config_path),
                "double_annotated_cases": len(cases),
                "core_evidence_f1": mean_evidence_f1,
            },
        )
    else:
        (completion / "WP5_agreement.done").unlink(missing_ok=True)
    return report


def adjudicate_annotations(
    config: TraceAnchorConfig,
    config_path: Path,
    *,
    incident_id: str | None = None,
) -> dict[str, Any]:
    resolved = config.resolved_dict(config_path)
    annotations_dir = Path(resolved["paths"]["annotations_dir"])
    completion = Path(resolved["paths"]["completion_markers_dir"])
    agreement_path = (
        Path(resolved["paths"]["artifacts_dir"])
        / "reports"
        / "WP05_agreement.json"
    )
    if not agreement_path.exists():
        raise RuntimeError("agreement must be computed before adjudication")
    agreement = json.loads(agreement_path.read_text(encoding="utf-8"))
    if not agreement.get("ok") or agreement.get("errors"):
        raise RuntimeError(
            "final agreement gate must pass before adjudication (20 pairs and F1 gate)"
        )
    marker_path = completion / "WP5_agreement.done"
    if not marker_path.exists():
        raise RuntimeError("final agreement completion marker is missing")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    current_config_hash = config_hash(config, config_path)
    if (
        marker.get("report_sha256") != sha256_file(agreement_path)
        or marker.get("config_sha256") != current_config_hash
        or agreement.get("config_sha256") != current_config_hash
    ):
        raise RuntimeError("final agreement report or marker is stale")
    agreement_cases = {item["incident_id"]: item for item in agreement["cases"]}
    annotations = _load_completed_independent(annotations_dir)
    groups: dict[str, list[GoldAnnotation]] = defaultdict(list)
    for annotation in annotations:
        groups[annotation.incident_id].append(annotation)
    targets = [incident_id] if incident_id else sorted(agreement_cases)
    created = []
    skipped = []
    for target in targets:
        if target not in agreement_cases or len(groups.get(target, [])) != 2:
            raise ValueError(f"{target} has no valid double-annotation agreement record")
        left, right = sorted(groups[target], key=lambda item: item.annotation_id)
        source_errors = _agreement_source_errors(
            agreement_cases[target], [left, right]
        )
        if source_errors:
            raise ValueError(f"{target} agreement is stale: {'; '.join(source_errors)}")
        packet = {
            "schema_version": 1,
            "classification": "PRIVATE GROUND TRUTH - EVALUATOR ONLY",
            "incident_id": target,
            "source_annotation_ids": [left.annotation_id, right.annotation_id],
            "agreement": agreement_cases[target],
            "differences": {
                "core_only_left": sorted(
                    set(left.core_evidence_ids).difference(right.core_evidence_ids)
                ),
                "core_only_right": sorted(
                    set(right.core_evidence_ids).difference(left.core_evidence_ids)
                ),
                "root_only_left": sorted(
                    {item.entity_id for item in left.root_cause_entities}.difference(
                        item.entity_id for item in right.root_cause_entities
                    )
                ),
                "root_only_right": sorted(
                    {item.entity_id for item in right.root_cause_entities}.difference(
                        item.entity_id for item in left.root_cause_entities
                    )
                ),
                "step_types_left": [item.step_type for item in left.steps],
                "step_types_right": [item.step_type for item in right.steps],
                "attack_left": [item.technique_id for item in left.attack_techniques],
                "attack_right": [item.technique_id for item in right.attack_techniques],
            },
            "instruction": "Resolve each difference from source evidence. Do not choose by majority or automatically copy intersections.",
        }
        packet_path = annotations_dir / "adjudication_queue" / f"{target}.json"
        atomic_write_json(packet_path, packet)
        output = annotations_dir / "adjudicated" / f"{target}.json"
        if output.exists():
            skipped.append(str(output))
            continue
        now = datetime.now(timezone.utc)
        draft = GoldAnnotation.model_validate(
            {
                "schema_version": 1,
                "status": "draft",
                "annotation_id": f"ann:{target}:adjudicator",
                "incident_id": target,
                "private_family": left.private_family,
                "scenario_token": left.scenario_token,
                "agent_split": left.agent_split,
                "investigation_start_ts_ns": min(
                    left.investigation_start_ts_ns, right.investigation_start_ts_ns
                ),
                "investigation_end_ts_ns": max(
                    left.investigation_end_ts_ns, right.investigation_end_ts_ns
                ),
                "anchor_times_ns": left.anchor_times_ns,
                "metadata": {
                    "annotator_id": "adjudicator",
                    "annotation_mode": "adjudicated",
                    "started_at": now,
                    "human_verified": False,
                    "source_annotation_ids": [left.annotation_id, right.annotation_id],
                },
            }
        )
        atomic_write_json(output, draft.model_dump(mode="json"))
        created.append(str(output))
    return {
        "schema_version": 1,
        "classification": "PRIVATE GROUND TRUTH - EVALUATOR ONLY",
        "created": created,
        "skipped_existing_human_files": skipped,
        "automatically_adjudicated": False,
    }


__all__ = [
    "adjudicate_annotations",
    "compute_agreement",
    "create_annotation_draft",
    "sample_gold",
    "validate_gold",
]
