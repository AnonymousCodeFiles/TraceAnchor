from __future__ import annotations

import hashlib
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from traceanchor.config import TraceAnchorConfig, config_hash
from traceanchor.ingest.common import atomic_write_json, atomic_write_parquet, sha256_file


GOLD_SCHEMA = pa.schema(
    [
        ("scenario_uid", pa.string()),
        ("scenario_token", pa.string()),
        ("family_private", pa.string()),
        ("family_selection_rank", pa.int32()),
        ("selection_reason", pa.string()),
        ("recording_time_seconds", pa.float64()),
        ("anchor_count", pa.int32()),
        ("raw_manifest_sha256", pa.string()),
    ]
)

TRIGGER_SPLIT_SCHEMA = pa.schema(
    [
        ("scenario_uid", pa.string()),
        ("scenario_token", pa.string()),
        ("family_private", pa.string()),
        ("trigger_split", pa.string()),
        ("gold_reserved", pa.bool_()),
        ("hash_rank", pa.string()),
    ]
)

AGENT_SPLIT_SCHEMA = pa.schema(
    [
        ("incident_id", pa.string()),
        ("scenario_uid", pa.string()),
        ("scenario_token", pa.string()),
        ("family_private", pa.string()),
        ("agent_split", pa.string()),
        ("locked_until_protocol_freeze", pa.bool_()),
    ]
)


def _seeded_hash(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()


def choose_gold_rows(
    rows: list[dict[str, object]],
    selection_seed: int,
    per_family: int = 3,
) -> list[dict[str, object]]:
    by_family: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if int(row["anchor_count"]) > 0:
            by_family[str(row["family_private"])].append(row)
    selected: list[dict[str, object]] = []
    reasons = ("short_recording", "median_recording", "long_recording")
    for family in sorted(by_family):
        candidates = sorted(
            by_family[family],
            key=lambda row: (
                float(row["recording_time_seconds"]),
                int(row["raw_size_bytes"]),
                _seeded_hash(selection_seed, str(row["scenario_uid"])),
            ),
        )
        if len(candidates) < per_family:
            raise ValueError(f"family {family} has fewer than {per_family} anchor-covered scenarios")
        indexes = [0, (len(candidates) - 1) // 2, len(candidates) - 1]
        chosen_indexes: list[int] = []
        for index in indexes:
            while index in chosen_indexes and index + 1 < len(candidates):
                index += 1
            while index in chosen_indexes and index > 0:
                index -= 1
            chosen_indexes.append(index)
        for rank, (index, reason) in enumerate(zip(chosen_indexes, reasons), start=1):
            row = dict(candidates[index])
            row["family_selection_rank"] = rank
            row["selection_reason"] = reason
            selected.append(row)
    return selected


def _collect_scenario_metadata(
    config: TraceAnchorConfig, config_path: Path
) -> list[dict[str, object]]:
    resolved = config.resolved_dict(config_path)
    manifests = Path(resolved["paths"]["manifests_dir"])
    evaluator = Path(resolved["paths"]["evaluator_dir"])
    scenario_rows = pq.read_table(manifests / "scenarios.parquet").to_pylist()
    raw_rows = pq.read_table(manifests / "raw_files.parquet").to_pylist()
    raw_sizes: dict[str, int] = defaultdict(int)
    for row in raw_rows:
        raw_sizes[str(row["scenario_token"])] += int(row["size_bytes"])
    metadata = []
    for row in scenario_rows:
        if row["quality_status"] != "ok":
            continue
        token = str(row["scenario_token"])
        private_path = evaluator / token / "scenario_private.parquet"
        anchors_path = evaluator / token / "exploit_anchors.parquet"
        if not private_path.exists() or not anchors_path.exists():
            raise FileNotFoundError(f"evaluator ingestion output missing for {token}")
        private = pq.read_table(private_path).to_pylist()[0]
        metadata.append(
            {
                "scenario_uid": str(row["scenario_uid"]),
                "scenario_token": token,
                "family_private": str(private["family_private"]),
                "recording_time_seconds": float(private["recording_time_seconds"]),
                "anchor_count": pq.read_metadata(anchors_path).num_rows,
                "raw_size_bytes": raw_sizes[token],
            }
        )
    return metadata


def make_splits(config: TraceAnchorConfig, config_path: Path) -> dict[str, object]:
    resolved = config.resolved_dict(config_path)
    splits_dir = Path(resolved["paths"]["splits_dir"])
    manifests_dir = Path(resolved["paths"]["manifests_dir"])
    markers_dir = Path(resolved["paths"]["completion_markers_dir"])
    metadata = _collect_scenario_metadata(config, config_path)
    selection_seed = int(config.splits.gold.selection_seed)
    selected = choose_gold_rows(
        metadata,
        selection_seed=selection_seed,
        per_family=int(config.splits.gold.scenarios_per_family),
    )
    gold_tokens = {str(row["scenario_token"]) for row in selected}
    raw_manifest_hash = sha256_file(manifests_dir / "raw_files.parquet")
    gold_rows = [
        {
            "scenario_uid": row["scenario_uid"],
            "scenario_token": row["scenario_token"],
            "family_private": row["family_private"],
            "family_selection_rank": row["family_selection_rank"],
            "selection_reason": row["selection_reason"],
            "recording_time_seconds": row["recording_time_seconds"],
            "anchor_count": row["anchor_count"],
            "raw_manifest_sha256": raw_manifest_hash,
        }
        for row in selected
    ]
    atomic_write_parquet(splits_dir / "gold_reserved.parquet", GOLD_SCHEMA, gold_rows)

    by_family: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in metadata:
        if row["scenario_token"] not in gold_tokens:
            row = dict(row)
            row["hash_rank"] = _seeded_hash(
                int(config.splits.hash_seed), str(row["scenario_uid"])
            )
            by_family[str(row["family_private"])].append(row)
    split_rows = []
    for family in sorted(by_family):
        candidates = sorted(by_family[family], key=lambda row: row["hash_rank"])
        size = len(candidates)
        train_end = int(size * float(config.splits.trigger.train_ratio))
        validation_end = train_end + int(size * float(config.splits.trigger.validation_ratio))
        for index, row in enumerate(candidates):
            split = "train" if index < train_end else "validation" if index < validation_end else "test"
            split_rows.append(
                {
                    "scenario_uid": row["scenario_uid"],
                    "scenario_token": row["scenario_token"],
                    "family_private": family,
                    "trigger_split": split,
                    "gold_reserved": False,
                    "hash_rank": row["hash_rank"],
                }
            )
    for row in selected:
        split_rows.append(
            {
                "scenario_uid": row["scenario_uid"],
                "scenario_token": row["scenario_token"],
                "family_private": row["family_private"],
                "trigger_split": "test",
                "gold_reserved": True,
                "hash_rank": _seeded_hash(
                    int(config.splits.hash_seed), str(row["scenario_uid"])
                ),
            }
        )
    split_rows.sort(key=lambda row: (str(row["trigger_split"]), str(row["scenario_token"])))
    atomic_write_parquet(
        splits_dir / "trigger_split.parquet", TRIGGER_SPLIT_SCHEMA, split_rows
    )

    dev_families = set(config.splits.agent_development_families)
    agent_rows = []
    for index, row in enumerate(
        sorted(selected, key=lambda item: (str(item["family_private"]), str(item["scenario_token"]))),
        start=1,
    ):
        agent_split = (
            "development" if row["family_private"] in dev_families else "test"
        )
        agent_rows.append(
            {
                "incident_id": f"INC-{index:03d}",
                "scenario_uid": row["scenario_uid"],
                "scenario_token": row["scenario_token"],
                "family_private": row["family_private"],
                "agent_split": agent_split,
                "locked_until_protocol_freeze": agent_split == "test",
            }
        )
    atomic_write_parquet(splits_dir / "agent_split.parquet", AGENT_SPLIT_SCHEMA, agent_rows)

    counts = {
        split: sum(row["trigger_split"] == split for row in split_rows)
        for split in ("train", "validation", "test")
    }
    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_sha256": config_hash(config, config_path),
        "scenarios": len(split_rows),
        "gold_reserved": len(gold_rows),
        "trigger_counts": counts,
        "agent_development": sum(row["agent_split"] == "development" for row in agent_rows),
        "agent_test": sum(row["agent_split"] == "test" for row in agent_rows),
        "raw_manifest_sha256": raw_manifest_hash,
    }
    report_path = splits_dir / "split_report.json"
    atomic_write_json(report_path, report)
    marker = {
        "schema_version": 1,
        "step": "splits_frozen",
        "config_sha256": report["config_sha256"],
        "gold_sha256": sha256_file(splits_dir / "gold_reserved.parquet"),
        "trigger_split_sha256": sha256_file(splits_dir / "trigger_split.parquet"),
        "agent_split_sha256": sha256_file(splits_dir / "agent_split.parquet"),
        "report_sha256": sha256_file(report_path),
    }
    atomic_write_json(markers_dir / "WP2_splits.done", marker)
    return report


def validate_splits(config: TraceAnchorConfig, config_path: Path) -> dict[str, object]:
    resolved = config.resolved_dict(config_path)
    splits_dir = Path(resolved["paths"]["splits_dir"])
    gold = pq.read_table(splits_dir / "gold_reserved.parquet").to_pylist()
    trigger = pq.read_table(splits_dir / "trigger_split.parquet").to_pylist()
    agent = pq.read_table(splits_dir / "agent_split.parquet").to_pylist()
    errors = []
    tokens = [str(row["scenario_token"]) for row in trigger]
    if len(tokens) != len(set(tokens)):
        errors.append("trigger split contains duplicate scenario_token")
    uid_values = [str(row["scenario_uid"]) for row in trigger]
    if len(uid_values) != len(set(uid_values)):
        errors.append("trigger split contains duplicate scenario_uid")
    gold_tokens = {str(row["scenario_token"]) for row in gold}
    trigger_by_token = {str(row["scenario_token"]): row for row in trigger}
    if any(trigger_by_token[token]["trigger_split"] != "test" for token in gold_tokens):
        errors.append("one or more gold scenarios are not in Trigger Test")
    if len(gold) != int(config.splits.gold.total_scenarios):
        errors.append(f"gold count is {len(gold)}")
    family_gold: dict[str, int] = defaultdict(int)
    for row in gold:
        family_gold[str(row["family_private"])] += 1
    if set(family_gold.values()) != {int(config.splits.gold.scenarios_per_family)}:
        errors.append(f"gold family counts invalid: {dict(family_gold)}")
    dev_families = set(config.splits.agent_development_families)
    for row in agent:
        expected = "development" if row["family_private"] in dev_families else "test"
        if row["agent_split"] != expected:
            errors.append(f"agent family assignment mismatch for {row['scenario_token']}")
    dev_count = sum(row["agent_split"] == "development" for row in agent)
    test_count = sum(row["agent_split"] == "test" for row in agent)
    if dev_count != int(config.splits.agent_development_cases):
        errors.append(f"agent development count is {dev_count}")
    if test_count != int(config.splits.agent_test_cases):
        errors.append(f"agent test count is {test_count}")
    return {
        "ok": not errors,
        "errors": errors,
        "gold": len(gold),
        "trigger": len(trigger),
        "agent_development": dev_count,
        "agent_test": test_count,
    }
