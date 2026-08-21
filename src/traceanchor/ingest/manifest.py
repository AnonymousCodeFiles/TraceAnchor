from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pyarrow.parquet as pq

from traceanchor.config import TraceAnchorConfig, config_hash
from traceanchor.ingest.common import (
    atomic_write_json,
    atomic_write_parquet,
    scenario_token,
    scenario_uid,
    sha256_file,
)
from traceanchor.ingest.schemas import RAW_FILE_SCHEMA, SCENARIO_SCHEMA


REQUIRED_EXTENSIONS = ("json", "pcap", "res", "sc")


@dataclass(frozen=True)
class Candidate:
    directory: Path
    relative_directory: str
    stem: str
    files: dict[str, Path]
    uid: str
    token: str

    @property
    def complete(self) -> bool:
        return all(extension in self.files for extension in REQUIRED_EXTENSIONS)


def discover_scenarios(root: Path, seed: int) -> list[Candidate]:
    if not root.is_dir():
        raise FileNotFoundError(f"raw_data_root is not a directory: {root}")
    candidates: list[Candidate] = []
    for directory, _, filenames in os.walk(root):
        by_stem: dict[str, dict[str, Path]] = {}
        for filename in filenames:
            path = Path(directory) / filename
            extension = path.suffix.lower().lstrip(".")
            if extension not in REQUIRED_EXTENSIONS:
                continue
            by_stem.setdefault(path.stem, {})[extension] = path
        for stem, files in sorted(by_stem.items()):
            relative = Path(directory).relative_to(root).as_posix()
            uid = scenario_uid(relative)
            candidates.append(
                Candidate(
                    directory=Path(directory),
                    relative_directory=relative,
                    stem=stem,
                    files=files,
                    uid=uid,
                    token=scenario_token(uid, seed),
                )
            )
    return sorted(candidates, key=lambda item: (item.relative_directory, item.stem))


def _hash_record(root: Path, candidate: Candidate, extension: str) -> dict[str, object]:
    path = candidate.files[extension]
    stat = path.stat()
    return {
        "scenario_uid": candidate.uid,
        "scenario_token": candidate.token,
        "relative_path": path.relative_to(root).as_posix(),
        "file_type": extension,
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": sha256_file(path),
    }


def _all_hash_jobs(root: Path, candidates: Iterable[Candidate]) -> list[tuple[Path, Candidate, str]]:
    return [
        (root, candidate, extension)
        for candidate in candidates
        for extension in REQUIRED_EXTENSIONS
        if extension in candidate.files
    ]


def build_raw_manifest(config: TraceAnchorConfig, config_path: Path) -> dict[str, object]:
    resolved = config.resolved_dict(config_path)
    root = Path(resolved["paths"]["raw_data_root"])
    manifests_dir = Path(resolved["paths"]["manifests_dir"])
    markers_dir = Path(resolved["paths"]["completion_markers_dir"])
    candidates = discover_scenarios(root, config.project.global_seed)
    jobs = _all_hash_jobs(root, candidates)
    worker_count = int(config.ingestion.parser_workers)
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        raw_records = list(executor.map(lambda args: _hash_record(*args), jobs))
    raw_records.sort(key=lambda row: (str(row["scenario_token"]), str(row["file_type"])))

    scenario_records = []
    incomplete = []
    for candidate in candidates:
        missing = sorted(set(REQUIRED_EXTENSIONS).difference(candidate.files))
        empty = sorted(
            extension
            for extension, path in candidate.files.items()
            if path.stat().st_size == 0
        )
        status = "ok" if not missing and not empty else "incomplete"
        scenario_records.append(
            {
                "scenario_uid": candidate.uid,
                "scenario_token": candidate.token,
                **{f"has_{extension}": extension in candidate.files for extension in REQUIRED_EXTENSIONS},
                "total_size_bytes": sum(path.stat().st_size for path in candidate.files.values()),
                "quality_status": status,
            }
        )
        if status != "ok":
            incomplete.append(
                {
                    "relative_directory": candidate.relative_directory,
                    "missing": missing,
                    "empty": empty,
                }
            )

    raw_path = manifests_dir / "raw_files.parquet"
    scenarios_path = manifests_dir / "scenarios.parquet"
    raw_count = atomic_write_parquet(raw_path, RAW_FILE_SCHEMA, raw_records)
    scenario_count = atomic_write_parquet(scenarios_path, SCENARIO_SCHEMA, scenario_records)
    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_sha256": config_hash(config, config_path),
        "raw_root": str(root),
        "scenario_candidates": scenario_count,
        "complete_scenarios": sum(row["quality_status"] == "ok" for row in scenario_records),
        "incomplete_scenarios": len(incomplete),
        "raw_files": raw_count,
        "bytes": sum(int(row["size_bytes"]) for row in raw_records),
        "incomplete": incomplete,
    }
    report_path = manifests_dir / "discovery_report.json"
    atomic_write_json(report_path, report)
    marker = {
        "schema_version": 1,
        "step": "raw_manifest",
        "config_sha256": report["config_sha256"],
        "raw_files_sha256": sha256_file(raw_path),
        "scenarios_sha256": sha256_file(scenarios_path),
        "report_sha256": sha256_file(report_path),
    }
    atomic_write_json(markers_dir / "WP1_manifest.done", marker)
    return report


def load_candidate_from_manifest(
    config: TraceAnchorConfig,
    config_path: Path,
    token: str,
) -> Candidate:
    resolved = config.resolved_dict(config_path)
    root = Path(resolved["paths"]["raw_data_root"])
    raw_path = Path(resolved["paths"]["manifests_dir"]) / "raw_files.parquet"
    if not raw_path.exists():
        raise FileNotFoundError("raw manifest missing; run the manifest command first")
    table = pq.read_table(raw_path, filters=[("scenario_token", "=", token)])
    rows = table.to_pylist()
    if not rows:
        raise KeyError(f"unknown scenario token: {token}")
    files = {str(row["file_type"]): root / str(row["relative_path"]) for row in rows}
    relative_directory = Path(str(rows[0]["relative_path"])).parent.as_posix()
    stems = {path.stem for path in files.values()}
    if len(stems) != 1:
        raise ValueError(f"manifest token {token} has inconsistent stems")
    return Candidate(
        directory=next(iter(files.values())).parent,
        relative_directory=relative_directory,
        stem=next(iter(stems)),
        files=files,
        uid=str(rows[0]["scenario_uid"]),
        token=token,
    )


def select_family_sample_from_manifest(
    config: TraceAnchorConfig,
    config_path: Path,
) -> list[Candidate]:
    resolved = config.resolved_dict(config_path)
    raw_path = Path(resolved["paths"]["manifests_dir"]) / "raw_files.parquet"
    if not raw_path.exists():
        raise FileNotFoundError("raw manifest missing; run the manifest command first")
    rows = pq.read_table(raw_path).to_pylist()
    grouped: dict[tuple[str, str], dict[str, object]] = {}
    for row in rows:
        relative = Path(str(row["relative_path"]))
        family = relative.parts[0]
        key = (family, str(row["scenario_token"]))
        item = grouped.setdefault(
            key,
            {"family": family, "token": str(row["scenario_token"]), "bytes": 0, "files": 0},
        )
        item["bytes"] = int(item["bytes"]) + int(row["size_bytes"])
        item["files"] = int(item["files"]) + 1
    selected = []
    for family in sorted({str(item["family"]) for item in grouped.values()}):
        complete = [
            item
            for item in grouped.values()
            if item["family"] == family and item["files"] == len(REQUIRED_EXTENSIONS)
        ]
        if not complete:
            raise ValueError(f"no complete scenario for family {family}")
        selected.append(min(complete, key=lambda item: (item["bytes"], item["token"])))
    selection_path = Path(resolved["paths"]["manifests_dir"]) / "ingestion_family_sample.json"
    atomic_write_json(
        selection_path,
        {
            "schema_version": 1,
            "selection_rule": "minimum_total_bytes_then_scenario_token",
            "family_count": len(selected),
            "total_bytes": sum(int(item["bytes"]) for item in selected),
            "scenarios": selected,
        },
    )
    return [
        load_candidate_from_manifest(config, config_path, str(item["token"]))
        for item in selected
    ]


def candidate_from_example(config: TraceAnchorConfig, config_path: Path) -> Candidate:
    resolved = config.resolved_dict(config_path)
    directory = Path(resolved["paths"]["example_scenario"])
    files = {
        extension: directory / f"{directory.name}.{extension}"
        for extension in REQUIRED_EXTENSIONS
    }
    missing = [extension for extension, path in files.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"configured example is missing files: {missing}")
    relative = f"example/{directory.parent.name}/{directory.name}"
    uid = scenario_uid(relative)
    return Candidate(
        directory=directory,
        relative_directory=relative,
        stem=directory.name,
        files=files,
        uid=uid,
        token=scenario_token(uid, config.project.global_seed),
    )
