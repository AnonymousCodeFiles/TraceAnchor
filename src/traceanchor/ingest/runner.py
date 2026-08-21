from __future__ import annotations

import resource
import shutil
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from traceanchor.config import TraceAnchorConfig, config_hash
from traceanchor.ingest.common import atomic_write_json, atomic_write_parquet, sha256_file
from traceanchor.ingest.json_meta import parse_json_metadata
from traceanchor.ingest.manifest import Candidate, REQUIRED_EXTENSIONS
from traceanchor.ingest.pcap import iter_packets
from traceanchor.ingest.resources import iter_resources
from traceanchor.ingest.schemas import (
    EXPLOIT_ANCHOR_SCHEMA,
    PACKET_SCHEMA,
    RESOURCE_SCHEMA,
    SCENARIO_PRIVATE_SCHEMA,
    SCENARIO_PUBLIC_SCHEMA,
    SYSCALL_SCHEMA,
)
from traceanchor.ingest.syscalls import iter_syscalls


def _peak_ram_gib() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / (1024 * 1024)


def ingest_candidate(
    config: TraceAnchorConfig,
    config_path: Path,
    candidate: Candidate,
    resume: bool = False,
) -> dict[str, object]:
    missing = sorted(set(REQUIRED_EXTENSIONS).difference(candidate.files))
    if missing:
        raise ValueError(f"scenario {candidate.token} is incomplete: {missing}")
    resolved = config.resolved_dict(config_path)
    artifacts = Path(resolved["paths"]["artifacts_dir"])
    public_target = Path(resolved["paths"]["parquet_dir"]) / candidate.token
    private_target = Path(resolved["paths"]["evaluator_dir"]) / candidate.token
    marker_path = (
        Path(resolved["paths"]["completion_markers_dir"])
        / "ingest"
        / f"{candidate.token}.done"
    )
    if marker_path.exists():
        if resume:
            return {"scenario_token": candidate.token, "status": "skipped_complete"}
        raise FileExistsError(f"scenario already complete: {candidate.token}; use --resume")
    if public_target.exists() or private_target.exists():
        raise FileExistsError(
            f"partial output exists for {candidate.token}; inspect it before retrying"
        )

    stage_root = artifacts / ".staging" / f"ingest-{candidate.token}-{uuid.uuid4().hex}"
    public_stage = stage_root / "public"
    private_stage = stage_root / "private"
    started = time.monotonic()
    try:
        family_private = candidate.relative_directory.split("/")[-2]
        public, private, anchors = parse_json_metadata(
            candidate.files["json"], candidate.uid, candidate.token, family_private
        )
        counts = {
            "scenario_public": atomic_write_parquet(
                public_stage / "scenario_public.parquet", SCENARIO_PUBLIC_SCHEMA, [public]
            ),
            "scenario_private": atomic_write_parquet(
                private_stage / "scenario_private.parquet", SCENARIO_PRIVATE_SCHEMA, [private]
            ),
            "exploit_anchors": atomic_write_parquet(
                private_stage / "exploit_anchors.parquet", EXPLOIT_ANCHOR_SCHEMA, anchors
            ),
            "syscall_event": atomic_write_parquet(
                public_stage / "syscall_event.parquet",
                SYSCALL_SCHEMA,
                iter_syscalls(
                    candidate.files["sc"],
                    candidate.token,
                    int(config.ingestion.syscall_max_args_chars),
                    int(config.ingestion.time_bucket_seconds),
                ),
            ),
            "network_packet": atomic_write_parquet(
                public_stage / "network_packet.parquet",
                PACKET_SCHEMA,
                iter_packets(
                    candidate.files["pcap"],
                    candidate.token,
                    int(config.ingestion.time_bucket_seconds),
                ),
            ),
            "resource_sample": atomic_write_parquet(
                public_stage / "resource_sample.parquet",
                RESOURCE_SCHEMA,
                iter_resources(
                    candidate.files["res"],
                    candidate.token,
                    int(config.ingestion.time_bucket_seconds),
                ),
            ),
        }
        public_target.parent.mkdir(parents=True, exist_ok=True)
        private_target.parent.mkdir(parents=True, exist_ok=True)
        os_replace_directory(public_stage, public_target)
        os_replace_directory(private_stage, private_target)
        elapsed = time.monotonic() - started
        output_files = sorted(public_target.glob("*.parquet")) + sorted(
            private_target.glob("*.parquet")
        )
        result = {
            "schema_version": 1,
            "scenario_uid": candidate.uid,
            "scenario_token": candidate.token,
            "status": "complete",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "config_sha256": config_hash(config, config_path),
            "input_sha256": {
                extension: sha256_file(candidate.files[extension])
                for extension in REQUIRED_EXTENSIONS
            },
            "counts": counts,
            "outputs": {
                str(path.relative_to(artifacts)): sha256_file(path) for path in output_files
            },
            "elapsed_seconds": elapsed,
            "peak_ram_gib": _peak_ram_gib(),
            "public_interval_version": 2,
        }
        atomic_write_json(marker_path, result)
        report_path = artifacts / "reports" / "ingestion" / f"{candidate.token}.json"
        atomic_write_json(report_path, result)
        return result
    except BaseException:
        shutil.rmtree(stage_root, ignore_errors=True)
        raise
    finally:
        try:
            stage_root.rmdir()
        except OSError:
            pass


def os_replace_directory(source: Path, target: Path) -> None:
    source.replace(target)
