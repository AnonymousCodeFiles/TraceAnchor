from __future__ import annotations

import hashlib
import json
import os
import resource
import stat
import tempfile
import time
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from traceanchor.config import TraceAnchorConfig, config_hash
from traceanchor.ingest.common import atomic_write_json, atomic_write_table, sha256_file
from traceanchor.ingest.manifest import (
    candidate_from_example,
    select_family_sample_from_manifest,
)


ENTITY_SCHEMA = pa.schema(
    [
        ("entity_id", pa.string()),
        ("scenario_token", pa.string()),
        ("entity_type", pa.string()),
        ("canonical_key", pa.string()),
        ("canonical_attributes_json", pa.string()),
        ("first_ts_ns", pa.int64()),
        ("last_ts_ns", pa.int64()),
        ("evidence_count", pa.int64()),
    ]
)

EDGE_SCHEMA = pa.schema(
    [
        ("edge_id", pa.string()),
        ("scenario_token", pa.string()),
        ("source_entity_id", pa.string()),
        ("target_entity_id", pa.string()),
        ("relation_type", pa.string()),
        ("first_ts_ns", pa.int64()),
        ("last_ts_ns", pa.int64()),
        ("representative_evidence_id", pa.string()),
        ("evidence_count", pa.int64()),
        ("confidence", pa.float64()),
        ("generation_method", pa.string()),
    ]
)

AUDIT_SCHEMA = pa.schema(
    [
        ("run_id", pa.string()),
        ("agent", pa.string()),
        ("tool", pa.string()),
        ("parameter_sha256", pa.string()),
        ("started_at", pa.string()),
        ("finished_at", pa.string()),
        ("duration_ms", pa.float64()),
        ("result_count", pa.int64()),
        ("truncated", pa.bool_()),
        ("error", pa.string()),
        ("cache_hit", pa.bool_()),
        ("input_sha256", pa.string()),
        ("output_sha256", pa.string()),
    ]
)


ENTITY_QUERY = r"""
with s as materialized (select * from read_parquet(?)),
     n as materialized (select * from read_parquet(?)),
entities as (
  select
    'ent_' || substr(sha256(scenario_token || '|process|pid:' || pid::varchar), 1, 24) entity_id,
    scenario_token, 'process' entity_type, 'pid:' || pid::varchar canonical_key,
    to_json(struct_pack(pid := pid, process_name := arg_max(process_name, ts_ns))) canonical_attributes_json,
    min(ts_ns)::bigint first_ts_ns, max(ts_ns)::bigint last_ts_ns, count(*)::bigint evidence_count
  from s where pid is not null group by scenario_token, pid
  union all
  select
    'ent_' || substr(sha256(scenario_token || '|thread|tid:' || tid::varchar), 1, 24),
    scenario_token, 'thread', 'tid:' || tid::varchar,
    to_json(struct_pack(tid := tid, pid := arg_max(pid, ts_ns))),
    min(ts_ns)::bigint, max(ts_ns)::bigint, count(*)::bigint
  from s where tid is not null group by scenario_token, tid
  union all
  select
    'ent_' || substr(sha256(scenario_token || '|file|path:' || file_path), 1, 24),
    scenario_token, 'file',
      'path:file_' || substr(sha256(scenario_token || '|file|' || file_path), 1, 24),
    to_json(struct_pack(path := 'file_' || substr(sha256(scenario_token || '|file|' || file_path), 1, 24))),
    min(ts_ns)::bigint, max(ts_ns)::bigint, count(*)::bigint
  from s where file_path is not null and file_path <> '' group by scenario_token, file_path
  union all
  select
    'ent_' || substr(sha256(scenario_token || '|socket|' || canonical_key), 1, 24),
    scenario_token, 'socket', canonical_key,
    to_json(struct_pack(src_ip := socket_src_ip, src_port := socket_src_port,
                        dst_ip := socket_dst_ip, dst_port := socket_dst_port,
                        protocol := protocol_hint)),
    min(ts_ns)::bigint, max(ts_ns)::bigint, count(*)::bigint
  from (
    select *, concat_ws('|', coalesce(protocol_hint, 'unknown'),
      coalesce(socket_src_ip, '?'), coalesce(socket_src_port::varchar, '?'),
      coalesce(socket_dst_ip, '?'), coalesce(socket_dst_port::varchar, '?')) canonical_key
    from s where socket_src_ip is not null or socket_dst_ip is not null
  ) sockets
  group by scenario_token, canonical_key, socket_src_ip, socket_src_port,
           socket_dst_ip, socket_dst_port, protocol_hint
  union all
  select
    'ent_' || substr(sha256(scenario_token || '|connection|' || canonical_key), 1, 24),
    scenario_token, 'connection', canonical_key,
    to_json(struct_pack(src_ip := src_ip, src_port := src_port, dst_ip := dst_ip,
                        dst_port := dst_port, protocol := ip_protocol)),
    min(ts_ns)::bigint, max(ts_ns)::bigint, count(*)::bigint
  from (
    select *, concat_ws('|', coalesce(ip_protocol::varchar, 'unknown'),
      coalesce(src_ip, '?'), coalesce(src_port::varchar, '?'),
      coalesce(dst_ip, '?'), coalesce(dst_port::varchar, '?')) canonical_key
    from n where src_ip is not null or dst_ip is not null
  ) connections
  group by scenario_token, canonical_key, src_ip, src_port, dst_ip, dst_port, ip_protocol
)
select * from entities order by entity_type, canonical_key
"""


EDGE_QUERY = r"""
with s as materialized (select * from read_parquet(?)),
relations as (
  select
    scenario_token,
    'ent_' || substr(sha256(scenario_token || '|process|pid:' || pid::varchar), 1, 24) source_entity_id,
    'ent_' || substr(sha256(scenario_token || '|thread|tid:' || tid::varchar), 1, 24) target_entity_id,
    'PROCESS_HAS_THREAD' relation_type, ts_ns, evidence_id
  from s where pid is not null and tid is not null
  union all
  select
    scenario_token,
    'ent_' || substr(sha256(scenario_token || '|process|pid:' || parent_pid::varchar), 1, 24),
    'ent_' || substr(sha256(scenario_token || '|process|pid:' || child_pid::varchar), 1, 24),
    'PROCESS_PARENT_OF', ts_ns, evidence_id
  from s where parent_pid is not null and child_pid is not null
  union all
  select
    scenario_token,
    'ent_' || substr(sha256(scenario_token || '|process|pid:' || pid::varchar), 1, 24),
    'ent_' || substr(sha256(scenario_token || '|file|path:' || file_path), 1, 24),
    case
      when syscall in ('read','readv','pread','pread64','preadv','preadv2') then 'PROCESS_READ_FILE'
      when syscall in ('write','writev','pwrite','pwrite64','pwritev','pwritev2') then 'PROCESS_WROTE_FILE'
      else 'PROCESS_OPENED_FILE'
    end,
    ts_ns, evidence_id
  from s where pid is not null and file_path is not null and file_path <> ''
  union all
  select
    scenario_token,
    'ent_' || substr(sha256(scenario_token || '|process|pid:' || pid::varchar), 1, 24),
    'ent_' || substr(sha256(scenario_token || '|socket|' || concat_ws('|',
      coalesce(protocol_hint, 'unknown'), coalesce(socket_src_ip, '?'),
      coalesce(socket_src_port::varchar, '?'), coalesce(socket_dst_ip, '?'),
      coalesce(socket_dst_port::varchar, '?'))), 1, 24),
    case when syscall in ('accept','accept4') then 'PROCESS_ACCEPTED_SOCKET'
         else 'PROCESS_CONNECTED_SOCKET' end,
    ts_ns, evidence_id
  from s where pid is not null and (socket_src_ip is not null or socket_dst_ip is not null)
), compact as (
  select scenario_token, source_entity_id, target_entity_id, relation_type,
         min(ts_ns)::bigint first_ts_ns, max(ts_ns)::bigint last_ts_ns,
         arg_min(evidence_id, ts_ns) representative_evidence_id,
         count(*)::bigint evidence_count
  from relations group by scenario_token, source_entity_id, target_entity_id, relation_type
)
select
  'edge_' || substr(sha256(scenario_token || '|' || relation_type || '|' ||
    source_entity_id || '|' || target_entity_id), 1, 24) edge_id,
  scenario_token, source_entity_id, target_entity_id, relation_type,
  first_ts_ns, last_ts_ns, representative_evidence_id, evidence_count,
  1.0::double confidence, 'deterministic_field_relation' generation_method
from compact order by relation_type, source_entity_id, target_entity_id
"""


KNOWLEDGE_ROWS = [
    ("T1046", "Network Service Scanning", "Probe remote services and ports.", "scan port service probe"),
    ("T1071", "Application Layer Protocol", "Communicate using application-layer protocols.", "http dns web protocol connection"),
    ("T1105", "Ingress Tool Transfer", "Transfer files or tools into an environment.", "download upload transfer file network"),
    ("T1059", "Command and Scripting Interpreter", "Execute commands through an interpreter.", "command shell script exec"),
    ("T1059.004", "Unix Shell", "Execute commands through a Unix shell.", "bash sh shell command"),
    ("T1082", "System Information Discovery", "Gather operating-system and host information.", "uname hostname system information"),
    ("T1083", "File and Directory Discovery", "Enumerate files and directories.", "directory file list stat open"),
    ("T1005", "Data from Local System", "Collect data available on the local system.", "read file local data"),
    ("T1053", "Scheduled Task or Job", "Use scheduled execution mechanisms.", "cron schedule timer job"),
    ("T1547", "Boot or Logon Autostart Execution", "Establish execution during boot or logon.", "startup boot profile service"),
    ("T1068", "Exploitation for Privilege Escalation", "Use a software flaw to gain privileges.", "privilege uid root capability"),
    ("T1021", "Remote Services", "Use remote services to access another system.", "ssh remote service login"),
    ("T1070", "Indicator Removal", "Remove or alter artifacts that could support detection.", "unlink delete clear log"),
    ("T1486", "Data Encrypted for Impact", "Encrypt data to interrupt availability.", "encrypt rename write impact"),
    ("T1190", "Exploit Public-Facing Application", "Target a weakness in an exposed application.", "web server request public application"),
]


def _connection() -> duckdb.DuckDBPyConnection:
    connection = duckdb.connect()
    connection.execute("SET enable_progress_bar=false")
    connection.execute("SET threads=2")
    connection.execute("SET memory_limit='8GB'")
    return connection


def _cast_table(table: pa.Table, schema: pa.Schema) -> pa.Table:
    return table.select(schema.names).cast(schema)


def _build_scenario(
    token: str,
    parquet_dir: Path,
    evidence_dir: Path,
    marker_path: Path,
    *,
    resume: bool,
) -> dict[str, object]:
    entity_path = evidence_dir / "agent" / "entity" / f"{token}.parquet"
    edge_path = evidence_dir / "agent" / "provenance_edge" / f"{token}.parquet"
    if resume and marker_path.exists() and entity_path.exists() and edge_path.exists():
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if int(marker.get("schema_version", 0)) >= 2:
            return marker
    scenario_dir = parquet_dir / token
    syscall_path = scenario_dir / "syscall_event.parquet"
    packet_path = scenario_dir / "network_packet.parquet"
    started = time.monotonic()
    connection = _connection()
    try:
        entities = connection.execute(
            ENTITY_QUERY, [str(syscall_path), str(packet_path)]
        ).fetch_arrow_table()
        edges = connection.execute(EDGE_QUERY, [str(syscall_path)]).fetch_arrow_table()
    finally:
        connection.close()
    atomic_write_table(entity_path, _cast_table(entities, ENTITY_SCHEMA))
    atomic_write_table(edge_path, _cast_table(edges, EDGE_SCHEMA))
    result = {
        "schema_version": 2,
        "scenario_token": token,
        "entities": entities.num_rows,
        "provenance_edges": edges.num_rows,
        "entity_sha256": sha256_file(entity_path),
        "provenance_edge_sha256": sha256_file(edge_path),
        "elapsed_seconds": time.monotonic() - started,
    }
    atomic_write_json(marker_path, result)
    return result


def _write_knowledge(path: Path) -> None:
    rows = [
        {
            "technique_id": technique_id,
            "name": name,
            "description": description,
            "keywords": keywords,
            "source": f"https://attack.mitre.org/techniques/{technique_id.replace('.', '/')}/",
            "version": "traceanchor-attack-subset-2026-07-26",
            "untrusted_text": True,
        }
        for technique_id, name, description, keywords in KNOWLEDGE_ROWS
    ]
    atomic_write_table(path, pa.Table.from_pylist(rows))


def _consolidate_alerts(alerts_dir: Path, output_path: Path) -> None:
    rows = []
    for path in sorted(alerts_dir.glob("*/*.parquet")):
        rows.extend(pq.read_table(path).to_pylist())
    rows.sort(key=lambda row: str(row["alert_id"]))
    if rows:
        schema = pq.read_schema(next(iter(sorted(alerts_dir.glob("*/*.parquet")))))
        table = pa.Table.from_pylist(rows, schema=schema)
    else:
        table = pa.table({"alert_id": pa.array([], type=pa.string())})
    atomic_write_table(output_path, table)


def _sql_literal(value: Path | str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _new_database(path: Path, statements: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(fd)
    os.unlink(temporary)
    try:
        connection = duckdb.connect(temporary)
        try:
            for statement in statements:
                connection.execute(statement)
            connection.execute("CHECKPOINT")
        finally:
            connection.close()
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _write_databases(
    config: TraceAnchorConfig,
    config_path: Path,
    evidence_dir: Path,
    ledger_version: str,
) -> None:
    resolved = config.resolved_dict(config_path)
    parquet_dir = Path(resolved["paths"]["parquet_dir"])
    evaluator_dir = Path(resolved["paths"]["evaluator_dir"])
    splits_dir = Path(resolved["paths"]["splits_dir"])
    agent_db = Path(resolved["paths"]["evidence_db"])
    evaluator_db = Path(resolved["paths"]["evaluator_db"])
    if agent_db.exists():
        agent_db.chmod(stat.S_IRUSR | stat.S_IWUSR)
    public_glob = _sql_literal(parquet_dir / "tw_*" / "scenario_public.parquet")
    syscall_glob = _sql_literal(parquet_dir / "tw_*" / "syscall_event.parquet")
    packet_glob = _sql_literal(parquet_dir / "tw_*" / "network_packet.parquet")
    resource_glob = _sql_literal(parquet_dir / "tw_*" / "resource_sample.parquet")
    entity_glob = _sql_literal(evidence_dir / "agent" / "entity" / "tw_*.parquet")
    edge_glob = _sql_literal(evidence_dir / "agent" / "provenance_edge" / "tw_*.parquet")
    alert_path = _sql_literal(evidence_dir / "agent" / "model_alert.parquet")
    audit_path = _sql_literal(evidence_dir / "agent" / "tool_audit_empty.parquet")
    knowledge_path = _sql_literal(evidence_dir / "agent" / "attack_knowledge.parquet")
    agent_statements = [
        f"CREATE TABLE ledger_metadata AS SELECT '{ledger_version}'::varchar ledger_version, 1::integer schema_version",
        f"CREATE VIEW scenario_public AS SELECT * FROM read_parquet({public_glob}, union_by_name=true)",
        f"""CREATE VIEW syscall_event AS SELECT evidence_id, scenario_token, line_no, ts_ns,
            uid, pid, tid, process_name, syscall, direction, result_class, fd,
            case when file_path is null then null
                 else 'file_' || substr(sha256(scenario_token || '|file|' || file_path), 1, 24)
            end file_path,
            socket_src_ip, socket_src_port, socket_dst_ip, socket_dst_port, protocol_hint,
            child_pid, parent_pid, payload_present, time_bucket, parse_status
            FROM read_parquet({syscall_glob}, union_by_name=true)""",
        f"""CREATE VIEW network_packet AS SELECT evidence_id, scenario_token, frame_no, ts_ns,
            captured_len, wire_len, eth_type, src_ip, dst_ip, src_port, dst_port,
            ip_protocol, tcp_flags, seq, ack, payload_len, time_bucket, parse_status
            FROM read_parquet({packet_glob}, union_by_name=true)""",
        f"""CREATE VIEW resource_sample AS SELECT evidence_id, scenario_token, row_no, ts_ns,
            cpu_usage, memory_usage, network_received, network_send, storage_read,
            storage_written, missing_mask, time_bucket, parse_status
            FROM read_parquet({resource_glob}, union_by_name=true)""",
        f"CREATE VIEW entity AS SELECT * FROM read_parquet({entity_glob}, union_by_name=true)",
        f"CREATE VIEW provenance_edge AS SELECT * FROM read_parquet({edge_glob}, union_by_name=true)",
        f"CREATE VIEW model_alert AS SELECT * FROM read_parquet({alert_path})",
        f"CREATE VIEW tool_audit AS SELECT * FROM read_parquet({audit_path})",
        f"CREATE VIEW attack_knowledge AS SELECT * FROM read_parquet({knowledge_path})",
    ]
    _new_database(agent_db, agent_statements)
    agent_db.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)

    private_glob = _sql_literal(evaluator_dir / "tw_*" / "scenario_private.parquet")
    anchor_glob = _sql_literal(evaluator_dir / "tw_*" / "exploit_anchors.parquet")
    trigger_path = _sql_literal(splits_dir / "trigger_split.parquet")
    agent_split_path = _sql_literal(splits_dir / "agent_split.parquet")
    evaluator_statements = [
        f"CREATE TABLE ledger_metadata AS SELECT '{ledger_version}'::varchar ledger_version, 1::integer schema_version",
        f"CREATE VIEW scenario_private AS SELECT * FROM read_parquet({private_glob}, union_by_name=true)",
        f"CREATE VIEW exploit_anchor AS SELECT * FROM read_parquet({anchor_glob}, union_by_name=true)",
        f"""CREATE VIEW split_private AS
            SELECT t.scenario_uid, t.scenario_token, t.family_private, t.trigger_split,
                   t.gold_reserved, a.incident_id, a.agent_split,
                   a.locked_until_protocol_freeze
            FROM read_parquet({trigger_path}) t
            LEFT JOIN read_parquet({agent_split_path}) a USING (scenario_uid, scenario_token)""",
    ]
    _new_database(evaluator_db, evaluator_statements)
    evaluator_db.chmod(stat.S_IRUSR | stat.S_IWUSR)


def build_evidence_store(
    config: TraceAnchorConfig,
    config_path: Path,
    *,
    selection: str = "all",
    resume: bool = False,
) -> dict[str, object]:
    started = time.monotonic()
    resolved = config.resolved_dict(config_path)
    parquet_dir = Path(resolved["paths"]["parquet_dir"])
    artifacts_dir = Path(resolved["paths"]["artifacts_dir"])
    evidence_dir = artifacts_dir / "evidence"
    completion_dir = Path(resolved["paths"]["completion_markers_dir"])
    all_tokens = sorted(path.name for path in parquet_dir.glob("tw_*") if path.is_dir())
    if selection == "example":
        tokens = [candidate_from_example(config, config_path).token]
    elif selection == "family_sample":
        tokens = [item.token for item in select_family_sample_from_manifest(config, config_path)]
    elif selection == "all":
        tokens = all_tokens
    else:
        raise ValueError(f"invalid evidence-store selection: {selection}")
    progress_path = Path(resolved["paths"]["logs_dir"]) / "evidence_store_progress.json"
    results = []
    for index, token in enumerate(tokens, start=1):
        if not (parquet_dir / token / "scenario_public.parquet").exists():
            raise FileNotFoundError(f"public scenario shard missing: {token}")
        result = _build_scenario(
            token,
            parquet_dir,
            evidence_dir,
            completion_dir / "evidence" / f"{token}.done",
            resume=resume,
        )
        results.append(result)
        atomic_write_json(
            progress_path,
            {
                "selection": selection,
                "completed_scenarios": index,
                "total_scenarios": len(tokens),
                "last_scenario_token": token,
            },
        )

    agent_dir = evidence_dir / "agent"
    _write_knowledge(agent_dir / "attack_knowledge.parquet")
    _consolidate_alerts(Path(resolved["paths"]["alerts_dir"]), agent_dir / "model_alert.parquet")
    atomic_write_table(
        agent_dir / "tool_audit_empty.parquet",
        pa.Table.from_pylist([], schema=AUDIT_SCHEMA),
    )
    completed_markers = [
        completion_dir / "evidence" / f"{token}.done" for token in all_tokens
    ]
    complete = all(path.exists() for path in completed_markers)
    index = {
        "schema_version": 2,
        "selection": selection,
        "requested_scenarios": len(tokens),
        "all_public_scenarios": len(all_tokens),
        "completed_all_scenarios": complete,
        "entities": sum(int(json.loads(path.read_text())["entities"]) for path in completed_markers if path.exists()),
        "provenance_edges": sum(
            int(json.loads(path.read_text())["provenance_edges"])
            for path in completed_markers
            if path.exists()
        ),
        "provenance_edges": sum(int(json.loads(path.read_text())["provenance_edges"]) for path in completed_markers if path.exists()),
        "scenario_marker_hashes": {
            path.stem: sha256_file(path) for path in completed_markers if path.exists()
        },
    }
    index_path = evidence_dir / "ledger_index.json"
    atomic_write_json(index_path, index)
    ledger_version = "ledger_" + hashlib.sha256(
        json.dumps(index, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    _write_databases(config, config_path, evidence_dir, ledger_version)
    result = {
        **index,
        "ledger_version": ledger_version,
        "ledger_index_sha256": sha256_file(index_path),
        "agent_database": str(Path(resolved["paths"]["evidence_db"])),
        "evaluator_database": str(Path(resolved["paths"]["evaluator_db"])),
        "elapsed_seconds": time.monotonic() - started,
        "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
        "processed": results,
    }
    if complete:
        atomic_write_json(completion_dir / "WP4_store.done", result)
    return result


__all__ = [
    "AUDIT_SCHEMA",
    "EDGE_SCHEMA",
    "ENTITY_SCHEMA",
    "build_evidence_store",
]
