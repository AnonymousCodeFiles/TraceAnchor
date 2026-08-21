from __future__ import annotations

import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import duckdb

from traceanchor.config import TraceAnchorConfig
from traceanchor.evidence.schemas import (
    AlertContextRequest,
    AttackKnowledgeRequest,
    ConnectionRequest,
    FileActivityRequest,
    JoinEvidenceRequest,
    PacketMetadataRequest,
    ProcessTreeRequest,
    SocketOwnerRequest,
    SyscallRequest,
    TimelineRequest,
    ToolEnvelope,
    ToolModel,
    ValidateEvidenceRequest,
)
from traceanchor.ingest.common import atomic_write_text


_ID_PATTERN = re.compile(r"^(?:sc|pcap|res):tw_[0-9a-f]{24}:[^:]+(?::[0-9]+)?$")
_PRIVATE_QUERY_PATTERN = re.compile(
    r"(?:cve[-_]?\d|family|exploit[_ ]?(?:name|time|flag)?|gold|label|split|role|anchor)",
    re.IGNORECASE,
)


class EvidenceToolError(ValueError):
    """Sanitized tool failure that never exposes SQL or filesystem details."""


def _hash_json(value: object) -> str:
    canonical = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class EvidenceTools:
    """Fixed-query, read-only Agent View tools.

    The class intentionally has no method accepting SQL, paths, shell text, or
    a DuckDB connection from its caller. Every invocation opens the Agent DB in
    read-only mode and records a hash-only audit event outside that database.
    """

    def __init__(
        self,
        config: TraceAnchorConfig,
        config_path: Path,
        *,
        run_id: str = "smoke",
        agent: str = "tool_smoke",
    ) -> None:
        resolved = config.resolved_dict(config_path)
        self.db_path = Path(resolved["paths"]["evidence_db"])
        self.audit_dir = Path(resolved["paths"]["artifacts_dir"]) / "tool_audit"
        self.max_range_ns = int(config.evidence_store.max_query_time_range_seconds) * 1_000_000_000
        self.max_records = int(config.evidence_store.max_records_per_tool_call)
        self.run_id = run_id
        self.agent = agent
        if not self.db_path.exists():
            raise FileNotFoundError("Agent Evidence Ledger is not built")
        with duckdb.connect(str(self.db_path), read_only=True) as connection:
            self.ledger_version = str(
                connection.execute("select ledger_version from ledger_metadata").fetchone()[0]
            )

    def _connection(self) -> duckdb.DuckDBPyConnection:
        return duckdb.connect(str(self.db_path), read_only=True)

    def _check_scenario(self, connection: duckdb.DuckDBPyConnection, token: str) -> None:
        if connection.execute(
            "select 1 from scenario_public where scenario_token=? limit 1", [token]
        ).fetchone() is None:
            raise EvidenceToolError("unknown scenario_token")

    def _check_range(self, start_ts_ns: int, end_ts_ns: int) -> None:
        if end_ts_ns - start_ts_ns > self.max_range_ns:
            raise EvidenceToolError("time range exceeds tool limit")

    @staticmethod
    def _offset(cursor: str | None) -> int:
        if cursor is None:
            return 0
        try:
            offset = int(cursor)
        except ValueError as exc:
            raise EvidenceToolError("invalid cursor") from exc
        if offset < 0 or offset > 10_000_000:
            raise EvidenceToolError("invalid cursor")
        return offset

    @staticmethod
    def _rows(connection: duckdb.DuckDBPyConnection, query: str, params: list[Any]) -> list[dict[str, Any]]:
        result = connection.execute(query, params)
        names = [str(item[0]) for item in result.description]
        return [dict(zip(names, row)) for row in result.fetchall()]

    def _paged(
        self,
        connection: duckdb.DuckDBPyConnection,
        query: str,
        params: list[Any],
        *,
        max_records: int,
        cursor: str | None,
    ) -> tuple[list[dict[str, Any]], bool, str | None]:
        offset = self._offset(cursor)
        rows = self._rows(connection, query, [*params, max_records + 1, offset])
        truncated = len(rows) > max_records
        if truncated:
            rows = rows[:max_records]
        return rows, truncated, str(offset + max_records) if truncated else None

    def _audit(
        self,
        *,
        tool: str,
        parameter_sha256: str,
        started_at: str,
        finished_at: str,
        duration_ms: float,
        result_count: int,
        truncated: bool,
        error: str | None,
        input_sha256: str,
        output_sha256: str,
    ) -> None:
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        path = self.audit_dir / f"{self.run_id}.jsonl"
        record = {
            "run_id": self.run_id,
            "agent": self.agent,
            "tool": tool,
            "parameter_sha256": parameter_sha256,
            "started_at": started_at,
            "finished_at": finished_at,
            "duration_ms": duration_ms,
            "result_count": result_count,
            "truncated": truncated,
            "error": error,
            "cache_hit": False,
            "input_sha256": input_sha256,
            "output_sha256": output_sha256,
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()

    def _run(
        self,
        tool: str,
        request: ToolModel,
        function: Callable[[], ToolEnvelope],
    ) -> ToolEnvelope:
        payload = request.model_dump(mode="json")
        parameter_hash = _hash_json(payload)
        started = datetime.now(timezone.utc)
        start_clock = time.monotonic()
        error: str | None = None
        result_count = 0
        truncated = False
        output_hash = ""
        try:
            result = function()
            result_count = result.result_count
            truncated = result.truncated
            output_hash = _hash_json(result.model_dump(mode="json"))
            return result
        except EvidenceToolError as exc:
            error = str(exc)
            raise
        except Exception as exc:  # never expose SQL paths or provider internals
            error = type(exc).__name__
            raise EvidenceToolError("tool query failed") from None
        finally:
            finished = datetime.now(timezone.utc)
            self._audit(
                tool=tool,
                parameter_sha256=parameter_hash,
                started_at=started.isoformat(),
                finished_at=finished.isoformat(),
                duration_ms=(time.monotonic() - start_clock) * 1000.0,
                result_count=result_count,
                truncated=truncated,
                error=error,
                input_sha256=parameter_hash,
                output_sha256=output_hash,
            )

    def _envelope(
        self,
        tool: str,
        token: str,
        records: list[dict[str, Any]],
        truncated: bool = False,
        next_cursor: str | None = None,
    ) -> ToolEnvelope:
        return ToolEnvelope(
            tool=tool,
            scenario_token=token,
            records=records,
            result_count=len(records),
            truncated=truncated,
            next_cursor=next_cursor,
            ledger_version=self.ledger_version,
        )

    def get_alert_context(self, request: AlertContextRequest) -> ToolEnvelope:
        def query() -> ToolEnvelope:
            with self._connection() as connection:
                row = self._rows(
                    connection,
                    """select alert_id, scenario_token, channels, start_ts, end_ts,
                              peak_score_network, peak_score_host, threshold_version,
                              model_hashes
                       from model_alert where alert_id=? and scenario_token=?""",
                    [request.alert_id, request.scenario_token],
                )
                if not row:
                    raise EvidenceToolError("alert not found for scenario_token")
                public = self._rows(
                    connection,
                    "select start_ts_ns,end_ts_ns from scenario_public where scenario_token=?",
                    [request.scenario_token],
                )[0]
                item = row[0]
                item["allowed_start_ts"] = max(
                    int(public["start_ts_ns"]), int(item["start_ts"]) - 30_000_000_000
                )
                item["allowed_end_ts"] = min(
                    int(public["end_ts_ns"]), int(item["end_ts"]) + 60_000_000_000
                )
                return self._envelope("get_alert_context", request.scenario_token, [item])

        return self._run("get_alert_context", request, query)

    def list_connections(self, request: ConnectionRequest) -> ToolEnvelope:
        def query() -> ToolEnvelope:
            self._check_range(request.start_ts_ns, request.end_ts_ns)
            with self._connection() as connection:
                self._check_scenario(connection, request.scenario_token)
                filters = ["scenario_token=?", "ts_ns between ? and ?"]
                params: list[Any] = [request.scenario_token, request.start_ts_ns, request.end_ts_ns]
                if request.ip is not None:
                    filters.append("(src_ip=? or dst_ip=?)")
                    params.extend([request.ip, request.ip])
                if request.port is not None:
                    filters.append("(src_port=? or dst_port=?)")
                    params.extend([request.port, request.port])
                if request.protocol is not None:
                    filters.append("ip_protocol=?")
                    params.append(request.protocol)
                where = " and ".join(filters)
                rows, truncated, cursor = self._paged(
                    connection,
                    f"""select concat_ws('|',coalesce(src_ip,'?'),coalesce(src_port::varchar,'?'),
                                      coalesce(dst_ip,'?'),coalesce(dst_port::varchar,'?'),
                                      coalesce(ip_protocol::varchar,'?')) connection_key,
                              min(ts_ns)::bigint first_ts_ns, max(ts_ns)::bigint last_ts_ns,
                              count(*)::bigint packet_count, sum(wire_len)::bigint byte_count,
                              arg_min(evidence_id,ts_ns) first_evidence_id,
                              arg_max(evidence_id,ts_ns) last_evidence_id
                       from network_packet where {where}
                       group by all order by first_ts_ns, connection_key limit ? offset ?""",
                    params,
                    max_records=request.max_records,
                    cursor=request.cursor,
                )
                return self._envelope("list_connections", request.scenario_token, rows, truncated, cursor)

        return self._run("list_connections", request, query)

    def get_packet_metadata(self, request: PacketMetadataRequest) -> ToolEnvelope:
        def query() -> ToolEnvelope:
            self._check_range(request.start_ts_ns, request.end_ts_ns)
            with self._connection() as connection:
                self._check_scenario(connection, request.scenario_token)
                filters = ["scenario_token=?", "ts_ns between ? and ?"]
                params: list[Any] = [request.scenario_token, request.start_ts_ns, request.end_ts_ns]
                if request.ip is not None:
                    filters.append("(src_ip=? or dst_ip=?)")
                    params.extend([request.ip, request.ip])
                if request.port is not None:
                    filters.append("(src_port=? or dst_port=?)")
                    params.extend([request.port, request.port])
                if request.protocol is not None:
                    filters.append("ip_protocol=?")
                    params.append(request.protocol)
                rows, truncated, cursor = self._paged(
                    connection,
                    f"""select evidence_id, frame_no, ts_ns, captured_len, wire_len,
                              eth_type, src_ip, dst_ip, src_port, dst_port, ip_protocol,
                              tcp_flags, payload_len, time_bucket, parse_status
                       from network_packet where {' and '.join(filters)}
                       order by ts_ns, frame_no limit ? offset ?""",
                    params,
                    max_records=request.max_records,
                    cursor=request.cursor,
                )
                return self._envelope("get_packet_metadata", request.scenario_token, rows, truncated, cursor)

        return self._run("get_packet_metadata", request, query)

    def find_socket_owner(self, request: SocketOwnerRequest) -> ToolEnvelope:
        def query() -> ToolEnvelope:
            tolerance_ns = request.tolerance_ms * 1_000_000
            with self._connection() as connection:
                self._check_scenario(connection, request.scenario_token)
                rows = self._rows(
                    connection,
                    """select evidence_id, ts_ns, pid, tid, process_name, syscall,
                              socket_src_ip, socket_src_port, socket_dst_ip, socket_dst_port,
                              abs(ts_ns-?)::bigint time_delta_ns
                       from syscall_event
                       where scenario_token=? and abs(ts_ns-?)<=?
                         and ((socket_src_ip=? and socket_src_port=? and socket_dst_ip=? and socket_dst_port=?)
                              or (socket_src_ip=? and socket_src_port=? and socket_dst_ip=? and socket_dst_port=?))
                       order by time_delta_ns, ts_ns limit ?""",
                    [
                        request.ts_ns,
                        request.scenario_token,
                        request.ts_ns,
                        tolerance_ns,
                        request.src_ip,
                        request.src_port,
                        request.dst_ip,
                        request.dst_port,
                        request.dst_ip,
                        request.dst_port,
                        request.src_ip,
                        request.src_port,
                        request.max_records + 1,
                    ],
                )
                truncated = len(rows) > request.max_records
                if truncated:
                    rows = rows[: request.max_records]
                for row in rows:
                    row["match_method"] = "exact_tuple_time_window"
                    row["confidence"] = max(
                        0.0, 1.0 - float(row["time_delta_ns"]) / max(tolerance_ns, 1)
                    )
                return self._envelope(
                    "find_socket_owner", request.scenario_token, rows, truncated
                )

        return self._run("find_socket_owner", request, query)

    def get_process_tree(self, request: ProcessTreeRequest) -> ToolEnvelope:
        def query() -> ToolEnvelope:
            with self._connection() as connection:
                self._check_scenario(connection, request.scenario_token)
                rows = self._rows(
                    connection,
                    """select e.source_entity_id, e.target_entity_id,
                              s.canonical_key source_key, t.canonical_key target_key,
                              e.first_ts_ns, e.last_ts_ns,
                              e.representative_evidence_id evidence_id, e.relation_type
                       from provenance_edge e
                       join entity s on s.entity_id=e.source_entity_id
                       join entity t on t.entity_id=e.target_entity_id
                       where e.scenario_token=? and e.relation_type='PROCESS_PARENT_OF'
                       order by e.first_ts_ns""",
                    [request.scenario_token],
                )
                current = {f"pid:{request.pid}"}
                seen = set()
                selected = []
                for depth in range(request.depth):
                    next_keys = set()
                    for row in rows:
                        if row["source_key"] in current and row["target_key"] not in seen:
                            row = dict(row)
                            row["depth"] = depth + 1
                            selected.append(row)
                            seen.add(row["target_key"])
                            next_keys.add(row["target_key"])
                    current = next_keys
                if not selected and not any(row["target_key"] == f"pid:{request.pid}" for row in rows):
                    # A root process with no explicit child relation is a valid negative result.
                    return self._envelope("get_process_tree", request.scenario_token, [])
                return self._envelope("get_process_tree", request.scenario_token, selected[:200], len(selected) > 200, None)

        return self._run("get_process_tree", request, query)

    def list_syscalls(self, request: SyscallRequest) -> ToolEnvelope:
        def query() -> ToolEnvelope:
            self._check_range(request.start_ts_ns, request.end_ts_ns)
            with self._connection() as connection:
                self._check_scenario(connection, request.scenario_token)
                filters = ["scenario_token=?", "ts_ns between ? and ?"]
                params: list[Any] = [request.scenario_token, request.start_ts_ns, request.end_ts_ns]
                if request.pid is not None:
                    filters.append("pid=?")
                    params.append(request.pid)
                if request.tid is not None:
                    filters.append("tid=?")
                    params.append(request.tid)
                if request.syscall is not None:
                    filters.append("syscall=?")
                    params.append(request.syscall)
                rows, truncated, cursor = self._paged(
                    connection,
                    f"""select evidence_id,line_no,ts_ns,uid,pid,tid,process_name,syscall,
                              direction,result_class,fd,file_path,socket_src_ip,socket_src_port,
                              socket_dst_ip,socket_dst_port,protocol_hint,child_pid,parent_pid,
                              payload_present,time_bucket,parse_status
                       from syscall_event where {' and '.join(filters)}
                       order by ts_ns,line_no limit ? offset ?""",
                    params,
                    max_records=request.max_records,
                    cursor=request.cursor,
                )
                return self._envelope("list_syscalls", request.scenario_token, rows, truncated, cursor)

        return self._run("list_syscalls", request, query)

    def get_file_activity(self, request: FileActivityRequest) -> ToolEnvelope:
        def query() -> ToolEnvelope:
            self._check_range(request.start_ts_ns, request.end_ts_ns)
            with self._connection() as connection:
                self._check_scenario(connection, request.scenario_token)
                filters = [
                    "scenario_token=?",
                    "ts_ns between ? and ?",
                    "file_path is not null",
                ]
                params: list[Any] = [request.scenario_token, request.start_ts_ns, request.end_ts_ns]
                if request.pid is not None:
                    filters.append("pid=?")
                    params.append(request.pid)
                if request.path_contains is not None:
                    filters.append("contains(file_path, ?)")
                    params.append(request.path_contains)
                rows, truncated, cursor = self._paged(
                    connection,
                    f"""select evidence_id,ts_ns,pid,tid,process_name,syscall,direction,
                              result_class,fd,file_path,time_bucket,parse_status
                       from syscall_event where {' and '.join(filters)}
                       order by ts_ns,line_no limit ? offset ?""",
                    params,
                    max_records=request.max_records,
                    cursor=request.cursor,
                )
                return self._envelope("get_file_activity", request.scenario_token, rows, truncated, cursor)

        return self._run("get_file_activity", request, query)

    def join_host_network_evidence(self, request: JoinEvidenceRequest) -> ToolEnvelope:
        def query() -> ToolEnvelope:
            self._check_range(request.start_ts_ns, request.end_ts_ns)
            tolerance_ns = int(1_000_000_000)
            with self._connection() as connection:
                self._check_scenario(connection, request.scenario_token)
                filters = ["true"]
                filter_params: list[Any] = []
                if request.src_ip is not None:
                    filters.append("(s.socket_src_ip=? or n.src_ip=? or s.socket_dst_ip=? or n.dst_ip=?)")
                    filter_params.extend([request.src_ip] * 4)
                if request.src_port is not None:
                    filters.append("(s.socket_src_port=? or n.src_port=? or s.socket_dst_port=? or n.dst_port=?)")
                    filter_params.extend([request.src_port] * 4)
                if request.dst_ip is not None:
                    filters.append("(s.socket_dst_ip=? or n.dst_ip=? or s.socket_src_ip=? or n.src_ip=?)")
                    filter_params.extend([request.dst_ip] * 4)
                if request.dst_port is not None:
                    filters.append("(s.socket_dst_port=? or n.dst_port=? or s.socket_src_port=? or n.src_port=?)")
                    filter_params.extend([request.dst_port] * 4)
                network_params = [
                    request.scenario_token,
                    max(0, request.start_ts_ns - tolerance_ns),
                    request.end_ts_ns + tolerance_ns,
                ]
                params = [
                    request.scenario_token,
                    request.start_ts_ns,
                    request.end_ts_ns,
                    *network_params,
                    *network_params,
                    *filter_params,
                    request.max_records + 1,
                    self._offset(request.cursor),
                ]
                rows = self._rows(
                    connection,
                    f"""with s_filtered as materialized (
                           select * from syscall_event
                           where scenario_token=? and ts_ns between ? and ?
                             and socket_src_ip is not null and socket_src_port is not null
                             and socket_dst_ip is not null and socket_dst_port is not null
                         ), n_oriented as materialized (
                           select *,src_ip match_src_ip,src_port match_src_port,
                                    dst_ip match_dst_ip,dst_port match_dst_port
                           from network_packet
                           where scenario_token=? and ts_ns between ? and ?
                             and src_ip is not null and src_port is not null
                             and dst_ip is not null and dst_port is not null
                           union all
                           select *,dst_ip match_src_ip,dst_port match_src_port,
                                    src_ip match_dst_ip,src_port match_dst_port
                           from network_packet
                           where scenario_token=? and ts_ns between ? and ?
                             and src_ip is not null and src_port is not null
                             and dst_ip is not null and dst_port is not null
                         )
                       select s.evidence_id syscall_evidence_id,n.evidence_id packet_evidence_id,
                              s.ts_ns syscall_ts_ns,n.ts_ns packet_ts_ns,abs(s.ts_ns-n.ts_ns)::bigint time_delta_ns,
                              s.pid,s.tid,s.process_name,s.syscall,n.src_ip,n.src_port,n.dst_ip,n.dst_port,
                              'SOCKET_MATCHES_FLOW' relation_type,
                              greatest(0.0,1.0-abs(s.ts_ns-n.ts_ns)::double/{tolerance_ns}) confidence
                       from s_filtered s join n_oriented n
                         on s.socket_src_ip=n.match_src_ip
                        and s.socket_src_port=n.match_src_port
                        and s.socket_dst_ip=n.match_dst_ip
                        and s.socket_dst_port=n.match_dst_port
                        and abs(s.ts_ns-n.ts_ns)<={tolerance_ns}
                       where {' and '.join(filters)}
                       order by time_delta_ns,syscall_ts_ns,s.evidence_id,n.evidence_id
                       limit ? offset ?""",
                    params,
                )
                truncated = len(rows) > request.max_records
                if truncated:
                    rows = rows[: request.max_records]
                next_cursor = (
                    str(self._offset(request.cursor) + request.max_records)
                    if truncated
                    else None
                )
                return self._envelope(
                    "join_host_network_evidence",
                    request.scenario_token,
                    rows,
                    truncated,
                    next_cursor,
                )

        return self._run("join_host_network_evidence", request, query)

    def build_timeline(self, request: TimelineRequest) -> ToolEnvelope:
        def query() -> ToolEnvelope:
            if any(not _ID_PATTERN.match(value) for value in request.evidence_ids):
                raise EvidenceToolError("malformed Evidence ID")
            if any(f":{request.scenario_token}:" not in value for value in request.evidence_ids):
                raise EvidenceToolError("cross-scene Evidence ID")
            ids = request.evidence_ids
            placeholders = ",".join("?" for _ in ids)
            with self._connection() as connection:
                rows = self._rows(
                    connection,
                    f"""select evidence_id,ts_ns,'syscall' event_type,process_name,syscall,
                              pid,tid,file_path as object_key
                       from syscall_event where scenario_token=? and evidence_id in ({placeholders})
                       union all
                       select evidence_id,ts_ns,'packet',null,null,null,null,
                              concat_ws('|',src_ip,src_port::varchar,dst_ip,dst_port::varchar)
                       from network_packet where scenario_token=? and evidence_id in ({placeholders})
                       union all
                       select evidence_id,ts_ns,'resource',null,null,null,null,null
                       from resource_sample where scenario_token=? and evidence_id in ({placeholders})
                       order by ts_ns,evidence_id""",
                    [request.scenario_token, *ids, request.scenario_token, *ids, request.scenario_token, *ids],
                )
                found = {str(row["evidence_id"]) for row in rows}
                if found != set(ids):
                    raise EvidenceToolError("one or more Evidence IDs do not exist")
                return self._envelope("build_timeline", request.scenario_token, rows)

        return self._run("build_timeline", request, query)

    def retrieve_attack_knowledge(self, request: AttackKnowledgeRequest) -> ToolEnvelope:
        def query() -> ToolEnvelope:
            if _PRIVATE_QUERY_PATTERN.search(request.observed_behavior):
                raise EvidenceToolError("knowledge query contains a private field or label")
            terms = {term.lower() for term in re.findall(r"[A-Za-z0-9_]{3,}", request.observed_behavior)}
            with self._connection() as connection:
                self._check_scenario(connection, request.scenario_token)
                rows = self._rows(
                    connection,
                    "select technique_id,name,description,keywords,source,version,untrusted_text from attack_knowledge",
                    [],
                )
            ranked = []
            for row in rows:
                haystack = " ".join(
                    str(row.get(key, "")).lower() for key in ("name", "description", "keywords")
                )
                score = sum(1 for term in terms if term in haystack)
                if score:
                    ranked.append((score, str(row["technique_id"]), row))
            ranked.sort(key=lambda item: (-item[0], item[1]))
            return self._envelope(
                "retrieve_attack_knowledge",
                request.scenario_token,
                [row for _score, _id, row in ranked[: request.max_records]],
            )

        return self._run("retrieve_attack_knowledge", request, query)

    def validate_evidence_ids(self, request: ValidateEvidenceRequest) -> ToolEnvelope:
        def query() -> ToolEnvelope:
            if any(not _ID_PATTERN.match(value) for value in request.evidence_ids):
                raise EvidenceToolError("malformed Evidence ID")
            if any(f":{request.scenario_token}:" not in value for value in request.evidence_ids):
                raise EvidenceToolError("cross-scene Evidence ID")
            ids = request.evidence_ids
            placeholders = ",".join("?" for _ in ids)
            with self._connection() as connection:
                rows = self._rows(
                    connection,
                    f"""select evidence_id,ts_ns,'syscall' evidence_type from syscall_event
                       where scenario_token=? and evidence_id in ({placeholders})
                       union all
                       select evidence_id,ts_ns,'packet' from network_packet
                       where scenario_token=? and evidence_id in ({placeholders})
                       union all
                       select evidence_id,ts_ns,'resource' from resource_sample
                       where scenario_token=? and evidence_id in ({placeholders})""",
                    [request.scenario_token, *ids, request.scenario_token, *ids, request.scenario_token, *ids],
                )
                found = {str(row["evidence_id"]) for row in rows}
                entities = set()
                for entity_id in request.required_entity_ids:
                    if not entity_id.startswith("ent_"):
                        raise EvidenceToolError("malformed entity ID")
                    if connection.execute(
                        "select 1 from entity where scenario_token=? and entity_id=?",
                        [request.scenario_token, entity_id],
                    ).fetchone() is None:
                        raise EvidenceToolError("required entity is absent")
                    entities.add(entity_id)
                output = [
                    {
                        "evidence_id": evidence_id,
                        "valid": evidence_id in found,
                        "scenario_token": request.scenario_token,
                        "supports_required_entities": bool(entities),
                        **next(
                            (
                                {"ts_ns": row["ts_ns"], "evidence_type": row["evidence_type"]}
                                for row in rows
                                if row["evidence_id"] == evidence_id
                            ),
                            {"ts_ns": None, "evidence_type": None},
                        ),
                    }
                    for evidence_id in ids
                ]
                return self._envelope("validate_evidence_ids", request.scenario_token, output)

        return self._run("validate_evidence_ids", request, query)


__all__ = ["EvidenceToolError", "EvidenceTools"]
