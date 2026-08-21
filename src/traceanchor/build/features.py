from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from traceanchor.config import FORBIDDEN_AGENT_COLUMNS, TraceAnchorConfig, config_hash
from traceanchor.ingest.common import atomic_write_json, atomic_write_table, sha256_file


TEMPLATE_SQL = """
coalesce(syscall, 'unknown') || '|' || coalesce(direction, '?') || '|' ||
coalesce(result_class, 'unknown') || '|' ||
case
  when file_path is not null then 'file'
  when socket_src_ip is not null then 'socket'
  when child_pid is not null or parent_pid is not null then 'process'
  else 'other'
end
"""

HOST_STAT_FEATURES = [
    "event_count",
    "unique_pid_count",
    "unique_tid_count",
    "unique_file_count",
    "unique_socket_count",
    "process_create_count",
    "exec_count",
    "connect_count",
    "accept_count",
    "file_write_count",
    "permission_change_count",
    "delete_count",
    "failed_operation_count",
    "missing",
    "age_seconds",
]

NETWORK_FEATURES = [
    "packet_count",
    "byte_count",
    "packet_length_mean",
    "packet_length_std",
    "packet_length_p25",
    "packet_length_p50",
    "packet_length_p75",
    "packet_length_max",
    "interarrival_mean_ms",
    "interarrival_std_ms",
    "interarrival_p50_ms",
    "interarrival_p95_ms",
    "connection_count",
    "new_connection_count",
    "closed_connection_count",
    "unique_ip_count",
    "unique_port_count",
    "tcp_syn_count",
    "tcp_rst_count",
    "tcp_fin_count",
    "tcp_ack_count",
    "retransmission_estimate",
    "tcp_count",
    "udp_count",
    "icmp_count",
    "other_protocol_count",
    "missing",
    "age_seconds",
]

META_COLUMNS = [
    "scenario_token",
    "second_ts",
    "label_private",
    "loss_mask",
    "max_observed_ts_ns",
]


def _connection() -> duckdb.DuckDBPyConnection:
    connection = duckdb.connect()
    connection.execute("SET enable_progress_bar=false")
    connection.execute("SET threads=2")
    connection.execute("SET memory_limit='8GB'")
    return connection


def _load_split_rows(config: TraceAnchorConfig, config_path: Path, split: str) -> list[dict]:
    splits_dir = Path(config.resolved_dict(config_path)["paths"]["splits_dir"])
    table = pq.read_table(
        splits_dir / "trigger_split.parquet", filters=[("trigger_split", "=", split)]
    )
    return sorted(table.to_pylist(), key=lambda row: str(row["scenario_token"]))


def build_host_vocabulary(config: TraceAnchorConfig, config_path: Path) -> dict[str, object]:
    resolved = config.resolved_dict(config_path)
    features_dir = Path(resolved["paths"]["features_dir"])
    parquet_dir = Path(resolved["paths"]["parquet_dir"])
    splits_dir = Path(resolved["paths"]["splits_dir"])
    train_rows = _load_split_rows(config, config_path, "train")
    vocab_dir = features_dir / "vocab"
    unigram_path = vocab_dir / "host_unigram.json"
    bigram_path = vocab_dir / "host_bigram.json"
    if unigram_path.exists() and bigram_path.exists():
        unigram_doc = json.loads(unigram_path.read_text())
        bigram_doc = json.loads(bigram_path.read_text())
        return {
            "train_scenarios": len(train_rows),
            "unigram_size": len(unigram_doc["items"]),
            "bigram_size": len(bigram_doc["items"]),
            "trigger_split_sha256": unigram_doc["trigger_split_sha256"],
            "status": "reused_complete",
        }
    partial_path = Path(resolved["paths"]["logs_dir"]) / "host_vocab_partial.json"
    completed_tokens: set[str] = set()
    unigram = Counter()
    bigram = Counter()
    if partial_path.exists():
        partial = json.loads(partial_path.read_text())
        completed_tokens = set(partial.get("completed_tokens", []))
        unigram.update({str(key): int(value) for key, value in partial.get("unigram", {}).items()})
        bigram.update({str(key): int(value) for key, value in partial.get("bigram", {}).items()})
    progress_path = Path(resolved["paths"]["logs_dir"]) / "host_vocab_progress.json"
    connection = _connection()
    try:
        for index, row in enumerate(train_rows, start=1):
            token = str(row["scenario_token"])
            if token in completed_tokens:
                continue
            path = parquet_dir / token / "syscall_event.parquet"
            query = f"""
                with base as (
                    select line_no, {TEMPLATE_SQL} as template
                    from read_parquet(?)
                ), sequenced as (
                    select template, lag(template) over (order by line_no) as previous
                    from base
                )
                select 'unigram' as kind, template as item, count(*) as frequency
                from sequenced group by template
                union all
                select 'bigram' as kind, previous || '->' || template as item, count(*)
                from sequenced where previous is not null group by previous, template
            """
            for kind, item, frequency in connection.execute(query, [str(path)]).fetchall():
                (unigram if kind == "unigram" else bigram)[str(item)] += int(frequency)
            completed_tokens.add(token)
            if len(completed_tokens) % 10 == 0 or len(completed_tokens) == len(train_rows):
                atomic_write_json(
                    partial_path,
                    {
                        "completed_tokens": sorted(completed_tokens),
                        "unigram": dict(unigram),
                        "bigram": dict(bigram),
                    },
                )
                atomic_write_json(
                    progress_path,
                    {
                        "completed_scenarios": len(completed_tokens),
                        "total_scenarios": len(train_rows),
                        "last_scenario_token": token,
                    },
                )
    finally:
        connection.close()

    unigram_limit = int(config.features.host.unigram_vocab_max)
    bigram_limit = int(config.features.host.bigram_vocab_max)
    unigram_items = sorted(unigram.items(), key=lambda item: (-item[1], item[0]))[:unigram_limit]
    bigram_items = sorted(bigram.items(), key=lambda item: (-item[1], item[0]))[:bigram_limit]
    split_hash = sha256_file(splits_dir / "trigger_split.parquet")
    common = {
        "schema_version": 1,
        "source_split": "train",
        "trigger_split_sha256": split_hash,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    unigram_document = {
        **common,
        "oov_index": len(unigram_items),
        "items": [
            {"template": template, "index": index, "count": count}
            for index, (template, count) in enumerate(unigram_items)
        ],
    }
    bigram_document = {
        **common,
        "oov_index": len(bigram_items),
        "items": [
            {"transition": transition, "index": index, "count": count}
            for index, (transition, count) in enumerate(bigram_items)
        ],
    }
    atomic_write_json(vocab_dir / "host_unigram.json", unigram_document)
    atomic_write_json(vocab_dir / "host_bigram.json", bigram_document)
    return {
        "train_scenarios": len(train_rows),
        "unigram_size": len(unigram_items),
        "bigram_size": len(bigram_items),
        "trigger_split_sha256": split_hash,
    }


def _load_vocab(features_dir: Path) -> tuple[dict[str, int], dict[str, int]]:
    unigram_doc = json.loads((features_dir / "vocab" / "host_unigram.json").read_text())
    bigram_doc = json.loads((features_dir / "vocab" / "host_bigram.json").read_text())
    unigram = {str(item["template"]): int(item["index"]) for item in unigram_doc["items"]}
    bigram = {str(item["transition"]): int(item["index"]) for item in bigram_doc["items"]}
    return unigram, bigram


def host_feature_names(unigram: dict[str, int], bigram: dict[str, int]) -> list[str]:
    return (
        [f"unigram_{index:04d}" for index in range(len(unigram))]
        + ["unigram_oov"]
        + [f"bigram_{index:04d}" for index in range(len(bigram))]
        + ["bigram_oov"]
        + HOST_STAT_FEATURES
    )


def _feature_schema(feature_names: list[str]) -> pa.Schema:
    return pa.schema(
        [
            ("scenario_token", pa.string()),
            ("second_ts", pa.int64()),
            *[(name, pa.float32()) for name in feature_names],
            ("label_private", pa.string()),
            ("loss_mask", pa.int8()),
            ("max_observed_ts_ns", pa.int64()),
        ]
    )


def _time_and_labels(
    public_path: Path, anchors_path: Path, observed_seconds: set[int]
) -> tuple[list[int], dict[int, tuple[str, int]]]:
    public = pq.read_table(public_path).to_pylist()[0]
    start = int(public["start_ts_ns"]) // 1_000_000_000
    end = int(public["end_ts_ns"]) // 1_000_000_000
    if observed_seconds:
        start = min(start, min(observed_seconds))
        end = max(end, max(observed_seconds))
    seconds = list(range(start, end + 1))
    anchors = pq.read_table(anchors_path, columns=["ts_ns"]).column("ts_ns").to_pylist()
    anchor_seconds = {int(value) // 1_000_000_000 for value in anchors}
    first_anchor = min(anchor_seconds) if anchor_seconds else None
    labels = {}
    for second in seconds:
        if second in anchor_seconds:
            labels[second] = ("positive_core", 1)
        elif first_anchor is not None and second < first_anchor - 5:
            labels[second] = ("known_negative", 1)
        else:
            labels[second] = ("unknown", 0)
    return seconds, labels


NETWORK_QUERY = """
with base as (
  select *, ts_ns - lag(ts_ns) over (order by frame_no) as iat_ns
  from read_parquet(?)
), grouped as (
  select
    floor(ts_ns / 1000000000)::bigint as second,
    count(*)::double as packet_count,
    sum(wire_len)::double as byte_count,
    avg(wire_len)::double as packet_length_mean,
    coalesce(stddev_pop(wire_len), 0)::double as packet_length_std,
    quantile_cont(wire_len, 0.25)::double as packet_length_p25,
    quantile_cont(wire_len, 0.50)::double as packet_length_p50,
    quantile_cont(wire_len, 0.75)::double as packet_length_p75,
    max(wire_len)::double as packet_length_max,
    coalesce(avg(iat_ns) / 1000000, 0)::double as interarrival_mean_ms,
    coalesce(stddev_pop(iat_ns) / 1000000, 0)::double as interarrival_std_ms,
    coalesce(quantile_cont(iat_ns, 0.50) / 1000000, 0)::double as interarrival_p50_ms,
    coalesce(quantile_cont(iat_ns, 0.95) / 1000000, 0)::double as interarrival_p95_ms,
    count(distinct concat_ws('|', src_ip, src_port, dst_ip, dst_port, ip_protocol))::double as connection_count,
    count(*) filter (where ip_protocol=6 and (tcp_flags & 2) != 0 and (tcp_flags & 16) = 0)::double as new_connection_count,
    count(*) filter (where ip_protocol=6 and (tcp_flags & 5) != 0)::double as closed_connection_count,
    (count(distinct src_ip) + count(distinct dst_ip))::double as unique_ip_count,
    (count(distinct src_port) + count(distinct dst_port))::double as unique_port_count,
    count(*) filter (where ip_protocol=6 and (tcp_flags & 2) != 0)::double as tcp_syn_count,
    count(*) filter (where ip_protocol=6 and (tcp_flags & 4) != 0)::double as tcp_rst_count,
    count(*) filter (where ip_protocol=6 and (tcp_flags & 1) != 0)::double as tcp_fin_count,
    count(*) filter (where ip_protocol=6 and (tcp_flags & 16) != 0)::double as tcp_ack_count,
    greatest(0, count(*) filter (where ip_protocol=6) - count(distinct case when ip_protocol=6 then concat_ws('|', src_ip, src_port, dst_ip, dst_port, seq) end))::double as retransmission_estimate,
    count(*) filter (where ip_protocol=6)::double as tcp_count,
    count(*) filter (where ip_protocol=17)::double as udp_count,
    count(*) filter (where ip_protocol in (1,58))::double as icmp_count,
    count(*) filter (where ip_protocol is null or ip_protocol not in (1,6,17,58))::double as other_protocol_count,
    max(ts_ns)::bigint as max_observed_ts_ns
  from base group by second
)
select * from grouped order by second
"""


def _network_rows(
    token: str, public_path: Path, anchors_path: Path, packet_path: Path
) -> list[dict[str, object]]:
    connection = _connection()
    try:
        query_rows = connection.execute(NETWORK_QUERY, [str(packet_path)]).fetchall()
        names = [item[0] for item in connection.description]
    finally:
        connection.close()
    observed = {int(row[0]) for row in query_rows}
    seconds, labels = _time_and_labels(public_path, anchors_path, observed)
    by_second = {int(row[0]): dict(zip(names, row)) for row in query_rows}
    output = []
    last_observed: int | None = None
    for second in seconds:
        source = by_second.get(second)
        if source is not None:
            last_observed = second
        row = {"scenario_token": token, "second_ts": second}
        for name in NETWORK_FEATURES:
            if name == "missing":
                value = 0.0 if source is not None else 1.0
            elif name == "age_seconds":
                value = 0.0 if source is not None else float(second - last_observed) if last_observed is not None else float(second - seconds[0] + 1)
            else:
                value = float(source.get(name, 0.0)) if source is not None else 0.0
            row[name] = value
        row["label_private"], row["loss_mask"] = labels[second]
        row["max_observed_ts_ns"] = int(source["max_observed_ts_ns"]) if source else None
        output.append(row)
    return output


HOST_STATS_QUERY = """
select
  floor(ts_ns / 1000000000)::bigint as second,
  count(*)::double as event_count,
  count(distinct pid)::double as unique_pid_count,
  count(distinct tid)::double as unique_tid_count,
  count(distinct file_path)::double as unique_file_count,
  count(distinct concat_ws('|', socket_src_ip, socket_src_port, socket_dst_ip, socket_dst_port)) filter (where socket_src_ip is not null)::double as unique_socket_count,
  count(*) filter (where syscall in ('clone','clone3','fork','vfork'))::double as process_create_count,
  count(*) filter (where syscall in ('execve','execveat'))::double as exec_count,
  count(*) filter (where syscall='connect')::double as connect_count,
  count(*) filter (where syscall in ('accept','accept4'))::double as accept_count,
  count(*) filter (where syscall in ('write','writev','pwrite','pwrite64') and file_path is not null)::double as file_write_count,
  count(*) filter (where syscall in ('chmod','fchmod','fchmodat'))::double as permission_change_count,
  count(*) filter (where syscall in ('unlink','unlinkat','rename','renameat','renameat2'))::double as delete_count,
  count(*) filter (where result_class='error')::double as failed_operation_count,
  max(ts_ns)::bigint as max_observed_ts_ns
from read_parquet(?) group by second order by second
"""


def _host_rows(
    token: str,
    public_path: Path,
    anchors_path: Path,
    syscall_path: Path,
    unigram: dict[str, int],
    bigram: dict[str, int],
) -> list[dict[str, object]]:
    connection = _connection()
    try:
        unigrams = connection.execute(
            f"""select floor(ts_ns/1000000000)::bigint as second,
                       {TEMPLATE_SQL} as template, count(*)::double
                from read_parquet(?) group by second, template""",
            [str(syscall_path)],
        ).fetchall()
        transitions = connection.execute(
            f"""with base as (
                    select line_no, floor(ts_ns/1000000000)::bigint as second,
                           {TEMPLATE_SQL} as template
                    from read_parquet(?)
                ), seq as (
                    select second, template, lag(template) over (order by line_no) as previous
                    from base
                )
                select second, previous || '->' || template as transition, count(*)::double
                from seq where previous is not null group by second, transition""",
            [str(syscall_path)],
        ).fetchall()
        stats_rows = connection.execute(HOST_STATS_QUERY, [str(syscall_path)]).fetchall()
        stats_names = [item[0] for item in connection.description]
    finally:
        connection.close()
    observed = {int(row[0]) for row in stats_rows}
    seconds, labels = _time_and_labels(public_path, anchors_path, observed)
    features = host_feature_names(unigram, bigram)
    by_second: dict[int, dict[str, float]] = {
        second: {name: 0.0 for name in features} for second in seconds
    }
    unigram_oov = len(unigram)
    bigram_oov = len(bigram)
    for second, template, count in unigrams:
        index = unigram.get(str(template), unigram_oov)
        name = "unigram_oov" if index == unigram_oov else f"unigram_{index:04d}"
        if int(second) in by_second:
            by_second[int(second)][name] += float(count)
    for second, transition, count in transitions:
        index = bigram.get(str(transition), bigram_oov)
        name = "bigram_oov" if index == bigram_oov else f"bigram_{index:04d}"
        if int(second) in by_second:
            by_second[int(second)][name] += float(count)
    stats = {int(row[0]): dict(zip(stats_names, row)) for row in stats_rows}
    output = []
    last_observed: int | None = None
    for second in seconds:
        stat = stats.get(second)
        if stat is not None:
            last_observed = second
        row = {"scenario_token": token, "second_ts": second, **by_second[second]}
        for name in HOST_STAT_FEATURES:
            if name == "missing":
                value = 0.0 if stat is not None else 1.0
            elif name == "age_seconds":
                value = 0.0 if stat is not None else float(second - last_observed) if last_observed is not None else float(second - seconds[0] + 1)
            else:
                value = float(stat.get(name, 0.0)) if stat is not None else 0.0
            row[name] = value
        row["label_private"], row["loss_mask"] = labels[second]
        row["max_observed_ts_ns"] = int(stat["max_observed_ts_ns"]) if stat else None
        output.append(row)
    return output


def _fit_scaler(paths: list[Path], feature_names: list[str]) -> dict[str, object]:
    matrices = []
    for path in paths:
        table = pq.read_table(path, columns=feature_names)
        matrix = np.column_stack(
            [np.asarray(table[name].to_numpy(zero_copy_only=False), dtype=np.float32) for name in feature_names]
        )
        matrices.append(matrix)
    combined = np.concatenate(matrices, axis=0)
    median = np.median(combined, axis=0)
    q25 = np.quantile(combined, 0.25, axis=0)
    q75 = np.quantile(combined, 0.75, axis=0)
    iqr = q75 - q25
    iqr[iqr == 0] = 1.0
    return {
        "schema_version": 1,
        "feature_names": feature_names,
        "median": median.astype(float).tolist(),
        "iqr": iqr.astype(float).tolist(),
        "fit_rows": int(combined.shape[0]),
        "source_split": "train",
    }


def _normalize_shard(raw_path: Path, output_path: Path, scaler: dict[str, object]) -> None:
    table = pq.read_table(raw_path)
    features = list(scaler["feature_names"])
    median = np.asarray(scaler["median"], dtype=np.float32)
    iqr = np.asarray(scaler["iqr"], dtype=np.float32)
    arrays: dict[str, pa.Array] = {
        "scenario_token": table["scenario_token"],
        "second_ts": table["second_ts"],
    }
    for index, name in enumerate(features):
        values = np.asarray(table[name].to_numpy(zero_copy_only=False), dtype=np.float32)
        arrays[name] = pa.array((values - median[index]) / iqr[index], type=pa.float32())
    arrays["label_private"] = table["label_private"]
    arrays["loss_mask"] = table["loss_mask"]
    arrays["max_observed_ts_ns"] = table["max_observed_ts_ns"]
    atomic_write_table(output_path, pa.table(arrays))


def build_features_for_split(
    config: TraceAnchorConfig,
    config_path: Path,
    split: str,
    resume: bool = False,
) -> dict[str, object]:
    if split not in {"train", "validation", "test"}:
        raise ValueError(f"invalid split: {split}")
    resolved = config.resolved_dict(config_path)
    features_dir = Path(resolved["paths"]["features_dir"])
    parquet_dir = Path(resolved["paths"]["parquet_dir"])
    evaluator_dir = Path(resolved["paths"]["evaluator_dir"])
    markers_dir = Path(resolved["paths"]["completion_markers_dir"])
    split_rows = _load_split_rows(config, config_path, split)
    if not (features_dir / "vocab" / "host_unigram.json").exists():
        if split != "train":
            raise FileNotFoundError("host vocabulary missing; build train features first")
        build_host_vocabulary(config, config_path)
    unigram, bigram = _load_vocab(features_dir)
    host_names = host_feature_names(unigram, bigram)
    raw_network_paths = []
    raw_host_paths = []
    progress_path = Path(resolved["paths"]["logs_dir"]) / f"feature_{split}_progress.json"
    for index, item in enumerate(split_rows, start=1):
        token = str(item["scenario_token"])
        network_raw = features_dir / "raw" / "network" / split / f"{token}.parquet"
        host_raw = features_dir / "raw" / "host" / split / f"{token}.parquet"
        marker_path = markers_dir / "features" / split / f"{token}.done"
        raw_network_paths.append(network_raw)
        raw_host_paths.append(host_raw)
        if resume and marker_path.exists() and network_raw.exists() and host_raw.exists():
            continue
        public_path = parquet_dir / token / "scenario_public.parquet"
        anchors_path = evaluator_dir / token / "exploit_anchors.parquet"
        network_rows = _network_rows(
            token,
            public_path,
            anchors_path,
            parquet_dir / token / "network_packet.parquet",
        )
        host_rows = _host_rows(
            token,
            public_path,
            anchors_path,
            parquet_dir / token / "syscall_event.parquet",
            unigram,
            bigram,
        )
        atomic_write_table(
            network_raw, pa.Table.from_pylist(network_rows, schema=_feature_schema(NETWORK_FEATURES))
        )
        atomic_write_table(
            host_raw, pa.Table.from_pylist(host_rows, schema=_feature_schema(host_names))
        )
        atomic_write_json(
            marker_path,
            {
                "scenario_token": token,
                "split": split,
                "network_rows": len(network_rows),
                "host_rows": len(host_rows),
                "network_raw_sha256": sha256_file(network_raw),
                "host_raw_sha256": sha256_file(host_raw),
            },
        )
        atomic_write_json(
            progress_path,
            {"completed_scenarios": index, "total_scenarios": len(split_rows), "last_scenario_token": token},
        )

    scaler_dir = features_dir / "scalers"
    network_scaler_path = scaler_dir / "network_scaler.json"
    host_scaler_path = scaler_dir / "host_scaler.json"
    if split == "train":
        network_scaler = _fit_scaler(raw_network_paths, NETWORK_FEATURES)
        host_scaler = _fit_scaler(raw_host_paths, host_names)
        split_hash = sha256_file(Path(resolved["paths"]["splits_dir"]) / "trigger_split.parquet")
        network_scaler["trigger_split_sha256"] = split_hash
        host_scaler["trigger_split_sha256"] = split_hash
        atomic_write_json(network_scaler_path, network_scaler)
        atomic_write_json(host_scaler_path, host_scaler)
    else:
        network_scaler = json.loads(network_scaler_path.read_text())
        host_scaler = json.loads(host_scaler_path.read_text())

    for raw_path in raw_network_paths:
        _normalize_shard(
            raw_path,
            features_dir / "network" / split / raw_path.name,
            network_scaler,
        )
    for raw_path in raw_host_paths:
        _normalize_shard(
            raw_path,
            features_dir / "host" / split / raw_path.name,
            host_scaler,
        )
    summary = {
        "schema_version": 1,
        "split": split,
        "scenarios": len(split_rows),
        "network_feature_count": len(NETWORK_FEATURES),
        "host_feature_count": len(host_names),
        "network_rows": sum(pq.read_metadata(path).num_rows for path in raw_network_paths),
        "host_rows": sum(pq.read_metadata(path).num_rows for path in raw_host_paths),
    }
    atomic_write_json(features_dir / "index" / f"{split}.json", summary)
    atomic_write_json(markers_dir / f"WP2_features_{split}.done", summary)
    return summary


def validate_datasets(config: TraceAnchorConfig, config_path: Path) -> dict[str, object]:
    resolved = config.resolved_dict(config_path)
    features_dir = Path(resolved["paths"]["features_dir"])
    splits_dir = Path(resolved["paths"]["splits_dir"])
    markers_dir = Path(resolved["paths"]["completion_markers_dir"])
    errors = []
    trigger = pq.read_table(splits_dir / "trigger_split.parquet").to_pylist()
    forbidden = {value.lower() for value in FORBIDDEN_AGENT_COLUMNS}
    totals = {}
    for split in ("train", "validation", "test"):
        expected = {str(row["scenario_token"]) for row in trigger if row["trigger_split"] == split}
        network_paths = list((features_dir / "network" / split).glob("*.parquet"))
        host_paths = list((features_dir / "host" / split).glob("*.parquet"))
        network_tokens = {path.stem for path in network_paths}
        host_tokens = {path.stem for path in host_paths}
        if network_tokens != expected:
            errors.append(f"network {split} shard tokens differ from split manifest")
        if host_tokens != expected:
            errors.append(f"host {split} shard tokens differ from split manifest")
        for path in network_paths + host_paths:
            schema_names = pq.read_schema(path).names
            leaked = forbidden.intersection(name.lower() for name in schema_names)
            if leaked:
                errors.append(f"forbidden feature columns in {path}: {sorted(leaked)}")
            table = pq.read_table(path, columns=["second_ts", "max_observed_ts_ns"])
            for second, maximum in zip(
                table["second_ts"].to_pylist(), table["max_observed_ts_ns"].to_pylist()
            ):
                if maximum is not None and maximum >= (int(second) + 1) * 1_000_000_000:
                    errors.append(f"future leakage in {path.name} at second {second}")
                    break
        totals[split] = {"scenarios": len(expected), "network_shards": len(network_paths), "host_shards": len(host_paths)}
    split_hash = sha256_file(splits_dir / "trigger_split.parquet")
    for filename in ("network_scaler.json", "host_scaler.json"):
        document = json.loads((features_dir / "scalers" / filename).read_text())
        if document["source_split"] != "train" or document["trigger_split_sha256"] != split_hash:
            errors.append(f"invalid train-only provenance in {filename}")
    for filename in ("host_unigram.json", "host_bigram.json"):
        document = json.loads((features_dir / "vocab" / filename).read_text())
        if document["source_split"] != "train" or document["trigger_split_sha256"] != split_hash:
            errors.append(f"invalid train-only provenance in {filename}")
    result = {"ok": not errors, "errors": errors, "splits": totals}
    report_path = Path(resolved["paths"]["artifacts_dir"]) / "reports" / "dataset_qa.json"
    atomic_write_json(report_path, result)
    if not errors:
        atomic_write_json(
            markers_dir / "WP2.done",
            {
                "schema_version": 1,
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "config_sha256": config_hash(config, config_path),
                "dataset_qa_sha256": sha256_file(report_path),
                "trigger_split_sha256": split_hash,
            },
        )
    return result
