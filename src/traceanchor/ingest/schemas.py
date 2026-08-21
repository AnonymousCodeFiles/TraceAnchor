from __future__ import annotations

import pyarrow as pa


SCHEMA_METADATA = {b"traceanchor_schema_version": b"1"}


def _schema(fields: list[tuple[str, pa.DataType]]) -> pa.Schema:
    return pa.schema(fields, metadata=SCHEMA_METADATA)


RAW_FILE_SCHEMA = _schema(
    [
        ("scenario_uid", pa.string()),
        ("scenario_token", pa.string()),
        ("relative_path", pa.string()),
        ("file_type", pa.string()),
        ("size_bytes", pa.int64()),
        ("mtime_ns", pa.int64()),
        ("sha256", pa.string()),
    ]
)

SCENARIO_SCHEMA = _schema(
    [
        ("scenario_uid", pa.string()),
        ("scenario_token", pa.string()),
        ("has_json", pa.bool_()),
        ("has_pcap", pa.bool_()),
        ("has_res", pa.bool_()),
        ("has_sc", pa.bool_()),
        ("total_size_bytes", pa.int64()),
        ("quality_status", pa.string()),
    ]
)

SCENARIO_PUBLIC_SCHEMA = _schema(
    [
        ("scenario_token", pa.string()),
        ("start_ts_ns", pa.int64()),
        ("end_ts_ns", pa.int64()),
        ("recording_time_seconds", pa.float64()),
        ("network_available", pa.bool_()),
        ("host_available", pa.bool_()),
        ("resource_available", pa.bool_()),
        ("quality_status", pa.string()),
    ]
)

SCENARIO_PRIVATE_SCHEMA = _schema(
    [
        ("scenario_uid", pa.string()),
        ("scenario_token", pa.string()),
        ("family_private", pa.string()),
        ("exploit", pa.bool_()),
        ("exploit_name", pa.string()),
        ("image", pa.string()),
        ("container_ready_ts_ns", pa.int64()),
        ("warmup_end_ts_ns", pa.int64()),
        ("recording_time_seconds", pa.float64()),
        ("containers_json", pa.string()),
    ]
)

EXPLOIT_ANCHOR_SCHEMA = _schema(
    [
        ("anchor_id", pa.string()),
        ("scenario_uid", pa.string()),
        ("scenario_token", pa.string()),
        ("ts_ns", pa.int64()),
        ("raw_timestamp", pa.string()),
        ("name_private", pa.string()),
        ("source_private", pa.string()),
    ]
)

SYSCALL_SCHEMA = _schema(
    [
        ("evidence_id", pa.string()),
        ("scenario_token", pa.string()),
        ("line_no", pa.int64()),
        ("ts_ns", pa.int64()),
        ("uid", pa.int64()),
        ("pid", pa.int64()),
        ("tid", pa.int64()),
        ("process_name", pa.string()),
        ("syscall", pa.string()),
        ("direction", pa.string()),
        ("result_class", pa.string()),
        ("fd", pa.int64()),
        ("file_path", pa.string()),
        ("socket_src_ip", pa.string()),
        ("socket_src_port", pa.int32()),
        ("socket_dst_ip", pa.string()),
        ("socket_dst_port", pa.int32()),
        ("protocol_hint", pa.string()),
        ("child_pid", pa.int64()),
        ("parent_pid", pa.int64()),
        ("args_json", pa.string()),
        ("payload_present", pa.bool_()),
        ("payload_hash", pa.string()),
        ("raw_line_hash", pa.string()),
        ("time_bucket", pa.int64()),
        ("parse_status", pa.string()),
    ]
)

PACKET_SCHEMA = _schema(
    [
        ("evidence_id", pa.string()),
        ("scenario_token", pa.string()),
        ("frame_no", pa.int64()),
        ("ts_ns", pa.int64()),
        ("captured_len", pa.int32()),
        ("wire_len", pa.int32()),
        ("eth_type", pa.int32()),
        ("src_ip", pa.string()),
        ("dst_ip", pa.string()),
        ("src_port", pa.int32()),
        ("dst_port", pa.int32()),
        ("ip_protocol", pa.int32()),
        ("tcp_flags", pa.int32()),
        ("seq", pa.int64()),
        ("ack", pa.int64()),
        ("payload_len", pa.int32()),
        ("payload_hash", pa.string()),
        ("time_bucket", pa.int64()),
        ("parse_status", pa.string()),
    ]
)

RESOURCE_SCHEMA = _schema(
    [
        ("evidence_id", pa.string()),
        ("scenario_token", pa.string()),
        ("row_no", pa.int64()),
        ("ts_ns", pa.int64()),
        ("raw_timestamp", pa.string()),
        ("cpu_usage", pa.float64()),
        ("memory_usage", pa.float64()),
        ("network_received", pa.float64()),
        ("network_send", pa.float64()),
        ("storage_read", pa.float64()),
        ("storage_written", pa.float64()),
        ("missing_mask", pa.int32()),
        ("time_bucket", pa.int64()),
        ("parse_status", pa.string()),
    ]
)

