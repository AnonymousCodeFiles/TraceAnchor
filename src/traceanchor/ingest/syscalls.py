from __future__ import annotations

import base64
import binascii
import json
import re
from pathlib import Path
from typing import Iterator

from traceanchor.ingest.common import sha256_bytes


ARG_KEY = re.compile(r"(?:^|\s)([A-Za-z_][A-Za-z0-9_.]*)=")
INTEGER_PREFIX = re.compile(r"^-?\d+")
SOCKET = re.compile(r"<(?P<version>[46])(?P<transport>[tu])>(?P<src>.+?)->(?P<dst>[^)\s]+)")
PAYLOAD_KEYS = {"data", "buffer", "payload"}
FILE_KEYS = ("path", "name", "oldpath", "newpath")
FORK_SYSCALLS = {"clone", "clone3", "fork", "vfork"}


def parse_arguments(text: str) -> dict[str, str]:
    matches = list(ARG_KEY.finditer(text))
    values: dict[str, str] = {}
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        values[match.group(1)] = text[start:end].strip()
    return values


def _leading_int(value: str | None) -> int | None:
    if value is None:
        return None
    match = INTEGER_PREFIX.match(value)
    return int(match.group(0)) if match else None


def _endpoint(value: str) -> tuple[str | None, int | None]:
    value = value.strip()
    if value.startswith("[") and "]:" in value:
        host, _, port = value[1:].rpartition("]:")
    else:
        host, separator, port = value.rpartition(":")
        if not separator:
            return value or None, None
    try:
        return host, int(port)
    except ValueError:
        return host, None


def _payload_metadata(arguments: dict[str, str]) -> tuple[bool, str | None, dict[str, object]]:
    safe: dict[str, object] = {}
    payload_present = False
    payload_hash: str | None = None
    for key, value in arguments.items():
        if key in PAYLOAD_KEYS:
            if value:
                payload_present = True
                try:
                    decoded = base64.b64decode(value, validate=True)
                    decodable = True
                except (binascii.Error, ValueError):
                    decoded = value.encode("utf-8", errors="replace")
                    decodable = False
                digest = sha256_bytes(decoded)
                payload_hash = payload_hash or digest
                safe[f"{key}_metadata"] = {
                    "encoded_length": len(value),
                    "decoded_length": len(decoded),
                    "sha256": digest,
                    "base64_decodable": decodable,
                }
            else:
                safe[f"{key}_metadata"] = {
                    "encoded_length": 0,
                    "decoded_length": 0,
                    "sha256": None,
                    "base64_decodable": True,
                }
            continue
        if len(value) > 512:
            safe[key] = {
                "length": len(value),
                "sha256": sha256_bytes(value.encode("utf-8", errors="replace")),
                "truncated": True,
            }
        else:
            safe[key] = value
    return payload_present, payload_hash, safe


def _bounded_json(values: dict[str, object], limit: int) -> str:
    encoded = json.dumps(values, sort_keys=True, separators=(",", ":"))
    if len(encoded) <= limit:
        return encoded
    compact: dict[str, object] = {}
    for key, value in values.items():
        candidate = dict(compact)
        candidate[key] = value
        rendered = json.dumps(candidate, sort_keys=True, separators=(",", ":"))
        if len(rendered) > max(0, limit - 80):
            break
        compact = candidate
    compact["_truncated"] = True
    compact["_original_json_length"] = len(encoded)
    return json.dumps(compact, sort_keys=True, separators=(",", ":"))


def parse_syscall_line(
    raw_line: bytes,
    token: str,
    line_no: int,
    max_args_chars: int = 4096,
    bucket_seconds: int = 60,
) -> dict[str, object]:
    raw_hash = sha256_bytes(raw_line)
    line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
    parts = line.split(maxsplit=7)
    if len(parts) < 7:
        return {
            "evidence_id": f"sc:{token}:invalid:{line_no}",
            "scenario_token": token,
            "line_no": line_no,
            "ts_ns": 0,
            "uid": None,
            "pid": None,
            "tid": None,
            "process_name": None,
            "syscall": None,
            "direction": None,
            "result_class": "unknown",
            "fd": None,
            "file_path": None,
            "socket_src_ip": None,
            "socket_src_port": None,
            "socket_dst_ip": None,
            "socket_dst_port": None,
            "protocol_hint": None,
            "child_pid": None,
            "parent_pid": None,
            "args_json": "{}",
            "payload_present": False,
            "payload_hash": None,
            "raw_line_hash": raw_hash,
            "time_bucket": 0,
            "parse_status": "malformed_prefix",
        }
    try:
        ts_ns, uid, pid, process_name, tid, syscall, direction = parts[:7]
        ts_value = int(ts_ns)
        uid_value = int(uid)
        pid_value = int(pid)
        tid_value = int(tid)
    except ValueError:
        malformed = parse_syscall_line(b"", token, line_no, max_args_chars, bucket_seconds)
        malformed["raw_line_hash"] = raw_hash
        malformed["parse_status"] = "invalid_numeric_prefix"
        return malformed
    arguments = parse_arguments(parts[7] if len(parts) == 8 else "")
    result = _leading_int(arguments.get("res"))
    if direction == ">":
        result_class = "enter"
    elif result is None:
        result_class = "unknown"
    elif result < 0:
        result_class = "error"
    else:
        result_class = "success"

    fd = _leading_int(arguments.get("fd"))
    socket_text = arguments.get("tuple") or arguments.get("fd") or ""
    socket_match = SOCKET.search(socket_text)
    src_ip = dst_ip = protocol = None
    src_port = dst_port = None
    if socket_match:
        src_ip, src_port = _endpoint(socket_match.group("src"))
        dst_ip, dst_port = _endpoint(socket_match.group("dst"))
        protocol = "tcp" if socket_match.group("transport") == "t" else "udp"

    file_path = next((arguments[key] for key in FILE_KEYS if arguments.get(key)), None)
    parent_pid = child_pid = None
    if syscall in FORK_SYSCALLS and direction == "<":
        ptid = _leading_int(arguments.get("ptid"))
        if ptid is not None and ptid != pid_value:
            parent_pid, child_pid = ptid, pid_value
        elif result is not None and result > 0:
            parent_pid, child_pid = pid_value, result

    payload_present, payload_hash, safe_arguments = _payload_metadata(arguments)
    return {
        "evidence_id": f"sc:{token}:{ts_value}:{line_no}",
        "scenario_token": token,
        "line_no": line_no,
        "ts_ns": ts_value,
        "uid": uid_value,
        "pid": pid_value,
        "tid": tid_value,
        "process_name": process_name,
        "syscall": syscall,
        "direction": direction,
        "result_class": result_class,
        "fd": fd,
        "file_path": file_path,
        "socket_src_ip": src_ip,
        "socket_src_port": src_port,
        "socket_dst_ip": dst_ip,
        "socket_dst_port": dst_port,
        "protocol_hint": protocol,
        "child_pid": child_pid,
        "parent_pid": parent_pid,
        "args_json": _bounded_json(safe_arguments, max_args_chars),
        "payload_present": payload_present,
        "payload_hash": payload_hash,
        "raw_line_hash": raw_hash,
        "time_bucket": ts_value // (bucket_seconds * 1_000_000_000),
        "parse_status": "ok",
    }


def iter_syscalls(
    path: Path,
    token: str,
    max_args_chars: int = 4096,
    bucket_seconds: int = 60,
) -> Iterator[dict[str, object]]:
    with path.open("rb") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            yield parse_syscall_line(
                raw_line, token, line_no, max_args_chars, bucket_seconds
            )

