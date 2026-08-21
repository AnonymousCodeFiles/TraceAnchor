from __future__ import annotations

import hashlib
import ipaddress
from pathlib import Path
from typing import Iterator

import dpkt


def _ip_text(value: bytes) -> str:
    return str(ipaddress.ip_address(value))


def _decode_link(linktype: int, raw: bytes) -> tuple[int | None, object | None, str]:
    try:
        if linktype == dpkt.pcap.DLT_EN10MB:
            frame = dpkt.ethernet.Ethernet(raw)
            return int(frame.type), frame.data, "ok"
        if linktype == dpkt.pcap.DLT_LINUX_SLL:
            frame = dpkt.sll.SLL(raw)
            return int(frame.ethtype), frame.data, "ok"
        if linktype == getattr(dpkt.pcap, "DLT_LINUX_SLL2", -1):
            return None, None, "unsupported_sll2"
        return None, None, "unsupported_linktype"
    except (dpkt.UnpackError, ValueError):
        return None, None, "malformed_frame"


def _packet_fields(payload: object | None) -> dict[str, object]:
    fields: dict[str, object] = {
        "src_ip": None,
        "dst_ip": None,
        "src_port": None,
        "dst_port": None,
        "ip_protocol": None,
        "tcp_flags": None,
        "seq": None,
        "ack": None,
        "payload_len": 0,
        "payload_hash": None,
    }
    while isinstance(payload, dpkt.ethernet.VLANtag8021Q):
        payload = payload.data
    if not isinstance(payload, (dpkt.ip.IP, dpkt.ip6.IP6)):
        return fields
    fields["src_ip"] = _ip_text(payload.src)
    fields["dst_ip"] = _ip_text(payload.dst)
    protocol = int(payload.p if isinstance(payload, dpkt.ip.IP) else payload.nxt)
    fields["ip_protocol"] = protocol
    transport = payload.data
    transport_payload = b""
    if isinstance(transport, dpkt.tcp.TCP):
        fields.update(
            {
                "src_port": int(transport.sport),
                "dst_port": int(transport.dport),
                "tcp_flags": int(transport.flags),
                "seq": int(transport.seq),
                "ack": int(transport.ack),
            }
        )
        transport_payload = bytes(transport.data)
    elif isinstance(transport, dpkt.udp.UDP):
        fields["src_port"] = int(transport.sport)
        fields["dst_port"] = int(transport.dport)
        transport_payload = bytes(transport.data)
    elif isinstance(transport, dpkt.icmp.ICMP):
        transport_payload = bytes(transport.data)
    if transport_payload:
        fields["payload_len"] = len(transport_payload)
        fields["payload_hash"] = hashlib.sha256(transport_payload).hexdigest()
    return fields


def iter_packets(
    path: Path,
    token: str,
    bucket_seconds: int = 60,
) -> Iterator[dict[str, object]]:
    with path.open("rb") as handle:
        reader = dpkt.pcap.Reader(handle)
        linktype = reader.datalink()
        packet_header = reader._Reader__ph
        packet_file = reader._Reader__f
        divisor = int(reader._divisor)
        scale = 1_000_000_000 // divisor
        frame_no = 0
        while True:
            header_bytes = packet_file.read(packet_header.__hdr_len__)
            if not header_bytes:
                break
            frame_no += 1
            try:
                header = packet_header(header_bytes)
                raw = packet_file.read(header.caplen)
                if len(raw) != header.caplen:
                    raise dpkt.NeedData("truncated captured frame")
                ts_ns = int(header.tv_sec) * 1_000_000_000 + int(header.tv_usec) * scale
                eth_type, payload, status = _decode_link(linktype, raw)
                fields = _packet_fields(payload)
                if status == "ok" and fields["src_ip"] is None:
                    status = "non_ip"
                captured_len = int(header.caplen)
                wire_len = int(header.len)
            except (dpkt.UnpackError, ValueError):
                ts_ns = 0
                eth_type, status = None, "malformed_frame"
                fields = _packet_fields(None)
                captured_len = 0
                wire_len = 0
            yield {
                "evidence_id": f"pcap:{token}:{frame_no}",
                "scenario_token": token,
                "frame_no": frame_no,
                "ts_ns": ts_ns,
                "captured_len": captured_len,
                "wire_len": wire_len,
                "eth_type": eth_type,
                **fields,
                "time_bucket": ts_ns // (bucket_seconds * 1_000_000_000),
                "parse_status": status,
            }

