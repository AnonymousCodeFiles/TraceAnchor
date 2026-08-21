from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ToolModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ScenarioRequest(ToolModel):
    scenario_token: str = Field(pattern=r"^tw_[0-9a-f]{24}$")


class TimeRangeRequest(ScenarioRequest):
    start_ts_ns: int = Field(ge=0)
    end_ts_ns: int = Field(ge=0)
    max_records: int = Field(default=200, gt=0, le=200)
    cursor: str | None = Field(default=None, max_length=32)

    @model_validator(mode="after")
    def ordered_range(self) -> "TimeRangeRequest":
        if self.end_ts_ns < self.start_ts_ns:
            raise ValueError("end_ts_ns precedes start_ts_ns")
        return self


class AlertContextRequest(ScenarioRequest):
    alert_id: str = Field(pattern=r"^al_[0-9a-f]{24}$")


class ConnectionRequest(TimeRangeRequest):
    ip: str | None = Field(default=None, max_length=64)
    port: int | None = Field(default=None, ge=0, le=65535)
    protocol: int | None = Field(default=None, ge=0, le=255)


class PacketMetadataRequest(ConnectionRequest):
    pass


class SocketOwnerRequest(ScenarioRequest):
    ts_ns: int = Field(ge=0)
    src_ip: str = Field(min_length=1, max_length=64)
    src_port: int = Field(ge=0, le=65535)
    dst_ip: str = Field(min_length=1, max_length=64)
    dst_port: int = Field(ge=0, le=65535)
    tolerance_ms: int = Field(default=1000, ge=0, le=1000)
    max_records: int = Field(default=200, gt=0, le=200)


class ProcessTreeRequest(ScenarioRequest):
    pid: int = Field(ge=0)
    depth: int = Field(default=2, ge=1, le=4)


class SyscallRequest(TimeRangeRequest):
    pid: int | None = Field(default=None, ge=0)
    tid: int | None = Field(default=None, ge=0)
    syscall: str | None = Field(default=None, max_length=64)


class FileActivityRequest(TimeRangeRequest):
    pid: int | None = Field(default=None, ge=0)
    path_contains: str | None = Field(default=None, max_length=256)


class JoinEvidenceRequest(TimeRangeRequest):
    src_ip: str | None = Field(default=None, max_length=64)
    src_port: int | None = Field(default=None, ge=0, le=65535)
    dst_ip: str | None = Field(default=None, max_length=64)
    dst_port: int | None = Field(default=None, ge=0, le=65535)


class TimelineRequest(ScenarioRequest):
    evidence_ids: list[str] = Field(min_length=1, max_length=200)


class AttackKnowledgeRequest(ScenarioRequest):
    observed_behavior: str = Field(min_length=3, max_length=500)
    max_records: int = Field(default=10, gt=0, le=50)


class ValidateEvidenceRequest(ScenarioRequest):
    evidence_ids: list[str] = Field(min_length=1, max_length=200)
    required_entity_ids: list[str] = Field(default_factory=list, max_length=200)


class ToolEnvelope(ToolModel):
    tool: str
    scenario_token: str
    records: list[dict[str, Any]]
    result_count: int = Field(ge=0)
    truncated: bool
    next_cursor: str | None
    ledger_version: str


__all__ = [
    "AlertContextRequest",
    "AttackKnowledgeRequest",
    "ConnectionRequest",
    "FileActivityRequest",
    "JoinEvidenceRequest",
    "PacketMetadataRequest",
    "ProcessTreeRequest",
    "SocketOwnerRequest",
    "SyscallRequest",
    "TimelineRequest",
    "ToolEnvelope",
    "ValidateEvidenceRequest",
]
