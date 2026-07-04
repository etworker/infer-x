"""Instance-related models."""

from __future__ import annotations

from pydantic import BaseModel, Field

from .enums import BackendType, InstanceStatus
from .params import BackendParams


class InstanceStartRequest(BackendParams):
    model: str
    backend: BackendType | None = None
    port: int | None = None
    host: str | None = None
    preset: str | None = None
    extra_args: list[str] = Field(default_factory=list)


class InstanceInfo(BaseModel):
    id: str
    model: str
    backend: BackendType = BackendType.llamacpp
    status: InstanceStatus
    port: int
    host: str
    pid: int | None = None
    ctx_size: int = 4096
    n_gpu_layers: str = "auto"
    n_parallel: int = 1
    started_at: str | None = None
    uptime_seconds: float | None = None
    gpu_memory_mb: int | None = None
    gpu_utilization_pct: float | None = None
    ram_usage_mb: float | None = None
    extra_args: list[str] = Field(default_factory=list)
    restart_count: int = 0
    tags: dict[str, str] = Field(default_factory=dict)


class InstanceList(BaseModel):
    instances: list[InstanceInfo]
    total: int
