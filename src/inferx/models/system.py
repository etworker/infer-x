"""System-related models: GPU, system info, health."""

from __future__ import annotations

from pydantic import BaseModel


class GPUInfo(BaseModel):
    index: int
    name: str
    total_memory_mb: int
    used_memory_mb: int
    free_memory_mb: int
    utilization_pct: float | None = None


class SystemInfo(BaseModel):
    gpus: list[GPUInfo]
    total_ram_mb: int
    used_ram_mb: int
    available_ram_mb: int
    cpu_count: int
    cpu_percent: float
    server_paths: dict[str, str]


class HealthResponse(BaseModel):
    status: str = "ok"
    instances_running: int = 0
    instances_total: int = 0
