"""Pydantic data models for request/response schemas."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class InstanceStatus(str, Enum):
    starting = "starting"
    running = "running"
    stopping = "stopping"
    stopped = "stopped"
    error = "error"


class DownloadSource(str, Enum):
    huggingface = "hf"
    modelscope = "ms"
    hf_mirror = "hf_mirror"
    url = "url"


class DownloadStatus(str, Enum):
    pending = "pending"
    downloading = "downloading"
    completed = "completed"
    failed = "failed"


# ---------------------------------------------------------------------------
# System
# ---------------------------------------------------------------------------

class GPUInfo(BaseModel):
    index: int
    name: str
    total_memory_mb: int
    used_memory_mb: int
    free_memory_mb: int
    utilization_pct: Optional[float] = None


class SystemInfo(BaseModel):
    gpus: List[GPUInfo]
    total_ram_mb: int
    used_ram_mb: int
    available_ram_mb: int
    cpu_count: int
    cpu_percent: float
    llama_server_path: str


class HealthResponse(BaseModel):
    status: str = "ok"
    instances_running: int = 0
    instances_total: int = 0


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class DefaultConfig(BaseModel):
    model_dir: str = "/home/ec2-user/models"
    llama_server_bin: str = "/home/ec2-user/llama.cpp/build/bin/llama-server"
    port_range_start: int = 8080
    port_range_end: int = 8180
    default_ctx_size: int = 4096
    default_n_gpu_layers: str = "auto"
    default_threads: Optional[int] = None
    default_batch_size: int = 2048
    default_n_parallel: int = 1
    default_flash_attn: str = "auto"
    default_host: str = "0.0.0.0"
    default_sleep_idle_seconds: int = -1
    max_instances: int = 4
    health_check_interval: int = 10
    auto_restart: bool = True
    auto_restart_max_retries: int = 3
    auto_restart_delay: int = 5
    hf_mirror_url: Optional[str] = "https://hf-mirror.com"
    download_max_concurrent: int = 2


class ConfigUpdate(BaseModel):
    model_dir: Optional[str] = None
    llama_server_bin: Optional[str] = None
    port_range_start: Optional[int] = None
    port_range_end: Optional[int] = None
    default_ctx_size: Optional[int] = None
    default_n_gpu_layers: Optional[str] = None
    default_threads: Optional[int] = None
    default_batch_size: Optional[int] = None
    default_n_parallel: Optional[int] = None
    default_flash_attn: Optional[str] = None
    default_host: Optional[str] = None
    default_sleep_idle_seconds: Optional[int] = None
    max_instances: Optional[int] = None
    health_check_interval: Optional[int] = None
    auto_restart: Optional[bool] = None
    auto_restart_max_retries: Optional[int] = None
    auto_restart_delay: Optional[int] = None
    hf_mirror_url: Optional[str] = None
    download_max_concurrent: Optional[int] = None


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class ModelFileInfo(BaseModel):
    name: str
    path: str
    size_mb: float
    quantization: Optional[str] = None
    family: Optional[str] = None


class ModelInfo(BaseModel):
    name: str
    path: str
    size_mb: float
    quantization: Optional[str] = None
    family: Optional[str] = None
    architecture: Optional[str] = None
    parameter_count: Optional[str] = None
    context_length: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

class DownloadRequest(BaseModel):
    source: DownloadSource
    repo: Optional[str] = None
    filename: Optional[str] = None
    url: Optional[str] = None
    quantization: Optional[str] = None
    save_name: Optional[str] = None


class DownloadProgress(BaseModel):
    task_id: str
    source: str
    repo: Optional[str] = None
    filename: Optional[str] = None
    status: DownloadStatus
    progress_pct: float = 0.0
    downloaded_bytes: int = 0
    total_bytes: int = 0
    speed_bytes_per_sec: float = 0.0
    error: Optional[str] = None
    save_path: Optional[str] = None


# ---------------------------------------------------------------------------
# Instance
# ---------------------------------------------------------------------------

class InstanceStartRequest(BaseModel):
    model: str
    port: Optional[int] = None
    ctx_size: Optional[int] = None
    n_gpu_layers: Optional[str] = None
    threads: Optional[int] = None
    batch_size: Optional[int] = None
    n_parallel: Optional[int] = None
    flash_attn: Optional[str] = None
    host: Optional[str] = None
    alias: Optional[str] = None
    sleep_idle_seconds: Optional[int] = None
    preset: Optional[str] = None
    extra_args: List[str] = Field(default_factory=list)


class InstanceInfo(BaseModel):
    id: str
    model: str
    status: InstanceStatus
    port: int
    host: str
    pid: Optional[int] = None
    ctx_size: int = 4096
    n_gpu_layers: str = "auto"
    n_parallel: int = 1
    started_at: Optional[str] = None
    uptime_seconds: Optional[float] = None
    gpu_memory_mb: Optional[int] = None
    gpu_utilization_pct: Optional[float] = None
    ram_usage_mb: Optional[float] = None
    extra_args: List[str] = Field(default_factory=list)
    restart_count: int = 0


class InstanceList(BaseModel):
    instances: List[InstanceInfo]
    total: int


# ---------------------------------------------------------------------------
# Preset
# ---------------------------------------------------------------------------

class Preset(BaseModel):
    name: str
    description: str = ""
    ctx_size: Optional[int] = None
    n_gpu_layers: Optional[str] = None
    threads: Optional[int] = None
    batch_size: Optional[int] = None
    n_parallel: Optional[int] = None
    flash_attn: Optional[str] = None
    sleep_idle_seconds: Optional[int] = None
    extra_args: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Log
# ---------------------------------------------------------------------------

class LogEntry(BaseModel):
    timestamp: str
    level: str
    message: str


class InstanceLogs(BaseModel):
    instance_id: str
    logs: List[LogEntry]
    total_lines: int
