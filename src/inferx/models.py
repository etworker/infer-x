"""Pydantic data models for request/response schemas."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


def _default_model_dir() -> str:
    """Get default model directory from environment or home directory."""
    import os
    return os.environ.get("INFER_HELPER_MODEL_DIR", str(Path.home() / "models"))


def _default_llamacpp_bin() -> str:
    """Get default llama-server binary path."""
    import os
    import shutil
    # Check environment variable first
    env_bin = os.environ.get("LLAMA_SERVER_BIN")
    if env_bin:
        return env_bin
    # Check if llama-server is in PATH
    if shutil.which("llama-server"):
        return "llama-server"
    # Default to common installation path
    return str(Path.home() / "llama.cpp" / "build" / "bin" / "llama-server")


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class BackendType(str, Enum):
    llamacpp = "llamacpp"
    vllm = "vllm"
    sglang = "sglang"
    tgi = "tgi"
    ollama = "ollama"
    tensorrt_llm = "tensorrt_llm"
    lmdeploy = "lmdeploy"
    openvino = "openvino"


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


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class DefaultConfig(BaseModel):
    model_dir: str = Field(default_factory=_default_model_dir)
    default_backend: BackendType = BackendType.llamacpp
    llama_server_bin: str = Field(default_factory=_default_llamacpp_bin)
    vllm_server_bin: str = "python -m vllm.entrypoints.openai.api_server"
    sglang_server_bin: str = "python -m sglang.launch_server"
    tgi_bin: str = "text-generation-launcher"
    ollama_bin: str = "ollama"
    tensorrt_llm_bin: str = "python -m tensorrt_llm.commands.tritonserver"
    lmdeploy_bin: str = "lmdeploy serve api_server"
    openvino_bin: str = "ovms"
    port_range_start: int = 8080
    port_range_end: int = 8180
    default_host: str = "0.0.0.0"
    max_instances: int = 4
    health_check_interval: int = 10
    auto_restart: bool = True
    auto_restart_max_retries: int = 3
    auto_restart_delay: int = 5
    hf_mirror_url: str | None = "https://hf-mirror.com"
    download_max_concurrent: int = 2

    # llama.cpp defaults
    default_ctx_size: int = 4096
    default_n_gpu_layers: str = "auto"
    default_threads: int | None = None
    default_batch_size: int = 2048
    default_n_parallel: int = 1
    default_flash_attn: str = "auto"
    default_sleep_idle_seconds: int = -1
    default_mlock: bool = False
    default_no_mmap: bool = False
    default_numa: str | None = None
    default_cont_batching: bool = True

    # vLLM defaults
    default_tensor_parallel_size: int = 1
    default_pipeline_parallel_size: int = 1
    default_max_model_len: int | None = None
    default_gpu_memory_utilization: float = 0.9
    default_max_num_seqs: int = 64
    default_max_num_batched_tokens: int | None = None
    default_vllm_dtype: str = "auto"
    default_quantization: str | None = None
    default_trust_remote_code: bool = False
    default_chat_template: str | None = None
    default_seed: int | None = None
    default_disable_log_requests: bool = False
    default_enforce_eager: bool = False
    default_max_context_len_to_capture: int | None = None

    # SGLang defaults
    default_tp: int = 1
    default_mem_fraction_static: float = 0.8
    default_max_num_reqs: int = 64
    default_nnodes: int = 1
    default_nccl_nvls: bool = False
    default_chunked_prefill_size: int | None = None
    default_mem_cache_size: int | None = None
    default_token_logprob_threshold: float | None = None
    default_schedule_policy: str | None = None
    default_schedule_conservativeness: float = 1.0
    default_server_worker_path: str | None = None

    # TGI defaults
    default_tgi_model_id: str | None = None
    default_tgi_max_batch_prefill_tokens: int = 4096
    default_tgi_max_batch_total_tokens: int | None = None
    default_tgi_max_concurrent_requests: int = 64
    default_tgi_max_input_length: int = 4096
    default_tgi_max_total_tokens: int = 8192
    default_tgi_sharded: bool = False
    default_tgi_num_shard: int | None = None
    default_tgi_quantize: str | None = None
    default_tgi_dtype: str = "auto"
    default_tgi_cuda_flash_attention: bool = True
    default_tgi_disable_grammar: bool = False

    # Ollama defaults
    default_ollama_num_parallel: int = 1
    default_ollama_num_gpu: int = 99
    default_ollama_num_ctx: int = 2048
    default_ollama_num_batch: int = 512
    default_ollama_low_vram: bool = False
    default_ollama_flash_attention: bool = False

    # TensorRT-LLM defaults
    default_trt_max_batch_size: int = 8
    default_trt_max_input_len: int = 2048
    default_trt_max_output_len: int = 512
    default_trt_max_seq_len: int = 2560
    default_trt_dtype: str = "float16"
    default_trt_deprecate_legacy: bool = True

    # LMDeploy defaults
    default_lmdeploy_tp: int = 1
    default_lmdeploy_session_len: int = 2048
    default_lmdeploy_max_batch_size: int = 128
    default_lmdeploy_cache_max_entry_count: float = 0.5
    default_lmdeploy_quant_policy: int = 0
    default_lmdeploy_rope_scaling_factor: float = 0.0
    default_lmdeploy_turbomind_tp: int = 1

    # OpenVINO defaults
    default_ov_model_name: str = "model"
    default_ov_batch_size: int = 1
    default_ov_max_model_len: int = 256
    default_ov_nireq: int = 1
    default_ov_plugin_config: str | None = None
    default_ov_model_section: str | None = None


class ConfigUpdate(BaseModel):
    model_dir: str | None = None
    default_backend: BackendType | None = None
    llama_server_bin: str | None = None
    vllm_server_bin: str | None = None
    sglang_server_bin: str | None = None
    port_range_start: int | None = None
    port_range_end: int | None = None
    default_host: str | None = None
    max_instances: int | None = None
    health_check_interval: int | None = None
    auto_restart: bool | None = None
    auto_restart_max_retries: int | None = None
    auto_restart_delay: int | None = None
    hf_mirror_url: str | None = None
    download_max_concurrent: int | None = None

    # llama.cpp defaults
    default_ctx_size: int | None = None
    default_n_gpu_layers: str | None = None
    default_threads: int | None = None
    default_batch_size: int | None = None
    default_n_parallel: int | None = None
    default_flash_attn: str | None = None
    default_sleep_idle_seconds: int | None = None
    default_mlock: bool | None = None
    default_no_mmap: bool | None = None
    default_numa: str | None = None
    default_cont_batching: bool | None = None

    # vLLM defaults
    default_tensor_parallel_size: int | None = None
    default_pipeline_parallel_size: int | None = None
    default_max_model_len: int | None = None
    default_gpu_memory_utilization: float | None = None
    default_max_num_seqs: int | None = None
    default_max_num_batched_tokens: int | None = None
    default_vllm_dtype: str | None = None
    default_quantization: str | None = None
    default_trust_remote_code: bool | None = None
    default_chat_template: str | None = None
    default_seed: int | None = None
    default_disable_log_requests: bool | None = None
    default_enforce_eager: bool | None = None
    default_max_context_len_to_capture: int | None = None

    # SGLang defaults
    default_tp: int | None = None
    default_mem_fraction_static: float | None = None
    default_max_num_reqs: int | None = None
    default_nnodes: int | None = None
    default_nccl_nvls: bool | None = None
    default_chunked_prefill_size: int | None = None
    default_mem_cache_size: int | None = None
    default_token_logprob_threshold: float | None = None
    default_schedule_policy: str | None = None
    default_schedule_conservativeness: float | None = None
    default_server_worker_path: str | None = None


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class ModelFileInfo(BaseModel):
    name: str
    path: str
    size_mb: float
    quantization: str | None = None
    family: str | None = None


class ModelInfo(BaseModel):
    name: str
    path: str
    size_mb: float
    quantization: str | None = None
    family: str | None = None
    architecture: str | None = None
    parameter_count: str | None = None
    context_length: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

class DownloadRequest(BaseModel):
    source: DownloadSource
    repo: str | None = None
    filename: str | None = None
    url: str | None = None
    quantization: str | None = None
    save_name: str | None = None


class DownloadProgress(BaseModel):
    task_id: str
    source: str
    repo: str | None = None
    filename: str | None = None
    status: DownloadStatus
    progress_pct: float = 0.0
    downloaded_bytes: int = 0
    total_bytes: int = 0
    speed_bytes_per_sec: float = 0.0
    error: str | None = None
    save_path: str | None = None


# ---------------------------------------------------------------------------
# Instance
# ---------------------------------------------------------------------------

class InstanceStartRequest(BaseModel):
    model: str
    backend: BackendType | None = None
    port: int | None = None
    host: str | None = None
    preset: str | None = None
    extra_args: list[str] = Field(default_factory=list)

    # llama.cpp parameters
    ctx_size: int | None = None
    n_gpu_layers: str | None = None
    threads: int | None = None
    batch_size: int | None = None
    n_parallel: int | None = None
    flash_attn: str | None = None
    alias: str | None = None
    sleep_idle_seconds: int | None = None
    mlock: bool | None = None
    no_mmap: bool | None = None
    numa: str | None = None
    cont_batching: bool | None = None

    # vLLM parameters
    tensor_parallel_size: int | None = None
    pipeline_parallel_size: int | None = None
    max_model_len: int | None = None
    gpu_memory_utilization: float | None = None
    max_num_seqs: int | None = None
    max_num_batched_tokens: int | None = None
    dtype: str | None = None
    quantization: str | None = None
    trust_remote_code: bool | None = None
    chat_template: str | None = None
    seed: int | None = None
    disable_log_requests: bool | None = None
    enforce_eager: bool | None = None
    max_context_len_to_capture: int | None = None

    # SGLang parameters
    tp: int | None = None
    mem_fraction_static: float | None = None
    max_num_reqs: int | None = None
    nnodes: int | None = None
    nccl_nvls: bool | None = None
    chunked_prefill_size: int | None = None
    mem_cache_size: int | None = None
    token_logprob_threshold: float | None = None
    schedule_policy: str | None = None
    schedule_conservativeness: float | None = None
    server_worker_path: str | None = None

    # TGI parameters
    tgi_model_id: str | None = None
    tgi_max_batch_prefill_tokens: int | None = None
    tgi_max_batch_total_tokens: int | None = None
    tgi_max_concurrent_requests: int | None = None
    tgi_max_input_length: int | None = None
    tgi_max_total_tokens: int | None = None
    tgi_sharded: bool | None = None
    tgi_num_shard: int | None = None
    tgi_quantize: str | None = None
    tgi_dtype: str | None = None
    tgi_cuda_flash_attention: bool | None = None
    tgi_disable_grammar: bool | None = None

    # Ollama parameters
    ollama_num_parallel: int | None = None
    ollama_num_gpu: int | None = None
    ollama_num_ctx: int | None = None
    ollama_num_batch: int | None = None
    ollama_low_vram: bool | None = None
    ollama_flash_attention: bool | None = None

    # TensorRT-LLM parameters
    trt_max_batch_size: int | None = None
    trt_max_input_len: int | None = None
    trt_max_output_len: int | None = None
    trt_max_seq_len: int | None = None
    trt_dtype: str | None = None
    trt_deprecate_legacy: bool | None = None

    # LMDeploy parameters
    lmdeploy_tp: int | None = None
    lmdeploy_session_len: int | None = None
    lmdeploy_max_batch_size: int | None = None
    lmdeploy_cache_max_entry_count: float | None = None
    lmdeploy_quant_policy: int | None = None
    lmdeploy_rope_scaling_factor: float | None = None
    lmdeploy_turbomind_tp: int | None = None

    # OpenVINO parameters
    ov_model_name: str | None = None
    ov_batch_size: int | None = None
    ov_max_model_len: int | None = None
    ov_nireq: int | None = None
    ov_plugin_config: str | None = None
    ov_model_section: str | None = None


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


class InstanceList(BaseModel):
    instances: list[InstanceInfo]
    total: int


# ---------------------------------------------------------------------------
# Preset
# ---------------------------------------------------------------------------

class Preset(BaseModel):
    name: str
    description: str = ""
    backend: BackendType | None = None
    extra_args: list[str] = Field(default_factory=list)

    # llama.cpp parameters
    ctx_size: int | None = None
    n_gpu_layers: str | None = None
    threads: int | None = None
    batch_size: int | None = None
    n_parallel: int | None = None
    flash_attn: str | None = None
    sleep_idle_seconds: int | None = None
    mlock: bool | None = None
    no_mmap: bool | None = None
    numa: str | None = None
    cont_batching: bool | None = None

    # vLLM parameters
    tensor_parallel_size: int | None = None
    pipeline_parallel_size: int | None = None
    max_model_len: int | None = None
    gpu_memory_utilization: float | None = None
    max_num_seqs: int | None = None
    max_num_batched_tokens: int | None = None
    dtype: str | None = None
    quantization: str | None = None
    trust_remote_code: bool | None = None
    chat_template: str | None = None
    seed: int | None = None
    disable_log_requests: bool | None = None
    enforce_eager: bool | None = None
    max_context_len_to_capture: int | None = None

    # SGLang parameters
    tp: int | None = None
    mem_fraction_static: float | None = None
    max_num_reqs: int | None = None
    nnodes: int | None = None
    nccl_nvls: bool | None = None
    chunked_prefill_size: int | None = None
    mem_cache_size: int | None = None
    token_logprob_threshold: float | None = None
    schedule_policy: str | None = None
    schedule_conservativeness: float | None = None
    server_worker_path: str | None = None

    # TGI parameters
    tgi_model_id: str | None = None
    tgi_max_batch_prefill_tokens: int | None = None
    tgi_max_batch_total_tokens: int | None = None
    tgi_max_concurrent_requests: int | None = None
    tgi_max_input_length: int | None = None
    tgi_max_total_tokens: int | None = None
    tgi_sharded: bool | None = None
    tgi_num_shard: int | None = None
    tgi_quantize: str | None = None
    tgi_dtype: str | None = None
    tgi_cuda_flash_attention: bool | None = None
    tgi_disable_grammar: bool | None = None

    # Ollama parameters
    ollama_num_parallel: int | None = None
    ollama_num_gpu: int | None = None
    ollama_num_ctx: int | None = None
    ollama_num_batch: int | None = None
    ollama_low_vram: bool | None = None
    ollama_flash_attention: bool | None = None

    # TensorRT-LLM parameters
    trt_max_batch_size: int | None = None
    trt_max_input_len: int | None = None
    trt_max_output_len: int | None = None
    trt_max_seq_len: int | None = None
    trt_dtype: str | None = None
    trt_deprecate_legacy: bool | None = None

    # LMDeploy parameters
    lmdeploy_tp: int | None = None
    lmdeploy_session_len: int | None = None
    lmdeploy_max_batch_size: int | None = None
    lmdeploy_cache_max_entry_count: float | None = None
    lmdeploy_quant_policy: int | None = None
    lmdeploy_rope_scaling_factor: float | None = None
    lmdeploy_turbomind_tp: int | None = None

    # OpenVINO parameters
    ov_model_name: str | None = None
    ov_batch_size: int | None = None
    ov_max_model_len: int | None = None
    ov_nireq: int | None = None
    ov_plugin_config: str | None = None
    ov_model_section: str | None = None


# ---------------------------------------------------------------------------
# Log
# ---------------------------------------------------------------------------

class LogEntry(BaseModel):
    timestamp: str
    level: str
    message: str


class InstanceLogs(BaseModel):
    instance_id: str
    logs: list[LogEntry]
    total_lines: int
