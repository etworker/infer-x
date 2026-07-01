"""Pydantic data models for request/response schemas."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    utilization_pct: Optional[float] = None


class SystemInfo(BaseModel):
    gpus: List[GPUInfo]
    total_ram_mb: int
    used_ram_mb: int
    available_ram_mb: int
    cpu_count: int
    cpu_percent: float
    server_paths: Dict[str, str]


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
    hf_mirror_url: Optional[str] = "https://hf-mirror.com"
    download_max_concurrent: int = 2

    # llama.cpp defaults
    default_ctx_size: int = 4096
    default_n_gpu_layers: str = "auto"
    default_threads: Optional[int] = None
    default_batch_size: int = 2048
    default_n_parallel: int = 1
    default_flash_attn: str = "auto"
    default_sleep_idle_seconds: int = -1
    default_mlock: bool = False
    default_no_mmap: bool = False
    default_numa: Optional[str] = None
    default_cont_batching: bool = True

    # vLLM defaults
    default_tensor_parallel_size: int = 1
    default_pipeline_parallel_size: int = 1
    default_max_model_len: Optional[int] = None
    default_gpu_memory_utilization: float = 0.9
    default_max_num_seqs: int = 64
    default_max_num_batched_tokens: Optional[int] = None
    default_vllm_dtype: str = "auto"
    default_quantization: Optional[str] = None
    default_trust_remote_code: bool = False
    default_chat_template: Optional[str] = None
    default_seed: Optional[int] = None
    default_disable_log_requests: bool = False
    default_enforce_eager: bool = False
    default_max_context_len_to_capture: Optional[int] = None

    # SGLang defaults
    default_tp: int = 1
    default_mem_fraction_static: float = 0.8
    default_max_num_reqs: int = 64
    default_nnodes: int = 1
    default_nccl_nvls: bool = False
    default_chunked_prefill_size: Optional[int] = None
    default_mem_cache_size: Optional[int] = None
    default_token_logprob_threshold: Optional[float] = None
    default_schedule_policy: Optional[str] = None
    default_schedule_conservativeness: float = 1.0
    default_server_worker_path: Optional[str] = None

    # TGI defaults
    default_tgi_model_id: Optional[str] = None
    default_tgi_max_batch_prefill_tokens: int = 4096
    default_tgi_max_batch_total_tokens: Optional[int] = None
    default_tgi_max_concurrent_requests: int = 64
    default_tgi_max_input_length: int = 4096
    default_tgi_max_total_tokens: int = 8192
    default_tgi_sharded: bool = False
    default_tgi_num_shard: Optional[int] = None
    default_tgi_quantize: Optional[str] = None
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
    default_ov_plugin_config: Optional[str] = None
    default_ov_model_section: Optional[str] = None


class ConfigUpdate(BaseModel):
    model_dir: Optional[str] = None
    default_backend: Optional[BackendType] = None
    llama_server_bin: Optional[str] = None
    vllm_server_bin: Optional[str] = None
    sglang_server_bin: Optional[str] = None
    port_range_start: Optional[int] = None
    port_range_end: Optional[int] = None
    default_host: Optional[str] = None
    max_instances: Optional[int] = None
    health_check_interval: Optional[int] = None
    auto_restart: Optional[bool] = None
    auto_restart_max_retries: Optional[int] = None
    auto_restart_delay: Optional[int] = None
    hf_mirror_url: Optional[str] = None
    download_max_concurrent: Optional[int] = None

    # llama.cpp defaults
    default_ctx_size: Optional[int] = None
    default_n_gpu_layers: Optional[str] = None
    default_threads: Optional[int] = None
    default_batch_size: Optional[int] = None
    default_n_parallel: Optional[int] = None
    default_flash_attn: Optional[str] = None
    default_sleep_idle_seconds: Optional[int] = None
    default_mlock: Optional[bool] = None
    default_no_mmap: Optional[bool] = None
    default_numa: Optional[str] = None
    default_cont_batching: Optional[bool] = None

    # vLLM defaults
    default_tensor_parallel_size: Optional[int] = None
    default_pipeline_parallel_size: Optional[int] = None
    default_max_model_len: Optional[int] = None
    default_gpu_memory_utilization: Optional[float] = None
    default_max_num_seqs: Optional[int] = None
    default_max_num_batched_tokens: Optional[int] = None
    default_vllm_dtype: Optional[str] = None
    default_quantization: Optional[str] = None
    default_trust_remote_code: Optional[bool] = None
    default_chat_template: Optional[str] = None
    default_seed: Optional[int] = None
    default_disable_log_requests: Optional[bool] = None
    default_enforce_eager: Optional[bool] = None
    default_max_context_len_to_capture: Optional[int] = None

    # SGLang defaults
    default_tp: Optional[int] = None
    default_mem_fraction_static: Optional[float] = None
    default_max_num_reqs: Optional[int] = None
    default_nnodes: Optional[int] = None
    default_nccl_nvls: Optional[bool] = None
    default_chunked_prefill_size: Optional[int] = None
    default_mem_cache_size: Optional[int] = None
    default_token_logprob_threshold: Optional[float] = None
    default_schedule_policy: Optional[str] = None
    default_schedule_conservativeness: Optional[float] = None
    default_server_worker_path: Optional[str] = None


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
    backend: Optional[BackendType] = None
    port: Optional[int] = None
    host: Optional[str] = None
    preset: Optional[str] = None
    extra_args: List[str] = Field(default_factory=list)

    # llama.cpp parameters
    ctx_size: Optional[int] = None
    n_gpu_layers: Optional[str] = None
    threads: Optional[int] = None
    batch_size: Optional[int] = None
    n_parallel: Optional[int] = None
    flash_attn: Optional[str] = None
    alias: Optional[str] = None
    sleep_idle_seconds: Optional[int] = None
    mlock: Optional[bool] = None
    no_mmap: Optional[bool] = None
    numa: Optional[str] = None
    cont_batching: Optional[bool] = None

    # vLLM parameters
    tensor_parallel_size: Optional[int] = None
    pipeline_parallel_size: Optional[int] = None
    max_model_len: Optional[int] = None
    gpu_memory_utilization: Optional[float] = None
    max_num_seqs: Optional[int] = None
    max_num_batched_tokens: Optional[int] = None
    dtype: Optional[str] = None
    quantization: Optional[str] = None
    trust_remote_code: Optional[bool] = None
    chat_template: Optional[str] = None
    seed: Optional[int] = None
    disable_log_requests: Optional[bool] = None
    enforce_eager: Optional[bool] = None
    max_context_len_to_capture: Optional[int] = None

    # SGLang parameters
    tp: Optional[int] = None
    mem_fraction_static: Optional[float] = None
    max_num_reqs: Optional[int] = None
    nnodes: Optional[int] = None
    nccl_nvls: Optional[bool] = None
    chunked_prefill_size: Optional[int] = None
    mem_cache_size: Optional[int] = None
    token_logprob_threshold: Optional[float] = None
    schedule_policy: Optional[str] = None
    schedule_conservativeness: Optional[float] = None
    server_worker_path: Optional[str] = None

    # TGI parameters
    tgi_model_id: Optional[str] = None
    tgi_max_batch_prefill_tokens: Optional[int] = None
    tgi_max_batch_total_tokens: Optional[int] = None
    tgi_max_concurrent_requests: Optional[int] = None
    tgi_max_input_length: Optional[int] = None
    tgi_max_total_tokens: Optional[int] = None
    tgi_sharded: Optional[bool] = None
    tgi_num_shard: Optional[int] = None
    tgi_quantize: Optional[str] = None
    tgi_dtype: Optional[str] = None
    tgi_cuda_flash_attention: Optional[bool] = None
    tgi_disable_grammar: Optional[bool] = None

    # Ollama parameters
    ollama_num_parallel: Optional[int] = None
    ollama_num_gpu: Optional[int] = None
    ollama_num_ctx: Optional[int] = None
    ollama_num_batch: Optional[int] = None
    ollama_low_vram: Optional[bool] = None
    ollama_flash_attention: Optional[bool] = None

    # TensorRT-LLM parameters
    trt_max_batch_size: Optional[int] = None
    trt_max_input_len: Optional[int] = None
    trt_max_output_len: Optional[int] = None
    trt_max_seq_len: Optional[int] = None
    trt_dtype: Optional[str] = None
    trt_deprecate_legacy: Optional[bool] = None

    # LMDeploy parameters
    lmdeploy_tp: Optional[int] = None
    lmdeploy_session_len: Optional[int] = None
    lmdeploy_max_batch_size: Optional[int] = None
    lmdeploy_cache_max_entry_count: Optional[float] = None
    lmdeploy_quant_policy: Optional[int] = None
    lmdeploy_rope_scaling_factor: Optional[float] = None
    lmdeploy_turbomind_tp: Optional[int] = None

    # OpenVINO parameters
    ov_model_name: Optional[str] = None
    ov_batch_size: Optional[int] = None
    ov_max_model_len: Optional[int] = None
    ov_nireq: Optional[int] = None
    ov_plugin_config: Optional[str] = None
    ov_model_section: Optional[str] = None


class InstanceInfo(BaseModel):
    id: str
    model: str
    backend: BackendType = BackendType.llamacpp
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
    backend: Optional[BackendType] = None
    extra_args: List[str] = Field(default_factory=list)

    # llama.cpp parameters
    ctx_size: Optional[int] = None
    n_gpu_layers: Optional[str] = None
    threads: Optional[int] = None
    batch_size: Optional[int] = None
    n_parallel: Optional[int] = None
    flash_attn: Optional[str] = None
    sleep_idle_seconds: Optional[int] = None
    mlock: Optional[bool] = None
    no_mmap: Optional[bool] = None
    numa: Optional[str] = None
    cont_batching: Optional[bool] = None

    # vLLM parameters
    tensor_parallel_size: Optional[int] = None
    pipeline_parallel_size: Optional[int] = None
    max_model_len: Optional[int] = None
    gpu_memory_utilization: Optional[float] = None
    max_num_seqs: Optional[int] = None
    max_num_batched_tokens: Optional[int] = None
    dtype: Optional[str] = None
    quantization: Optional[str] = None
    trust_remote_code: Optional[bool] = None
    chat_template: Optional[str] = None
    seed: Optional[int] = None
    disable_log_requests: Optional[bool] = None
    enforce_eager: Optional[bool] = None
    max_context_len_to_capture: Optional[int] = None

    # SGLang parameters
    tp: Optional[int] = None
    mem_fraction_static: Optional[float] = None
    max_num_reqs: Optional[int] = None
    nnodes: Optional[int] = None
    nccl_nvls: Optional[bool] = None
    chunked_prefill_size: Optional[int] = None
    mem_cache_size: Optional[int] = None
    token_logprob_threshold: Optional[float] = None
    schedule_policy: Optional[str] = None
    schedule_conservativeness: Optional[float] = None
    server_worker_path: Optional[str] = None

    # TGI parameters
    tgi_model_id: Optional[str] = None
    tgi_max_batch_prefill_tokens: Optional[int] = None
    tgi_max_batch_total_tokens: Optional[int] = None
    tgi_max_concurrent_requests: Optional[int] = None
    tgi_max_input_length: Optional[int] = None
    tgi_max_total_tokens: Optional[int] = None
    tgi_sharded: Optional[bool] = None
    tgi_num_shard: Optional[int] = None
    tgi_quantize: Optional[str] = None
    tgi_dtype: Optional[str] = None
    tgi_cuda_flash_attention: Optional[bool] = None
    tgi_disable_grammar: Optional[bool] = None

    # Ollama parameters
    ollama_num_parallel: Optional[int] = None
    ollama_num_gpu: Optional[int] = None
    ollama_num_ctx: Optional[int] = None
    ollama_num_batch: Optional[int] = None
    ollama_low_vram: Optional[bool] = None
    ollama_flash_attention: Optional[bool] = None

    # TensorRT-LLM parameters
    trt_max_batch_size: Optional[int] = None
    trt_max_input_len: Optional[int] = None
    trt_max_output_len: Optional[int] = None
    trt_max_seq_len: Optional[int] = None
    trt_dtype: Optional[str] = None
    trt_deprecate_legacy: Optional[bool] = None

    # LMDeploy parameters
    lmdeploy_tp: Optional[int] = None
    lmdeploy_session_len: Optional[int] = None
    lmdeploy_max_batch_size: Optional[int] = None
    lmdeploy_cache_max_entry_count: Optional[float] = None
    lmdeploy_quant_policy: Optional[int] = None
    lmdeploy_rope_scaling_factor: Optional[float] = None
    lmdeploy_turbomind_tp: Optional[int] = None

    # OpenVINO parameters
    ov_model_name: Optional[str] = None
    ov_batch_size: Optional[int] = None
    ov_max_model_len: Optional[int] = None
    ov_nireq: Optional[int] = None
    ov_plugin_config: Optional[str] = None
    ov_model_section: Optional[str] = None


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
