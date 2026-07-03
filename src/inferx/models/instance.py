"""Instance-related models."""

from __future__ import annotations

from pydantic import BaseModel, Field

from .enums import BackendType, InstanceStatus


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
