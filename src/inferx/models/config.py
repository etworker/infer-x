"""Configuration models: DefaultConfig, ConfigUpdate."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from .enums import BackendType


def _default_model_dir() -> str:
    import os
    return os.environ.get("INFER_HELPER_MODEL_DIR", str(Path.home() / "models"))


def _default_llamacpp_bin() -> str:
    import os
    import shutil
    env_bin = os.environ.get("LLAMA_SERVER_BIN")
    if env_bin:
        return env_bin
    if shutil.which("llama-server"):
        return "llama-server"
    return str(Path.home() / "llama.cpp" / "build" / "bin" / "llama-server")


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
    health_check_timeout: int = 5
    startup_timeout_seconds: int = 120
    shutdown_timeout_seconds: int = 10
    auto_restart: bool = True
    auto_restart_max_retries: int = 3
    auto_restart_delay: int = 5
    hf_mirror_url: str | None = "https://hf-mirror.com"
    download_max_concurrent: int = 2
    hf_model_repos: dict[str, str] = Field(default_factory=lambda: {
        "qwen2.5": "Qwen/Qwen2.5-{size}-Instruct",
        "qwen3": "Qwen/Qwen3-{size}",
        "qwen3.5": "Qwen/Qwen3.5-{size}",
        "gemma-4": "google/gemma-4-{size}-it",
        "llama": "meta-llama/Llama-{size}",
        "mistral": "mistralai/Mistral-{size}",
    })
    ms_model_repos: dict[str, str] = Field(default_factory=lambda: {
        "qwen2.5": "Qwen/Qwen2.5-{size}-Instruct",
        "qwen3": "Qwen/Qwen3-{size}",
        "qwen3.5": "Qwen/Qwen3.5-{size}",
        "gemma-4": "AI-ModelScope/gemma-4-{size}-it",
        "llama": "meta-llama/Llama-{size}",
        "mistral": "mistralai/Mistral-{size}",
    })

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
