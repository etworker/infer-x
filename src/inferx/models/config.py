"""Configuration models: DefaultConfig, ConfigUpdate."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

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
    model_config = ConfigDict(extra="forbid")

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

    # Common defaults
    default_ctx_size: int = 4096
    default_n_gpu_layers: str = "auto"
    default_threads: int | None = None
    default_batch_size: int = 2048
    default_n_parallel: int = 1
    default_flash_attn: str = "auto"
    default_tensor_parallel_size: int = 1
    default_max_model_len: int | None = None
    default_gpu_memory_utilization: float = 0.9
    default_vllm_dtype: str = "auto"
    default_quantization: str | None = None


class ConfigUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

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

    # Common defaults
    default_ctx_size: int | None = None
    default_n_gpu_layers: str | None = None
    default_threads: int | None = None
    default_batch_size: int | None = None
    default_n_parallel: int | None = None
    default_flash_attn: str | None = None
    default_tensor_parallel_size: int | None = None
    default_max_model_len: int | None = None
    default_gpu_memory_utilization: float | None = None
    default_vllm_dtype: str | None = None
    default_quantization: str | None = None
