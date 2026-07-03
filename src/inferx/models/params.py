"""Shared BackendParams model — single source of truth for common backend parameters."""

from __future__ import annotations

from pydantic import BaseModel, Field


class BackendParams(BaseModel):
    """Common backend parameters shared across start requests, presets, and config.

    Use extra_args for backend-specific flags that are not listed here.
    Naming conventions match the CLI flags of popular backends (llama.cpp, vLLM, etc.).
    """
    ctx_size: int | None = Field(default=None, description="Context size (-c for llama.cpp)")
    n_gpu_layers: str | None = Field(default=None, description="GPU layers (-ngl for llama.cpp)")
    n_parallel: int | None = Field(default=None, description="Parallel slots (-np for llama.cpp)")
    threads: int | None = Field(default=None, description="Thread count (-t for llama.cpp)")
    batch_size: int | None = Field(default=None, description="Batch size (-b for llama.cpp)")
    flash_attn: str | None = Field(default=None, description="Flash attention mode")
    tensor_parallel_size: int | None = Field(default=None, description="Tensor parallel size")
    max_model_len: int | None = Field(default=None, description="Maximum model length")
    gpu_memory_utilization: float | None = Field(default=None, description="GPU memory utilization (0-1)")
    dtype: str | None = Field(default=None, description="Data type (float16, bfloat16, etc.)")
    quantization: str | None = Field(default=None, description="Quantization method")
    alias: str | None = Field(default=None, description="Model alias (-a for llama.cpp)")
