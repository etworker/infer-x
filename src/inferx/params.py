"""Data-driven parameter resolution for backend instances."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .models import BackendType, BackendParams, DefaultConfig, InstanceStartRequest, Preset


# Mapping from BackendParams field names -> config field names (prefixed with "default_")
_PARAM_TO_CONFIG = {
    "ctx_size": "default_ctx_size",
    "n_gpu_layers": "default_n_gpu_layers",
    "n_parallel": "default_n_parallel",
    "threads": "default_threads",
    "batch_size": "default_batch_size",
    "flash_attn": "default_flash_attn",
    "tensor_parallel_size": "default_tensor_parallel_size",
    "max_model_len": "default_max_model_len",
    "gpu_memory_utilization": "default_gpu_memory_utilization",
    "dtype": "default_vllm_dtype",
    "quantization": "default_quantization",
}


def resolve_params(
    backend_type: BackendType,
    req: InstanceStartRequest,
    preset: Preset | None,
    config: DefaultConfig,
) -> dict[str, Any]:
    """Resolve all parameters for a backend using 3-level fallback (request > preset > config).

    Returns a dict suitable for passing to backend.build_command().
    """
    resolved: dict[str, Any] = {}
    for field_name in BackendParams.model_fields:
        req_val = getattr(req, field_name, None)
        preset_val = getattr(preset, field_name, None) if preset else None
        cfg_field = _PARAM_TO_CONFIG.get(field_name)
        cfg_val = getattr(config, cfg_field, None) if cfg_field else None
        resolved[field_name] = req_val if req_val is not None else (preset_val if preset_val is not None else cfg_val)
    return resolved
