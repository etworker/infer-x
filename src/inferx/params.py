"""Data-driven parameter resolution for backend instances."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .models import BackendType, DefaultConfig, InstanceStartRequest, Preset


@dataclass(frozen=True)
class ParamSpec:
    """Specification for a single backend parameter."""
    param_key: str           # Key in the params dict passed to build_command()
    req_field: str           # Field name on InstanceStartRequest
    preset_field: str | None # Field name on Preset (None if preset doesn't have it)
    cfg_field: str           # Field name on DefaultConfig


# ── Per-backend parameter registries ────────────────────────────────────────

_LLAMACPP_PARAMS = [
    ParamSpec("ctx_size", "ctx_size", "ctx_size", "default_ctx_size"),
    ParamSpec("n_gpu_layers", "n_gpu_layers", "n_gpu_layers", "default_n_gpu_layers"),
    ParamSpec("threads", "threads", "threads", "default_threads"),
    ParamSpec("batch_size", "batch_size", "batch_size", "default_batch_size"),
    ParamSpec("n_parallel", "n_parallel", "n_parallel", "default_n_parallel"),
    ParamSpec("flash_attn", "flash_attn", "flash_attn", "default_flash_attn"),
    ParamSpec("sleep_idle_seconds", "sleep_idle_seconds", "sleep_idle_seconds", "default_sleep_idle_seconds"),
    ParamSpec("mlock", "mlock", "mlock", "default_mlock"),
    ParamSpec("no_mmap", "no_mmap", "no_mmap", "default_no_mmap"),
    ParamSpec("numa", "numa", "numa", "default_numa"),
    ParamSpec("cont_batching", "cont_batching", "cont_batching", "default_cont_batching"),
]

_VLLM_PARAMS = [
    ParamSpec("tensor_parallel_size", "tensor_parallel_size", "tensor_parallel_size", "default_tensor_parallel_size"),
    ParamSpec("pipeline_parallel_size", "pipeline_parallel_size", "pipeline_parallel_size", "default_pipeline_parallel_size"),
    ParamSpec("max_model_len", "max_model_len", "max_model_len", "default_max_model_len"),
    ParamSpec("gpu_memory_utilization", "gpu_memory_utilization", "gpu_memory_utilization", "default_gpu_memory_utilization"),
    ParamSpec("max_num_seqs", "max_num_seqs", "max_num_seqs", "default_max_num_seqs"),
    ParamSpec("max_num_batched_tokens", "max_num_batched_tokens", "max_num_batched_tokens", "default_max_num_batched_tokens"),
    ParamSpec("dtype", "dtype", "dtype", "default_vllm_dtype"),
    ParamSpec("quantization", "quantization", "quantization", "default_quantization"),
    ParamSpec("trust_remote_code", "trust_remote_code", "trust_remote_code", "default_trust_remote_code"),
    ParamSpec("chat_template", "chat_template", "chat_template", "default_chat_template"),
    ParamSpec("seed", "seed", "seed", "default_seed"),
    ParamSpec("disable_log_requests", "disable_log_requests", "disable_log_requests", "default_disable_log_requests"),
    ParamSpec("enforce_eager", "enforce_eager", "enforce_eager", "default_enforce_eager"),
    ParamSpec("max_context_len_to_capture", "max_context_len_to_capture", "max_context_len_to_capture", "default_max_context_len_to_capture"),
]

_SGLANG_PARAMS = [
    ParamSpec("tp", "tp", "tp", "default_tp"),
    ParamSpec("mem_fraction_static", "mem_fraction_static", "mem_fraction_static", "default_mem_fraction_static"),
    ParamSpec("max_num_reqs", "max_num_reqs", "max_num_reqs", "default_max_num_reqs"),
    ParamSpec("nnodes", "nnodes", "nnodes", "default_nnodes"),
    ParamSpec("nccl_nvls", "nccl_nvls", "nccl_nvls", "default_nccl_nvls"),
    ParamSpec("chunked_prefill_size", "chunked_prefill_size", "chunked_prefill_size", "default_chunked_prefill_size"),
    ParamSpec("mem_cache_size", "mem_cache_size", "mem_cache_size", "default_mem_cache_size"),
    ParamSpec("token_logprob_threshold", "token_logprob_threshold", "token_logprob_threshold", "default_token_logprob_threshold"),
    ParamSpec("schedule_policy", "schedule_policy", "schedule_policy", "default_schedule_policy"),
    ParamSpec("schedule_conservativeness", "schedule_conservativeness", "schedule_conservativeness", "default_schedule_conservativeness"),
    ParamSpec("server_worker_path", "server_worker_path", "server_worker_path", "default_server_worker_path"),
]

_TGI_PARAMS = [
    ParamSpec("tgi_model_id", "tgi_model_id", "tgi_model_id", "default_tgi_model_id"),
    ParamSpec("tgi_max_batch_prefill_tokens", "tgi_max_batch_prefill_tokens", "tgi_max_batch_prefill_tokens", "default_tgi_max_batch_prefill_tokens"),
    ParamSpec("tgi_max_batch_total_tokens", "tgi_max_batch_total_tokens", "tgi_max_batch_total_tokens", "default_tgi_max_batch_total_tokens"),
    ParamSpec("tgi_max_concurrent_requests", "tgi_max_concurrent_requests", "tgi_max_concurrent_requests", "default_tgi_max_concurrent_requests"),
    ParamSpec("tgi_max_input_length", "tgi_max_input_length", "tgi_max_input_length", "default_tgi_max_input_length"),
    ParamSpec("tgi_max_total_tokens", "tgi_max_total_tokens", "tgi_max_total_tokens", "default_tgi_max_total_tokens"),
    ParamSpec("tgi_sharded", "tgi_sharded", "tgi_sharded", "default_tgi_sharded"),
    ParamSpec("tgi_num_shard", "tgi_num_shard", "tgi_num_shard", "default_tgi_num_shard"),
    ParamSpec("tgi_quantize", "tgi_quantize", "tgi_quantize", "default_tgi_quantize"),
    ParamSpec("tgi_dtype", "tgi_dtype", "tgi_dtype", "default_tgi_dtype"),
    ParamSpec("tgi_cuda_flash_attention", "tgi_cuda_flash_attention", "tgi_cuda_flash_attention", "default_tgi_cuda_flash_attention"),
    ParamSpec("tgi_disable_grammar", "tgi_disable_grammar", "tgi_disable_grammar", "default_tgi_disable_grammar"),
]

_OLLAMA_PARAMS = [
    ParamSpec("ollama_num_parallel", "ollama_num_parallel", "ollama_num_parallel", "default_ollama_num_parallel"),
    ParamSpec("ollama_num_gpu", "ollama_num_gpu", "ollama_num_gpu", "default_ollama_num_gpu"),
    ParamSpec("ollama_num_ctx", "ollama_num_ctx", "ollama_num_ctx", "default_ollama_num_ctx"),
    ParamSpec("ollama_num_batch", "ollama_num_batch", "ollama_num_batch", "default_ollama_num_batch"),
    ParamSpec("ollama_low_vram", "ollama_low_vram", "ollama_low_vram", "default_ollama_low_vram"),
    ParamSpec("ollama_flash_attention", "ollama_flash_attention", "ollama_flash_attention", "default_ollama_flash_attention"),
]

_TRT_PARAMS = [
    ParamSpec("trt_max_batch_size", "trt_max_batch_size", "trt_max_batch_size", "default_trt_max_batch_size"),
    ParamSpec("trt_max_input_len", "trt_max_input_len", "trt_max_input_len", "default_trt_max_input_len"),
    ParamSpec("trt_max_output_len", "trt_max_output_len", "trt_max_output_len", "default_trt_max_output_len"),
    ParamSpec("trt_max_seq_len", "trt_max_seq_len", "trt_max_seq_len", "default_trt_max_seq_len"),
    ParamSpec("trt_dtype", "trt_dtype", "trt_dtype", "default_trt_dtype"),
    ParamSpec("trt_deprecate_legacy", "trt_deprecate_legacy", "trt_deprecate_legacy", "default_trt_deprecate_legacy"),
]

_LMDEPLOY_PARAMS = [
    ParamSpec("lmdeploy_tp", "lmdeploy_tp", "lmdeploy_tp", "default_lmdeploy_tp"),
    ParamSpec("lmdeploy_session_len", "lmdeploy_session_len", "lmdeploy_session_len", "default_lmdeploy_session_len"),
    ParamSpec("lmdeploy_max_batch_size", "lmdeploy_max_batch_size", "lmdeploy_max_batch_size", "default_lmdeploy_max_batch_size"),
    ParamSpec("lmdeploy_cache_max_entry_count", "lmdeploy_cache_max_entry_count", "lmdeploy_cache_max_entry_count", "default_lmdeploy_cache_max_entry_count"),
    ParamSpec("lmdeploy_quant_policy", "lmdeploy_quant_policy", "lmdeploy_quant_policy", "default_lmdeploy_quant_policy"),
    ParamSpec("lmdeploy_rope_scaling_factor", "lmdeploy_rope_scaling_factor", "lmdeploy_rope_scaling_factor", "default_lmdeploy_rope_scaling_factor"),
    ParamSpec("lmdeploy_turbomind_tp", "lmdeploy_turbomind_tp", "lmdeploy_turbomind_tp", "default_lmdeploy_turbomind_tp"),
]

_OPENVINO_PARAMS = [
    ParamSpec("ov_model_name", "ov_model_name", "ov_model_name", "default_ov_model_name"),
    ParamSpec("ov_batch_size", "ov_batch_size", "ov_batch_size", "default_ov_batch_size"),
    ParamSpec("ov_max_model_len", "ov_max_model_len", "ov_max_model_len", "default_ov_max_model_len"),
    ParamSpec("ov_nireq", "ov_nireq", "ov_nireq", "default_ov_nireq"),
    ParamSpec("ov_plugin_config", "ov_plugin_config", "ov_plugin_config", "default_ov_plugin_config"),
    ParamSpec("ov_model_section", "ov_model_section", "ov_model_section", "default_ov_model_section"),
]


BACKEND_PARAMS: dict[BackendType, list[ParamSpec]] = {
    BackendType.llamacpp: _LLAMACPP_PARAMS,
    BackendType.vllm: _VLLM_PARAMS,
    BackendType.sglang: _SGLANG_PARAMS,
    BackendType.tgi: _TGI_PARAMS,
    BackendType.ollama: _OLLAMA_PARAMS,
    BackendType.tensorrt_llm: _TRT_PARAMS,
    BackendType.lmdeploy: _LMDEPLOY_PARAMS,
    BackendType.openvino: _OPENVINO_PARAMS,
}


def _resolve_value(req_val: Any, preset_val: Any, cfg_val: Any) -> Any:
    """3-level fallback: request -> preset -> config."""
    if req_val is not None:
        return req_val
    if preset_val is not None:
        return preset_val
    return cfg_val


def resolve_params(
    backend_type: BackendType,
    req: InstanceStartRequest,
    preset: Preset | None,
    config: DefaultConfig,
) -> dict[str, Any]:
    """Resolve all parameters for a backend using data-driven registry.

    Returns a dict suitable for passing to backend.build_command().
    """
    specs = BACKEND_PARAMS.get(backend_type, [])
    resolved: dict[str, Any] = {}
    for spec in specs:
        req_val = getattr(req, spec.req_field, None)
        preset_val = getattr(preset, spec.preset_field, None) if preset and spec.preset_field else None
        cfg_val = getattr(config, spec.cfg_field, None)
        resolved[spec.param_key] = _resolve_value(req_val, preset_val, cfg_val)
    return resolved
