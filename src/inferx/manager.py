"""Core instance manager: process lifecycle, health checks, auto-restart."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import socket
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import psutil

from .config import ConfigManager
from .downloader import ModelDownloader
from .monitor import ResourceMonitor
from .models import (
    BackendType,
    InstanceInfo,
    InstanceStartRequest,
    InstanceStatus,
    Preset,
)
from .backends import get_backend

logger = logging.getLogger("infer_helper")


@dataclass
class InstanceProcess:
    info: InstanceInfo
    process: Optional[subprocess.Popen] = field(default=None, repr=False)
    log_file: Optional[Path] = field(default=None, repr=False)
    _health_task: Optional[asyncio.Task] = field(default=None, repr=False)
    _restart_task: Optional[asyncio.Task] = field(default=None, repr=False)


class InstanceManager:
    def __init__(self, config_manager: ConfigManager):
        self._config = config_manager
        self._monitor = ResourceMonitor()
        self._downloader = ModelDownloader(
            model_dir=config_manager.config.model_dir,
            hf_mirror_url=config_manager.config.hf_mirror_url,
            max_concurrent=config_manager.config.download_max_concurrent,
        )
        self._instances: Dict[str, InstanceProcess] = {}
        self._ports_in_use: set = set()
        self._logs_dir = Path(__file__).parent / "logs"
        self._logs_dir.mkdir(exist_ok=True)

    @property
    def monitor(self) -> ResourceMonitor:
        return self._monitor

    @property
    def downloader(self) -> ModelDownloader:
        return self._downloader

    # ---- port management ----------------------------------------------------

    def _find_available_port(self) -> int:
        cfg = self._config.config
        for port in range(cfg.port_range_start, cfg.port_range_end + 1):
            if port in self._ports_in_use:
                continue
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind(("127.0.0.1", port))
                    return port
                except OSError:
                    continue
        raise RuntimeError("No available ports in configured range")

    def _release_port(self, port: int) -> None:
        self._ports_in_use.discard(port)

    # ---- model discovery ----------------------------------------------------

    def list_models(self, backend_type: Optional[BackendType] = None) -> List[Dict[str, Any]]:
        """List available models, optionally filtered by backend type."""
        model_dir = Path(self._config.config.model_dir).expanduser()
        models = []

        if backend_type:
            # Use backend-specific model discovery
            backend = get_backend(backend_type.value)
            models = backend.get_model_paths(model_dir)
        else:
            # Discover models for all backends
            for bt in BackendType:
                try:
                    backend = get_backend(bt.value)
                    backend_models = backend.get_model_paths(model_dir)
                    for m in backend_models:
                        m["backend"] = bt.value
                    models.extend(backend_models)
                except Exception:
                    continue

        return models

    def get_model_info(self, name: str) -> Optional[Dict[str, Any]]:
        model_dir = Path(self._config.config.model_dir).expanduser()
        target = model_dir / name

        # Try direct path first
        if target.exists():
            if target.is_file() and target.suffix == ".gguf":
                return {
                    "name": name,
                    "path": str(target),
                    "size_mb": round(target.stat().st_size / (1024 * 1024), 1),
                    "family": self._guess_family(target.name),
                    "quantization": self._guess_quantization(target.name),
                    "backend": BackendType.llamacpp.value,
                }
            elif target.is_dir():
                # Check if it's a HuggingFace model directory
                config_file = target / "config.json"
                has_safetensors = any(target.glob("*.safetensors"))
                has_bin = any(target.glob("*.bin"))

                if config_file.exists() and (has_safetensors or has_bin):
                    total_size = 0
                    for f in target.glob("*.safetensors"):
                        total_size += f.stat().st_size
                    for f in target.glob("*.bin"):
                        total_size += f.stat().st_size

                    # Determine backend based on available tools
                    backend_type = BackendType.vllm  # Default to vllm for HF models

                    return {
                        "name": name,
                        "path": str(target),
                        "size_mb": round(total_size / (1024 * 1024), 1) if total_size > 0 else 0,
                        "family": self._guess_family(target.name),
                        "quantization": self._guess_quantization(target.name),
                        "backend": backend_type.value,
                    }

                # Check for .gguf files in directory
                ggufs = list(target.glob("*.gguf"))
                if ggufs:
                    main = ggufs[0]
                    return {
                        "name": f"{target.name}/{main.name}",
                        "path": str(main),
                        "size_mb": round(main.stat().st_size / (1024 * 1024), 1),
                        "family": self._guess_family(main.name),
                        "quantization": self._guess_quantization(main.name),
                        "backend": BackendType.llamacpp.value,
                    }

        # Search recursively
        for p in model_dir.rglob(name):
            if p.suffix == ".gguf":
                return {
                    "name": name,
                    "path": str(p),
                    "size_mb": round(p.stat().st_size / (1024 * 1024), 1),
                    "family": self._guess_family(p.name),
                    "quantization": self._guess_quantization(p.name),
                    "backend": BackendType.llamacpp.value,
                }

        return None

    def delete_model(self, name: str) -> bool:
        model_dir = Path(self._config.config.model_dir).expanduser()
        target = model_dir / name
        if target.exists() and target.is_file():
            target.unlink()
            return True
        if target.is_dir():
            import shutil
            shutil.rmtree(target)
            return True
        return False

    @staticmethod
    def _guess_family(name: str) -> Optional[str]:
        name_lower = name.lower()
        for family in ["qwen", "gemma", "llama", "mistral", "phi", "deepseek", "yi", "baichuan"]:
            if family in name_lower:
                return family
        return None

    @staticmethod
    def _guess_quantization(name: str) -> Optional[str]:
        import re
        m = re.search(r"(Q[0-9]+_[A-Z0-9]+|F16|F32|BF16|IQ[0-9]+_[A-Z0-9]+)", name, re.IGNORECASE)
        return m.group(1).upper() if m else None

    # ---- instance lifecycle -------------------------------------------------

    async def start_instance(self, req: InstanceStartRequest) -> InstanceInfo:
        cfg = self._config.config

        if len(self._instances) >= cfg.max_instances:
            raise RuntimeError(f"Max instances ({cfg.max_instances}) reached")

        # Determine backend
        backend_type = req.backend or cfg.default_backend
        backend = get_backend(backend_type)

        # Check if backend is installed
        from .backends import check_backend_installed
        if not check_backend_installed(backend_type):
            raise RuntimeError(
                f"Backend '{backend_type.value}' is not installed. "
                f"Please install it first."
            )

        # Get binary path based on backend
        binary_map = {
            BackendType.llamacpp: cfg.llama_server_bin,
            BackendType.vllm: cfg.vllm_server_bin,
            BackendType.sglang: cfg.sglang_server_bin,
            BackendType.tgi: cfg.tgi_bin,
            BackendType.ollama: cfg.ollama_bin,
            BackendType.tensorrt_llm: cfg.tensorrt_llm_bin,
            BackendType.lmdeploy: cfg.lmdeploy_bin,
            BackendType.openvino: cfg.openvino_bin,
        }
        binary = binary_map.get(backend_type, "")

        # For llamacpp, model_path is the .gguf file
        # For vllm/sglang/tgi/lmdeploy, model_path is the model directory or HF model name
        model_path = Path(cfg.model_dir).expanduser() / req.model
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {req.model} (looked in {Path(cfg.model_dir).expanduser()})")

        port = req.port or self._find_available_port()
        if port in self._ports_in_use and (req.port is None or port != req.port):
            raise RuntimeError(f"Port {port} already in use")
        self._ports_in_use.add(port)

        # resolve preset defaults
        preset = None
        if req.preset:
            preset = self._config.get_preset(req.preset)

        inst_id = f"inst-{uuid.uuid4().hex[:8]}"
        log_path = self._logs_dir / f"{inst_id}.log"

        # Build params dict for backend
        params = {
            "binary": binary,
            "log_file": str(log_path),
        }

        # Helper function to resolve parameter from request -> preset -> config
        def resolve_param(req_val, preset_val, cfg_val):
            if req_val is not None:
                return req_val
            if preset_val is not None:
                return preset_val
            return cfg_val

        # llama.cpp parameters
        if backend_type == BackendType.llamacpp:
            params["ctx_size"] = resolve_param(req.ctx_size, preset.ctx_size if preset else None, cfg.default_ctx_size)
            params["n_gpu_layers"] = resolve_param(req.n_gpu_layers, preset.n_gpu_layers if preset else None, cfg.default_n_gpu_layers)
            params["threads"] = resolve_param(req.threads, preset.threads if preset else None, cfg.default_threads)
            params["batch_size"] = resolve_param(req.batch_size, preset.batch_size if preset else None, cfg.default_batch_size)
            params["n_parallel"] = resolve_param(req.n_parallel, preset.n_parallel if preset else None, cfg.default_n_parallel)
            params["flash_attn"] = resolve_param(req.flash_attn, preset.flash_attn if preset else None, cfg.default_flash_attn)
            params["sleep_idle_seconds"] = resolve_param(req.sleep_idle_seconds, preset.sleep_idle_seconds if preset else None, cfg.default_sleep_idle_seconds)
            params["alias"] = req.alias
            params["mlock"] = resolve_param(req.mlock, preset.mlock if preset else None, cfg.default_mlock)
            params["no_mmap"] = resolve_param(req.no_mmap, preset.no_mmap if preset else None, cfg.default_no_mmap)
            params["numa"] = resolve_param(req.numa, preset.numa if preset else None, cfg.default_numa)
            params["cont_batching"] = resolve_param(req.cont_batching, preset.cont_batching if preset else None, cfg.default_cont_batching)

        # vLLM parameters
        elif backend_type == BackendType.vllm:
            params["tensor_parallel_size"] = resolve_param(req.tensor_parallel_size, preset.tensor_parallel_size if preset else None, cfg.default_tensor_parallel_size)
            params["pipeline_parallel_size"] = resolve_param(req.pipeline_parallel_size, preset.pipeline_parallel_size if preset else None, cfg.default_pipeline_parallel_size)
            params["max_model_len"] = resolve_param(req.max_model_len, preset.max_model_len if preset else None, cfg.default_max_model_len)
            params["gpu_memory_utilization"] = resolve_param(req.gpu_memory_utilization, preset.gpu_memory_utilization if preset else None, cfg.default_gpu_memory_utilization)
            params["max_num_seqs"] = resolve_param(req.max_num_seqs, preset.max_num_seqs if preset else None, cfg.default_max_num_seqs)
            params["max_num_batched_tokens"] = resolve_param(req.max_num_batched_tokens, preset.max_num_batched_tokens if preset else None, cfg.default_max_num_batched_tokens)
            params["dtype"] = resolve_param(req.dtype, preset.dtype if preset else None, cfg.default_vllm_dtype)
            params["quantization"] = resolve_param(req.quantization, preset.quantization if preset else None, cfg.default_quantization)
            params["trust_remote_code"] = resolve_param(req.trust_remote_code, preset.trust_remote_code if preset else None, cfg.default_trust_remote_code)
            params["chat_template"] = resolve_param(req.chat_template, preset.chat_template if preset else None, cfg.default_chat_template)
            params["seed"] = resolve_param(req.seed, preset.seed if preset else None, cfg.default_seed)
            params["disable_log_requests"] = resolve_param(req.disable_log_requests, preset.disable_log_requests if preset else None, cfg.default_disable_log_requests)
            params["enforce_eager"] = resolve_param(req.enforce_eager, preset.enforce_eager if preset else None, cfg.default_enforce_eager)
            params["max_context_len_to_capture"] = resolve_param(req.max_context_len_to_capture, preset.max_context_len_to_capture if preset else None, cfg.default_max_context_len_to_capture)

        # SGLang parameters
        elif backend_type == BackendType.sglang:
            params["tp"] = resolve_param(req.tp, preset.tp if preset else None, cfg.default_tp)
            params["mem_fraction_static"] = resolve_param(req.mem_fraction_static, preset.mem_fraction_static if preset else None, cfg.default_mem_fraction_static)
            params["max_num_reqs"] = resolve_param(req.max_num_reqs, preset.max_num_reqs if preset else None, cfg.default_max_num_reqs)
            params["nnodes"] = resolve_param(req.nnodes, preset.nnodes if preset else None, cfg.default_nnodes)
            params["nccl_nvls"] = resolve_param(req.nccl_nvls, preset.nccl_nvls if preset else None, cfg.default_nccl_nvls)
            params["chunked_prefill_size"] = resolve_param(req.chunked_prefill_size, preset.chunked_prefill_size if preset else None, cfg.default_chunked_prefill_size)
            params["mem_cache_size"] = resolve_param(req.mem_cache_size, preset.mem_cache_size if preset else None, cfg.default_mem_cache_size)
            params["token_logprob_threshold"] = resolve_param(req.token_logprob_threshold, preset.token_logprob_threshold if preset else None, cfg.default_token_logprob_threshold)
            params["schedule_policy"] = resolve_param(req.schedule_policy, preset.schedule_policy if preset else None, cfg.default_schedule_policy)
            params["schedule_conservativeness"] = resolve_param(req.schedule_conservativeness, preset.schedule_conservativeness if preset else None, cfg.default_schedule_conservativeness)
            params["server_worker_path"] = resolve_param(req.server_worker_path, preset.server_worker_path if preset else None, cfg.default_server_worker_path)

        # TGI parameters
        elif backend_type == BackendType.tgi:
            params["tgi_model_id"] = resolve_param(req.tgi_model_id, preset.tgi_model_id if preset else None, cfg.default_tgi_model_id)
            params["tgi_max_batch_prefill_tokens"] = resolve_param(req.tgi_max_batch_prefill_tokens, preset.tgi_max_batch_prefill_tokens if preset else None, cfg.default_tgi_max_batch_prefill_tokens)
            params["tgi_max_batch_total_tokens"] = resolve_param(req.tgi_max_batch_total_tokens, preset.tgi_max_batch_total_tokens if preset else None, cfg.default_tgi_max_batch_total_tokens)
            params["tgi_max_concurrent_requests"] = resolve_param(req.tgi_max_concurrent_requests, preset.tgi_max_concurrent_requests if preset else None, cfg.default_tgi_max_concurrent_requests)
            params["tgi_max_input_length"] = resolve_param(req.tgi_max_input_length, preset.tgi_max_input_length if preset else None, cfg.default_tgi_max_input_length)
            params["tgi_max_total_tokens"] = resolve_param(req.tgi_max_total_tokens, preset.tgi_max_total_tokens if preset else None, cfg.default_tgi_max_total_tokens)
            params["tgi_sharded"] = resolve_param(req.tgi_sharded, preset.tgi_sharded if preset else None, cfg.default_tgi_sharded)
            params["tgi_num_shard"] = resolve_param(req.tgi_num_shard, preset.tgi_num_shard if preset else None, cfg.default_tgi_num_shard)
            params["tgi_quantize"] = resolve_param(req.tgi_quantize, preset.tgi_quantize if preset else None, cfg.default_tgi_quantize)
            params["tgi_dtype"] = resolve_param(req.tgi_dtype, preset.tgi_dtype if preset else None, cfg.default_tgi_dtype)
            params["tgi_cuda_flash_attention"] = resolve_param(req.tgi_cuda_flash_attention, preset.tgi_cuda_flash_attention if preset else None, cfg.default_tgi_cuda_flash_attention)
            params["tgi_disable_grammar"] = resolve_param(req.tgi_disable_grammar, preset.tgi_disable_grammar if preset else None, cfg.default_tgi_disable_grammar)

        # Ollama parameters
        elif backend_type == BackendType.ollama:
            params["ollama_num_parallel"] = resolve_param(req.ollama_num_parallel, preset.ollama_num_parallel if preset else None, cfg.default_ollama_num_parallel)
            params["ollama_num_gpu"] = resolve_param(req.ollama_num_gpu, preset.ollama_num_gpu if preset else None, cfg.default_ollama_num_gpu)
            params["ollama_num_ctx"] = resolve_param(req.ollama_num_ctx, preset.ollama_num_ctx if preset else None, cfg.default_ollama_num_ctx)
            params["ollama_num_batch"] = resolve_param(req.ollama_num_batch, preset.ollama_num_batch if preset else None, cfg.default_ollama_num_batch)
            params["ollama_low_vram"] = resolve_param(req.ollama_low_vram, preset.ollama_low_vram if preset else None, cfg.default_ollama_low_vram)
            params["ollama_flash_attention"] = resolve_param(req.ollama_flash_attention, preset.ollama_flash_attention if preset else None, cfg.default_ollama_flash_attention)

        # TensorRT-LLM parameters
        elif backend_type == BackendType.tensorrt_llm:
            params["trt_max_batch_size"] = resolve_param(req.trt_max_batch_size, preset.trt_max_batch_size if preset else None, cfg.default_trt_max_batch_size)
            params["trt_max_input_len"] = resolve_param(req.trt_max_input_len, preset.trt_max_input_len if preset else None, cfg.default_trt_max_input_len)
            params["trt_max_output_len"] = resolve_param(req.trt_max_output_len, preset.trt_max_output_len if preset else None, cfg.default_trt_max_output_len)
            params["trt_max_seq_len"] = resolve_param(req.trt_max_seq_len, preset.trt_max_seq_len if preset else None, cfg.default_trt_max_seq_len)
            params["trt_dtype"] = resolve_param(req.trt_dtype, preset.trt_dtype if preset else None, cfg.default_trt_dtype)
            params["trt_deprecate_legacy"] = resolve_param(req.trt_deprecate_legacy, preset.trt_deprecate_legacy if preset else None, cfg.default_trt_deprecate_legacy)

        # LMDeploy parameters
        elif backend_type == BackendType.lmdeploy:
            params["lmdeploy_tp"] = resolve_param(req.lmdeploy_tp, preset.lmdeploy_tp if preset else None, cfg.default_lmdeploy_tp)
            params["lmdeploy_session_len"] = resolve_param(req.lmdeploy_session_len, preset.lmdeploy_session_len if preset else None, cfg.default_lmdeploy_session_len)
            params["lmdeploy_max_batch_size"] = resolve_param(req.lmdeploy_max_batch_size, preset.lmdeploy_max_batch_size if preset else None, cfg.default_lmdeploy_max_batch_size)
            params["lmdeploy_cache_max_entry_count"] = resolve_param(req.lmdeploy_cache_max_entry_count, preset.lmdeploy_cache_max_entry_count if preset else None, cfg.default_lmdeploy_cache_max_entry_count)
            params["lmdeploy_quant_policy"] = resolve_param(req.lmdeploy_quant_policy, preset.lmdeploy_quant_policy if preset else None, cfg.default_lmdeploy_quant_policy)
            params["lmdeploy_rope_scaling_factor"] = resolve_param(req.lmdeploy_rope_scaling_factor, preset.lmdeploy_rope_scaling_factor if preset else None, cfg.default_lmdeploy_rope_scaling_factor)
            params["lmdeploy_turbomind_tp"] = resolve_param(req.lmdeploy_turbomind_tp, preset.lmdeploy_turbomind_tp if preset else None, cfg.default_lmdeploy_turbomind_tp)

        # OpenVINO parameters
        elif backend_type == BackendType.openvino:
            params["ov_model_name"] = resolve_param(req.ov_model_name, preset.ov_model_name if preset else None, cfg.default_ov_model_name)
            params["ov_batch_size"] = resolve_param(req.ov_batch_size, preset.ov_batch_size if preset else None, cfg.default_ov_batch_size)
            params["ov_max_model_len"] = resolve_param(req.ov_max_model_len, preset.ov_max_model_len if preset else None, cfg.default_ov_max_model_len)
            params["ov_nireq"] = resolve_param(req.ov_nireq, preset.ov_nireq if preset else None, cfg.default_ov_nireq)
            params["ov_plugin_config"] = resolve_param(req.ov_plugin_config, preset.ov_plugin_config if preset else None, cfg.default_ov_plugin_config)
            params["ov_model_section"] = resolve_param(req.ov_model_section, preset.ov_model_section if preset else None, cfg.default_ov_model_section)

        host = req.host or cfg.default_host
        extra_args = req.extra_args or (preset.extra_args if preset else [])

        # Use backend to build command
        cmd = backend.build_command(
            model_path=str(model_path),
            port=port,
            host=host,
            log_file=str(log_path),
            params=params,
            extra_args=extra_args,
        )

        logger.info("Starting instance %s (backend=%s): %s", inst_id, backend_type.value, " ".join(cmd))

        env = os.environ.copy()
        env.update(backend.get_env(binary))

        try:
            # Capture stderr for error reporting
            stderr_file = self._logs_dir / f"{inst_id}.stderr"
            proc = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=open(stderr_file, "w"),
            )
        except FileNotFoundError as e:
            self._release_port(port)
            raise RuntimeError(
                f"Failed to start {backend_type.value} server: "
                f"binary not found or command invalid. "
                f"Command: {' '.join(cmd)}. Error: {e}"
            )
        except Exception as e:
            self._release_port(port)
            raise RuntimeError(f"Failed to start {backend_type.value} server: {e}")

        info = InstanceInfo(
            id=inst_id,
            model=req.model,
            backend=backend_type,
            status=InstanceStatus.starting,
            port=port,
            host=host,
            pid=proc.pid,
            ctx_size=params["ctx_size"],
            n_gpu_layers=params["n_gpu_layers"],
            n_parallel=params["n_parallel"],
            started_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
            extra_args=extra_args,
        )

        inst = InstanceProcess(info=info, process=proc, log_file=log_path)
        self._instances[inst_id] = inst

        # background: wait for ready + health check
        inst._health_task = asyncio.create_task(self._wait_and_monitor(inst_id))

        return info

    async def _wait_and_monitor(self, inst_id: str) -> None:
        inst = self._instances.get(inst_id)
        if not inst:
            return
        cfg = self._config.config
        url = f"http://{inst.info.host}:{inst.info.port}"

        # wait for server to be ready
        for _ in range(60):
            await asyncio.sleep(1)
            if not self._monitor.is_process_alive(inst.info.pid):
                inst.info.status = InstanceStatus.error
                return
            try:
                async with httpx.AsyncClient(timeout=2) as c:
                    r = await c.get(f"{url}/health")
                    if r.status_code == 200:
                        inst.info.status = InstanceStatus.running
                        break
            except Exception:
                continue
        else:
            inst.info.status = InstanceStatus.running

        # periodic health check
        while inst.info.status == InstanceStatus.running:
            await asyncio.sleep(cfg.health_check_interval)
            if not self._monitor.is_process_alive(inst.info.pid):
                inst.info.status = InstanceStatus.error
                logger.warning("Instance %s process died", inst_id)
                if cfg.auto_restart and inst.info.restart_count < cfg.auto_restart_max_retries:
                    inst.info.restart_count += 1
                    await asyncio.sleep(cfg.auto_restart_delay)
                    await self._restart_instance(inst_id)
                break
            # update memory usage
            ram = self._monitor.get_process_memory_mb(inst.info.pid)
            inst.info.ram_usage_mb = round(ram, 1) if ram else None

    async def _restart_instance(self, inst_id: str) -> None:
        inst = self._instances.get(inst_id)
        if not inst:
            return
        old_req = InstanceStartRequest(
            model=inst.info.model,
            backend=inst.info.backend,
            port=inst.info.port,
            ctx_size=inst.info.ctx_size,
            n_gpu_layers=inst.info.n_gpu_layers,
            n_parallel=inst.info.n_parallel,
            extra_args=inst.info.extra_args,
        )
        # stop old process
        self._kill_process(inst)
        self._release_port(inst.info.port)
        del self._instances[inst_id]
        # restart
        try:
            await self.start_instance(old_req)
            logger.info("Instance %s restarted successfully", inst_id)
        except Exception as e:
            logger.error("Failed to restart instance %s: %s", inst_id, e)

    def _kill_process(self, inst: InstanceProcess) -> None:
        if inst.process and inst.process.poll() is None:
            try:
                inst.process.send_signal(signal.SIGTERM)
                inst.process.wait(timeout=10)
            except (subprocess.TimeoutExpired, ProcessLookupError):
                inst.process.kill()
            except Exception:
                pass

    async def stop_instance(self, inst_id: str) -> bool:
        inst = self._instances.get(inst_id)
        if not inst:
            return False
        inst.info.status = InstanceStatus.stopping
        if inst._health_task:
            inst._health_task.cancel()
        self._kill_process(inst)
        self._release_port(inst.info.port)
        del self._instances[inst_id]
        return True

    async def restart_instance(self, inst_id: str) -> InstanceInfo:
        inst = self._instances.get(inst_id)
        if not inst:
            raise KeyError(f"Instance {inst_id} not found")
        old_req = InstanceStartRequest(
            model=inst.info.model,
            backend=inst.info.backend,
            port=inst.info.port,
            ctx_size=inst.info.ctx_size,
            n_gpu_layers=inst.info.n_gpu_layers,
            n_parallel=inst.info.n_parallel,
            extra_args=inst.info.extra_args,
        )
        self._kill_process(inst)
        self._release_port(inst.info.port)
        del self._instances[inst_id]
        return await self.start_instance(old_req)

    def list_instances(self) -> List[InstanceInfo]:
        return [inst.info for inst in self._instances.values()]

    def get_instance(self, inst_id: str) -> Optional[InstanceInfo]:
        inst = self._instances.get(inst_id)
        if inst:
            # refresh stats
            if inst.process and inst.info.pid:
                alive = self._monitor.is_process_alive(inst.info.pid)
                if not alive and inst.info.status == InstanceStatus.running:
                    inst.info.status = InstanceStatus.error
                ram = self._monitor.get_process_memory_mb(inst.info.pid)
                inst.info.ram_usage_mb = round(ram, 1) if ram else None
            return inst.info
        return None

    def get_instance_logs(self, inst_id: str, lines: int = 100) -> List[str]:
        inst = self._instances.get(inst_id)
        if not inst or not inst.log_file or not inst.log_file.exists():
            return []
        try:
            with open(inst.log_file, "r", encoding="utf-8", errors="replace") as f:
                all_lines = f.readlines()
            return [l.rstrip("\n") for l in all_lines[-lines:]]
        except Exception:
            return []

    # ---- shutdown -----------------------------------------------------------

    async def shutdown(self) -> None:
        inst_ids = list(self._instances.keys())
        for inst_id in inst_ids:
            await self.stop_instance(inst_id)
        logger.info("All instances stopped")
