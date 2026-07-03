"""Core instance manager: process lifecycle, health checks, auto-restart."""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import socket
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from .backends import get_backend
from .cache import TTLCache
from .config import ConfigManager
from .downloader import ModelDownloader
from .models import (
    BackendType,
    InstanceInfo,
    InstanceStartRequest,
    InstanceStatus,
)
from .monitor import ResourceMonitor
from .params import resolve_params
from .utils import guess_family, guess_quantization, get_binary_path

logger = logging.getLogger("inferx")


@dataclass
class InstanceProcess:
    info: InstanceInfo
    process: subprocess.Popen | None = field(default=None, repr=False)
    log_file: Path | None = field(default=None, repr=False)
    _health_task: asyncio.Task | None = field(default=None, repr=False)
    _restart_task: asyncio.Task | None = field(default=None, repr=False)


class InstanceManager:
    def __init__(self, config_manager: ConfigManager):
        self._config = config_manager
        self._monitor = ResourceMonitor()
        self._downloader = ModelDownloader(
            model_dir=config_manager.config.model_dir,
            hf_mirror_url=config_manager.config.hf_mirror_url,
            max_concurrent=config_manager.config.download_max_concurrent,
        )
        self._instances: dict[str, InstanceProcess] = {}
        self._ports_in_use: set = set()
        self._logs_dir = Path(__file__).parent / "logs"
        self._logs_dir.mkdir(exist_ok=True)
        self._model_cache = TTLCache(default_ttl=5.0)

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

    def list_models(self, backend_type: BackendType | None = None) -> list[dict[str, Any]]:
        """List available models, optionally filtered by backend type."""
        cache_key = f"models:{backend_type}"
        cached = self._model_cache.get(cache_key)
        if cached is not None:
            return cached

        model_dir = Path(self._config.config.model_dir).expanduser()
        models = []

        if backend_type:
            backend = get_backend(backend_type.value)
            models = backend.get_model_paths(model_dir)
        else:
            seen_paths = set()
            for bt in BackendType:
                try:
                    backend = get_backend(bt.value)
                    backend_models = backend.get_model_paths(model_dir)
                    for m in backend_models:
                        model_path = m.get("path", "")
                        if model_path not in seen_paths:
                            seen_paths.add(model_path)
                            m["backend"] = bt.value
                            models.append(m)
                except Exception:
                    continue

        self._model_cache.set(cache_key, models)
        return models

    def get_model_info(self, name: str) -> dict[str, Any] | None:
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
                        "family": guess_family(target.name),
                        "quantization": guess_quantization(target.name),
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

    # --- Model info helpers (delegated to utils) ---

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
        binary = get_binary_path(backend_type, cfg)

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

        # Data-driven parameter resolution: request -> preset -> config
        params.update(resolve_params(backend_type, req, preset, cfg))
        # alias is request-only (not in preset/config)
        if req.alias:
            params["alias"] = req.alias

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
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=open(stderr_file, "w"),
                start_new_session=True,
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
            ctx_size=params.get("ctx_size", 4096),
            n_gpu_layers=params.get("n_gpu_layers", "auto"),
            n_parallel=params.get("n_parallel", 1),
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

        logger.warning("[DEBUG-MONITOR] _wait_and_monitor started for %s (pid=%s)", inst_id, inst.info.pid)

        # wait for server to be ready
        for i in range(60):
            await asyncio.sleep(1)
            alive = self._monitor.is_process_alive(inst.info.pid)
            if not alive:
                logger.warning("[DEBUG-MONITOR] %s process DEAD during startup (iter %d)", inst_id, i)
                inst.info.status = InstanceStatus.error
                return
            try:
                async with httpx.AsyncClient(timeout=2) as c:
                    r = await c.get(f"{url}/health")
                    if r.status_code == 200:
                        logger.warning("[DEBUG-MONITOR] %s health OK at iter %d", inst_id, i)
                        inst.info.status = InstanceStatus.running
                        break
            except Exception:
                continue
        else:
            logger.warning("[DEBUG-MONITOR] %s health check exhausted 60 iters, forcing running", inst_id)
            inst.info.status = InstanceStatus.running

        logger.warning("[DEBUG-MONITOR] %s entering health loop, status=%s", inst_id, inst.info.status)

        # periodic health check
        while inst.info.status == InstanceStatus.running:
            await asyncio.sleep(cfg.health_check_interval)
            alive = self._monitor.is_process_alive(inst.info.pid)
            logger.warning("[DEBUG-MONITOR] %s periodic check: alive=%s", inst_id, alive)
            if not alive:
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
        import traceback as _tb
        logger.warning("[DEBUG-KILL] _kill_process called for pid=%s, id=%s, caller:\n%s",
                       inst.process.pid if inst.process else None,
                       inst.info.id,
                       "".join(_tb.format_stack()[-4:-1]))
        if inst.process and inst.process.poll() is None:
            try:
                # Kill the entire process tree
                os.killpg(os.getpgid(inst.process.pid), signal.SIGTERM)
                inst.process.wait(timeout=10)
            except (subprocess.TimeoutExpired, ProcessLookupError, OSError):
                try:
                    os.killpg(os.getpgid(inst.process.pid), signal.SIGKILL)
                except Exception:
                    pass
            except Exception:
                pass

    async def stop_instance(self, inst_id: str) -> bool:
        import traceback as _tb
        logger.warning("[DEBUG-STOP] stop_instance called for id=%s, caller:\n%s",
                       inst_id,
                       "".join(_tb.format_stack()[-4:-1]))
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

    def list_instances(self) -> list[InstanceInfo]:
        return [inst.info for inst in self._instances.values()]

    def get_instance(self, inst_id: str) -> InstanceInfo | None:
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

    def get_instance_logs(self, inst_id: str, lines: int = 100) -> list[str]:
        inst = self._instances.get(inst_id)
        if not inst or not inst.log_file or not inst.log_file.exists():
            return []
        try:
            with open(inst.log_file, encoding="utf-8", errors="replace") as f:
                all_lines = f.readlines()
            return [line.rstrip("\n") for line in all_lines[-lines:]]
        except Exception:
            return []

    # ---- shutdown -----------------------------------------------------------

    async def shutdown(self) -> None:
        inst_ids = list(self._instances.keys())
        for inst_id in inst_ids:
            await self.stop_instance(inst_id)
        logger.info("All instances stopped")
