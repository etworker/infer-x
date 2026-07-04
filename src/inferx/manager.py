"""Instance manager — thin orchestrator over PortManager, ModelService, ProcessManager."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx

from .config import ConfigManager
from .downloader import ModelDownloader
from .logging import logger
from .model_service import ModelService
from .models import BackendType, InstanceInfo, InstanceStartRequest
from .monitor import ResourceMonitor
from .port_manager import PortManager
from .process_manager import ProcessManager


class InstanceManager:
    def __init__(self, config_manager: ConfigManager):
        self._config = config_manager
        self._monitor = ResourceMonitor()
        self._downloader = ModelDownloader(
            model_dir=config_manager.config.model_dir,
            hf_mirror_url=config_manager.config.hf_mirror_url,
            max_concurrent=config_manager.config.download_max_concurrent,
            hf_model_repos=config_manager.config.hf_model_repos,
            ms_model_repos=config_manager.config.ms_model_repos,
        )
        self._port_mgr = PortManager(config_manager)
        self._model_svc = ModelService(config_manager)
        logs_dir = Path(__file__).parent / "logs"
        logs_dir.mkdir(exist_ok=True)
        self._proc_mgr = ProcessManager(config_manager, self._monitor, logs_dir, port_manager=self._port_mgr)
        self._http_client = httpx.AsyncClient(timeout=10)

    # ---- properties ---------------------------------------------------------

    @property
    def monitor(self) -> ResourceMonitor:
        return self._monitor

    @property
    def http_client(self) -> httpx.AsyncClient:
        return self._http_client

    @property
    def downloader(self) -> ModelDownloader:
        return self._downloader

    # ---- model operations (delegated) ---------------------------------------

    def list_models(self, backend_type: BackendType | None = None) -> list[dict[str, Any]]:
        return self._model_svc.list_models(backend_type)

    def get_model_info(self, name: str) -> dict[str, Any] | None:
        return self._model_svc.get_model_info(name)

    def delete_model(self, name: str) -> bool:
        return self._model_svc.delete_model(name)

    # ---- instance operations (delegated) ------------------------------------

    async def start_instance(self, req: InstanceStartRequest) -> InstanceInfo:
        cfg = self._config.config
        if len(self._proc_mgr.instances) >= cfg.max_instances:
            raise RuntimeError(f"Max instances ({cfg.max_instances}) reached")

        port = req.port or self._port_mgr.find_available()
        if req.port is None or port != req.port:
            self._port_mgr.acquire(port)

        preset = self._config.get_preset(req.preset) if req.preset else None
        host = req.host or cfg.default_host
        extra_args = req.extra_args or (preset.extra_args if preset else [])

        try:
            return await self._proc_mgr.start(req, port, host, extra_args, preset)
        except Exception:
            self._port_mgr.release(port)
            raise

    async def stop_instance(self, inst_id: str) -> bool:
        inst = self._proc_mgr.instances.get(inst_id)
        if inst:
            self._port_mgr.release(inst.info.port)
        return await self._proc_mgr.stop(inst_id)

    async def restart_instance(self, inst_id: str) -> InstanceInfo:
        inst = self._proc_mgr.instances.get(inst_id)
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
        port = inst.info.port
        host = inst.info.host
        extra_args = inst.info.extra_args
        await self._proc_mgr.kill(inst)
        del self._proc_mgr.instances[inst_id]
        # 端口不释放，直接复用，避免竞争条件
        try:
            return await self._proc_mgr.start(old_req, port, host, extra_args)
        except Exception:
            self._port_mgr.release(port)
            raise

    def list_instances(self) -> list[InstanceInfo]:
        return self._proc_mgr.list_all()

    def get_instance(self, inst_id: str) -> InstanceInfo | None:
        return self._proc_mgr.get_info(inst_id)

    def get_instance_logs(self, inst_id: str, lines: int = 100) -> list[str]:
        return self._proc_mgr.get_logs(inst_id, lines)

    # ---- shutdown -----------------------------------------------------------

    async def shutdown(self) -> None:
        for inst_id in list(self._proc_mgr.instances.keys()):
            await self.stop_instance(inst_id)
        await self._http_client.aclose()
        logger.info("All instances stopped")
