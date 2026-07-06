"""Route dependency injection and sub-router aggregation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter

from ..config import ConfigManager
from ..logging import logger
from ..manager import InstanceManager

# Shared state — initialized once in init_routes()
_state: dict[str, Any] = {}


def init_routes(config: ConfigManager, manager: InstanceManager) -> None:
    """Initialize all shared state and register sub-routers."""
    _state["config"] = config
    _state["manager"] = manager

    # Initialize benchmark manager
    from .benchmark import _bench
    _bench().set_manager(manager)


def get_config() -> ConfigManager:
    return _state["config"]


def get_manager() -> InstanceManager:
    return _state["manager"]


def audit_log(action: str, resource: str, resource_id: str, success: bool = True, **extra: Any) -> None:
    """Inline audit log entry."""
    status = "OK" if success else "FAIL"
    extra_str = f" {extra}" if extra else ""
    logger.info("[AUDIT] {} {} {} {}{}", action, resource, resource_id, status, extra_str)


def create_api_router() -> APIRouter:
    """Create and return the combined API router with all sub-routers."""
    from .system import router as system_router
    from .models import router as models_router
    from .instances import router as instances_router
    from .presets import router as presets_router
    from .benchmark import router as benchmark_router

    api = APIRouter(prefix="/api")
    api.include_router(system_router)
    api.include_router(models_router)
    api.include_router(instances_router)
    api.include_router(presets_router)
    api.include_router(benchmark_router)
    return api
