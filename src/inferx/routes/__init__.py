"""Route dependency injection and sub-router aggregation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter

from ..config import ConfigManager
from ..manager import InstanceManager
from ..monitoring import AlertManager, AuditLogger, UsageTracker

# Shared state — initialized once in init_routes()
_state: dict[str, Any] = {}


def init_routes(config: ConfigManager, manager: InstanceManager) -> None:
    """Initialize all shared state and register sub-routers."""
    data_dir = Path(__file__).parent.parent / "data"
    data_dir.mkdir(exist_ok=True)

    _state["config"] = config
    _state["manager"] = manager
    _state["alert_manager"] = AlertManager(data_dir)
    _state["usage_tracker"] = UsageTracker(data_dir)
    _state["audit_logger"] = AuditLogger(data_dir)

    # Initialize benchmark manager
    from .benchmark import _bench
    _bench().set_manager(manager)


def get_config() -> ConfigManager:
    return _state["config"]


def get_manager() -> InstanceManager:
    return _state["manager"]


def get_alert_manager() -> AlertManager:
    return _state["alert_manager"]


def get_usage_tracker() -> UsageTracker:
    return _state["usage_tracker"]


def get_audit_logger() -> AuditLogger:
    return _state["audit_logger"]


def create_api_router() -> APIRouter:
    """Create and return the combined API router with all sub-routers."""
    from .system import router as system_router
    from .models import router as models_router
    from .instances import router as instances_router
    from .presets import router as presets_router
    from .monitoring import router as monitoring_router
    from .benchmark import router as benchmark_router

    api = APIRouter(prefix="/api")
    api.include_router(system_router)
    api.include_router(models_router)
    api.include_router(instances_router)
    api.include_router(presets_router)
    api.include_router(monitoring_router)
    api.include_router(benchmark_router)
    return api
