"""FastAPI route definitions — thin aggregator for sub-routers."""

from __future__ import annotations

from fastapi import APIRouter

from .config import ConfigManager
from .manager import InstanceManager
from .routes import create_api_router, init_routes

# Re-export for backward compatibility
router = create_api_router()

__all__ = ["router", "init_routes"]
