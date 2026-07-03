"""Backend registry — pluggable registration mechanism."""

from __future__ import annotations

import time
from typing import Any

from ..models import BackendType


class _BackendRegistry:
    def __init__(self) -> None:
        self._backends: dict[BackendType, type] = {}
        self._installed_cache: dict[str, tuple[bool, float]] = {}

    def register(self, backend_type: BackendType):
        """Decorator to register a backend class."""
        def decorator(cls):
            self._backends[backend_type] = cls
            return cls
        return decorator

    def _resolve_type(self, backend_type: BackendType | str) -> BackendType:
        """Resolve string to BackendType, or pass through."""
        if isinstance(backend_type, str):
            try:
                backend_type = BackendType(backend_type)
            except ValueError:
                raise ValueError(f"Unknown backend: {backend_type}")
        return backend_type

    def get(self, backend_type: BackendType | str):
        """Get a backend instance by type."""
        bt = self._resolve_type(backend_type)
        cls = self._backends.get(bt)
        if not cls:
            raise ValueError(f"Unknown backend: {bt}")
        return cls()

    def get_class(self, backend_type: BackendType | str) -> type:
        """Get backend class (not instance) by type."""
        bt = self._resolve_type(backend_type)
        cls = self._backends.get(bt)
        if not cls:
            raise ValueError(f"Unknown backend: {bt}")
        return cls

    def is_installed(self, backend_type: BackendType | str, *, force: bool = False) -> bool:
        """Check if a backend is installed (cached for 60s)."""
        try:
            bt = self._resolve_type(backend_type)
        except ValueError:
            return False

        now = time.monotonic()
        cached = self._installed_cache.get(bt.value)
        if not force and cached and now - cached[1] < 60:
            return cached[0]

        try:
            cls = self.get_class(bt)
            result = cls.is_installed()
        except Exception:
            result = False

        self._installed_cache[bt.value] = (result, now)
        return result

    def get_all_status(self) -> list[dict[str, Any]]:
        """Get installation status of all registered backends."""
        result = []
        for bt, cls in self._backends.items():
            installed = self.is_installed(bt)
            result.append({
                "id": bt.value,
                "name": getattr(cls, "backend_name", bt.value),
                "description": getattr(cls, "description", ""),
                "model_types": getattr(cls, "model_types", []),
                "check_type": getattr(cls, "check_type", "unknown"),
                "installed": installed,
            })
        return result

    def get_binary_config_attr(self, backend_type: BackendType | str) -> str | None:
        """Get the config field name for the binary path of a backend."""
        if isinstance(backend_type, str):
            backend_type = BackendType(backend_type)
        cls = self._backends.get(backend_type)
        if cls:
            return getattr(cls, "binary_config_attr", None)
        return None

    def get_all_binary_attrs(self) -> dict[str, str]:
        """Get a dict mapping backend ID -> config field name for all backends."""
        return {
            bt.value: attr
            for bt, cls in self._backends.items()
            if (attr := getattr(cls, "binary_config_attr", None))
        }

    @property
    def backends(self) -> dict[BackendType, type]:
        return dict(self._backends)


# Module-level singleton
registry = _BackendRegistry()
register_backend = registry.register
