"""Simple in-memory TTL cache for expensive operations."""

from __future__ import annotations

import time
from threading import Lock
from typing import Any


class TTLCache:
    """In-memory cache with per-key TTL."""

    def __init__(self, default_ttl: float = 5.0):
        self._default_ttl = default_ttl
        self._store: dict[str, tuple[Any, float]] = {}
        self._lock = Lock()

    def get(self, key: str) -> Any | None:
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            value, expires_at = entry
            if time.monotonic() > expires_at:
                del self._store[key]
                return None
            return value

    def set(self, key: str, value: Any, ttl: float | None = None) -> None:
        with self._lock:
            self._store[key] = (value, time.monotonic() + (ttl if ttl is not None else self._default_ttl))

    def invalidate(self, key: str) -> None:
        with self._lock:
            self._store.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
