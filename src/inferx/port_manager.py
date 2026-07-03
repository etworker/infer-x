"""Port allocation and release for backend instances."""

from __future__ import annotations

import socket

from .config import ConfigManager


class PortManager:
    def __init__(self, config: ConfigManager):
        self._config = config
        self._ports_in_use: set[int] = set()

    def find_available(self) -> int:
        cfg = self._config.config
        for port in range(cfg.port_range_start, cfg.port_range_end + 1):
            if port in self._ports_in_use:
                continue
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind((cfg.default_host, port))
                    return port
                except OSError:
                    continue
        raise RuntimeError("No available ports in configured range")

    def acquire(self, port: int) -> None:
        self._ports_in_use.add(port)

    def release(self, port: int) -> None:
        self._ports_in_use.discard(port)
