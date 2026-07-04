"""Abstract base class for inference backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from ..utils import guess_family, guess_quantization


class Backend(ABC):
    """Abstract base class for inference backends.

    Subclasses should set these class-level metadata attributes:
      backend_id, backend_name, description, model_types, check_type, binary_config_attr
    """

    backend_id: str = ""
    backend_name: str = ""
    description: str = ""
    model_types: list[str] = []
    check_type: str = "unknown"
    binary_config_attr: str = ""

    @abstractmethod
    def build_command(
        self,
        model_path: str,
        port: int,
        host: str,
        log_file: str,
        params: dict[str, Any],
        extra_args: list[str],
    ) -> list[str]:
        """Build the command line arguments for the inference server."""
        pass

    @abstractmethod
    def get_env(self, binary_path: str, host: str = "localhost", port: int = 8080) -> dict[str, str]:
        """Get environment variables needed for the backend."""
        pass

    @abstractmethod
    def get_model_paths(self, model_dir: Path) -> list[dict[str, Any]]:
        """Discover available models in the model directory."""
        pass

    @staticmethod
    def _guess_family(name: str) -> str | None:
        return guess_family(name)

    @staticmethod
    def _guess_quantization(name: str) -> str | None:
        return guess_quantization(name)

    @classmethod
    @abstractmethod
    def is_installed(cls) -> bool:
        """Check if this backend is installed on the system."""
        pass


from .registry import registry

get_backend = registry.get
check_backend_installed = registry.is_installed
get_all_backends_status = registry.get_all_status
