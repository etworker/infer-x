"""Abstract base class for inference backends."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from ..utils import guess_family, guess_quantization


class Backend(ABC):
    """Abstract base class for inference backends.

    Subclasses set these metadata as class attributes:
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
        pass

    @abstractmethod
    def get_env(self, binary_path: str) -> dict[str, str]:
        pass

    @abstractmethod
    def get_model_paths(self, model_dir: Path) -> list[dict[str, Any]]:
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
        pass


# ---- Simple registry (dict-based, no decorator magic) ----

_backends: dict[str, type[Backend]] = {}
_installed_cache: dict[str, tuple[bool, float]] = {}


def register_backend(cls: type[Backend]) -> type[Backend]:
    _backends[cls.backend_id] = cls
    return cls


def get_backend(backend_type: str) -> Backend:
    # Trigger imports so backends register themselves
    from .llamacpp import LlamaCppBackend  # noqa: F401
    from .lmdeploy import LMDeployBackend  # noqa: F401
    from .ollama import OllamaBackend  # noqa: F401
    from .openvino import OpenVINOBackend  # noqa: F401
    from .sglang import SGLangBackend  # noqa: F401
    from .tensorrt_llm import TensorRTLLMBackend  # noqa: F401
    from .tgi import TGIBackend  # noqa: F401
    from .vllm import VLLMBackend  # noqa: F401

    cls = _backends.get(backend_type)
    if not cls:
        raise ValueError(f"Unknown backend: {backend_type}")
    return cls()


def check_backend_installed(backend_type: str) -> bool:
    now = time.monotonic()
    cached = _installed_cache.get(backend_type)
    if cached and now - cached[1] < 60:
        return cached[0]
    try:
        backend = get_backend(backend_type)
        result = backend.is_installed()
    except Exception:
        result = False
    _installed_cache[backend_type] = (result, now)
    return result


def get_all_backends_status() -> list[dict[str, Any]]:
    from .llamacpp import LlamaCppBackend  # noqa: F401
    from .lmdeploy import LMDeployBackend  # noqa: F401
    from .ollama import OllamaBackend  # noqa: F401
    from .openvino import OpenVINOBackend  # noqa: F401
    from .sglang import SGLangBackend  # noqa: F401
    from .tensorrt_llm import TensorRTLLMBackend  # noqa: F401
    from .tgi import TGIBackend  # noqa: F401
    from .vllm import VLLMBackend  # noqa: F401

    result = []
    for cls in _backends.values():
        installed = check_backend_installed(cls.backend_id)
        result.append({
            "id": cls.backend_id,
            "name": cls.backend_name,
            "description": cls.description,
            "model_types": cls.model_types,
            "check_type": cls.check_type,
            "installed": installed,
        })
    return result
