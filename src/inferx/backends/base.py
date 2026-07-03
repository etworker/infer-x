"""Abstract base class for inference backends."""

from __future__ import annotations

import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from ..utils import guess_family, guess_quantization


class Backend(ABC):
    """Abstract base class for inference backends."""

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
    def get_env(self, binary_path: str) -> dict[str, str]:
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


def get_backend(backend_type: str) -> Backend:
    """Factory function to get backend instance."""
    from .llamacpp import LlamaCppBackend
    from .lmdeploy import LMDeployBackend
    from .ollama import OllamaBackend
    from .openvino import OpenVINOBackend
    from .sglang import SGLangBackend
    from .tensorrt_llm import TensorRTLLMBackend
    from .tgi import TGIBackend
    from .vllm import VLLMBackend

    backends = {
        "llamacpp": LlamaCppBackend,
        "vllm": VLLMBackend,
        "sglang": SGLangBackend,
        "tgi": TGIBackend,
        "ollama": OllamaBackend,
        "tensorrt_llm": TensorRTLLMBackend,
        "lmdeploy": LMDeployBackend,
        "openvino": OpenVINOBackend,
    }

    backend_cls = backends.get(backend_type)
    if not backend_cls:
        raise ValueError(f"Unknown backend: {backend_type}")
    return backend_cls()


_installed_cache: dict[str, tuple[bool, float]] = {}


def check_backend_installed(backend_type: str) -> bool:
    """Check if a backend is installed on the system (cached for 60s)."""
    import os
    import subprocess
    import time

    now = time.monotonic()
    cached = _installed_cache.get(backend_type)
    if cached and now - cached[1] < 60:
        return cached[0]

    result = False
    try:
        if backend_type == "llamacpp":
            if shutil.which("llama-server"):
                result = True
            else:
                env_bin = os.environ.get("LLAMA_SERVER_BIN")
                if env_bin and Path(env_bin).exists():
                    result = True
                else:
                    home = Path.home()
                    common_paths = [
                        home / "llama.cpp" / "build" / "bin" / "llama-server",
                        Path("/usr/local/bin/llama-server"),
                        Path("/usr/bin/llama-server"),
                    ]
                    result = any(p.exists() for p in common_paths)

        elif backend_type == "vllm":
            try:
                import importlib
                importlib.import_module("vllm")
                result = True
            except ImportError:
                result = False

        elif backend_type == "sglang":
            try:
                import importlib
                importlib.import_module("sglang")
                result = True
            except ImportError:
                result = False

        elif backend_type == "tgi":
            proc = subprocess.run(
                ["docker", "images", "-q", "ghcr.io/huggingface/text-generation-inference"],
                capture_output=True, text=True, timeout=5
            )
            result = bool(proc.stdout.strip())

        elif backend_type == "ollama":
            result = shutil.which("ollama") is not None

        elif backend_type == "tensorrt_llm":
            try:
                import importlib
                importlib.import_module("tensorrt_llm")
                result = True
            except ImportError:
                result = False

        elif backend_type == "lmdeploy":
            try:
                import importlib
                importlib.import_module("lmdeploy")
                result = True
            except ImportError:
                result = False

        elif backend_type == "openvino":
            try:
                import importlib
                importlib.import_module("openvino")
                result = True
            except ImportError:
                result = False

    except Exception:
        result = False

    _installed_cache[backend_type] = (result, now)
    return result


def get_all_backends_status() -> list[dict[str, Any]]:
    """Get status of all backends."""
    backends_info = [
        {
            "id": "llamacpp",
            "name": "llama.cpp",
            "description": "Local GGUF model inference, lightweight",
            "model_types": ["gguf"],
            "check_type": "binary",
        },
        {
            "id": "vllm",
            "name": "vLLM",
            "description": "High-performance LLM serving with PagedAttention",
            "model_types": ["huggingface", "safetensors"],
            "check_type": "python_module",
        },
        {
            "id": "sglang",
            "name": "SGLang",
            "description": "High-performance LLM framework with RadixAttention",
            "model_types": ["huggingface", "safetensors"],
            "check_type": "python_module",
        },
        {
            "id": "tgi",
            "name": "TGI",
            "description": "HuggingFace Text Generation Inference, production-grade",
            "model_types": ["huggingface", "safetensors"],
            "check_type": "docker_image",
        },
        {
            "id": "ollama",
            "name": "Ollama",
            "description": "User-friendly local LLM runner",
            "model_types": ["ollama"],
            "check_type": "binary",
        },
        {
            "id": "tensorrt_llm",
            "name": "TensorRT-LLM",
            "description": "NVIDIA optimized inference for maximum performance",
            "model_types": ["tensorrt_engine"],
            "check_type": "python_module",
        },
        {
            "id": "lmdeploy",
            "name": "LMDeploy",
            "description": "Shanghai AI Lab inference, optimized for Chinese models",
            "model_types": ["huggingface", "safetensors"],
            "check_type": "python_module",
        },
        {
            "id": "openvino",
            "name": "OpenVINO",
            "description": "Intel optimized inference for Intel hardware",
            "model_types": ["openvino"],
            "check_type": "python_module",
        },
    ]

    for backend in backends_info:
        backend["installed"] = check_backend_installed(backend["id"])

    return backends_info
