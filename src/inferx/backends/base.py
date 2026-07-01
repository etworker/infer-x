"""Abstract base class for inference backends."""

from __future__ import annotations

import re
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional


class Backend(ABC):
    """Abstract base class for inference backends."""

    @abstractmethod
    def build_command(
        self,
        model_path: str,
        port: int,
        host: str,
        log_file: str,
        params: Dict[str, Any],
        extra_args: List[str],
    ) -> List[str]:
        """Build the command line arguments for the inference server."""
        pass

    @abstractmethod
    def get_env(self, binary_path: str) -> Dict[str, str]:
        """Get environment variables needed for the backend."""
        pass

    @abstractmethod
    def get_model_paths(self, model_dir: Path) -> List[Dict[str, Any]]:
        """Discover available models in the model directory."""
        pass

    @staticmethod
    def _guess_family(name: str) -> Optional[str]:
        """Guess model family from filename."""
        name_lower = name.lower()
        for family in ["qwen", "gemma", "llama", "mistral", "phi", "deepseek", "yi", "baichuan"]:
            if family in name_lower:
                return family
        return None

    @staticmethod
    def _guess_quantization(name: str) -> Optional[str]:
        """Guess quantization from filename."""
        m = re.search(r"(Q[0-9]+_[A-Z0-9]+|F16|F32|BF16|IQ[0-9]+_[A-Z0-9]+)", name, re.IGNORECASE)
        return m.group(1).upper() if m else None


def get_backend(backend_type: str) -> Backend:
    """Factory function to get backend instance."""
    from .llamacpp import LlamaCppBackend
    from .vllm import VLLMBackend
    from .sglang import SGLangBackend
    from .tgi import TGIBackend
    from .ollama import OllamaBackend
    from .tensorrt_llm import TensorRTLLMBackend
    from .lmdeploy import LMDeployBackend
    from .openvino import OpenVINOBackend

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


def check_backend_installed(backend_type: str) -> bool:
    """Check if a backend is installed on the system."""
    import os
    import subprocess

    try:
        if backend_type == "llamacpp":
            if shutil.which("llama-server"):
                return True
            env_bin = os.environ.get("LLAMA_SERVER_BIN")
            if env_bin and Path(env_bin).exists():
                return True
            home = Path.home()
            common_paths = [
                home / "llama.cpp" / "build" / "bin" / "llama-server",
                Path("/usr/local/bin/llama-server"),
                Path("/usr/bin/llama-server"),
            ]
            return any(p.exists() for p in common_paths)

        elif backend_type == "vllm":
            try:
                import importlib
                importlib.import_module("vllm")
                return True
            except ImportError:
                return False

        elif backend_type == "sglang":
            try:
                import importlib
                importlib.import_module("sglang")
                return True
            except ImportError:
                return False

        elif backend_type == "tgi":
            result = subprocess.run(
                ["docker", "images", "-q", "ghcr.io/huggingface/text-generation-inference"],
                capture_output=True, text=True, timeout=5
            )
            return bool(result.stdout.strip())

        elif backend_type == "ollama":
            return shutil.which("ollama") is not None

        elif backend_type == "tensorrt_llm":
            try:
                import importlib
                importlib.import_module("tensorrt_llm")
                return True
            except ImportError:
                return False

        elif backend_type == "lmdeploy":
            try:
                import importlib
                importlib.import_module("lmdeploy")
                return True
            except ImportError:
                return False

        elif backend_type == "openvino":
            try:
                import importlib
                importlib.import_module("openvino")
                return True
            except ImportError:
                return False

    except Exception:
        return False

    return False


def get_all_backends_status() -> List[Dict[str, Any]]:
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
