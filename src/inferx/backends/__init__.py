"""Backend implementations for different inference engines."""

from .base import Backend, check_backend_installed, get_all_backends_status, get_backend
from .registry import registry

# Import backends to trigger registration
from .llamacpp import LlamaCppBackend  # noqa: F401
from .lmdeploy import LMDeployBackend  # noqa: F401
from .ollama import OllamaBackend  # noqa: F401
from .openvino import OpenVINOBackend  # noqa: F401
from .sglang import SGLangBackend  # noqa: F401
from .tensorrt_llm import TensorRTLLMBackend  # noqa: F401
from .tgi import TGIBackend  # noqa: F401
from .vllm import VLLMBackend  # noqa: F401

__all__ = [
    "Backend",
    "get_backend",
    "check_backend_installed",
    "get_all_backends_status",
    "registry",
    "LlamaCppBackend",
    "VLLMBackend",
    "SGLangBackend",
    "TGIBackend",
    "OllamaBackend",
    "TensorRTLLMBackend",
    "LMDeployBackend",
    "OpenVINOBackend",
]
