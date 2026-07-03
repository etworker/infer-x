"""Enumerations for backend types, instance status, download sources."""

from __future__ import annotations

from enum import Enum


class BackendType(str, Enum):
    llamacpp = "llamacpp"
    vllm = "vllm"
    sglang = "sglang"
    tgi = "tgi"
    ollama = "ollama"
    tensorrt_llm = "tensorrt_llm"
    lmdeploy = "lmdeploy"
    openvino = "openvino"


class InstanceStatus(str, Enum):
    starting = "starting"
    running = "running"
    stopping = "stopping"
    stopped = "stopped"
    error = "error"


class DownloadSource(str, Enum):
    huggingface = "hf"
    modelscope = "ms"
    hf_mirror = "hf_mirror"
    url = "url"


class DownloadStatus(str, Enum):
    pending = "pending"
    downloading = "downloading"
    completed = "completed"
    failed = "failed"
