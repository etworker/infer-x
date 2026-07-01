"""Tests for backend implementations."""

import pytest
from pathlib import Path
from infer_helper.backends import get_backend, check_backend_installed
from infer_helper.backends.llamacpp import LlamaCppBackend
from infer_helper.backends.vllm import VLLMBackend
from infer_helper.backends.sglang import SGLangBackend


class TestLlamaCppBackend:
    def setup_method(self):
        self.backend = LlamaCppBackend()

    def test_build_command_minimal(self):
        cmd = self.backend.build_command(
            model_path="/path/to/model.gguf",
            port=8080,
            host="0.0.0.0",
            log_file="/tmp/test.log",
            params={"binary": "llama-server"},
            extra_args=[],
        )
        assert cmd[0] == "llama-server"
        assert "-m" in cmd
        assert "/path/to/model.gguf" in cmd
        assert "--port" in cmd
        assert "8080" in cmd

    def test_build_command_with_options(self):
        cmd = self.backend.build_command(
            model_path="/path/to/model.gguf",
            port=8080,
            host="0.0.0.0",
            log_file="/tmp/test.log",
            params={
                "binary": "llama-server",
                "ctx_size": 8192,
                "n_gpu_layers": 32,
                "threads": 8,
            },
            extra_args=[],
        )
        assert "-c" in cmd
        assert "8192" in cmd
        assert "-ngl" in cmd
        assert "32" in cmd
        assert "-t" in cmd
        assert "8" in cmd

    def test_get_model_paths(self, tmp_path):
        # Create test GGUF file
        model_file = tmp_path / "test-model.gguf"
        model_file.touch()
        
        models = self.backend.get_model_paths(tmp_path)
        assert len(models) == 1
        assert models[0]["name"] == "test-model.gguf"

    def test_guess_family(self):
        assert self.backend._guess_family("qwen3-8b") == "qwen"
        assert self.backend._guess_family("llama-7b") == "llama"
        assert self.backend._guess_family("mistral-7b") == "mistral"
        assert self.backend._guess_family("unknown-model") is None


class TestVLLMBackend:
    def setup_method(self):
        self.backend = VLLMBackend()

    def test_build_command_minimal(self):
        cmd = self.backend.build_command(
            model_path="/path/to/model",
            port=8080,
            host="0.0.0.0",
            log_file="/tmp/test.log",
            params={"binary": "python -m vllm.entrypoints.openai.api_server"},
            extra_args=[],
        )
        assert cmd[0] == "python"
        assert "-m" in cmd
        assert "vllm.entrypoints.openai.api_server" in cmd
        assert "--model" in cmd

    def test_build_command_with_tp(self):
        cmd = self.backend.build_command(
            model_path="/path/to/model",
            port=8080,
            host="0.0.0.0",
            log_file="/tmp/test.log",
            params={
                "binary": "python -m vllm.entrypoints.openai.api_server",
                "tensor_parallel_size": 2,
            },
            extra_args=[],
        )
        assert "--tensor-parallel-size" in cmd
        assert "2" in cmd


class TestCheckBackendInstalled:
    def test_check_llamacpp(self):
        # This test depends on the system
        result = check_backend_installed("llamacpp")
        assert isinstance(result, bool)

    def test_check_unknown_backend(self):
        result = check_backend_installed("unknown_backend")
        assert result is False
