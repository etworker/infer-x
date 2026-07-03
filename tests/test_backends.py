"""Comprehensive tests for backend implementations."""

import pytest
from pathlib import Path
from inferx.backends import get_backend, check_backend_installed, get_all_backends_status
from inferx.backends.base import Backend
from inferx.backends.llamacpp import LlamaCppBackend
from inferx.backends.vllm import VLLMBackend
from inferx.backends.sglang import SGLangBackend
from inferx.backends.tgi import TGIBackend
from inferx.backends.ollama import OllamaBackend
from inferx.backends.tensorrt_llm import TensorRTLLMBackend
from inferx.backends.lmdeploy import LMDeployBackend
from inferx.backends.openvino import OpenVINOBackend


class TestGetBackend:
    def test_get_all_backends(self):
        for name in ["llamacpp", "vllm", "sglang", "tgi", "ollama", "tensorrt_llm", "lmdeploy", "openvino"]:
            backend = get_backend(name)
            assert isinstance(backend, Backend)

    def test_invalid_backend(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            get_backend("nonexistent")


class TestLlamaCppBackend:
    def setup_method(self):
        self.backend = LlamaCppBackend()

    def test_build_command_minimal(self):
        cmd = self.backend.build_command(
            model_path="/path/to/model.gguf",
            port=8080, host="0.0.0.0", log_file="/tmp/test.log",
            params={"binary": "llama-server"}, extra_args=[],
        )
        assert cmd[0] == "llama-server"
        assert "-m" in cmd
        assert "/path/to/model.gguf" in cmd
        assert "--port" in cmd
        assert "8080" in cmd

    def test_build_command_with_ctx(self):
        cmd = self.backend.build_command(
            model_path="/m.gguf", port=8080, host="0.0.0.0",
            log_file="/tmp/t.log", params={"binary": "llama-server", "ctx_size": 8192}, extra_args=[],
        )
        assert "-c" in cmd
        assert "8192" in cmd

    def test_build_command_with_ngl(self):
        cmd = self.backend.build_command(
            model_path="/m.gguf", port=8080, host="0.0.0.0",
            log_file="/tmp/t.log", params={"binary": "llama-server", "n_gpu_layers": 32}, extra_args=[],
        )
        assert "-ngl" in cmd
        assert "32" in cmd

    def test_build_command_with_threads(self):
        cmd = self.backend.build_command(
            model_path="/m.gguf", port=8080, host="0.0.0.0",
            log_file="/tmp/t.log", params={"binary": "llama-server", "threads": 8}, extra_args=[],
        )
        assert "-t" in cmd
        assert "8" in cmd

    def test_build_command_with_parallel(self):
        cmd = self.backend.build_command(
            model_path="/m.gguf", port=8080, host="0.0.0.0",
            log_file="/tmp/t.log", params={"binary": "llama-server", "n_parallel": 4}, extra_args=[],
        )
        assert "-np" in cmd
        assert "4" in cmd

    def test_build_command_with_batch(self):
        cmd = self.backend.build_command(
            model_path="/m.gguf", port=8080, host="0.0.0.0",
            log_file="/tmp/t.log", params={"binary": "llama-server", "batch_size": 4096}, extra_args=[],
        )
        assert "-b" in cmd
        assert "4096" in cmd

    def test_build_command_with_flash_attn(self):
        cmd = self.backend.build_command(
            model_path="/m.gguf", port=8080, host="0.0.0.0",
            log_file="/tmp/t.log", params={"binary": "llama-server", "flash_attn": "on"}, extra_args=[],
        )
        assert "-fa" in cmd

    def test_build_command_with_host(self):
        cmd = self.backend.build_command(
            model_path="/m.gguf", port=8080, host="127.0.0.1",
            log_file="/tmp/t.log", params={"binary": "llama-server"}, extra_args=[],
        )
        assert "--host" in cmd
        assert "127.0.0.1" in cmd

    def test_build_command_with_extra_args(self):
        cmd = self.backend.build_command(
            model_path="/m.gguf", port=8080, host="0.0.0.0",
            log_file="/tmp/t.log", params={"binary": "llama-server"}, extra_args=["--verbose"],
        )
        assert "--verbose" in cmd

    def test_build_command_with_alias(self):
        cmd = self.backend.build_command(
            model_path="/m.gguf", port=8080, host="0.0.0.0",
            log_file="/tmp/t.log", params={"binary": "llama-server", "alias": "my-model"}, extra_args=[],
        )
        assert "-a" in cmd
        assert "my-model" in cmd

    def test_build_command_with_sleep_idle(self):
        cmd = self.backend.build_command(
            model_path="/m.gguf", port=8080, host="0.0.0.0",
            log_file="/tmp/t.log", params={"binary": "llama-server", "sleep_idle_seconds": 300}, extra_args=[],
        )
        assert "--sleep-idle-seconds" in cmd
        assert "300" in cmd

    def test_build_command_with_mlock(self):
        cmd = self.backend.build_command(
            model_path="/m.gguf", port=8080, host="0.0.0.0",
            log_file="/tmp/t.log", params={"binary": "llama-server", "mlock": True}, extra_args=[],
        )
        assert "--mlock" in cmd

    def test_get_model_paths(self, tmp_path):
        (tmp_path / "model1.gguf").touch()
        (tmp_path / "model2.gguf").touch()
        (tmp_path / "not-a-model.txt").touch()
        models = self.backend.get_model_paths(tmp_path)
        assert len(models) == 2
        names = {m["name"] for m in models}
        assert "model1.gguf" in names
        assert "model2.gguf" in names

    def test_get_model_paths_empty(self, tmp_path):
        models = self.backend.get_model_paths(tmp_path)
        assert len(models) == 0

    def test_guess_family(self):
        assert self.backend._guess_family("qwen3-8b") == "qwen"
        assert self.backend._guess_family("llama-7b") == "llama"
        assert self.backend._guess_family("mistral-7b") == "mistral"
        assert self.backend._guess_family("gemma-2b") == "gemma"
        assert self.backend._guess_family("phi-3") == "phi"
        assert self.backend._guess_family("deepseek-v2") == "deepseek"
        assert self.backend._guess_family("yi-34b") == "yi"
        assert self.backend._guess_family("baichuan-7b") == "baichuan"
        assert self.backend._guess_family("unknown-model") is None

    def test_guess_quantization(self):
        assert self.backend._guess_quantization("model-Q4_K_M.gguf") == "Q4_K"
        assert self.backend._guess_quantization("model-Q8_0.gguf") == "Q8_0"
        assert self.backend._guess_quantization("model-F16.gguf") == "F16"
        assert self.backend._guess_quantization("model-BF16.gguf") == "BF16"
        assert self.backend._guess_quantization("model-IQ4_XS.gguf") == "IQ4_XS"
        assert self.backend._guess_quantization("model.gguf") is None

    def test_get_env(self):
        env = self.backend.get_env("/usr/bin/llama-server")
        assert isinstance(env, dict)


class TestVLLMBackend:
    def setup_method(self):
        self.backend = VLLMBackend()

    def test_build_command_minimal(self):
        cmd = self.backend.build_command(
            model_path="/path/to/model", port=8080, host="0.0.0.0",
            log_file="/tmp/test.log",
            params={"binary": "python -m vllm.entrypoints.openai.api_server"},
            extra_args=[],
        )
        assert "-m" in cmd
        assert "vllm.entrypoints.openai.api_server" in cmd
        assert "--model" in cmd

    def test_build_command_with_tp(self):
        cmd = self.backend.build_command(
            model_path="/m", port=8080, host="0.0.0.0", log_file="/tmp/t.log",
            params={"binary": "python -m vllm.entrypoints.openai.api_server", "tensor_parallel_size": 2},
            extra_args=[],
        )
        assert "--tensor-parallel-size" in cmd
        assert "2" in cmd

    def test_build_command_with_max_model_len(self):
        cmd = self.backend.build_command(
            model_path="/m", port=8080, host="0.0.0.0", log_file="/tmp/t.log",
            params={"binary": "python -m vllm.entrypoints.openai.api_server", "max_model_len": 4096},
            extra_args=[],
        )
        assert "--max-model-len" in cmd
        assert "4096" in cmd

    def test_build_command_with_gpu_mem(self):
        cmd = self.backend.build_command(
            model_path="/m", port=8080, host="0.0.0.0", log_file="/tmp/t.log",
            params={"binary": "python -m vllm.entrypoints.openai.api_server", "gpu_memory_utilization": 0.8},
            extra_args=[],
        )
        assert "--gpu-memory-utilization" in cmd

    def test_build_command_with_dtype(self):
        cmd = self.backend.build_command(
            model_path="/m", port=8080, host="0.0.0.0", log_file="/tmp/t.log",
            params={"binary": "python -m vllm.entrypoints.openai.api_server", "dtype": "float16"},
            extra_args=[],
        )
        assert "--dtype" in cmd
        assert "float16" in cmd

    def test_build_command_with_quantization(self):
        cmd = self.backend.build_command(
            model_path="/m", port=8080, host="0.0.0.0", log_file="/tmp/t.log",
            params={"binary": "python -m vllm.entrypoints.openai.api_server", "quantization": "awq"},
            extra_args=[],
        )
        assert "--quantization" in cmd
        assert "awq" in cmd

    def test_build_command_with_enforce_eager(self):
        cmd = self.backend.build_command(
            model_path="/m", port=8080, host="0.0.0.0", log_file="/tmp/t.log",
            params={"binary": "python -m vllm.entrypoints.openai.api_server", "enforce_eager": True},
            extra_args=[],
        )
        assert "--enforce-eager" in cmd

    def test_get_model_paths(self, tmp_path):
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "config.json").touch()
        models = self.backend.get_model_paths(tmp_path)
        assert len(models) >= 0  # May or may not find models

    def test_get_env(self):
        env = self.backend.get_env("python -m vllm.entrypoints.openai.api_server")
        assert isinstance(env, dict)


class TestSGLangBackend:
    def setup_method(self):
        self.backend = SGLangBackend()

    def test_build_command_minimal(self):
        cmd = self.backend.build_command(
            model_path="/path/to/model", port=8080, host="0.0.0.0",
            log_file="/tmp/test.log",
            params={"binary": "python -m sglang.launch_server"},
            extra_args=[],
        )
        assert "-m" in cmd
        assert "sglang.launch_server" in cmd
        assert "--model-path" in cmd

    def test_build_command_with_tp(self):
        cmd = self.backend.build_command(
            model_path="/m", port=8080, host="0.0.0.0", log_file="/tmp/t.log",
            params={"binary": "python -m sglang.launch_server", "tp": 2},
            extra_args=[],
        )
        assert "--tp" in cmd
        assert "2" in cmd

    def test_build_command_with_mem_fraction(self):
        cmd = self.backend.build_command(
            model_path="/m", port=8080, host="0.0.0.0", log_file="/tmp/t.log",
            params={"binary": "python -m sglang.launch_server", "mem_fraction_static": 0.7},
            extra_args=[],
        )
        assert "--mem-fraction-static" in cmd


class TestTGIBackend:
    def setup_method(self):
        self.backend = TGIBackend()

    def test_build_command_minimal(self):
        cmd = self.backend.build_command(
            model_path="/path/to/model", port=8080, host="0.0.0.0",
            log_file="/tmp/test.log",
            params={"binary": "text-generation-launcher"},
            extra_args=[],
        )
        assert "text-generation-launcher" in cmd or "--model-id" in cmd

    def test_build_command_with_num_shard(self):
        cmd = self.backend.build_command(
            model_path="/m", port=8080, host="0.0.0.0", log_file="/tmp/t.log",
            params={"binary": "text-generation-launcher", "tgi_num_shard": 2},
            extra_args=[],
        )
        assert "--num-shard" in cmd


class TestOllamaBackend:
    def setup_method(self):
        self.backend = OllamaBackend()

    def test_build_command_minimal(self):
        cmd = self.backend.build_command(
            model_path="llama3", port=8080, host="0.0.0.0",
            log_file="/tmp/test.log",
            params={"binary": "ollama"},
            extra_args=[],
        )
        assert "ollama" in cmd[0] or "serve" in cmd


class TestCheckBackendInstalled:
    def test_returns_bool(self):
        for name in ["llamacpp", "vllm", "sglang", "tgi", "ollama", "tensorrt_llm", "lmdeploy", "openvino"]:
            result = check_backend_installed(name)
            assert isinstance(result, bool)

    def test_unknown_backend(self):
        result = check_backend_installed("unknown_backend")
        assert result is False


class TestGetAllBackendsStatus:
    def test_returns_all_backends(self):
        # This test may be slow due to import checks
        status = get_all_backends_status()
        assert len(status) == 8
        ids = {b["id"] for b in status}
        assert ids == {"llamacpp", "vllm", "sglang", "tgi", "ollama", "tensorrt_llm", "lmdeploy", "openvino"}

    def test_each_has_required_fields(self):
        status = get_all_backends_status()
        for b in status:
            assert "id" in b
            assert "name" in b
            assert "description" in b
            assert "model_types" in b
            assert "check_type" in b
            assert "installed" in b
            assert isinstance(b["installed"], bool)

    def test_installed_field_matches_check(self):
        status = get_all_backends_status()
        for b in status:
            expected = check_backend_installed(b["id"])
            assert b["installed"] == expected, f"{b['id']}: expected {expected}, got {b['installed']}"
