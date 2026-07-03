"""Model discovery, info, and deletion."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from .backends import get_backend
from .cache import TTLCache
from .config import ConfigManager
from .models import BackendType
from .utils import guess_family, guess_quantization


class ModelService:
    def __init__(self, config: ConfigManager):
        self._config = config
        self._cache = TTLCache(default_ttl=5.0)

    def list_models(self, backend_type: BackendType | None = None) -> list[dict[str, Any]]:
        cache_key = f"models:{backend_type}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        model_dir = Path(self._config.config.model_dir).expanduser()
        models: list[dict[str, Any]] = []

        if backend_type:
            backend = get_backend(backend_type.value)
            models = backend.get_model_paths(model_dir)
        else:
            seen_paths: set[str] = set()
            for bt in BackendType:
                if bt == BackendType.ollama:
                    continue
                try:
                    backend = get_backend(bt.value)
                    backend_models = backend.get_model_paths(model_dir)
                    for m in backend_models:
                        model_path = m.get("path", "")
                        if model_path not in seen_paths:
                            seen_paths.add(model_path)
                            m["backend"] = bt.value
                            models.append(m)
                except Exception:
                    continue

        self._cache.set(cache_key, models)
        return models

    def get_model_info(self, name: str) -> dict[str, Any] | None:
        model_dir = Path(self._config.config.model_dir).expanduser()
        target = model_dir / name

        if target.exists():
            if target.is_file() and target.suffix == ".gguf":
                return self._make_gguf_info(name, target, BackendType.llamacpp)

            if target.is_dir():
                # HuggingFace model directory
                config_file = target / "config.json"
                has_safetensors = any(target.glob("*.safetensors"))
                has_bin = any(target.glob("*.bin"))

                if config_file.exists() and (has_safetensors or has_bin):
                    total_size = sum(f.stat().st_size for f in target.glob("*.safetensors"))
                    total_size += sum(f.stat().st_size for f in target.glob("*.bin"))
                    return {
                        "name": name,
                        "path": str(target),
                        "size_mb": round(total_size / (1024 * 1024), 1) if total_size > 0 else 0,
                        "family": guess_family(target.name),
                        "quantization": guess_quantization(target.name),
                        "backend": BackendType.vllm.value,
                    }

                # GGUF files in directory
                ggufs = list(target.glob("*.gguf"))
                if ggufs:
                    main = ggufs[0]
                    return {
                        "name": f"{target.name}/{main.name}",
                        "path": str(main),
                        "size_mb": round(main.stat().st_size / (1024 * 1024), 1),
                        "family": guess_family(main.name),
                        "quantization": guess_quantization(main.name),
                        "backend": BackendType.llamacpp.value,
                    }

        # Recursive search
        for p in model_dir.rglob(name):
            if p.suffix == ".gguf":
                return self._make_gguf_info(name, p, BackendType.llamacpp)

        return None

    def delete_model(self, name: str) -> bool:
        model_dir = Path(self._config.config.model_dir).expanduser()
        target = model_dir / name
        if target.exists() and target.is_file():
            target.unlink()
            return True
        if target.is_dir():
            shutil.rmtree(target)
            return True
        return False

    @staticmethod
    def _make_gguf_info(name: str, path: Path, backend: BackendType) -> dict[str, Any]:
        return {
            "name": name,
            "path": str(path),
            "size_mb": round(path.stat().st_size / (1024 * 1024), 1),
            "family": guess_family(path.name),
            "quantization": guess_quantization(path.name),
            "backend": backend.value,
        }
