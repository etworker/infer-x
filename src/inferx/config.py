"""Configuration management with YAML persistence."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .models import DefaultConfig, Preset

DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent / "config.yaml"


class ConfigManager:
    def __init__(self, config_path: str | None = None):
        self._path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        self._config = DefaultConfig()
        self._presets: dict[str, Preset] = {}
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            self._save()
            return
        with open(self._path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        cfg_data = data.get("config", {})
        if cfg_data:
            self._config = DefaultConfig(**cfg_data)
        presets_data = data.get("presets", {})
        for name, p in presets_data.items():
            self._presets[name] = Preset(name=name, **p)

    def _save(self) -> None:
        data = {
            "config": self._config.model_dump(),
            "presets": {name: p.model_dump(exclude={"name"}) for name, p in self._presets.items()},
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    @property
    def config(self) -> DefaultConfig:
        return self._config

    def update_config(self, **kwargs: Any) -> DefaultConfig:
        for k, v in kwargs.items():
            if v is not None and hasattr(self._config, k):
                setattr(self._config, k, v)
        self._save()
        return self._config

    def list_presets(self) -> dict[str, Preset]:
        return dict(self._presets)

    def get_preset(self, name: str) -> Preset | None:
        return self._presets.get(name)

    def save_preset(self, preset: Preset) -> Preset:
        self._presets[preset.name] = preset
        self._save()
        return preset

    def delete_preset(self, name: str) -> bool:
        if name in self._presets:
            del self._presets[name]
            self._save()
            return True
        return False
