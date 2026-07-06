"""Shared utilities for backend command building."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable


def discover_hf_models(
    model_dir: Path,
    guess_family: Callable[[str], str | None],
    guess_quantization: Callable[[str], str | None],
) -> list[dict[str, Any]]:
    """Discover HuggingFace model directories (config.json + safetensors/bin)."""
    models = []
    if not model_dir.exists():
        return models
    for p in sorted(model_dir.iterdir()):
        if not p.is_dir():
            continue
        config_file = p / "config.json"
        safetensors_files = list(p.glob("*.safetensors"))
        bin_files = list(p.glob("*.bin"))
        if config_file.exists() and (safetensors_files or bin_files):
            total_size = (
                sum(f.stat().st_size for f in safetensors_files)
                + sum(f.stat().st_size for f in bin_files)
            )
            models.append({
                "name": p.name,
                "path": str(p),
                "size_mb": round(total_size / (1024 * 1024), 1) if total_size > 0 else 0,
                "family": guess_family(p.name),
                "quantization": guess_quantization(p.name),
            })
    return models
