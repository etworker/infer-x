"""Model-related models."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ModelFileInfo(BaseModel):
    name: str
    path: str
    size_mb: float
    quantization: str | None = None
    family: str | None = None


class ModelInfo(BaseModel):
    name: str
    path: str
    size_mb: float
    quantization: str | None = None
    family: str | None = None
    architecture: str | None = None
    parameter_count: str | None = None
    context_length: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
