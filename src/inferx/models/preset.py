"""Preset model."""

from __future__ import annotations

from pydantic import Field

from .enums import BackendType
from .params import BackendParams


class Preset(BackendParams):
    name: str
    description: str = ""
    backend: BackendType | None = None
    extra_args: list[str] = Field(default_factory=list)
