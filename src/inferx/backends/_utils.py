"""Shared utilities for backend command building."""

from __future__ import annotations

from typing import Any


def resolve_binary(binary: str) -> list[str]:
    """Split binary command string into parts.

    Handles patterns like:
      - "llama-server" -> ["llama-server"]
      - "python -m vllm.entrypoints.openai.api_server" -> ["python", "-m", "vllm.entrypoints.openai.api_server"]
      - "lmdeploy serve api_server" -> ["lmdeploy", "serve", "api_server"]
    """
    if binary.startswith("python"):
        return binary.split()
    if " " in binary:
        return binary.split()
    return [binary]


def add_flag(
    cmd: list[str],
    params: dict[str, Any],
    key: str,
    flag: str,
    *,
    condition: bool | None = None,
    flag_first: bool = True,
) -> None:
    """Append a CLI flag if the param value is present and satisfies condition.

    Patterns:
      add_flag(cmd, params, "threads", "-t")          -> {"threads": 8}     -> cmd += ["-t", "8"]
      add_flag(cmd, params, "mlock", "--mlock")        -> {"mlock": True}   -> cmd += ["--mlock"]
      add_flag(cmd, params, "numa", "--numa")          -> {"numa": "distribute"} -> cmd += ["--numa", "distribute"]
    """
    value = params.get(key)
    if value is None or value is False:
        return
    if condition is not None and not condition:
        return
    if flag_first:
        cmd.append(flag)
    if value is not True:
        cmd.append(str(value))


def add_flag_if(cmd: list[str], flag: str, value: Any, *, when: bool = True) -> None:
    """Append flag with value only when `when` is true and `value` is not None."""
    if not when or value is None:
        return
    cmd.append(flag)
    if value is not True:
        cmd.append(str(value))
