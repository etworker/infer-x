"""Log-related models."""

from __future__ import annotations

from pydantic import BaseModel


class LogEntry(BaseModel):
    timestamp: str
    level: str
    message: str


class InstanceLogs(BaseModel):
    instance_id: str
    logs: list[LogEntry]
    total_lines: int
