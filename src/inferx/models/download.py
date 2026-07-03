"""Download-related models."""

from __future__ import annotations

from pydantic import BaseModel

from .enums import DownloadSource, DownloadStatus


class DownloadRequest(BaseModel):
    source: DownloadSource
    repo: str | None = None
    filename: str | None = None
    url: str | None = None
    quantization: str | None = None
    save_name: str | None = None


class DownloadProgress(BaseModel):
    task_id: str
    source: str
    repo: str | None = None
    filename: str | None = None
    status: DownloadStatus
    progress_pct: float = 0.0
    downloaded_bytes: int = 0
    total_bytes: int = 0
    speed_bytes_per_sec: float = 0.0
    error: str | None = None
    save_path: str | None = None
