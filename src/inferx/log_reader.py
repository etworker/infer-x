"""Log reading utility for instance log files."""

from __future__ import annotations

import collections
from pathlib import Path


class LogReader:
    """Reads log files from disk."""

    def __init__(self, logs_dir: Path) -> None:
        self._logs_dir = logs_dir

    def get_logs(self, inst_id: str, inst_log_file: Path | None = None, lines: int = 100) -> list[str]:
        log_file = inst_log_file or self._logs_dir / f"{inst_id}.log"
        if not log_file.exists():
            return []
        try:
            with open(log_file, encoding="utf-8", errors="replace") as f:
                result = collections.deque(f, maxlen=lines)
            return [line.rstrip("\n") for line in result]
        except Exception:
            return []

    def get_error_logs(self, inst_id: str, lines: int = 50) -> list[str]:
        stderr_path = self._logs_dir / f"{inst_id}.stderr"
        if not stderr_path.exists():
            return []
        try:
            with open(stderr_path, encoding="utf-8", errors="replace") as f:
                result = collections.deque(f, maxlen=lines)
            return [line.rstrip("\n") for line in result]
        except Exception:
            return []
