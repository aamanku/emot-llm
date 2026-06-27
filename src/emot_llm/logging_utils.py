"""JSONL session logging."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


@dataclass
class SessionLogger:
    base_dir: str | Path = "logs"
    session_id: str = field(default_factory=utc_stamp)

    def __post_init__(self) -> None:
        self.session_dir = Path(self.base_dir) / self.session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.session_dir / "ticks.jsonl"

    def write(self, record: dict[str, Any]) -> None:
        record = {"logged_at": datetime.now(timezone.utc).isoformat(), **record}
        with self.jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def path(self) -> Path:
        return self.jsonl_path
