"""Flat-file structured logging shared by every pipeline layer.

Each layer logs one JSON object per line (JSONL) tagged with a request_id so a
full four-layer transaction can be reconstructed and scored later during
experiment evaluation.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from vish_agent.config import LOG_DIR

_write_lock = threading.Lock()


class TransactionLogger:
    def __init__(self, log_path: Path | str | None = None):
        self.log_path = Path(log_path) if log_path else LOG_DIR / "transactions.jsonl"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, *, request_id: str, layer: str, event: str, data: dict[str, Any]) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "request_id": request_id,
            "layer": layer,
            "event": event,
            "data": data,
        }
        line = json.dumps(record, default=str)
        with _write_lock:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")


default_logger = TransactionLogger()
