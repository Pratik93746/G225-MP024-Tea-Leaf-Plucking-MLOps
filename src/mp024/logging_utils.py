"""Structured JSON logging: one JSON object per line, written to stdout (never to a file)."""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone

# Standard LogRecord attributes; any other public attribute came from `extra=`.
_STANDARD_ATTRS = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {"message", "asctime"}

# Personal-detail keys that must never appear in emitted log lines.
_BLOCKED_PII_KEYS = frozenset({
    "name", "customer_name", "worker_name", "employee_name",
    "worker_id", "customer_id", "employee_id",
    "email", "phone", "phone_number", "mobile", "address", "ssn",
})


class JsonFormatter(logging.Formatter):
    """Format a log record as a single line of JSON."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _STANDARD_ATTRS and not key.startswith("_") and key.lower() not in _BLOCKED_PII_KEYS:
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class _StdoutHandler(logging.StreamHandler):
    """StreamHandler that resolves sys.stdout dynamically so redirected stdout is respected."""

    def __init__(self) -> None:
        logging.Handler.__init__(self)

    @property
    def stream(self):
        return sys.stdout


def get_json_logger(name: str = "mp024", level: int = logging.INFO) -> logging.Logger:
    """Return a logger that writes single-line JSON records to stdout (idempotent)."""
    logger = logging.getLogger(name)
    if not any(getattr(handler, "_mp024_json", False) for handler in logger.handlers):
        handler = _StdoutHandler()
        handler.setFormatter(JsonFormatter())
        handler._mp024_json = True
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger
