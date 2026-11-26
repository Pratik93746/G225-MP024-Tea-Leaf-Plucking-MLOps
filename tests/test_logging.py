"""Tests for structured JSON stdout logging."""
import json
import logging

from mp024.logging_utils import JsonFormatter, get_json_logger


def test_formatter_writes_one_valid_json_line_with_extras():
    record = logging.LogRecord("mp024.api", logging.INFO, __file__, 1, "prediction served", (), None)
    record.prediction_kg = 31.2
    line = JsonFormatter().format(record)
    assert "\n" not in line
    payload = json.loads(line)
    assert payload["message"] == "prediction served"
    assert payload["level"] == "INFO"
    assert payload["prediction_kg"] == 31.2
    assert "timestamp" in payload


def test_logger_writes_one_json_line_per_prediction_to_stdout_only(capsys):
    logger = get_json_logger("mp024.test.stdout")
    # Calling get_json_logger again must not attach duplicate handlers.
    logger_again = get_json_logger("mp024.test.stdout")
    assert logger is logger_again
    assert len(logger.handlers) == 1
    assert not any(isinstance(h, logging.FileHandler) for h in logger.handlers)

    logger.info(
        "prediction served",
        extra={
            "event": "prediction",
            "prediction_kg": 28.5,
            "section": "River Side",
            "flush": "first",
            "hours_worked": 9.0,
        },
    )
    captured = capsys.readouterr()
    assert captured.err == ""
    lines = [ln for ln in captured.out.splitlines() if ln.strip()]
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["event"] == "prediction"
    assert payload["prediction_kg"] == 28.5
    assert payload["section"] == "River Side"


def test_formatter_strips_personal_detail_fields():
    record = logging.LogRecord("mp024.api", logging.INFO, __file__, 1, "prediction served", (), None)
    record.worker_name = "Should Not Appear"
    record.email = "hidden@example.com"
    record.prediction_kg = 24.0
    payload = json.loads(JsonFormatter().format(record))
    assert "worker_name" not in payload
    assert "email" not in payload
    assert payload["prediction_kg"] == 24.0
