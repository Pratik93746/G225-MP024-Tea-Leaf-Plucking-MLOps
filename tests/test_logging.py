import json
import logging

from mp024.logging_utils import JsonFormatter


def test_formatter_writes_one_valid_json_line_with_extras():
    record = logging.LogRecord("mp024.api", logging.INFO, __file__, 1, "prediction served", (), None)
    record.prediction_kg = 31.2
    line = JsonFormatter().format(record)
    assert "\n" not in line
    payload = json.loads(line)
    assert payload["message"] == "prediction served" and payload["level"] == "INFO"
    assert payload["prediction_kg"] == 31.2 and "timestamp" in payload


def test_prediction_endpoint_logs_json_to_stdout(capsys):
    from fastapi.testclient import TestClient

    from app.main import app
    TestClient(app).post("/predict", json={"worker_exp_yr": 5, "rain_mm_week": 50, "slope_deg": 10, "temp_c": 20.0,
                                           "bush_age_yr": 30, "hours_worked": 8.0, "flush": "second",
                                           "section": "Upper Garden"})
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("{")]
    assert lines, "expected a JSON log line on stdout"
    assert json.loads(lines[-1])["event"] == "prediction"
