from __future__ import annotations

from app.backend import services
from app.backend.quality import data_quality_report


def test_data_quality_report_shape():
    report = data_quality_report()

    assert report["status"] in {"pass", "warn", "fail"}
    assert report["summary"]["checks"] > 0
    assert "live_results" in report["sections"]
    assert "artifacts" in report["sections"]


def test_health_payloads():
    model_health = services.model_health()
    pipeline_status = services.pipeline_status()
    models = services.model_registry()

    assert "data_quality" in model_health
    assert "files" in pipeline_status
    assert models
