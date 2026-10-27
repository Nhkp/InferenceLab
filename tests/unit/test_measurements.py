import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from inference_lab.benchmark.metrics import aggregate, percentile, request_metrics
from inference_lab.config.schemas import RequestMeasurement, WorkloadEvidence

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]


def request(**changes) -> RequestMeasurement:
    raw = json.loads((ROOT / "tests/data/synthetic-request.json").read_text())
    raw.update(
        request_start_ns=0,
        first_content_ns=100_000_000,
        last_content_ns=300_000_000,
        terminal_ns=1_000_000_000,
        output_tokens=5,
    )
    raw.update(changes)
    return RequestMeasurement.model_validate(raw)


def test_canonical_metrics_use_usage_and_one_shared_window_including_failures() -> None:
    first = request()
    second = request(
        request_id="second",
        request_start_ns=100_000_000,
        first_content_ns=150_000_000,
        last_content_ns=150_000_000,
        terminal_ns=500_000_000,
        content_events=1,
        output_tokens=1,
    )
    failed = request(
        request_id="failed",
        transport_status="timeout",
        error="idle_timeout",
        terminal_ns=2_000_000_000,
        output_tokens=999,
    )
    metrics = aggregate([first, second, failed])
    assert metrics["successful_requests_per_second"].value == 1
    assert metrics["successful_output_tokens_per_second"].value == 3
    assert metrics["error_rate"].value == pytest.approx(1 / 3)
    assert metrics["client_ttft_p50_ms"].value == 75
    assert metrics["client_ttft_p95_ms"].value == 97.5
    assert metrics["client_e2e_p50_ms"].value == 700
    assert metrics["client_tpot_estimate_p50_ms"].value == 50
    assert metrics["client_tpot_estimate_p50_ms"].sample_count == 1
    assert metrics["client_ttft_p50_ms"].sample_count == 2
    assert all(value is None for value in request_metrics(failed).values())


@pytest.mark.parametrize("tokens,events", [(0, 2), (1, 2), (5, 1)])
def test_tpot_requires_multiple_usage_tokens_and_content_events(tokens: int, events: int) -> None:
    raw = request(output_tokens=tokens, content_events=events)
    assert request_metrics(raw)["client_tpot_estimate_ms"] is None


def test_no_content_is_not_a_fake_zero_ttft() -> None:
    raw = request(
        first_content_ns=None,
        last_content_ns=None,
        content_events=0,
        missing_reasons={"first_content_ns": "no content", "last_content_ns": "no content"},
    )
    metrics = aggregate([raw])
    assert metrics["client_ttft_p50_ms"].value is None
    assert metrics["client_ttft_p50_ms"].reason
    assert metrics["client_e2e_p50_ms"].value == 1000


def test_empty_zero_duration_and_single_sample_windows() -> None:
    assert all(metric.value is None and metric.reason for metric in aggregate([]).values())
    zero = request(first_content_ns=0, last_content_ns=0, terminal_ns=0)
    assert aggregate([zero])["successful_requests_per_second"].value is None
    assert aggregate([request()])["client_ttft_p99_ms"].value == 100
    assert percentile([0, 10, 20, 30], 0.95) == pytest.approx(28.5)
    for values, q in [([float("nan")], 0.5), ([1], -1), ([1], float("inf"))]:
        with pytest.raises(ValueError):
            percentile(values, q)


def test_unknown_observed_lengths_are_explicit_and_not_conformant() -> None:
    data = json.loads((ROOT / "tests/data/synthetic-result.json").read_text())["workload"]
    fields = ("input_tokens_min", "input_tokens_max", "output_tokens_min", "output_tokens_max")
    data.update(
        {field: None for field in fields},
        conformant=False,
        missing_reasons={field: "No server usage" for field in fields},
    )
    observed = WorkloadEvidence.model_validate(data)
    assert observed.input_tokens_min is None
    for change in ({"conformant": True}, {"missing_reasons": {}}):
        with pytest.raises(ValidationError):
            WorkloadEvidence.model_validate(data | change)
