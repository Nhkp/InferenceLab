"""Contract 1.0 client metrics, recomputable solely from raw request records."""

import math
from collections import Counter
from collections.abc import Sequence
from typing import Literal

from inference_lab.config.schemas import METRIC_UNITS, MetricValue, RequestMeasurement


def request_metrics(request: RequestMeasurement) -> dict[str, float | None]:
    values: dict[str, float | None] = {
        "client_ttft_ms": None,
        "client_e2e_ms": None,
        "client_tpot_estimate_ms": None,
    }
    if request.transport_status != "success":
        return values
    start, first, last, terminal = (
        request.request_start_ns,
        request.first_content_ns,
        request.last_content_ns,
        request.terminal_ns,
    )
    assert start is not None and terminal is not None
    values["client_e2e_ms"] = (terminal - start) / 1e6
    if first is not None:
        values["client_ttft_ms"] = (first - start) / 1e6
    if (
        first is not None
        and last is not None
        and request.content_events >= 2
        and request.output_tokens is not None
        and request.output_tokens > 1
    ):
        values["client_tpot_estimate_ms"] = (last - first) / 1e6 / (request.output_tokens - 1)
    return values


def percentile(values: Sequence[float], q: float) -> float | None:
    if not 0 <= q <= 1 or not math.isfinite(q):
        raise ValueError("Percentile must be between zero and one")
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("Observations must be finite and nonnegative")
    if not values:
        return None
    ordered = sorted(values)
    h = (len(ordered) - 1) * q
    low, high = math.floor(h), math.ceil(h)
    return ordered[low] + (ordered[high] - ordered[low]) * (h - low)


def aggregate(requests: Sequence[RequestMeasurement]) -> dict[str, MetricValue]:
    starts = [r.request_start_ns for r in requests if r.request_start_ns is not None]
    ends = [r.terminal_ns for r in requests if r.terminal_ns is not None]
    valid_window = bool(requests) and len(starts) == len(ends) == len(requests)
    duration = (max(ends) - min(starts)) / 1e9 if valid_window else 0.0
    successes = [r for r in requests if r.transport_status == "success"]
    result: dict[str, MetricValue] = {}
    per_request = [request_metrics(r) for r in successes]
    for prefix in ("client_ttft", "client_e2e", "client_tpot_estimate"):
        observations = [value for r in per_request if (value := r[prefix + "_ms"]) is not None]
        for name, q in (("p50", 0.5), ("p95", 0.95), ("p99", 0.99)):
            value = percentile(observations, q)
            result[f"{prefix}_{name}_ms"] = MetricValue(
                value=value,
                unit="ms",
                source="client",
                sample_count=len(observations),
                population="successful_requests",
                reason=None
                if value is not None
                else "No successful requests with the required timing/token observations",
            )
    rates = {
        "successful_requests_per_second": len(successes) / duration if duration > 0 else None,
        "successful_output_tokens_per_second": (
            sum(r.output_tokens or 0 for r in successes) / duration if duration > 0 else None
        ),
        "error_rate": (len(requests) - len(successes)) / len(requests)
        if requests and duration > 0
        else None,
    }
    for name, value in rates.items():
        unit: Literal["requests/s", "tokens/s", "ratio"]
        unit = (
            "ratio" if name == "error_rate" else ("tokens/s" if "tokens" in name else "requests/s")
        )
        result[name] = MetricValue(
            value=value,
            unit=unit,
            source="client",
            sample_count=len(requests),
            population="all_attempts",
            reason=None if value is not None else "No complete, positive-duration attempt window",
        )
    assert set(result) == set(METRIC_UNITS)
    return result


def error_breakdown(requests: Sequence[RequestMeasurement]) -> dict[str, int]:
    return dict(Counter(r.error for r in requests if r.error is not None))
