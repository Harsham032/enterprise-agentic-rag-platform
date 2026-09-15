"""Latency measurement helpers used by the API and the evaluation harness."""

from __future__ import annotations

import time
from collections.abc import Sequence
from types import TracebackType


class Stopwatch:
    """Context manager measuring wall-clock duration in milliseconds.

    >>> with Stopwatch() as sw:
    ...     pass
    >>> sw.elapsed_ms >= 0.0
    True
    """

    __slots__ = ("_start", "elapsed_ms")

    def __init__(self) -> None:
        self._start = 0.0
        self.elapsed_ms = 0.0

    def __enter__(self) -> Stopwatch:
        self._start = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000.0


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolation percentile, matching ``numpy.percentile`` defaults.

    Implemented here so latency reporting does not depend on NumPy being loaded
    in the request path.
    """
    if not values:
        return 0.0
    if not 0.0 <= q <= 100.0:
        raise ValueError("q must be between 0 and 100")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * (q / 100.0)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def latency_summary(samples: Sequence[float]) -> dict[str, float]:
    """Return mean/p50/p90/p95/p99 for a list of millisecond timings."""
    if not samples:
        return {
            "count": 0,
            "mean_ms": 0.0,
            "p50_ms": 0.0,
            "p90_ms": 0.0,
            "p95_ms": 0.0,
            "p99_ms": 0.0,
        }
    return {
        "count": float(len(samples)),
        "mean_ms": sum(samples) / len(samples),
        "p50_ms": percentile(samples, 50),
        "p90_ms": percentile(samples, 90),
        "p95_ms": percentile(samples, 95),
        "p99_ms": percentile(samples, 99),
    }
