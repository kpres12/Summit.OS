"""
Observability Metrics for Summit.OS

Provides structured metrics collection compatible with Prometheus/OpenTelemetry.
Includes counters, gauges, histograms, and a health aggregator.
"""

from __future__ import annotations

import time
import math
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from collections import defaultdict

logger = logging.getLogger("observability")


class Counter:
    """Monotonically increasing counter."""

    def __init__(
        self, name: str, description: str = "", labels: Dict[str, str] | None = None
    ):
        self.name = name
        self.description = description
        self.labels = labels or {}
        self._value: float = 0.0

    def inc(self, amount: float = 1.0):
        self._value += amount

    @property
    def value(self) -> float:
        return self._value


class Gauge:
    """Value that can go up and down."""

    def __init__(
        self, name: str, description: str = "", labels: Dict[str, str] | None = None
    ):
        self.name = name
        self.description = description
        self.labels = labels or {}
        self._value: float = 0.0

    def set(self, value: float):
        self._value = value

    def inc(self, amount: float = 1.0):
        self._value += amount

    def dec(self, amount: float = 1.0):
        self._value -= amount

    @property
    def value(self) -> float:
        return self._value


class Histogram:
    """Distribution of values with configurable buckets."""

    DEFAULT_BUCKETS = [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]

    def __init__(
        self, name: str, description: str = "", buckets: List[float] | None = None
    ):
        self.name = name
        self.description = description
        self.buckets = sorted(buckets or self.DEFAULT_BUCKETS)
        self._count: int = 0
        self._sum: float = 0.0
        self._bucket_counts: Dict[float, int] = {b: 0 for b in self.buckets}
        self._bucket_counts[float("inf")] = 0

    def observe(self, value: float):
        self._count += 1
        self._sum += value
        for b in self.buckets:
            if value <= b:
                self._bucket_counts[b] += 1
        self._bucket_counts[float("inf")] += 1

    @property
    def count(self) -> int:
        return self._count

    @property
    def mean(self) -> float:
        return self._sum / max(1, self._count)


class MetricsRegistry:
    """Central registry for all metrics."""

    def __init__(self, service_name: str = "summit-os"):
        self.service_name = service_name
        self.counters: Dict[str, Counter] = {}
        self.gauges: Dict[str, Gauge] = {}
        self.histograms: Dict[str, Histogram] = {}

    def counter(self, name: str, description: str = "", **labels) -> Counter:
        key = f"{name}:{labels}"
        if key not in self.counters:
            self.counters[key] = Counter(name, description, labels)
        return self.counters[key]

    def gauge(self, name: str, description: str = "", **labels) -> Gauge:
        key = f"{name}:{labels}"
        if key not in self.gauges:
            self.gauges[key] = Gauge(name, description, labels)
        return self.gauges[key]

    def histogram(
        self, name: str, description: str = "", buckets: List[float] | None = None
    ) -> Histogram:
        if name not in self.histograms:
            self.histograms[name] = Histogram(name, description, buckets)
        return self.histograms[name]

    def to_prometheus_text(self) -> str:
        """Export metrics in Prometheus text exposition format."""
        lines = []
        for key, c in self.counters.items():
            labels_str = ",".join(f'{k}="{v}"' for k, v in c.labels.items())
            label_part = f"{{{labels_str}}}" if labels_str else ""
            lines.append(f"# HELP {c.name} {c.description}")
            lines.append(f"# TYPE {c.name} counter")
            lines.append(f"{c.name}{label_part} {c.value}")

        for key, g in self.gauges.items():
            labels_str = ",".join(f'{k}="{v}"' for k, v in g.labels.items())
            label_part = f"{{{labels_str}}}" if labels_str else ""
            lines.append(f"# HELP {g.name} {g.description}")
            lines.append(f"# TYPE {g.name} gauge")
            lines.append(f"{g.name}{label_part} {g.value}")

        for name, h in self.histograms.items():
            lines.append(f"# HELP {h.name} {h.description}")
            lines.append(f"# TYPE {h.name} histogram")
            for b, count in h._bucket_counts.items():
                le = "+Inf" if b == float("inf") else str(b)
                lines.append(f'{h.name}_bucket{{le="{le}"}} {count}')
            lines.append(f"{h.name}_sum {h._sum}")
            lines.append(f"{h.name}_count {h._count}")

        return "\n".join(lines)


@dataclass
class HealthStatus:
    """Health status for a component."""

    component: str
    healthy: bool
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    last_check: float = field(default_factory=time.time)


class HealthAggregator:
    """Aggregates health status from all components."""

    def __init__(self):
        self.components: Dict[str, HealthStatus] = {}

    def report(self, component: str, healthy: bool, message: str = "", **details):
        self.components[component] = HealthStatus(
            component=component,
            healthy=healthy,
            message=message,
            details=details,
            last_check=time.time(),
        )

    def is_healthy(self) -> bool:
        if not self.components:
            return True
        return all(c.healthy for c in self.components.values())

    def get_status(self) -> Dict:
        return {
            "healthy": self.is_healthy(),
            "components": {
                name: {
                    "healthy": s.healthy,
                    "message": s.message,
                    "details": s.details,
                    "last_check": s.last_check,
                }
                for name, s in self.components.items()
            },
        }


# ── Global singletons ────────────────────────────────────

_default_registry: Optional[MetricsRegistry] = None
_default_health: Optional[HealthAggregator] = None


def get_metrics(service_name: str = "summit-os") -> MetricsRegistry:
    global _default_registry
    if _default_registry is None:
        _default_registry = MetricsRegistry(service_name)
    return _default_registry


def get_health() -> HealthAggregator:
    global _default_health
    if _default_health is None:
        _default_health = HealthAggregator()
    return _default_health
