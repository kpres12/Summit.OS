"""
Anomaly Detection for Summit.OS

Detects anomalous behavior in entity telemetry streams:
- ZScoreDetector: statistical z-score on sliding windows
- MovingAverageDetector: deviation from exponential moving average
- IsolationForestDetector: sklearn wrapper (when available) + pure-Python fallback
- EnsembleDetector: combines multiple detectors with voting

Feeds into the intelligence service for automated alerting.
"""

from __future__ import annotations

import math
import time
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from abc import ABC, abstractmethod
from collections import defaultdict, deque

logger = logging.getLogger("ai.anomaly")


@dataclass
class AnomalyResult:
    """Result of anomaly detection on a data point or window."""

    entity_id: str
    is_anomaly: bool
    score: float  # 0.0 = normal, 1.0 = extreme anomaly
    metric_name: str = ""
    value: float = 0.0
    threshold: float = 0.0
    description: str = ""
    detector_name: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        return {
            "entity_id": self.entity_id,
            "is_anomaly": self.is_anomaly,
            "score": round(self.score, 4),
            "metric": self.metric_name,
            "value": self.value,
            "threshold": self.threshold,
            "description": self.description,
            "detector": self.detector_name,
        }


@dataclass
class TimeSeriesPoint:
    """A single data point in a time series."""

    value: float
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ── Abstract Detector ───────────────────────────────────────


class AnomalyDetector(ABC):
    """Base class for anomaly detectors."""

    @abstractmethod
    def ingest(
        self, entity_id: str, metric: str, point: TimeSeriesPoint
    ) -> AnomalyResult:
        """Ingest a data point and check for anomaly."""
        ...

    @abstractmethod
    def check_window(self, entity_id: str, metric: str) -> AnomalyResult:
        """Check the current window for anomalies."""
        ...

    @abstractmethod
    def reset(self, entity_id: str, metric: Optional[str] = None) -> None:
        """Reset detector state."""
        ...


# ── Z-Score Detector ────────────────────────────────────────


class ZScoreDetector(AnomalyDetector):
    """
    Statistical z-score anomaly detector.

    Maintains a sliding window per (entity, metric) pair.
    Points with |z-score| > threshold are flagged anomalous.
    """

    def __init__(self, window_size: int = 100, z_threshold: float = 3.0):
        self.window_size = window_size
        self.z_threshold = z_threshold
        # (entity_id, metric) → deque of values
        self._windows: Dict[Tuple[str, str], deque] = defaultdict(
            lambda: deque(maxlen=window_size)
        )

    def ingest(
        self, entity_id: str, metric: str, point: TimeSeriesPoint
    ) -> AnomalyResult:
        key = (entity_id, metric)
        window = self._windows[key]
        window.append(point.value)
        return self._check(entity_id, metric, point.value)

    def check_window(self, entity_id: str, metric: str) -> AnomalyResult:
        key = (entity_id, metric)
        window = self._windows.get(key)
        if not window:
            return AnomalyResult(
                entity_id=entity_id,
                is_anomaly=False,
                score=0.0,
                metric_name=metric,
                detector_name="zscore",
            )
        return self._check(entity_id, metric, window[-1])

    def _check(self, entity_id: str, metric: str, value: float) -> AnomalyResult:
        key = (entity_id, metric)
        window = self._windows[key]

        if len(window) < 5:
            return AnomalyResult(
                entity_id=entity_id,
                is_anomaly=False,
                score=0.0,
                metric_name=metric,
                value=value,
                description="Insufficient data",
                detector_name="zscore",
            )

        mean = sum(window) / len(window)
        variance = sum((x - mean) ** 2 for x in window) / len(window)
        std = math.sqrt(variance) if variance > 0 else 1e-10

        z_score = abs(value - mean) / std
        normalized_score = min(z_score / (self.z_threshold * 2), 1.0)
        is_anomaly = z_score > self.z_threshold

        return AnomalyResult(
            entity_id=entity_id,
            is_anomaly=is_anomaly,
            score=normalized_score,
            metric_name=metric,
            value=value,
            threshold=self.z_threshold,
            description=f"z={z_score:.2f} (μ={mean:.2f}, σ={std:.2f})",
            detector_name="zscore",
        )

    def reset(self, entity_id: str, metric: Optional[str] = None) -> None:
        if metric:
            self._windows.pop((entity_id, metric), None)
        else:
            keys = [k for k in self._windows if k[0] == entity_id]
            for k in keys:
                del self._windows[k]


# ── Moving Average Detector ─────────────────────────────────


class MovingAverageDetector(AnomalyDetector):
    """
    Exponential moving average (EMA) anomaly detector.

    Flags points that deviate significantly from the EMA.
    Adaptive: learns the normal rate of change over time.
    """

    def __init__(self, alpha: float = 0.1, deviation_multiplier: float = 3.0):
        self.alpha = alpha
        self.deviation_multiplier = deviation_multiplier
        # (entity_id, metric) → (ema, ema_of_deviation)
        self._state: Dict[Tuple[str, str], Tuple[float, float, int]] = {}

    def ingest(
        self, entity_id: str, metric: str, point: TimeSeriesPoint
    ) -> AnomalyResult:
        key = (entity_id, metric)
        value = point.value

        if key not in self._state:
            self._state[key] = (value, 0.0, 1)
            return AnomalyResult(
                entity_id=entity_id,
                is_anomaly=False,
                score=0.0,
                metric_name=metric,
                value=value,
                description="First observation",
                detector_name="ema",
            )

        ema, ema_dev, count = self._state[key]

        # Update EMA
        new_ema = self.alpha * value + (1 - self.alpha) * ema
        deviation = abs(value - ema)
        new_ema_dev = self.alpha * deviation + (1 - self.alpha) * ema_dev

        self._state[key] = (new_ema, new_ema_dev, count + 1)

        # Check anomaly
        threshold = max(new_ema_dev * self.deviation_multiplier, 1e-6)
        score = min(deviation / (threshold * 2), 1.0) if threshold > 0 else 0.0
        is_anomaly = deviation > threshold and count > 10

        return AnomalyResult(
            entity_id=entity_id,
            is_anomaly=is_anomaly,
            score=score,
            metric_name=metric,
            value=value,
            threshold=threshold,
            description=f"dev={deviation:.2f}, ema={new_ema:.2f}, thresh={threshold:.2f}",
            detector_name="ema",
        )

    def check_window(self, entity_id: str, metric: str) -> AnomalyResult:
        key = (entity_id, metric)
        state = self._state.get(key)
        if not state:
            return AnomalyResult(
                entity_id=entity_id,
                is_anomaly=False,
                score=0.0,
                metric_name=metric,
                detector_name="ema",
            )
        ema, ema_dev, count = state
        return AnomalyResult(
            entity_id=entity_id,
            is_anomaly=False,
            score=0.0,
            metric_name=metric,
            value=ema,
            description=f"Current EMA: {ema:.2f}, EMA deviation: {ema_dev:.2f}",
            detector_name="ema",
        )

    def reset(self, entity_id: str, metric: Optional[str] = None) -> None:
        if metric:
            self._state.pop((entity_id, metric), None)
        else:
            keys = [k for k in self._state if k[0] == entity_id]
            for k in keys:
                del self._state[k]


# ── Isolation Forest Detector ───────────────────────────────


class IsolationForestDetector(AnomalyDetector):
    """
    Isolation Forest anomaly detector.

    Uses sklearn when available, falls back to a simple
    percentile-based approach otherwise.
    """

    def __init__(
        self,
        contamination: float = 0.05,
        n_estimators: int = 100,
        buffer_size: int = 500,
    ):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.buffer_size = buffer_size
        self._buffers: Dict[Tuple[str, str], List[float]] = defaultdict(list)
        self._models: Dict[Tuple[str, str], Any] = {}
        self._sklearn_available = self._check_sklearn()

    @staticmethod
    def _check_sklearn() -> bool:
        try:
            from sklearn.ensemble import IsolationForest

            return True
        except ImportError:
            return False

    def _fit_model(self, key: Tuple[str, str]):
        """Fit or refit the isolation forest model."""
        buffer = self._buffers[key]
        if len(buffer) < 20:
            return

        if self._sklearn_available:
            from sklearn.ensemble import IsolationForest
            import numpy as np

            X = np.array(buffer).reshape(-1, 1)
            model = IsolationForest(
                contamination=self.contamination,
                n_estimators=self.n_estimators,
                random_state=42,
            )
            model.fit(X)
            self._models[key] = ("sklearn", model)
        else:
            # Fallback: compute percentile thresholds
            sorted_vals = sorted(buffer)
            n = len(sorted_vals)
            low_idx = max(0, int(n * self.contamination / 2))
            high_idx = min(n - 1, int(n * (1 - self.contamination / 2)))
            self._models[key] = (
                "percentile",
                sorted_vals[low_idx],
                sorted_vals[high_idx],
            )

    def ingest(
        self, entity_id: str, metric: str, point: TimeSeriesPoint
    ) -> AnomalyResult:
        key = (entity_id, metric)
        buffer = self._buffers[key]
        buffer.append(point.value)

        # Trim buffer
        if len(buffer) > self.buffer_size:
            self._buffers[key] = buffer[-self.buffer_size :]

        # Refit periodically
        if len(buffer) % 50 == 0:
            self._fit_model(key)

        return self._predict(entity_id, metric, point.value)

    def _predict(self, entity_id: str, metric: str, value: float) -> AnomalyResult:
        key = (entity_id, metric)
        model_info = self._models.get(key)

        if model_info is None:
            return AnomalyResult(
                entity_id=entity_id,
                is_anomaly=False,
                score=0.0,
                metric_name=metric,
                value=value,
                description="Model not yet fitted",
                detector_name="iforest",
            )

        if model_info[0] == "sklearn":
            import numpy as np

            model = model_info[1]
            X = np.array([[value]])
            pred = model.predict(X)[0]
            score_val = -model.score_samples(X)[0]
            # Normalize score to [0, 1]
            normalized = min(max(score_val, 0), 1.0)
            is_anomaly = pred == -1

            return AnomalyResult(
                entity_id=entity_id,
                is_anomaly=is_anomaly,
                score=normalized,
                metric_name=metric,
                value=value,
                description=f"IF score={score_val:.3f}",
                detector_name="iforest",
            )
        else:
            # Percentile fallback
            _, low, high = model_info
            if value < low:
                score = min(abs(value - low) / max(abs(low), 1e-6), 1.0)
                return AnomalyResult(
                    entity_id=entity_id,
                    is_anomaly=True,
                    score=score,
                    metric_name=metric,
                    value=value,
                    threshold=low,
                    description=f"Below {self.contamination*50:.1f}th percentile ({low:.2f})",
                    detector_name="iforest_fallback",
                )
            elif value > high:
                score = min(abs(value - high) / max(abs(high), 1e-6), 1.0)
                return AnomalyResult(
                    entity_id=entity_id,
                    is_anomaly=True,
                    score=score,
                    metric_name=metric,
                    value=value,
                    threshold=high,
                    description=f"Above {(1-self.contamination/2)*100:.1f}th percentile ({high:.2f})",
                    detector_name="iforest_fallback",
                )
            else:
                return AnomalyResult(
                    entity_id=entity_id,
                    is_anomaly=False,
                    score=0.0,
                    metric_name=metric,
                    value=value,
                    description="Within normal range",
                    detector_name="iforest_fallback",
                )

    def check_window(self, entity_id: str, metric: str) -> AnomalyResult:
        key = (entity_id, metric)
        buffer = self._buffers.get(key)
        if not buffer:
            return AnomalyResult(
                entity_id=entity_id,
                is_anomaly=False,
                score=0.0,
                metric_name=metric,
                detector_name="iforest",
            )
        return self._predict(entity_id, metric, buffer[-1])

    def reset(self, entity_id: str, metric: Optional[str] = None) -> None:
        if metric:
            self._buffers.pop((entity_id, metric), None)
            self._models.pop((entity_id, metric), None)
        else:
            keys = [k for k in self._buffers if k[0] == entity_id]
            for k in keys:
                del self._buffers[k]
                self._models.pop(k, None)


# ── Ensemble Detector ───────────────────────────────────────


class EnsembleDetector(AnomalyDetector):
    """
    Combines multiple anomaly detectors via majority voting.

    An anomaly is flagged when >= vote_threshold detectors agree.
    The final score is the weighted average of individual scores.
    """

    def __init__(
        self, detectors: Optional[List[AnomalyDetector]] = None, vote_threshold: int = 2
    ):
        self.detectors = detectors or [
            ZScoreDetector(),
            MovingAverageDetector(),
            IsolationForestDetector(),
        ]
        self.vote_threshold = vote_threshold

    def ingest(
        self, entity_id: str, metric: str, point: TimeSeriesPoint
    ) -> AnomalyResult:
        results = [d.ingest(entity_id, metric, point) for d in self.detectors]
        return self._combine(entity_id, metric, point.value, results)

    def check_window(self, entity_id: str, metric: str) -> AnomalyResult:
        results = [d.check_window(entity_id, metric) for d in self.detectors]
        value = results[0].value if results else 0.0
        return self._combine(entity_id, metric, value, results)

    def _combine(
        self, entity_id: str, metric: str, value: float, results: List[AnomalyResult]
    ) -> AnomalyResult:
        votes = sum(1 for r in results if r.is_anomaly)
        avg_score = sum(r.score for r in results) / max(len(results), 1)
        is_anomaly = votes >= self.vote_threshold

        descriptions = [f"{r.detector_name}:{r.score:.2f}" for r in results]

        return AnomalyResult(
            entity_id=entity_id,
            is_anomaly=is_anomaly,
            score=avg_score,
            metric_name=metric,
            value=value,
            threshold=float(self.vote_threshold),
            description=f"votes={votes}/{len(results)} [{', '.join(descriptions)}]",
            detector_name="ensemble",
        )

    def reset(self, entity_id: str, metric: Optional[str] = None) -> None:
        for d in self.detectors:
            d.reset(entity_id, metric)
