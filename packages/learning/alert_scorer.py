"""
alert_scorer.py — Source reliability scoring for Heli.OS CyberSynetic engine.

Learns which signal sources (adapters, cameras, sensors) produce reliable alerts
vs. noisy ones. Adjusts priority weights automatically based on operator feedback.

The key invariant: we never fully silence a source (weight >= 0.1) because a
sensor might suddenly become relevant again. Humans can always override the
weight. But a source with a 90% false-positive rate will have its alerts quietly
downgraded so they don't pollute the operator's attention.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    insert,
    select,
    update,
)
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

logger = logging.getLogger("learning.alert_scorer")

UTC = timezone.utc

# Weight bounds — never fully silence a source, never give it more than 2x weight
WEIGHT_MIN = 0.1
WEIGHT_MAX = 2.0

# Bayesian prior pseudo-counts for new sources (slightly optimistic)
PRIOR_CONFIRMED = 2.0  # α₀
PRIOR_FALSE_POSITIVE = 0.5  # β₀


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------


@dataclass
class SourceReliabilityScore:
    source_id: str
    total_alerts: int
    confirmed: int
    dismissed: int
    false_positives: int
    investigated: int
    reliability: float  # Bayesian smoothed: confirmed / (confirmed + false_positives)
    noise_rate: float  # false_positives / total_alerts
    priority_weight: float  # multiplier to apply to alerts from this source
    last_updated: datetime

    def to_dict(self) -> dict:
        d = asdict(self)
        d["last_updated"] = self.last_updated.isoformat()
        return d


# ---------------------------------------------------------------------------
# SQLAlchemy table
# ---------------------------------------------------------------------------

_metadata = MetaData()

source_scores_table = Table(
    "alert_source_scores",
    _metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("source_id", String(128), nullable=False, unique=True),
    Column("total_alerts", Integer, nullable=False, default=0),
    Column("confirmed", Integer, nullable=False, default=0),
    Column("dismissed", Integer, nullable=False, default=0),
    Column("false_positives", Integer, nullable=False, default=0),
    Column("investigated", Integer, nullable=False, default=0),
    Column("reliability", Float, nullable=False, default=0.8),
    Column("noise_rate", Float, nullable=False, default=0.0),
    Column("priority_weight", Float, nullable=False, default=1.0),
    Column("last_updated", DateTime(timezone=True), nullable=False),
    Index("ix_ass_source_id", "source_id"),
)


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------


class AlertQualityScorer:
    """
    Tracks alert quality per source and adjusts priority weights.

    Weight update logic:
    - reliability is estimated via Beta-Binomial: (confirmed + α₀) / (confirmed + false_positives + α₀ + β₀)
    - priority_weight is then derived from reliability via a linear transform:
        weight = WEIGHT_MIN + reliability × (WEIGHT_MAX - WEIGHT_MIN)
      This maps 0% reliability → 0.1, 100% reliability → 2.0, 50% → 1.05.

    New sources start at weight=1.0 (neutral), which corresponds to ~47% reliability
    under the prior. They earn trust or lose it with each operator action.
    """

    def __init__(self, database_url: str) -> None:
        self._db_url = database_url
        self._engine: Optional[AsyncEngine] = None

    async def initialize(self) -> None:
        self._engine = create_async_engine(self._db_url, echo=False, future=True)
        async with self._engine.begin() as conn:
            await conn.run_sync(_metadata.create_all)
        logger.info("AlertQualityScorer initialized")

    def _ensure_engine(self) -> AsyncEngine:
        if self._engine is None:
            raise RuntimeError("AlertQualityScorer.initialize() has not been called")
        return self._engine

    @staticmethod
    def _compute_reliability(confirmed: int, false_positives: int) -> float:
        """Bayesian Beta-Binomial reliability estimate."""
        alpha = confirmed + PRIOR_CONFIRMED
        beta = false_positives + PRIOR_FALSE_POSITIVE
        return round(alpha / (alpha + beta), 4)

    @staticmethod
    def _weight_from_reliability(reliability: float) -> float:
        """Map [0,1] reliability to [WEIGHT_MIN, WEIGHT_MAX] priority weight."""
        w = WEIGHT_MIN + reliability * (WEIGHT_MAX - WEIGHT_MIN)
        return round(max(WEIGHT_MIN, min(WEIGHT_MAX, w)), 4)

    async def _load_or_create(self, source_id: str) -> SourceReliabilityScore:
        engine = self._ensure_engine()
        async with engine.connect() as conn:
            result = await conn.execute(
                select(source_scores_table).where(
                    source_scores_table.c.source_id == source_id
                )
            )
            row = result.first()

        if row:
            return SourceReliabilityScore(
                source_id=row.source_id,
                total_alerts=row.total_alerts,
                confirmed=row.confirmed,
                dismissed=row.dismissed,
                false_positives=row.false_positives,
                investigated=row.investigated,
                reliability=row.reliability,
                noise_rate=row.noise_rate,
                priority_weight=row.priority_weight,
                last_updated=row.last_updated,
            )

        # Brand new source — start neutral
        initial_reliability = self._compute_reliability(0, 0)
        return SourceReliabilityScore(
            source_id=source_id,
            total_alerts=0,
            confirmed=0,
            dismissed=0,
            false_positives=0,
            investigated=0,
            reliability=initial_reliability,
            noise_rate=0.0,
            priority_weight=1.0,
            last_updated=datetime.now(UTC),
        )

    async def _save(self, score: SourceReliabilityScore) -> None:
        engine = self._ensure_engine()
        row_data = {
            "total_alerts": score.total_alerts,
            "confirmed": score.confirmed,
            "dismissed": score.dismissed,
            "false_positives": score.false_positives,
            "investigated": score.investigated,
            "reliability": score.reliability,
            "noise_rate": score.noise_rate,
            "priority_weight": score.priority_weight,
            "last_updated": score.last_updated,
        }
        async with engine.begin() as conn:
            existing = await conn.execute(
                select(source_scores_table.c.id).where(
                    source_scores_table.c.source_id == score.source_id
                )
            )
            if existing.first():
                await conn.execute(
                    update(source_scores_table)
                    .where(source_scores_table.c.source_id == score.source_id)
                    .values(**row_data)
                )
            else:
                await conn.execute(
                    insert(source_scores_table).values(
                        source_id=score.source_id, **row_data
                    )
                )

    async def record_alert_outcome(
        self,
        alert_id: str,
        source_id: str,
        outcome: str,
    ) -> None:
        """
        Record an operator's verdict on an alert from a given source.

        outcome: 'confirmed' | 'dismissed' | 'false_positive' | 'investigated'
        """
        score = await self._load_or_create(source_id)
        score.total_alerts += 1

        if outcome == "confirmed":
            score.confirmed += 1
        elif outcome == "dismissed":
            score.dismissed += 1
        elif outcome == "false_positive":
            score.false_positives += 1
        elif outcome == "investigated":
            score.investigated += 1
        else:
            logger.warning(
                "Unknown alert outcome '%s' for source %s", outcome, source_id
            )
            return

        # Recompute derived fields
        score.reliability = self._compute_reliability(
            score.confirmed, score.false_positives
        )
        score.noise_rate = round(score.false_positives / max(score.total_alerts, 1), 4)
        score.priority_weight = self._weight_from_reliability(score.reliability)
        score.last_updated = datetime.now(UTC)

        await self._save(score)
        logger.debug(
            "Alert outcome '%s' recorded for source %s: reliability=%.3f weight=%.3f",
            outcome,
            source_id,
            score.reliability,
            score.priority_weight,
        )

    async def get_weight(self, source_id: str) -> float:
        """Returns the priority multiplier for a source. Returns 1.0 for unknown sources."""
        engine = self._ensure_engine()
        async with engine.connect() as conn:
            result = await conn.execute(
                select(source_scores_table.c.priority_weight).where(
                    source_scores_table.c.source_id == source_id
                )
            )
            row = result.first()
        return row.priority_weight if row else 1.0

    async def score_alert(self, alert: dict) -> dict:
        """
        Takes an alert dict and returns it enriched with reliability metadata.

        Added fields:
        - source_reliability_weight: float
        - adjusted_priority: str (may be downgraded if source is noisy)
        - confidence_note: str | None
        """
        source_id = (
            alert.get("adapter_id")
            or alert.get("source_id")
            or alert.get("source")
            or "unknown"
        )

        score = await self._load_or_create(source_id)
        weight = score.priority_weight

        # Priority ladder (high → medium → low → info)
        _priority_ladder = ["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"]
        original_priority = str(
            alert.get("priority") or alert.get("severity") or "MEDIUM"
        ).upper()

        adjusted_priority = original_priority
        confidence_note: Optional[str] = None

        if original_priority in _priority_ladder:
            idx = _priority_ladder.index(original_priority)

            if weight < 0.4:
                # Very noisy source — downgrade by 2 steps
                new_idx = min(idx + 2, len(_priority_ladder) - 1)
                adjusted_priority = _priority_ladder[new_idx]
                confidence_note = (
                    f"Source '{source_id}' has {score.noise_rate:.0%} false-positive rate "
                    f"({score.false_positives}/{score.total_alerts} alerts). Priority downgraded."
                )
            elif weight < 0.7:
                # Somewhat noisy — downgrade by 1 step
                new_idx = min(idx + 1, len(_priority_ladder) - 1)
                adjusted_priority = _priority_ladder[new_idx]
                confidence_note = (
                    f"Source '{source_id}' reliability: {score.reliability:.0%}. "
                    f"Priority adjusted."
                )
            elif weight > 1.5 and idx > 0:
                # Highly reliable source — upgrade by 1 step
                new_idx = idx - 1
                adjusted_priority = _priority_ladder[new_idx]
                confidence_note = (
                    f"Source '{source_id}' is highly reliable ({score.reliability:.0%}). "
                    f"Priority elevated."
                )

        result = dict(alert)
        result["source_reliability_weight"] = weight
        result["adjusted_priority"] = adjusted_priority
        result["confidence_note"] = confidence_note
        return result

    async def get_all_scores(self) -> list[SourceReliabilityScore]:
        engine = self._ensure_engine()
        async with engine.connect() as conn:
            result = await conn.execute(
                select(source_scores_table).order_by(
                    source_scores_table.c.reliability.desc()
                )
            )
            rows = result.fetchall()
        return [
            SourceReliabilityScore(
                source_id=r.source_id,
                total_alerts=r.total_alerts,
                confirmed=r.confirmed,
                dismissed=r.dismissed,
                false_positives=r.false_positives,
                investigated=r.investigated,
                reliability=r.reliability,
                noise_rate=r.noise_rate,
                priority_weight=r.priority_weight,
                last_updated=r.last_updated,
            )
            for r in rows
        ]
