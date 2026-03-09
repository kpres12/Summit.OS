"""
asset_model.py — Bayesian asset capability model for Summit.OS CyberSynetic engine.

Learns the real-world performance characteristics of each asset (drone, vehicle,
sensor) from observed mission outcomes. Starts with reasonable priors and updates
incrementally with each completed mission.

The key insight: manufacturer specs are the prior. Observed performance is the
evidence. After enough missions, the model reflects reality — not the brochure.
"""
from __future__ import annotations

import json
import logging
import math
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
    Text,
    insert,
    select,
    update,
)
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from .feedback import FeedbackEvent, FeedbackEventType

logger = logging.getLogger("learning.asset_model")

UTC = timezone.utc

# ---------------------------------------------------------------------------
# Priors per entity type — reasonable defaults before any observations
# ---------------------------------------------------------------------------

_PRIORS: dict[str, dict] = {
    "drone": {
        "estimated_range_m":     5000.0,
        "estimated_endurance_s": 1800.0,   # 30 min
        "battery_drain_rate":    2.5,       # % per minute
        "reliability_score":     0.85,
        "avg_speed_mps":         12.0,
    },
    "uav": {
        "estimated_range_m":     8000.0,
        "estimated_endurance_s": 3600.0,
        "battery_drain_rate":    1.8,
        "reliability_score":     0.88,
        "avg_speed_mps":         18.0,
    },
    "vehicle": {
        "estimated_range_m":     50000.0,
        "estimated_endurance_s": 14400.0,  # 4 hrs
        "battery_drain_rate":    0.3,
        "reliability_score":     0.92,
        "avg_speed_mps":         8.0,
    },
    "sensor": {
        "estimated_range_m":     500.0,
        "estimated_endurance_s": 86400.0,  # 24 hrs
        "battery_drain_rate":    0.1,
        "reliability_score":     0.95,
        "avg_speed_mps":         0.0,
    },
    "default": {
        "estimated_range_m":     3000.0,
        "estimated_endurance_s": 1800.0,
        "battery_drain_rate":    2.0,
        "reliability_score":     0.80,
        "avg_speed_mps":         5.0,
    },
}


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class AssetCapabilityEstimate:
    entity_id:             str
    entity_type:           str
    estimated_range_m:     float
    estimated_endurance_s: float
    battery_drain_rate:    float    # % per minute under load
    reliability_score:     float    # 0–1, Bayesian mission completion rate
    avg_speed_mps:         float
    observations:          int      # number of missions that informed this
    last_updated:          datetime
    confidence:            float    # 0–1, grows with observations

    def to_dict(self) -> dict:
        d = asdict(self)
        d["last_updated"] = self.last_updated.isoformat()
        return d


# ---------------------------------------------------------------------------
# SQLAlchemy table
# ---------------------------------------------------------------------------

_metadata = MetaData()

asset_estimates_table = Table(
    "asset_capability_estimates",
    _metadata,
    Column("id",                    Integer, primary_key=True, autoincrement=True),
    Column("entity_id",             String(128), nullable=False, unique=True),
    Column("entity_type",           String(64),  nullable=False, default="default"),
    Column("estimated_range_m",     Float,       nullable=False),
    Column("estimated_endurance_s", Float,       nullable=False),
    Column("battery_drain_rate",    Float,       nullable=False),
    Column("reliability_score",     Float,       nullable=False),
    Column("avg_speed_mps",         Float,       nullable=False),
    Column("observations",          Integer,     nullable=False, default=0),
    Column("last_updated",          DateTime(timezone=True), nullable=False),
    Column("confidence",            Float,       nullable=False, default=0.0),
    Index("ix_ace_entity_id", "entity_id"),
)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class AssetPerformanceModel:
    """
    Maintains capability estimates for all known assets.

    Bayesian update strategy:
    - Continuous metrics (range, endurance, speed, battery drain) use
      exponential moving average with weight inversely proportional to
      observation count — early observations move the needle more.
    - Reliability (binary: succeeded/failed) uses Beta distribution updating:
      reliability = (alpha) / (alpha + beta), where alpha = successes + prior_alpha
      and beta = failures + prior_beta.

    The confidence score is derived from observation count via a logistic curve
    so it asymptotes at ~1.0 and climbs quickly in the first 20 missions.
    """

    def __init__(self, database_url: str) -> None:
        self._db_url = database_url
        self._engine: Optional[AsyncEngine] = None

    async def initialize(self) -> None:
        self._engine = create_async_engine(self._db_url, echo=False, future=True)
        async with self._engine.begin() as conn:
            await conn.run_sync(_metadata.create_all)
        logger.info("AssetPerformanceModel initialized")

    def _ensure_engine(self) -> AsyncEngine:
        if self._engine is None:
            raise RuntimeError("AssetPerformanceModel.initialize() has not been called")
        return self._engine

    @staticmethod
    def _confidence(observations: int) -> float:
        """Logistic confidence: 0 at 0 obs, ~0.5 at 10, ~0.88 at 30, asymptote 1.0."""
        if observations <= 0:
            return 0.0
        return round(1.0 - 1.0 / (1.0 + observations / 10.0), 4)

    @staticmethod
    def _ema(current: float, new_value: float, n: int) -> float:
        """
        Bayesian-flavored EMA: the learning rate decreases as n grows.
        alpha = 2 / (n + 1), clamped so we move meaningfully on early data.
        """
        alpha = max(0.05, min(0.5, 2.0 / (n + 1)))
        return current * (1.0 - alpha) + new_value * alpha

    @staticmethod
    def _beta_reliability(current: float, n: int, success: bool) -> float:
        """
        Beta-Binomial update for reliability.
        Treats current estimate as derived from (alpha, beta) with n observations.
        Prior pseudo-counts: alpha0=2, beta0=0.5 (optimistic prior — assets
        generally work unless proven otherwise).
        """
        alpha0, beta0 = 2.0, 0.5
        # Reconstruct implied alpha/beta from current estimate and n
        alpha = current * n + alpha0
        beta  = (1.0 - current) * n + beta0
        if success:
            alpha += 1.0
        else:
            beta += 1.0
        new_estimate = alpha / (alpha + beta)
        return round(min(0.999, max(0.001, new_estimate)), 4)

    def _prior_for(self, entity_type: str) -> dict:
        t = (entity_type or "default").lower()
        return dict(_PRIORS.get(t, _PRIORS["default"]))

    async def _load_or_create(self, entity_id: str, entity_type: str = "default") -> AssetCapabilityEstimate:
        engine = self._ensure_engine()
        async with engine.connect() as conn:
            result = await conn.execute(
                select(asset_estimates_table).where(
                    asset_estimates_table.c.entity_id == entity_id
                )
            )
            row = result.first()

        if row:
            return AssetCapabilityEstimate(
                entity_id=row.entity_id,
                entity_type=row.entity_type,
                estimated_range_m=row.estimated_range_m,
                estimated_endurance_s=row.estimated_endurance_s,
                battery_drain_rate=row.battery_drain_rate,
                reliability_score=row.reliability_score,
                avg_speed_mps=row.avg_speed_mps,
                observations=row.observations,
                last_updated=row.last_updated,
                confidence=row.confidence,
            )

        # First time we see this asset — seed with priors
        prior = self._prior_for(entity_type)
        return AssetCapabilityEstimate(
            entity_id=entity_id,
            entity_type=entity_type,
            estimated_range_m=prior["estimated_range_m"],
            estimated_endurance_s=prior["estimated_endurance_s"],
            battery_drain_rate=prior["battery_drain_rate"],
            reliability_score=prior["reliability_score"],
            avg_speed_mps=prior["avg_speed_mps"],
            observations=0,
            last_updated=datetime.now(UTC),
            confidence=0.0,
        )

    async def _save(self, est: AssetCapabilityEstimate) -> None:
        engine = self._ensure_engine()
        row_data = {
            "entity_type":           est.entity_type,
            "estimated_range_m":     est.estimated_range_m,
            "estimated_endurance_s": est.estimated_endurance_s,
            "battery_drain_rate":    est.battery_drain_rate,
            "reliability_score":     est.reliability_score,
            "avg_speed_mps":         est.avg_speed_mps,
            "observations":          est.observations,
            "last_updated":          est.last_updated,
            "confidence":            est.confidence,
        }
        async with engine.begin() as conn:
            # Upsert pattern
            existing = await conn.execute(
                select(asset_estimates_table.c.id).where(
                    asset_estimates_table.c.entity_id == est.entity_id
                )
            )
            if existing.first():
                await conn.execute(
                    update(asset_estimates_table)
                    .where(asset_estimates_table.c.entity_id == est.entity_id)
                    .values(**row_data)
                )
            else:
                await conn.execute(
                    insert(asset_estimates_table).values(entity_id=est.entity_id, **row_data)
                )

    async def update(self, event: FeedbackEvent) -> None:
        """Update capability estimate from a feedback event."""
        if event.entity_id is None:
            return

        entity_type = event.extra.get("entity_type", "default")
        est = await self._load_or_create(event.entity_id, entity_type)
        n = est.observations
        changed = False

        if event.event_type == FeedbackEventType.ASSET_RETURNED:
            # Asset completed a mission — update range, endurance, battery drain, speed
            est.observations += 1
            n = est.observations
            changed = True

            if event.distance_m and event.distance_m > 0:
                est.estimated_range_m = round(
                    self._ema(est.estimated_range_m, event.distance_m, n), 2
                )

            if event.duration_seconds and event.duration_seconds > 0:
                est.estimated_endurance_s = round(
                    self._ema(est.estimated_endurance_s, event.duration_seconds, n), 2
                )
                if event.distance_m and event.distance_m > 0:
                    speed = event.distance_m / event.duration_seconds
                    est.avg_speed_mps = round(self._ema(est.avg_speed_mps, speed, n), 3)

            if event.battery_delta_pct and event.battery_delta_pct > 0:
                minutes = (event.duration_seconds or 1.0) / 60.0
                drain_rate = event.battery_delta_pct / max(minutes, 0.1)
                est.battery_drain_rate = round(
                    self._ema(est.battery_drain_rate, drain_rate, n), 3
                )

            # Successful return — update reliability positively
            est.reliability_score = self._beta_reliability(est.reliability_score, n - 1, success=True)

        elif event.event_type == FeedbackEventType.ASSET_MALFUNCTION:
            # Mission failure — reliability takes a hit
            est.observations += 1
            n = est.observations
            est.reliability_score = self._beta_reliability(est.reliability_score, n - 1, success=False)
            changed = True

        elif event.event_type == FeedbackEventType.ASSET_BATTERY_LOW:
            # Battery warning — drain rate is higher than estimated
            est.observations += 1
            n = est.observations
            # Penalise endurance estimate slightly: assume true endurance is
            # 80% of what we thought when battery is hitting low
            est.estimated_endurance_s = round(est.estimated_endurance_s * 0.97, 2)
            changed = True

        elif event.event_type == FeedbackEventType.MISSION_COMPLETED:
            # Mission-level completion — contributes to reliability even if
            # no direct ASSET_RETURNED event was emitted
            if event.duration_seconds and event.duration_seconds > 0:
                est.observations += 1
                n = est.observations
                est.estimated_endurance_s = round(
                    self._ema(est.estimated_endurance_s, event.duration_seconds, n), 2
                )
                est.reliability_score = self._beta_reliability(est.reliability_score, n - 1, success=True)
                changed = True

        if changed:
            est.confidence   = self._confidence(est.observations)
            est.last_updated = datetime.now(UTC)
            await self._save(est)
            logger.debug(
                "Updated asset estimate for %s: reliability=%.3f observations=%d",
                event.entity_id,
                est.reliability_score,
                est.observations,
            )

    async def get_estimate(self, entity_id: str) -> Optional[AssetCapabilityEstimate]:
        engine = self._ensure_engine()
        async with engine.connect() as conn:
            result = await conn.execute(
                select(asset_estimates_table).where(
                    asset_estimates_table.c.entity_id == entity_id
                )
            )
            row = result.first()
        if not row:
            return None
        return AssetCapabilityEstimate(
            entity_id=row.entity_id,
            entity_type=row.entity_type,
            estimated_range_m=row.estimated_range_m,
            estimated_endurance_s=row.estimated_endurance_s,
            battery_drain_rate=row.battery_drain_rate,
            reliability_score=row.reliability_score,
            avg_speed_mps=row.avg_speed_mps,
            observations=row.observations,
            last_updated=row.last_updated,
            confidence=row.confidence,
        )

    async def rank_assets_for_mission(
        self,
        candidates: list[str],
        required_range_m: float,
        required_endurance_s: float,
    ) -> list[tuple[str, float]]:
        """
        Rank candidates by suitability score.

        Score = reliability × range_headroom × endurance_headroom

        headroom is the ratio of estimated capability to required capability,
        clamped at 1.0 (no benefit to exceeding requirement by more than 100%).
        Assets that cannot meet requirements get a near-zero score but are still
        returned — the human decides.
        """
        scored: list[tuple[str, float]] = []
        for entity_id in candidates:
            est = await self._load_or_create(entity_id)

            range_ratio      = est.estimated_range_m / max(required_range_m, 1.0)
            endurance_ratio  = est.estimated_endurance_s / max(required_endurance_s, 1.0)

            # Headroom: clamped 0.1–1.0 (under-capability penalised, over-capability not rewarded)
            range_headroom     = max(0.1, min(1.0, range_ratio))
            endurance_headroom = max(0.1, min(1.0, endurance_ratio))

            score = est.reliability_score * range_headroom * endurance_headroom
            scored.append((entity_id, round(score, 4)))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    async def get_all_estimates(self) -> list[AssetCapabilityEstimate]:
        engine = self._ensure_engine()
        async with engine.connect() as conn:
            result = await conn.execute(
                select(asset_estimates_table).order_by(
                    asset_estimates_table.c.reliability_score.desc()
                )
            )
            rows = result.fetchall()
        return [
            AssetCapabilityEstimate(
                entity_id=r.entity_id,
                entity_type=r.entity_type,
                estimated_range_m=r.estimated_range_m,
                estimated_endurance_s=r.estimated_endurance_s,
                battery_drain_rate=r.battery_drain_rate,
                reliability_score=r.reliability_score,
                avg_speed_mps=r.avg_speed_mps,
                observations=r.observations,
                last_updated=r.last_updated,
                confidence=r.confidence,
            )
            for r in rows
        ]
