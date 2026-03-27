"""
engine.py — CyberSynetic learning engine for Summit.OS.

This is the single entry point for the rest of the system. Every operator
action flows through process_feedback(); every recommendation request comes
out through score_alert(), recommend_assets(), or suggest_mission_templates().

The engine is intentionally stateless between calls — all learning state lives
in the database. This means it survives restarts and can be horizontally scaled
without coordination.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Optional

from .feedback import FeedbackEvent, FeedbackEventType, FeedbackStore
from .asset_model import AssetCapabilityEstimate, AssetPerformanceModel
from .alert_scorer import AlertQualityScorer, SourceReliabilityScore
from .mission_patterns import MissionPattern, MissionPatternLibrary

logger = logging.getLogger("learning.engine")

UTC = timezone.utc

# Event types that should trigger mission pattern recording
_MISSION_TERMINAL = frozenset(
    {
        FeedbackEventType.MISSION_COMPLETED,
        FeedbackEventType.MISSION_FAILED,
        FeedbackEventType.MISSION_ABORTED,
    }
)

# Alert outcome map: FeedbackEventType → outcome string for AlertQualityScorer
_ALERT_OUTCOME_MAP: dict[FeedbackEventType, str] = {
    FeedbackEventType.ALERT_CONFIRMED: "confirmed",
    FeedbackEventType.ALERT_DISMISSED: "dismissed",
    FeedbackEventType.ALERT_FALSE_POSITIVE: "false_positive",
    FeedbackEventType.ALERT_INVESTIGATED: "investigated",
}


class CyberSyneticEngine:
    """
    The learning and adaptation engine for Summit.OS.

    Receives feedback events and routes them to the appropriate learning models.
    Provides a unified API for recommendations.

    This is what makes Summit.OS self-improving: every operator action feeds
    back into the models, making the next recommendation slightly better than
    the last.
    """

    def __init__(self, database_url: str) -> None:
        self._db_url = database_url
        self.feedback_store = FeedbackStore(database_url)
        self.asset_model = AssetPerformanceModel(database_url)
        self.alert_scorer = AlertQualityScorer(database_url)
        self.pattern_library = MissionPatternLibrary(database_url)

    async def initialize(self) -> None:
        """Initialize all stores and models. Must be called once at startup."""
        await asyncio.gather(
            self.feedback_store.initialize(),
            self.asset_model.initialize(),
            self.alert_scorer.initialize(),
            self.pattern_library.initialize(),
        )
        logger.info("CyberSynetic engine initialized — all four learning loops ready")

    async def process_feedback(self, event: FeedbackEvent) -> None:
        """
        Route a feedback event to all relevant learning models.

        Order of operations:
        1. Always persist the raw event (audit trail + future retraining).
        2. Update asset model if this is an asset lifecycle event.
        3. Update alert scorer if this is an alert lifecycle event.
        4. Record mission pattern if this is a terminal mission event.

        Each step is independent; a failure in one does not abort the others.
        """
        # 1. Persist — always first, never optional
        try:
            await self.feedback_store.record(event)
        except Exception as exc:
            logger.error("Failed to persist feedback event %s: %s", event.event_id, exc)
            # Continue — models should still get the signal even if persistence fails

        # 2. Asset model update
        _ASSET_EVENTS = {
            FeedbackEventType.ASSET_RETURNED,
            FeedbackEventType.ASSET_MALFUNCTION,
            FeedbackEventType.ASSET_BATTERY_LOW,
            FeedbackEventType.MISSION_COMPLETED,
        }
        if event.event_type in _ASSET_EVENTS and event.entity_id:
            try:
                await self.asset_model.update(event)
            except Exception as exc:
                logger.warning(
                    "Asset model update failed for event %s: %s", event.event_id, exc
                )

        # 3. Alert scorer update
        if event.alert_id and event.event_type in _ALERT_OUTCOME_MAP:
            source_id = event.adapter_id or event.entity_id or "unknown"
            outcome = _ALERT_OUTCOME_MAP[event.event_type]
            try:
                await self.alert_scorer.record_alert_outcome(
                    alert_id=event.alert_id,
                    source_id=source_id,
                    outcome=outcome,
                )
            except Exception as exc:
                logger.warning(
                    "Alert scorer update failed for event %s: %s", event.event_id, exc
                )

        # 4. Mission pattern recording
        if event.mission_id and event.event_type in _MISSION_TERMINAL:
            outcome_map = {
                FeedbackEventType.MISSION_COMPLETED: "completed",
                FeedbackEventType.MISSION_FAILED: "failed",
                FeedbackEventType.MISSION_ABORTED: "aborted",
            }
            outcome = outcome_map[event.event_type]
            duration_s = event.duration_seconds or 0.0
            mission_context = {
                "mission_id": event.mission_id,
                **(event.extra or {}),
            }
            try:
                await self.pattern_library.record_mission(
                    mission=mission_context,
                    outcome=outcome,
                    duration_s=duration_s,
                )
            except Exception as exc:
                logger.warning(
                    "Pattern library update failed for event %s: %s",
                    event.event_id,
                    exc,
                )

        logger.debug(
            "Processed feedback event %s (%s)", event.event_id, event.event_type
        )

    async def score_alert(self, alert: dict) -> dict:
        """Score an incoming alert using learned source reliability."""
        return await self.alert_scorer.score_alert(alert)

    async def recommend_assets(
        self,
        mission: dict,
        candidates: list[str],
    ) -> list[tuple[str, float]]:
        """
        Rank candidate assets for a mission using learned capability models.

        Reads required_range_m and required_endurance_s from the mission dict,
        falling back to sensible defaults.
        """
        required_range_m = float(
            mission.get("required_range_m") or mission.get("range_m") or 1000.0
        )
        required_endurance_s = float(
            mission.get("required_endurance_s") or mission.get("endurance_s") or 600.0
        )
        return await self.asset_model.rank_assets_for_mission(
            candidates=candidates,
            required_range_m=required_range_m,
            required_endurance_s=required_endurance_s,
        )

    async def suggest_mission_templates(
        self,
        context: dict,
        limit: int = 5,
    ) -> list[MissionPattern]:
        """
        Suggest relevant mission templates from the pattern library.

        context keys used:
        - asset_types: list[str]
        - domain_tags / tags / domain: list[str] | str
        """
        asset_types = context.get("asset_types") or []
        context_tags = (
            context.get("domain_tags")
            or context.get("tags")
            or ([context["domain"]] if context.get("domain") else [])
        )
        if isinstance(context_tags, str):
            context_tags = [context_tags]
        return await self.pattern_library.suggest_templates(
            asset_types=list(asset_types),
            context_tags=list(context_tags),
            limit=limit,
        )

    async def get_system_intelligence(self) -> dict[str, Any]:
        """
        Returns a summary of what the system has learned — for the DEV view.

        This is the "proof of life" readout: how much the system has observed
        and what the signal quality looks like across sources and assets.
        """
        try:
            all_estimates = await self.asset_model.get_all_estimates()
        except Exception:
            all_estimates = []

        try:
            all_scores = await self.alert_scorer.get_all_scores()
        except Exception:
            all_scores = []

        try:
            all_patterns = await self.pattern_library.get_all_patterns()
        except Exception:
            all_patterns = []

        try:
            total_events = await self.feedback_store.total_count()
        except Exception:
            total_events = 0

        total_missions = sum(p.use_count for p in all_patterns)

        # Top reliable sources (reliability ≥ 0.7, sorted desc)
        reliable_sources = sorted(
            [s for s in all_scores if s.reliability >= 0.7],
            key=lambda s: s.reliability,
            reverse=True,
        )[:5]

        # Top noisy sources (noise_rate ≥ 0.3, sorted desc)
        noisy_sources = sorted(
            [s for s in all_scores if s.noise_rate >= 0.3],
            key=lambda s: s.noise_rate,
            reverse=True,
        )[:5]

        return {
            "assets_modeled": len(all_estimates),
            "total_missions_learned": total_missions,
            "alert_sources_scored": len(all_scores),
            "mission_patterns": len(all_patterns),
            "total_feedback_events": total_events,
            "top_reliable_sources": [
                {
                    "source_id": s.source_id,
                    "reliability": s.reliability,
                    "total_alerts": s.total_alerts,
                    "priority_weight": s.priority_weight,
                }
                for s in reliable_sources
            ],
            "top_noisy_sources": [
                {
                    "source_id": s.source_id,
                    "noise_rate": s.noise_rate,
                    "false_positives": s.false_positives,
                    "total_alerts": s.total_alerts,
                    "priority_weight": s.priority_weight,
                }
                for s in noisy_sources
            ],
            "asset_summary": [
                {
                    "entity_id": e.entity_id,
                    "entity_type": e.entity_type,
                    "reliability_score": e.reliability_score,
                    "observations": e.observations,
                    "confidence": e.confidence,
                }
                for e in sorted(
                    all_estimates, key=lambda e: e.reliability_score, reverse=True
                )[:10]
            ],
        }
