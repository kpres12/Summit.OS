"""
routers/learning.py — CyberSynetic learning API for Summit.OS.

Exposes the four learning loops to the console UI and external integrators.
All endpoints are read-heavy; writes happen only via POST /learning/feedback.

Use init_learning_router(engine) in main.py lifespan before including the router.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger("routers.learning")

router = APIRouter(prefix="/learning", tags=["learning"])

# Module-level engine reference — set by init_learning_router()
_engine = None


def init_learning_router(engine: Any) -> None:
    """Wire the CyberSynetic engine into this router. Call once during lifespan."""
    global _engine
    _engine = engine
    logger.info("Learning router initialized with CyberSynetic engine")


def _get_engine():
    if _engine is None:
        raise HTTPException(
            status_code=503,
            detail="CyberSynetic engine is not initialized"
        )
    return _engine


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class FeedbackEventRequest(BaseModel):
    event_type:        str
    user_id:           Optional[str] = None
    user_role:         Optional[str] = None
    entity_id:         Optional[str] = None
    alert_id:          Optional[str] = None
    mission_id:        Optional[str] = None
    adapter_id:        Optional[str] = None
    duration_seconds:  Optional[float] = None
    success:           Optional[bool] = None
    distance_m:        Optional[float] = None
    battery_delta_pct: Optional[float] = None
    extra:             dict = {}


class AlertScoreRequest(BaseModel):
    alert: dict


class MissionTemplateSuggestRequest(BaseModel):
    asset_types:  list[str] = []
    domain_tags:  list[str] = []
    tags:         list[str] = []
    domain:       Optional[str] = None
    limit:        int = 5


class AssetRankRequest(BaseModel):
    mission:           dict
    candidates:        list[str]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/feedback", status_code=202)
async def submit_feedback(req: FeedbackEventRequest):
    """
    Submit a feedback event to the CyberSynetic engine.

    This is the primary write path — every operator action that should
    influence the learning models should be submitted here.

    Returns 202 Accepted immediately; processing is async.
    """
    engine = _get_engine()

    # Import here to avoid circular imports at module load time
    from packages.learning.feedback import FeedbackEvent, FeedbackEventType

    try:
        event_type = FeedbackEventType(req.event_type)
    except ValueError:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown event_type '{req.event_type}'. Valid types: {[e.value for e in FeedbackEventType]}"
        )

    event = FeedbackEvent(
        event_type=event_type,
        user_id=req.user_id,
        user_role=req.user_role,
        entity_id=req.entity_id,
        alert_id=req.alert_id,
        mission_id=req.mission_id,
        adapter_id=req.adapter_id,
        duration_seconds=req.duration_seconds,
        success=req.success,
        distance_m=req.distance_m,
        battery_delta_pct=req.battery_delta_pct,
        extra=req.extra,
    )

    try:
        await engine.process_feedback(event)
    except Exception as exc:
        logger.error("process_feedback failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Feedback processing error: {exc}")

    return {"status": "accepted", "event_id": event.event_id}


@router.get("/assets")
async def list_asset_estimates():
    """
    Return all asset capability estimates, ordered by reliability score.

    These are the learned performance models for every asset the system
    has observed operating. Confidence rises with observation count.
    """
    engine = _get_engine()
    try:
        estimates = await engine.asset_model.get_all_estimates()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {
        "assets": [e.to_dict() for e in estimates],
        "total":  len(estimates),
    }


@router.get("/assets/{entity_id}")
async def get_asset_estimate(entity_id: str):
    """Return the capability estimate for a single asset."""
    engine = _get_engine()
    try:
        estimate = await engine.asset_model.get_estimate(entity_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    if estimate is None:
        raise HTTPException(
            status_code=404,
            detail=f"No capability estimate for asset '{entity_id}'"
        )
    return estimate.to_dict()


@router.post("/assets/rank")
async def rank_assets(req: AssetRankRequest):
    """
    Rank candidate assets for a mission using learned capability models.

    Returns candidates sorted by suitability score (highest first).
    """
    engine = _get_engine()
    try:
        ranked = await engine.recommend_assets(
            mission=req.mission,
            candidates=req.candidates,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {
        "ranked_assets": [
            {"entity_id": entity_id, "score": score}
            for entity_id, score in ranked
        ]
    }


@router.get("/alert-sources")
async def list_alert_sources():
    """
    Return all source reliability scores, ordered by reliability.

    Use this in the DEV view to see which sensors/adapters the system
    has learned to trust and which are generating noise.
    """
    engine = _get_engine()
    try:
        scores = await engine.alert_scorer.get_all_scores()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {
        "sources": [s.to_dict() for s in scores],
        "total":   len(scores),
    }


@router.post("/alerts/score")
async def score_alert(req: AlertScoreRequest):
    """
    Score an alert using learned source reliability.

    Returns the original alert dict with three added fields:
    - source_reliability_weight: float (0.1–2.0)
    - adjusted_priority: str (may differ from original if source is noisy/reliable)
    - confidence_note: str | None (human-readable explanation of any adjustment)
    """
    engine = _get_engine()
    try:
        scored = await engine.score_alert(req.alert)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return scored


@router.get("/mission-patterns")
async def list_mission_patterns():
    """
    Return all mission patterns in the library, ordered by success rate.

    Patterns are built automatically from completed missions. High-use,
    high-success patterns are strong candidates for templating.
    """
    engine = _get_engine()
    try:
        patterns = await engine.pattern_library.get_all_patterns()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {
        "patterns": [p.to_dict() for p in patterns],
        "total":    len(patterns),
    }


@router.get("/mission-patterns/{pattern_id}")
async def get_mission_pattern(pattern_id: str):
    """Return a single mission pattern by ID."""
    engine = _get_engine()
    try:
        pattern = await engine.pattern_library.get_pattern(pattern_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    if pattern is None:
        raise HTTPException(status_code=404, detail=f"Pattern '{pattern_id}' not found")
    return pattern.to_dict()


@router.post("/mission-patterns/suggest")
async def suggest_mission_templates(req: MissionTemplateSuggestRequest):
    """
    Suggest relevant mission templates for the current planning context.

    Ranked by: context_relevance × success_rate × log(use_count).
    Returns at most `limit` patterns.
    """
    engine = _get_engine()
    context = {
        "asset_types": req.asset_types,
        "domain_tags": req.domain_tags or req.tags,
        "domain":      req.domain,
    }
    try:
        suggestions = await engine.suggest_mission_templates(context, limit=req.limit)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {
        "suggestions": [p.to_dict() for p in suggestions],
        "total":       len(suggestions),
    }


@router.get("/intelligence")
async def get_system_intelligence():
    """
    Return a summary of what the CyberSynetic system has learned.

    This is the primary data source for the DEV view Intelligence Dashboard.
    Shows total observations, source reliability rankings, and asset confidence.
    """
    engine = _get_engine()
    try:
        intel = await engine.get_system_intelligence()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return intel
