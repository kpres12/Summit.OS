"""
feedback.py — Core feedback event schema and collection for Summit.OS CyberSynetic engine.

Every operator action is a learning signal. This module defines the canonical
event schema and the store that persists all feedback for the learning loops.

Compatible with both SQLite (dev) and PostgreSQL (production) via SQLAlchemy Core.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    func,
    insert,
    select,
)
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

logger = logging.getLogger("learning.feedback")

UTC = timezone.utc


# ---------------------------------------------------------------------------
# Event schema
# ---------------------------------------------------------------------------


class FeedbackEventType(str, Enum):
    # Alert lifecycle
    ALERT_INVESTIGATED = "ALERT_INVESTIGATED"  # operator clicked investigate
    ALERT_DISMISSED = "ALERT_DISMISSED"  # operator dismissed as not relevant
    ALERT_FALSE_POSITIVE = "ALERT_FALSE_POSITIVE"  # operator marked as false positive
    ALERT_CONFIRMED = "ALERT_CONFIRMED"  # operator confirmed real event

    # Mission lifecycle
    MISSION_CREATED = "MISSION_CREATED"
    MISSION_DISPATCHED = "MISSION_DISPATCHED"  # assets sent
    MISSION_COMPLETED = "MISSION_COMPLETED"  # objective achieved
    MISSION_ABORTED = "MISSION_ABORTED"  # operator cancelled
    MISSION_FAILED = "MISSION_FAILED"  # failed to complete objective

    # Asset performance
    ASSET_DISPATCHED = "ASSET_DISPATCHED"
    ASSET_RETURNED = "ASSET_RETURNED"  # mission complete, asset back
    ASSET_BATTERY_LOW = "ASSET_BATTERY_LOW"  # battery warning during mission
    ASSET_MALFUNCTION = "ASSET_MALFUNCTION"  # asset failed during mission

    # Recommendation feedback
    RECOMMENDATION_ACCEPTED = "RECOMMENDATION_ACCEPTED"
    RECOMMENDATION_REJECTED = "RECOMMENDATION_REJECTED"
    RECOMMENDATION_MODIFIED = "RECOMMENDATION_MODIFIED"  # accepted but changed


class FeedbackEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type: FeedbackEventType
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    user_id: Optional[str] = None
    user_role: Optional[str] = None

    # What the event is about
    entity_id: Optional[str] = None  # asset involved
    alert_id: Optional[str] = None
    mission_id: Optional[str] = None
    adapter_id: Optional[str] = None  # signal source

    # Outcome data (filled in for completion events)
    duration_seconds: Optional[float] = None
    success: Optional[bool] = None
    distance_m: Optional[float] = None  # for asset missions
    battery_delta_pct: Optional[float] = None  # battery used

    # Free-form context
    extra: dict = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# SQLAlchemy table definition
# ---------------------------------------------------------------------------

_metadata = MetaData()

feedback_events_table = Table(
    "feedback_events",
    _metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("event_id", String(64), nullable=False, unique=True),
    Column("event_type", String(64), nullable=False),
    Column("timestamp", DateTime(timezone=True), nullable=False),
    Column("user_id", String(128)),
    Column("user_role", String(64)),
    Column("entity_id", String(128)),
    Column("alert_id", String(128)),
    Column("mission_id", String(128)),
    Column("adapter_id", String(128)),
    Column("duration_seconds", Float),
    Column("success", Boolean),
    Column("distance_m", Float),
    Column("battery_delta_pct", Float),
    Column("extra_json", Text, default="{}"),
    Index("ix_fe_entity_id", "entity_id"),
    Index("ix_fe_alert_id", "alert_id"),
    Index("ix_fe_mission_id", "mission_id"),
    Index("ix_fe_event_type", "event_type"),
    Index("ix_fe_timestamp", "timestamp"),
)


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class FeedbackStore:
    """
    Persistent store for all feedback events.

    All four learning loops read from this store; the engine writes to it
    on every operator action.
    """

    def __init__(self, database_url: str) -> None:
        self._db_url = database_url
        self._engine: Optional[AsyncEngine] = None

    async def initialize(self) -> None:
        """Create the feedback_events table if it does not exist."""
        self._engine = create_async_engine(self._db_url, echo=False, future=True)
        async with self._engine.begin() as conn:
            await conn.run_sync(_metadata.create_all)
        logger.info("FeedbackStore initialized")

    def _ensure_engine(self) -> AsyncEngine:
        if self._engine is None:
            raise RuntimeError("FeedbackStore.initialize() has not been called")
        return self._engine

    async def record(self, event: FeedbackEvent) -> None:
        """Persist a feedback event."""
        engine = self._ensure_engine()
        row = {
            "event_id": event.event_id,
            "event_type": event.event_type.value,
            "timestamp": event.timestamp,
            "user_id": event.user_id,
            "user_role": event.user_role,
            "entity_id": event.entity_id,
            "alert_id": event.alert_id,
            "mission_id": event.mission_id,
            "adapter_id": event.adapter_id,
            "duration_seconds": event.duration_seconds,
            "success": event.success,
            "distance_m": event.distance_m,
            "battery_delta_pct": event.battery_delta_pct,
            "extra_json": json.dumps(event.extra),
        }
        async with engine.begin() as conn:
            await conn.execute(insert(feedback_events_table).values(**row))
        logger.debug(
            "Recorded feedback event %s (%s)", event.event_id, event.event_type
        )

    async def query(
        self,
        entity_id: Optional[str] = None,
        alert_id: Optional[str] = None,
        mission_id: Optional[str] = None,
        event_type: Optional[FeedbackEventType] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> list[FeedbackEvent]:
        """Return feedback events matching the given filters."""
        engine = self._ensure_engine()
        stmt = select(feedback_events_table)
        if entity_id:
            stmt = stmt.where(feedback_events_table.c.entity_id == entity_id)
        if alert_id:
            stmt = stmt.where(feedback_events_table.c.alert_id == alert_id)
        if mission_id:
            stmt = stmt.where(feedback_events_table.c.mission_id == mission_id)
        if event_type:
            stmt = stmt.where(feedback_events_table.c.event_type == event_type.value)
        if since:
            stmt = stmt.where(feedback_events_table.c.timestamp >= since)
        stmt = stmt.order_by(feedback_events_table.c.timestamp.desc()).limit(limit)

        async with engine.connect() as conn:
            result = await conn.execute(stmt)
            rows = result.fetchall()

        events = []
        for row in rows:
            extra = {}
            try:
                extra = json.loads(row.extra_json or "{}")
            except Exception:
                pass
            events.append(
                FeedbackEvent(
                    event_id=row.event_id,
                    event_type=FeedbackEventType(row.event_type),
                    timestamp=row.timestamp,
                    user_id=row.user_id,
                    user_role=row.user_role,
                    entity_id=row.entity_id,
                    alert_id=row.alert_id,
                    mission_id=row.mission_id,
                    adapter_id=row.adapter_id,
                    duration_seconds=row.duration_seconds,
                    success=row.success,
                    distance_m=row.distance_m,
                    battery_delta_pct=row.battery_delta_pct,
                    extra=extra,
                )
            )
        return events

    async def get_stats(self, entity_id: str) -> dict[str, Any]:
        """
        Return aggregated performance statistics for a given entity (asset).

        Computed directly from the raw event stream so models can bootstrap
        from stored history without requiring a separate aggregation pass.
        """
        engine = self._ensure_engine()

        # Mission counts
        async with engine.connect() as conn:
            total_q = await conn.execute(
                select(func.count()).where(
                    feedback_events_table.c.entity_id == entity_id
                )
            )
            total = total_q.scalar() or 0

            completed_q = await conn.execute(
                select(func.count()).where(
                    (feedback_events_table.c.entity_id == entity_id)
                    & (
                        feedback_events_table.c.event_type
                        == FeedbackEventType.ASSET_RETURNED.value
                    )
                )
            )
            completed = completed_q.scalar() or 0

            malfunctions_q = await conn.execute(
                select(func.count()).where(
                    (feedback_events_table.c.entity_id == entity_id)
                    & (
                        feedback_events_table.c.event_type
                        == FeedbackEventType.ASSET_MALFUNCTION.value
                    )
                )
            )
            malfunctions = malfunctions_q.scalar() or 0

            dispatches_q = await conn.execute(
                select(func.count()).where(
                    (feedback_events_table.c.entity_id == entity_id)
                    & (
                        feedback_events_table.c.event_type
                        == FeedbackEventType.ASSET_DISPATCHED.value
                    )
                )
            )
            dispatches = dispatches_q.scalar() or 0

            avg_duration_q = await conn.execute(
                select(func.avg(feedback_events_table.c.duration_seconds)).where(
                    (feedback_events_table.c.entity_id == entity_id)
                    & (feedback_events_table.c.duration_seconds.isnot(None))
                )
            )
            avg_duration = avg_duration_q.scalar()

            avg_battery_q = await conn.execute(
                select(func.avg(feedback_events_table.c.battery_delta_pct)).where(
                    (feedback_events_table.c.entity_id == entity_id)
                    & (feedback_events_table.c.battery_delta_pct.isnot(None))
                )
            )
            avg_battery = avg_battery_q.scalar()

            avg_distance_q = await conn.execute(
                select(func.avg(feedback_events_table.c.distance_m)).where(
                    (feedback_events_table.c.entity_id == entity_id)
                    & (feedback_events_table.c.distance_m.isnot(None))
                )
            )
            avg_distance = avg_distance_q.scalar()

        missions_attempted = dispatches or completed
        failure_rate = (
            (malfunctions / missions_attempted) if missions_attempted > 0 else 0.0
        )

        return {
            "entity_id": entity_id,
            "total_events": total,
            "missions_completed": completed,
            "missions_attempted": missions_attempted,
            "malfunctions": malfunctions,
            "failure_rate": round(failure_rate, 4),
            "avg_duration_s": (
                round(avg_duration, 2) if avg_duration is not None else None
            ),
            "avg_battery_used_pct": (
                round(avg_battery, 2) if avg_battery is not None else None
            ),
            "avg_distance_m": (
                round(avg_distance, 2) if avg_distance is not None else None
            ),
        }

    async def total_count(self) -> int:
        """Return total number of feedback events recorded."""
        engine = self._ensure_engine()
        async with engine.connect() as conn:
            result = await conn.execute(
                select(func.count()).select_from(feedback_events_table)
            )
            return result.scalar() or 0
