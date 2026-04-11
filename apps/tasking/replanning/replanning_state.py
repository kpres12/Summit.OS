"""
Replanning State — Summit.OS Tasking

Tracks per-mission state used to detect replan triggers. Persists in Redis
so the trigger monitor and replanning engine can share state across restarts.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

logger = logging.getLogger("tasking.replanning.state")

_REDIS_KEY_PREFIX = "summit:replan:mission:"


@dataclass
class MissionReplanState:
    """Per-mission state snapshot for replan trigger detection."""

    mission_id: str
    asset_assignments: Dict[str, str]       # asset_id → role
    last_position: Dict[str, dict]          # asset_id → {lat, lon, ts}
    threat_proximity_m: float               # nearest threat distance
    battery_levels: Dict[str, float]        # asset_id → battery_pct (0–100)
    weather_score: float                    # 0–1 (1=good, 0=abort)
    replan_count: int = 0
    last_replan_ts: float = 0.0

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, data: str) -> "MissionReplanState":
        d = json.loads(data)
        return cls(**d)


class ReplanStateStore:
    """Persists MissionReplanState in Redis; falls back to in-memory dict."""

    def __init__(self, redis_client: Any = None):
        self._redis = redis_client
        self._memory: Dict[str, MissionReplanState] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    async def save(self, state: MissionReplanState) -> None:
        """Persist a MissionReplanState."""
        key = _REDIS_KEY_PREFIX + state.mission_id
        if self._redis is not None:
            try:
                await self._redis.set(key, state.to_json())
                return
            except Exception as e:
                logger.warning(f"Redis save failed, using memory: {e}")
        self._memory[state.mission_id] = state

    async def load(self, mission_id: str) -> Optional[MissionReplanState]:
        """Load a MissionReplanState by mission_id."""
        key = _REDIS_KEY_PREFIX + mission_id
        if self._redis is not None:
            try:
                raw = await self._redis.get(key)
                if raw:
                    return MissionReplanState.from_json(
                        raw if isinstance(raw, str) else raw.decode()
                    )
                return None
            except Exception as e:
                logger.warning(f"Redis load failed, using memory: {e}")
        return self._memory.get(mission_id)

    async def delete(self, mission_id: str) -> None:
        """Remove a mission's replan state."""
        key = _REDIS_KEY_PREFIX + mission_id
        if self._redis is not None:
            try:
                await self._redis.delete(key)
            except Exception as e:
                logger.warning(f"Redis delete failed, using memory: {e}")
        self._memory.pop(mission_id, None)

    async def all_active(self) -> List[MissionReplanState]:
        """Return all stored MissionReplanState objects."""
        if self._redis is not None:
            try:
                keys = await self._redis.keys(_REDIS_KEY_PREFIX + "*")
                states: List[MissionReplanState] = []
                for key in keys:
                    raw = await self._redis.get(key)
                    if raw:
                        states.append(
                            MissionReplanState.from_json(
                                raw if isinstance(raw, str) else raw.decode()
                            )
                        )
                return states
            except Exception as e:
                logger.warning(f"Redis all_active failed, using memory: {e}")
        return list(self._memory.values())
