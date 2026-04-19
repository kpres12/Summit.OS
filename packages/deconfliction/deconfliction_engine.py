"""
Deconfliction Engine — Heli.OS

Detects 3D airspace conflicts between active assets and planned flight paths.
Runs at 0.5s tick rate (asyncio task). Emits conflict alerts to MQTT
summit/deconfliction/conflicts.

Scales to ~200 active assets (O(n²) comparison, ~20k pairs at 200 assets,
well within 0.5s budget on any modern hardware).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .volume import CylinderVolume, overlaps
from .resolution import ConflictResolution, ResolutionPlanner

logger = logging.getLogger("deconfliction.engine")

TICK_INTERVAL_S: float = 0.5

# Severity thresholds (seconds to predicted conflict)
IMMINENT_S: float = 5.0
WARNING_S: float = 30.0
ADVISORY_S: float = 60.0


@dataclass
class Conflict:
    """A detected 3D airspace conflict between two entities."""

    conflict_id: str
    asset_a: str
    asset_b: str
    volume_a: CylinderVolume
    volume_b: CylinderVolume
    time_to_conflict_s: float   # estimated seconds until overlap (0 = already overlapping)
    severity: str               # IMMINENT / WARNING / ADVISORY
    ts: float = field(default_factory=time.time)


def _severity(time_to_conflict_s: float) -> str:
    if time_to_conflict_s <= IMMINENT_S:
        return "IMMINENT"
    if time_to_conflict_s <= WARNING_S:
        return "WARNING"
    return "ADVISORY"


class DeconflictionEngine:
    """Detects airspace conflicts and resolves them at 0.5s tick rate."""

    def __init__(
        self,
        tick_interval_s: float = TICK_INTERVAL_S,
        mqtt_client: Any = None,
    ):
        self._tick_interval_s = tick_interval_s
        self._mqtt = mqtt_client
        # entity_id → CylinderVolume
        self._active_volumes: Dict[str, CylinderVolume] = {}
        # conflict_id → Conflict (unresolved)
        self._active_conflicts: Dict[str, Conflict] = {}
        self._planner = ResolutionPlanner()

    # ── Volume management ─────────────────────────────────────────────────────

    def update_volume(self, entity_id: str, volume: CylinderVolume) -> None:
        """Register or update an entity's current airspace volume."""
        self._active_volumes[entity_id] = volume

    def remove_volume(self, entity_id: str) -> None:
        """Deregister an entity from the deconfliction engine."""
        self._active_volumes.pop(entity_id, None)
        # Clear any active conflicts involving this entity
        to_remove = [
            cid for cid, c in self._active_conflicts.items()
            if c.asset_a == entity_id or c.asset_b == entity_id
        ]
        for cid in to_remove:
            self._active_conflicts.pop(cid, None)

    # ── Main loop ─────────────────────────────────────────────────────────────

    async def run(self) -> None:
        """Main tick loop: detect conflicts, resolve, publish."""
        logger.info(
            f"DeconflictionEngine started (tick={self._tick_interval_s}s)"
        )
        while True:
            try:
                await self._tick()
            except Exception as e:
                logger.error(f"DeconflictionEngine tick error: {e}", exc_info=True)
            await asyncio.sleep(self._tick_interval_s)

    async def _tick(self) -> None:
        # Purge stale volumes
        now = time.time()
        stale = [eid for eid, v in self._active_volumes.items() if v.is_stale(now)]
        for eid in stale:
            self.remove_volume(eid)

        conflicts = self._detect_conflicts()
        self._active_conflicts.clear()

        for conflict in conflicts:
            self._active_conflicts[conflict.conflict_id] = conflict
            resolution = self._planner.resolve(conflict)
            await self._publish_conflict(conflict, resolution)

    # ── Detection ─────────────────────────────────────────────────────────────

    def _detect_conflicts(self) -> List[Conflict]:
        """O(n²) scan of active volumes for overlapping cylinders."""
        entities = list(self._active_volumes.items())
        detected: List[Conflict] = []
        seen: set[frozenset] = set()

        for i in range(len(entities)):
            eid_a, vol_a = entities[i]
            for j in range(i + 1, len(entities)):
                eid_b, vol_b = entities[j]
                pair = frozenset({eid_a, eid_b})
                if pair in seen:
                    continue
                seen.add(pair)

                if overlaps(vol_a, vol_b):
                    # Already overlapping → time_to_conflict = 0
                    ttc = 0.0
                    sev = _severity(ttc)
                    conflict = Conflict(
                        conflict_id=str(uuid.uuid4()),
                        asset_a=eid_a,
                        asset_b=eid_b,
                        volume_a=vol_a,
                        volume_b=vol_b,
                        time_to_conflict_s=ttc,
                        severity=sev,
                    )
                    detected.append(conflict)

        return detected

    # ── Publishing ────────────────────────────────────────────────────────────

    async def _publish_conflict(
        self, conflict: Conflict, resolution: ConflictResolution
    ) -> None:
        """Publish conflict + resolution to MQTT."""
        payload = json.dumps(
            {
                "conflict_id": conflict.conflict_id,
                "asset_a": conflict.asset_a,
                "asset_b": conflict.asset_b,
                "severity": conflict.severity,
                "time_to_conflict_s": conflict.time_to_conflict_s,
                "resolution": {
                    "strategy": resolution.strategy,
                    "instructions": resolution.instructions,
                },
                "ts": conflict.ts,
            }
        )
        topic = "summit/deconfliction/conflicts"
        if self._mqtt:
            try:
                self._mqtt.publish(topic, payload, qos=1)
            except Exception as e:
                logger.warning(f"MQTT publish failed for conflict {conflict.conflict_id}: {e}")
        else:
            logger.debug(f"Conflict detected [{conflict.severity}]: {conflict.asset_a} ↔ {conflict.asset_b}")

    # ── Queries ───────────────────────────────────────────────────────────────

    def get_active_conflicts(self) -> List[dict]:
        """Return all currently unresolved conflicts as plain dicts."""
        return [
            {
                "conflict_id": c.conflict_id,
                "asset_a": c.asset_a,
                "asset_b": c.asset_b,
                "severity": c.severity,
                "time_to_conflict_s": c.time_to_conflict_s,
                "ts": c.ts,
            }
            for c in self._active_conflicts.values()
        ]
