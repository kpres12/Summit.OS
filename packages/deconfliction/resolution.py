"""
Deconfliction Resolution — Heli.OS

Resolution strategies for detected airspace conflicts:
  ALTITUDE_SEPARATION  — assign different altitude layers (every 30m)
  TIME_SEPARATION      — stagger departure/arrival times
  ROUTE_OFFSET         — shift flight path laterally by conflict radius + margin
  PRIORITY_YIELD       — lower-priority asset holds, higher proceeds
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict

if TYPE_CHECKING:
    from .deconfliction_engine import Conflict

ALTITUDE_LAYER_M: float = 30.0      # altitude separation per layer
TIME_STAGGER_S: float = 30.0        # seconds to delay lower-priority asset
ROUTE_MARGIN_M: float = 10.0        # extra margin beyond combined radii


@dataclass
class ConflictResolution:
    """The computed resolution for a detected airspace conflict."""

    conflict_id: str
    strategy: str
    asset_a: str
    asset_b: str
    instructions: Dict[str, dict]   # asset_id → {action, params}
    ts: float = field(default_factory=time.time)


class ResolutionPlanner:
    """Selects and computes a resolution strategy for a given Conflict."""

    def resolve(self, conflict: "Conflict") -> ConflictResolution:
        """Pick the best resolution strategy and return instructions.

        Priority rules:
        - If one asset has strictly higher priority → PRIORITY_YIELD
        - If conflict is IMMINENT (< 5s) → ALTITUDE_SEPARATION (fastest)
        - Default → ALTITUDE_SEPARATION
        """
        if conflict.volume_a.priority != conflict.volume_b.priority:
            return self._priority_yield(conflict)
        if conflict.severity == "IMMINENT":
            return self._altitude_separation(conflict)
        return self._altitude_separation(conflict)

    def _altitude_separation(self, conflict: "Conflict") -> ConflictResolution:
        """Lower-priority asset climbs one altitude layer (30m)."""
        # Asset with lower priority gets the altitude bump; if equal, asset_b yields
        if conflict.volume_a.priority >= conflict.volume_b.priority:
            yield_asset = conflict.asset_b
            yield_vol = conflict.volume_b
            keep_asset = conflict.asset_a
        else:
            yield_asset = conflict.asset_a
            yield_vol = conflict.volume_a
            keep_asset = conflict.asset_b

        new_alt = yield_vol.alt_ceil_m + ALTITUDE_LAYER_M
        return ConflictResolution(
            conflict_id=conflict.conflict_id,
            strategy="ALTITUDE_SEPARATION",
            asset_a=conflict.asset_a,
            asset_b=conflict.asset_b,
            instructions={
                yield_asset: {
                    "action": "CHANGE_ALTITUDE",
                    "target_alt_m": new_alt,
                    "reason": "deconfliction_altitude_separation",
                },
                keep_asset: {
                    "action": "MAINTAIN",
                    "reason": "deconfliction_altitude_separation",
                },
            },
        )

    def _time_separation(self, conflict: "Conflict") -> ConflictResolution:
        """Lower-priority asset delays by TIME_STAGGER_S seconds."""
        if conflict.volume_a.priority >= conflict.volume_b.priority:
            yield_asset = conflict.asset_b
            keep_asset = conflict.asset_a
        else:
            yield_asset = conflict.asset_a
            keep_asset = conflict.asset_b

        return ConflictResolution(
            conflict_id=conflict.conflict_id,
            strategy="TIME_SEPARATION",
            asset_a=conflict.asset_a,
            asset_b=conflict.asset_b,
            instructions={
                yield_asset: {
                    "action": "HOLD",
                    "hold_s": TIME_STAGGER_S,
                    "reason": "deconfliction_time_separation",
                },
                keep_asset: {
                    "action": "PROCEED",
                    "reason": "deconfliction_time_separation",
                },
            },
        )

    def _priority_yield(self, conflict: "Conflict") -> ConflictResolution:
        """Lower-priority asset holds; higher-priority asset proceeds."""
        if conflict.volume_a.priority >= conflict.volume_b.priority:
            high_asset = conflict.asset_a
            low_asset = conflict.asset_b
        else:
            high_asset = conflict.asset_b
            low_asset = conflict.asset_a

        return ConflictResolution(
            conflict_id=conflict.conflict_id,
            strategy="PRIORITY_YIELD",
            asset_a=conflict.asset_a,
            asset_b=conflict.asset_b,
            instructions={
                low_asset: {
                    "action": "HOLD",
                    "hold_s": TIME_STAGGER_S,
                    "reason": "deconfliction_priority_yield",
                },
                high_asset: {
                    "action": "PROCEED",
                    "reason": "deconfliction_priority_yield",
                },
            },
        )
