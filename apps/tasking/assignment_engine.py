"""
Intent-Based Assignment Engine for Summit.OS

Given a mission intent (what you want done), this engine:
1. Queries the WorldStore for available assets
2. Scores each asset by capability match, proximity, battery, current load
3. Selects the optimal asset(s) and generates a flight/patrol plan

This replaces the simple "first N available" logic in tasking/main.py.

Intent types:
  - "survey"    → grid coverage pattern
  - "monitor"   → perimeter patrol
  - "search"    → expanding square search
  - "observe"   → orbit/loiter
  - "respond"   → direct approach + verification
  - "contain"   → containment ring
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from packages.entities.core import Entity, EntityType, EntityDomain, LifecycleState

logger = logging.getLogger("tasking.assignment")


# ── Intent → Pattern mapping ──────────────────────────────────

INTENT_PATTERN_MAP = {
    "survey":       "grid",
    "surveillance": "grid",
    "mapping":      "grid",
    "monitor":      "perimeter",
    "patrol":       "perimeter",
    "search":       "expanding_square",
    "sar":          "expanding_square",
    "observe":      "orbit",
    "loiter":       "orbit",
    "overwatch":    "orbit",
    "respond":      "direct",
    "verify":       "direct",
    "contain":      "containment",
    "suppress":     "direct",
}

# Capability weights for scoring
CAPABILITY_WEIGHTS = {
    "thermal":      0.3,
    "rgb_camera":   0.2,
    "lidar":        0.15,
    "gas_sensor":   0.1,
    "payload_drop": 0.25,
}


@dataclass
class AssetScore:
    """Scoring result for a single asset."""
    asset_id: str
    entity: Entity
    total_score: float = 0.0
    capability_score: float = 0.0
    proximity_score: float = 0.0
    battery_score: float = 0.0
    availability_score: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AssignmentResult:
    """Result of the assignment engine."""
    selected_assets: List[AssetScore]
    pattern: str
    intent: str
    all_scores: List[AssetScore]


class AssignmentEngine:
    """
    Scores and selects assets for mission assignment.

    Uses the WorldStore to find available assets and scores them
    based on multiple factors.
    """

    def __init__(self, world_store=None):
        self.world_store = world_store

    def assign(
        self,
        intent: str,
        target_lat: float,
        target_lon: float,
        num_assets: int = 1,
        required_capabilities: Optional[List[str]] = None,
        org_id: Optional[str] = None,
        available_assets: Optional[List[Dict[str, Any]]] = None,
    ) -> AssignmentResult:
        """
        Find and score the best assets for a mission intent.

        Args:
            intent: What the operator wants done (e.g., "survey", "monitor")
            target_lat: Target area latitude
            target_lon: Target area longitude
            num_assets: How many assets to assign
            required_capabilities: Must-have capabilities (e.g., ["thermal"])
            org_id: Organization filter
            available_assets: Pre-fetched asset list (for when WorldStore isn't available)

        Returns:
            AssignmentResult with selected assets and pattern
        """
        pattern = INTENT_PATTERN_MAP.get(intent.lower(), "loiter")

        # Get candidate assets
        if self.world_store:
            candidates = self._get_candidates_from_store(org_id)
        elif available_assets:
            candidates = self._assets_to_entities(available_assets)
        else:
            return AssignmentResult(
                selected_assets=[], pattern=pattern, intent=intent, all_scores=[],
            )

        # Score each candidate
        scores = []
        for entity in candidates:
            score = self._score_asset(
                entity, intent, target_lat, target_lon, required_capabilities,
            )
            scores.append(score)

        # Sort by total score descending
        scores.sort(key=lambda s: s.total_score, reverse=True)

        # Filter out assets that don't meet minimum threshold
        viable = [s for s in scores if s.total_score > 0.1]

        # Select top N
        selected = viable[:num_assets]

        return AssignmentResult(
            selected_assets=selected,
            pattern=pattern,
            intent=intent,
            all_scores=scores,
        )

    def _get_candidates_from_store(self, org_id: Optional[str]) -> List[Entity]:
        """Query WorldStore for available asset entities."""
        if not self.world_store:
            return []

        return self.world_store.query(
            entity_type=EntityType.ASSET,
            state=LifecycleState.ACTIVE,
            org_id=org_id,
            limit=200,
        )

    def _assets_to_entities(self, assets: List[Dict[str, Any]]) -> List[Entity]:
        """Convert raw asset dicts (from DB) to Entity objects for scoring."""
        from packages.entities.core import (
            Entity, EntityType, EntityDomain, LifecycleState,
            Kinematics, GeoPoint, Provenance, AerialData,
        )

        entities = []
        for a in assets:
            caps = a.get("capabilities") or {}
            if isinstance(caps, str):
                import json
                try:
                    caps = json.loads(caps) or {}
                except Exception:
                    caps = {}
            if not isinstance(caps, dict):
                caps = {}

            entity = Entity(
                id=a.get("asset_id", ""),
                entity_type=EntityType.ASSET,
                domain=EntityDomain.AERIAL,
                state=LifecycleState.ACTIVE,
                name=a.get("asset_id", ""),
                metadata={
                    "type": a.get("type", ""),
                    "link": a.get("link", ""),
                    **{k: str(v) for k, v in caps.items() if isinstance(v, (str, int, float, bool))},
                },
            )

            # Battery
            battery = a.get("battery") or 0
            if battery > 0:
                entity.aerial = AerialData(battery_pct=float(battery))

            # Constraints
            constraints = a.get("constraints") or {}
            if isinstance(constraints, str):
                import json
                try:
                    constraints = json.loads(constraints) or {}
                except Exception:
                    constraints = {}
            if not isinstance(constraints, dict):
                constraints = {}
            for k, v in constraints.items():
                entity.metadata[f"constraint_{k}"] = str(v)

            entities.append(entity)

        return entities

    def _score_asset(
        self,
        entity: Entity,
        intent: str,
        target_lat: float,
        target_lon: float,
        required_capabilities: Optional[List[str]],
    ) -> AssetScore:
        """Score a single asset for a mission."""
        score = AssetScore(asset_id=entity.id, entity=entity)

        # 1. Capability score (0-1)
        score.capability_score = self._score_capability(entity, intent, required_capabilities)

        # 2. Proximity score (0-1) — closer is better
        score.proximity_score = self._score_proximity(entity, target_lat, target_lon)

        # 3. Battery score (0-1)
        score.battery_score = self._score_battery(entity)

        # 4. Availability score (0-1) — not currently on a mission
        score.availability_score = self._score_availability(entity)

        # Weighted total
        score.total_score = (
            score.capability_score * 0.30
            + score.proximity_score * 0.25
            + score.battery_score * 0.25
            + score.availability_score * 0.20
        )

        # Hard filter: if required capabilities missing, zero out
        if required_capabilities:
            meta_caps = set(entity.metadata.keys())
            for cap in required_capabilities:
                if cap not in meta_caps and entity.metadata.get(cap) != "true":
                    score.total_score = 0.0
                    score.details["disqualified"] = f"Missing required: {cap}"
                    break

        return score

    def _score_capability(
        self, entity: Entity, intent: str, required: Optional[List[str]]
    ) -> float:
        """Score based on asset capabilities matching the intent."""
        # Intent-based capability mapping
        intent_needs = {
            "survey":   ["rgb_camera", "thermal"],
            "monitor":  ["rgb_camera", "thermal"],
            "search":   ["thermal", "rgb_camera"],
            "observe":  ["rgb_camera"],
            "respond":  ["thermal", "rgb_camera"],
            "contain":  ["payload_drop", "thermal"],
            "suppress": ["payload_drop"],
        }

        needed = intent_needs.get(intent.lower(), ["rgb_camera"])
        meta = entity.metadata

        matched = 0
        for cap in needed:
            if cap in meta or meta.get(cap) == "true":
                matched += 1

        if not needed:
            return 0.5

        return min(1.0, matched / len(needed))

    def _score_proximity(
        self, entity: Entity, target_lat: float, target_lon: float
    ) -> float:
        """Score based on distance to target. Closer = higher score."""
        if not entity.kinematics or not entity.kinematics.position:
            return 0.3  # Unknown position gets moderate score

        pos = entity.kinematics.position
        dist_m = self._haversine(pos.latitude, pos.longitude, target_lat, target_lon)

        # Score: 1.0 at 0m, 0.5 at 5km, 0.1 at 50km
        if dist_m <= 100:
            return 1.0
        elif dist_m <= 5000:
            return 1.0 - (dist_m / 5000) * 0.5
        elif dist_m <= 50000:
            return 0.5 - ((dist_m - 5000) / 45000) * 0.4
        else:
            return 0.1

    def _score_battery(self, entity: Entity) -> float:
        """Score based on battery level."""
        battery = 100.0  # Default if unknown

        if entity.aerial and entity.aerial.battery_pct > 0:
            battery = entity.aerial.battery_pct
        elif "battery" in entity.metadata:
            try:
                battery = float(entity.metadata["battery"])
            except (ValueError, TypeError):
                pass

        # Score: 1.0 at 100%, 0.0 at 20% (hard cutoff below 20%)
        if battery < 20:
            return 0.0
        return min(1.0, (battery - 20) / 80)

    def _score_availability(self, entity: Entity) -> float:
        """Score based on current assignment status."""
        # Check if entity has active mission relationships
        for rel in entity.relationships:
            if rel.relationship in ("assigned_to", "executing"):
                return 0.2  # Already assigned, low priority

        if entity.state != LifecycleState.ACTIVE:
            return 0.0

        return 1.0

    @staticmethod
    def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two points in meters."""
        R = 6371000  # Earth radius in meters
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(math.radians(lat1))
            * math.cos(math.radians(lat2))
            * math.sin(dlon / 2) ** 2
        )
        return R * 2 * math.asin(math.sqrt(a))


def resolve_pattern(intent: str) -> str:
    """Resolve an intent string to a coverage pattern name."""
    return INTENT_PATTERN_MAP.get(intent.lower(), "loiter")
