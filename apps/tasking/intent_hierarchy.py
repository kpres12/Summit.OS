"""
apps/tasking/intent_hierarchy.py — Multi-tier hierarchical intent propagation.

Extends single-level role decomposition into an N-tier command chain:
  COMMANDER → WING_LEAD → FLIGHT_LEAD → ASSET

Each tier receives intent + available assets + local constraints, applies
its own decomposition logic, and dispatches to the next tier down.
Lower tiers can modify, split, or veto assignments from above based on
local conditions (battery, comms, terrain).

CANVAS TA2: "conditions-based authorities delegated to lower tier nodes"
and "distributed teams execute local workflows through trust, shared
awareness, and understanding of commander's intent."

Usage:
    decomposer = HierarchicalDecomposer()
    result = decomposer.decompose(
        intent="search_and_rescue",
        area={"center": {"lat": 37.5, "lon": -122.1}, "radius_m": 2000},
        assets=available_assets,
        tiers=[CommandTier.WING_LEAD, CommandTier.FLIGHT_LEAD],
    )
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from role_decomposer import RoleDecomposer, classify_asset_domain

logger = logging.getLogger("tasking.intent_hierarchy")


class CommandTier(str, Enum):
    COMMANDER    = "COMMANDER"
    WING_LEAD    = "WING_LEAD"
    FLIGHT_LEAD  = "FLIGHT_LEAD"
    ASSET        = "ASSET"


@dataclass
class TierConstraints:
    """Local constraints a tier applies before passing intent downward."""
    max_assets:         Optional[int]   = None   # cap on assets this tier controls
    max_radius_m:       Optional[float] = None   # geographically limit the sub-area
    min_battery_pct:    float           = 20.0   # filter out low-battery assets
    comms_range_m:      Optional[float] = None   # only use assets within comms range
    allowed_domains:    Optional[List[str]] = None  # restrict to specific domains


@dataclass
class TierNode:
    """One node in the command hierarchy."""
    tier:        CommandTier
    tier_id:     str                             # e.g. "wing-alpha", "flight-1"
    intent:      str                             # intent received from parent
    area:        Dict[str, Any]                  # sub-area this tier is responsible for
    assets:      List[Dict[str, Any]]            # assets under this tier's command
    constraints: TierConstraints = field(default_factory=TierConstraints)
    children:    List["TierNode"] = field(default_factory=list)
    assignments: Dict[str, Any] = field(default_factory=dict)  # final per-asset plans


@dataclass
class HierarchyResult:
    """Full result of hierarchical decomposition."""
    intent:          str
    tier_structure:  List[Dict[str, Any]]   # human-readable tree for console
    assignments:     Dict[str, Any]         # flat asset_id → plan map (for dispatch)
    tier_count:      int
    asset_count:     int


class HierarchicalDecomposer:
    """
    Decomposes commander intent through configurable command tiers.

    Each tier:
      1. Filters assets using local constraints (battery, comms, domain)
      2. Splits its area into sub-areas for child tiers
      3. Decomposes intent into role assignments for its asset group
      4. Recurses into child tiers

    The result is a flat assignment map (same format as single-level decomposer)
    plus a tier_structure tree showing how intent flowed down.
    """

    def __init__(self) -> None:
        self._role_decomposer = RoleDecomposer()

    def decompose(
        self,
        intent: str,
        area: Dict[str, Any],
        assets: List[Dict[str, Any]],
        tiers: Optional[List[CommandTier]] = None,
        planning_params: Optional[Dict[str, Any]] = None,
    ) -> HierarchyResult:
        """
        Decompose intent through the given tier chain.

        Args:
            intent:          Commander's intent string
            area:            Full mission area
            assets:          All available assets
            tiers:           Tier chain below COMMANDER.
                             Defaults to [WING_LEAD, FLIGHT_LEAD].
            planning_params: Base planning params (passed through)

        Returns HierarchyResult with flat assignments + tier structure.
        """
        if tiers is None:
            tiers = [CommandTier.WING_LEAD, CommandTier.FLIGHT_LEAD]

        logger.info(
            "Hierarchical decompose: intent=%s tiers=%s assets=%d",
            intent, [t.value for t in tiers], len(assets),
        )

        root = TierNode(
            tier        = CommandTier.COMMANDER,
            tier_id     = "commander",
            intent      = intent,
            area        = area,
            assets      = assets,
            constraints = TierConstraints(),
        )

        self._decompose_tier(root, tiers, planning_params or {})

        flat_assignments: Dict[str, Any] = {}
        self._collect_assignments(root, flat_assignments)

        tier_structure = self._build_tree(root)

        return HierarchyResult(
            intent         = intent,
            tier_structure = tier_structure,
            assignments    = flat_assignments,
            tier_count     = len(tiers) + 1,
            asset_count    = len(flat_assignments),
        )

    # ── Internal recursion ────────────────────────────────────────────────────

    def _decompose_tier(
        self,
        node: TierNode,
        remaining_tiers: List[CommandTier],
        planning_params: Dict[str, Any],
    ) -> None:
        """Recursively decompose intent through tiers."""
        assets = self._filter_assets(node.assets, node.constraints)

        if not remaining_tiers:
            # Leaf tier — assign directly to assets
            manifest = self._role_decomposer.decompose(
                intent          = node.intent,
                available_assets = assets,
                area            = node.area,
                planning_params = planning_params,
            )
            for role in manifest.roles:
                for asset in role.assets:
                    node.assignments[asset["asset_id"]] = {
                        **role.planning_params,
                        "role":        role.role_name,
                        "domain":      role.domain,
                        "description": role.description,
                        "tier":        node.tier.value,
                        "tier_id":     node.tier_id,
                        "priority":    role.priority,
                    }
            logger.debug(
                "Tier %s/%s assigned %d assets",
                node.tier.value, node.tier_id, len(node.assignments),
            )
            return

        # Split assets and area across child tier nodes
        next_tier = remaining_tiers[0]
        sub_remaining = remaining_tiers[1:]

        child_groups = self._split_assets(assets, node.area, next_tier)

        for group_id, (group_assets, sub_area) in enumerate(child_groups):
            # Derive sub-intent: same intent, but child may specialize
            sub_intent = self._derive_sub_intent(node.intent, next_tier, group_id)

            child = TierNode(
                tier        = next_tier,
                tier_id     = f"{next_tier.value.lower()}-{group_id + 1}",
                intent      = sub_intent,
                area        = sub_area,
                assets      = group_assets,
                constraints = self._tier_constraints(next_tier),
            )
            node.children.append(child)
            self._decompose_tier(child, sub_remaining, planning_params)

    def _filter_assets(
        self,
        assets: List[Dict[str, Any]],
        constraints: TierConstraints,
    ) -> List[Dict[str, Any]]:
        filtered = assets

        # Battery filter
        filtered = [
            a for a in filtered
            if (a.get("battery") or 100) >= constraints.min_battery_pct
        ]

        # Domain filter
        if constraints.allowed_domains:
            filtered = [
                a for a in filtered
                if classify_asset_domain(a) in constraints.allowed_domains
            ]

        # Asset count cap
        if constraints.max_assets and len(filtered) > constraints.max_assets:
            filtered = filtered[:constraints.max_assets]

        return filtered

    def _split_assets(
        self,
        assets: List[Dict[str, Any]],
        area: Dict[str, Any],
        tier: CommandTier,
    ) -> List[tuple]:
        """
        Split asset pool and area for child tier nodes.

        WING_LEAD: split by domain (aerial group, ground group)
        FLIGHT_LEAD: split numerically into flights of ≤4 assets
        """
        if tier == CommandTier.WING_LEAD:
            return self._split_by_domain(assets, area)
        elif tier == CommandTier.FLIGHT_LEAD:
            return self._split_into_flights(assets, area, max_per_flight=4)
        else:
            return [(assets, area)]

    def _split_by_domain(
        self,
        assets: List[Dict[str, Any]],
        area: Dict[str, Any],
    ) -> List[tuple]:
        """Group assets by domain; each domain becomes its own wing."""
        domain_map: Dict[str, List[Dict]] = {}
        for asset in assets:
            domain = classify_asset_domain(asset)
            domain_map.setdefault(domain, []).append(asset)

        return [
            (domain_assets, area)
            for domain_assets in domain_map.values()
            if domain_assets
        ]

    def _split_into_flights(
        self,
        assets: List[Dict[str, Any]],
        area: Dict[str, Any],
        max_per_flight: int = 4,
    ) -> List[tuple]:
        """Split assets into flights of max_per_flight, each with a sub-area."""
        if not assets:
            return []

        flights = [
            assets[i:i + max_per_flight]
            for i in range(0, len(assets), max_per_flight)
        ]
        n = len(flights)
        sub_areas = self._partition_area(area, n)

        return list(zip(flights, sub_areas))

    def _partition_area(
        self,
        area: Dict[str, Any],
        n: int,
    ) -> List[Dict[str, Any]]:
        """Divide a circular area into n equal wedge sub-areas."""
        if n <= 1:
            return [area]

        center = area.get("center", {"lat": 0.0, "lon": 0.0})
        radius = area.get("radius_m", 500.0)
        sub_radius = radius / math.sqrt(n)  # rough equal-area partition

        sub_areas = []
        for i in range(n):
            angle = (2 * math.pi * i) / n
            offset_lat = (sub_radius / 111_111.0) * math.sin(angle)
            offset_lon = (sub_radius / (111_111.0 * max(0.01, math.cos(
                math.radians(center.get("lat", 0))
            )))) * math.cos(angle)

            sub_areas.append({
                "center": {
                    "lat": center.get("lat", 0) + offset_lat,
                    "lon": center.get("lon", 0) + offset_lon,
                },
                "radius_m": sub_radius,
            })

        return sub_areas

    def _derive_sub_intent(
        self,
        parent_intent: str,
        tier: CommandTier,
        group_id: int,
    ) -> str:
        """
        Lower tiers can specialize intent.
        For now, intent passes through unchanged — the tier applies it
        to its own asset group and area. Future: intent refinement rules.
        """
        return parent_intent

    def _tier_constraints(self, tier: CommandTier) -> TierConstraints:
        """Default constraints per tier level."""
        if tier == CommandTier.WING_LEAD:
            return TierConstraints(min_battery_pct=25.0)
        elif tier == CommandTier.FLIGHT_LEAD:
            return TierConstraints(min_battery_pct=30.0, max_assets=4)
        return TierConstraints()

    def _collect_assignments(
        self,
        node: TierNode,
        flat: Dict[str, Any],
    ) -> None:
        flat.update(node.assignments)
        for child in node.children:
            self._collect_assignments(child, flat)

    def _build_tree(self, node: TierNode) -> List[Dict[str, Any]]:
        return [{
            "tier":        node.tier.value,
            "tier_id":     node.tier_id,
            "intent":      node.intent,
            "asset_count": len(node.assets),
            "assigned":    len(node.assignments),
            "children":    [self._build_tree(c)[0] for c in node.children],
        }]
