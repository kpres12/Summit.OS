"""
Track-to-Weapon Decision Support Ranker

Given a confirmed threat track and the available weapon-capable assets,
produces a ranked list of WeaponOptions to surface to the operator.

This module is **decision support only** — it computes feasibility,
ranks options, and surfaces them. It does NOT execute. Execution requires
explicit operator authorization through
`packages/c2_intel/engagement_authorization.py`.

Inputs
------
    track:     Confirmed sensor-fused track (TrackEvidence)
    assets:    List of weapon-capable assets in the world model
    roe:       Current ROE context
    deconf:    Blue-force + airspace deconfliction context
    intent:    Optional commander's intent constraints

Outputs
-------
    Ranked list of WeaponOption objects, suitable for direct hand-off to
    EngagementAuthorizationGate.surface_options().

Ranking factors (linear weighted sum, configurable via assignment_weights):
    pk_estimate        — probability of mission effect (higher better)
    range_margin       — distance / max_range (closer to 1 better, >1 invalid)
    time_of_flight     — lower is better (responsiveness)
    collateral_risk    — lower is better
    blue_force_margin  — distance to nearest blue force (higher better)
    weapon_avail       — readiness/availability of the weapon system

Ranking does NOT decide who to engage — it provides options for human
authorization.
"""

from __future__ import annotations

import logging
import math
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .engagement_authorization import (
    DeconflictionContext, ROEContext, TrackEvidence, WeaponOption,
)

logger = logging.getLogger("c2_intel.weapon_target_ranker")

EARTH_R_KM = 6371.0


# ---------------------------------------------------------------------------
# Asset model
# ---------------------------------------------------------------------------


@dataclass
class WeaponAsset:
    """A weapon-capable asset in the world model.

    Note: the asset DOES NOT execute on its own. This struct exposes the
    parameters needed for ranking and decision support."""
    asset_id:         str
    weapon_class:     str        # "soft_kill" | "hard_kill" | "non_lethal" | "kinetic_air" | ...
    platform:         str        # "ground_fixed", "ground_mobile", "rotary", "fixed_wing", "ship", ...
    position:         Dict[str, float]   # {lat, lon, alt_m}
    max_range_m:      float
    min_range_m:      float = 0.0
    cruise_speed_ms:  float = 250.0
    weapon_class_pk:  float = 0.85       # nominal probability of mission effect (0..1)
    ready:            bool   = True
    rounds_available: int   = 1
    notes:            str   = ""


# ---------------------------------------------------------------------------
# Ranking config
# ---------------------------------------------------------------------------


@dataclass
class RankingWeights:
    pk_estimate:       float = 0.30
    range_margin:      float = 0.20
    time_of_flight:    float = 0.20
    collateral_risk:   float = 0.20
    blue_force_margin: float = 0.10


DEFAULT_WEIGHTS = RankingWeights()


# ---------------------------------------------------------------------------
# Geometry / timing helpers
# ---------------------------------------------------------------------------


def _haversine_m(a_lat: float, a_lon: float,
                 b_lat: float, b_lon: float) -> float:
    p1, p2 = math.radians(a_lat), math.radians(b_lat)
    dp = p2 - p1
    dl = math.radians(b_lon - a_lon)
    a = math.sin(dp/2)**2 + math.cos(p1) * math.cos(p2) * math.sin(dl/2)**2
    return float(2 * EARTH_R_KM * 1000 * math.asin(math.sqrt(a)))


def _track_position(track: TrackEvidence) -> Optional[Tuple[float, float]]:
    if not track.last_position:
        return None
    return float(track.last_position["lat"]), float(track.last_position["lon"])


# ---------------------------------------------------------------------------
# Per-option scoring
# ---------------------------------------------------------------------------


def _collateral_score(roe: ROEContext, deconf: DeconflictionContext) -> float:
    """Returns 0..1 score where higher = lower collateral risk = better."""
    if roe is None:
        return 0.5
    table = {
        "minimal":  1.00,
        "low":      0.90,
        "moderate": 0.65,
        "elevated": 0.30,
        "blocking": 0.0,
    }
    base = table.get(roe.collateral_estimate, 0.5)
    # Penalize for nearby civilians count (logarithmic dampener)
    if deconf.nearby_civilians_count > 0:
        base *= max(0.1, 1.0 - 0.1 * math.log10(deconf.nearby_civilians_count + 1))
    return float(base)


def _blue_force_margin(asset: WeaponAsset,
                       blue_positions: List[Dict[str, float]],
                       track_lat: float, track_lon: float) -> float:
    """Score 0..1: how far the closest blue-force is from the engagement geometry."""
    if not blue_positions:
        return 1.0
    # Distance from blue to TARGET position (proxy for collateral risk window)
    min_d = min(
        _haversine_m(b["lat"], b["lon"], track_lat, track_lon)
        for b in blue_positions
    )
    # Saturate: 5 km clean → 1.0, ≤ 100 m unsafe → 0
    if min_d <= 100.0:
        return 0.0
    if min_d >= 5000.0:
        return 1.0
    return (min_d - 100.0) / (5000.0 - 100.0)


def _build_option(track: TrackEvidence, asset: WeaponAsset,
                  roe: ROEContext, deconf: DeconflictionContext,
                  blue_positions: List[Dict[str, float]],
                  weights: RankingWeights) -> Optional[Tuple[WeaponOption, float]]:
    pos = _track_position(track)
    if pos is None:
        return None
    track_lat, track_lon = pos
    asset_lat, asset_lon = asset.position["lat"], asset.position["lon"]
    distance_m = _haversine_m(asset_lat, asset_lon, track_lat, track_lon)

    # Range feasibility — out-of-range options are NOT returned
    if distance_m > asset.max_range_m or distance_m < asset.min_range_m:
        return None
    if not asset.ready or asset.rounds_available <= 0:
        return None

    # Per-factor scores in [0, 1]
    pk_score      = float(asset.weapon_class_pk)
    range_score   = 1.0 - distance_m / asset.max_range_m  # closer = better margin
    tof_s         = distance_m / max(asset.cruise_speed_ms, 1.0)
    tof_score     = max(0.0, 1.0 - tof_s / 600.0)         # 10-minute saturation
    collateral    = _collateral_score(roe, deconf)
    blue_margin   = _blue_force_margin(asset, blue_positions, track_lat, track_lon)

    composite = (
        weights.pk_estimate       * pk_score   +
        weights.range_margin      * range_score +
        weights.time_of_flight    * tof_score   +
        weights.collateral_risk   * collateral  +
        weights.blue_force_margin * blue_margin
    )

    # Compliance flags
    roe_compliant = bool(
        roe is None or
        (roe.permits_engagement_type and roe.proportionality_passed
         and roe.collateral_estimate != "blocking")
    )
    deconf_ok = bool(deconf.blue_force_clear and deconf.airspace_clear)

    rationale = (
        f"{asset.platform} {asset.weapon_class} @ {distance_m/1000.0:.2f} km "
        f"(max {asset.max_range_m/1000.0:.1f} km), TOF≈{tof_s:.0f}s, "
        f"PK≈{pk_score:.2f}, collateral={collateral:.2f}, blue_margin={blue_margin:.2f}"
    )

    return WeaponOption(
        option_id=f"opt-{asset.asset_id}-{uuid.uuid4().hex[:8]}",
        weapon_asset_id=asset.asset_id,
        weapon_class=asset.weapon_class,
        range_m=distance_m,
        time_of_flight_s=tof_s,
        pk_estimate=pk_score,
        roe_compliant=roe_compliant,
        deconfliction_ok=deconf_ok,
        rationale=rationale,
    ), float(composite)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def rank_weapon_options(
    track: TrackEvidence,
    weapon_assets: List[WeaponAsset],
    roe: ROEContext,
    deconf: DeconflictionContext,
    blue_force_positions: Optional[List[Dict[str, float]]] = None,
    weights: Optional[RankingWeights] = None,
    top_k: int = 5,
) -> List[WeaponOption]:
    """Return up to top_k WeaponOptions ranked best-first.

    The returned options are decision support — they are surfaced to the
    operator via EngagementAuthorizationGate.surface_options(). Selecting
    one and authorizing requires a separate human authorization step.
    """
    weights = weights or DEFAULT_WEIGHTS
    blue_force_positions = blue_force_positions or []

    scored: List[Tuple[WeaponOption, float]] = []
    for asset in weapon_assets:
        result = _build_option(track, asset, roe, deconf,
                               blue_force_positions, weights)
        if result is not None:
            scored.append(result)

    scored.sort(key=lambda x: x[1], reverse=True)
    if not scored:
        logger.info("[ranker] no viable options for track=%s (n_assets=%d)",
                    track.track_id, len(weapon_assets))
    else:
        logger.info(
            "[ranker] track=%s -> %d viable, top option score=%.3f, asset=%s",
            track.track_id, len(scored), scored[0][1],
            scored[0][0].weapon_asset_id,
        )

    return [opt for opt, _ in scored[:top_k]]
