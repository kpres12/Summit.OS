"""
Mission Orchestrator — Heli.OS

Closes the loop between natural-language mission input and physical device action.

  Operator: "Find the missing hiker"
       ↓
  parse_mission_nlp()   — LLM extracts mission_type, search pattern, objectives
       ↓
  _discover_assets()    — WorldStore query for available, capable assets
       ↓
  _generate_waypoints() — Grid / lawnmower / spiral from search area
       ↓
  SwarmPlanner.replan() — Hungarian algorithm: best asset per waypoint
       ↓
  BehaviorTree per asset — Runs the mission tick loop per device
       ↓
  _dispatch_command()   — Translates BT nav_target → send_command() on adapter
       ↓
  c2_intel monitoring   — Battery critical → RTB, entity detected → redirect
       ↓
  Re-plan on failure    — Asset offline or target found triggers replan()

One orchestrator instance per active mission. Runs as an asyncio background task.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
import uuid
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("heli.orchestrator")

# ---------------------------------------------------------------------------
# Capability map — what each adapter type can do
# ---------------------------------------------------------------------------

ASSET_CAPABILITIES: Dict[str, List[str]] = {
    "mavlink":   ["fly", "goto", "search", "survey", "rtb", "camera"],
    "dji":       ["fly", "goto", "search", "survey", "rtb", "camera"],
    "spot":      ["walk", "search", "inspect", "camera"],
    "onvif":     ["camera", "ptz", "observe"],
    "thermal":   ["camera", "observe", "detect_thermal"],
    "ros2":      ["navigate", "search", "manipulate"],
    "ais":       ["observe"],
    "opensky":   ["observe"],
    "atak":      ["observe", "navigate", "coordinate"],
    "meshtastic": ["observe", "coordinate"],
    "nmea2000":  ["observe", "navigate"],
    "aisstream": ["observe"],
    "kraken":    ["observe", "underwater_survey"],
}

# BT mission type → required capability
MISSION_CAPABILITY: Dict[str, str] = {
    # Core mission types
    "SURVEY":          "goto",
    "RECON":           "search",
    "PATROL":          "goto",
    "SAR":             "search",
    "ESCORT":          "navigate",
    "INTERCEPT":       "goto",
    "OBSERVE":         "observe",
    "MONITOR":         "observe",
    # Infrastructure inspection
    "INSPECT":         "camera",
    # Military / government
    "HADR":            "search",
    "ACE":             "goto",
    "FORCE_PROTECT":   "observe",
    "CASEVAC_ESCORT":  "navigate",
    # Maritime
    "MARITIME_SAR":    "search",
    # Conservation / wildlife
    "ANTI_POACH":      "search",
    # Agriculture
    "PRECISION_AG":    "camera",
    # Oil & gas
    "PIPELINE_PATROL": "goto",
}


# ---------------------------------------------------------------------------
# Haversine (duplicated locally to avoid package import cycles)
# ---------------------------------------------------------------------------

def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6_371_000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ---------------------------------------------------------------------------
# Waypoint generators
# ---------------------------------------------------------------------------

def _grid_waypoints(area: List[Dict], alt_m: float, spacing_m: float = 80.0) -> List[Dict]:
    """Lawnmower grid over a bounding box."""
    if not area:
        return []
    lats = [p["lat"] for p in area]
    lons = [p["lon"] for p in area]
    min_lat, max_lat = min(lats), max(lats)
    min_lon, max_lon = min(lons), max(lons)

    lat_step = spacing_m / 111_320.0
    lon_step = spacing_m / (111_320.0 * math.cos(math.radians((min_lat + max_lat) / 2)))

    rows = max(2, int((max_lat - min_lat) / lat_step) + 1)
    cols = max(2, int((max_lon - min_lon) / lon_step) + 1)

    waypoints = []
    for r in range(rows):
        lat = min_lat + r * lat_step
        col_range = range(cols) if r % 2 == 0 else range(cols - 1, -1, -1)
        for c in col_range:
            lon = min_lon + c * lon_step
            waypoints.append({"lat": lat, "lon": lon, "alt": alt_m, "radius_m": 15.0})
    return waypoints


def _spiral_waypoints(center: Dict, alt_m: float, rings: int = 4, points_per_ring: int = 8) -> List[Dict]:
    """Expanding spiral from a center point."""
    waypoints = []
    for ring in range(1, rings + 1):
        radius_m = ring * 50.0
        for i in range(points_per_ring):
            angle = 2 * math.pi * i / points_per_ring
            dlat = (radius_m * math.cos(angle)) / 111_320.0
            dlon = (radius_m * math.sin(angle)) / (111_320.0 * math.cos(math.radians(center["lat"])))
            waypoints.append({
                "lat": center["lat"] + dlat,
                "lon": center["lon"] + dlon,
                "alt": alt_m,
                "radius_m": 15.0,
            })
    return waypoints


def _orbit_waypoints(center: Dict, alt_m: float, radius_m: float = 100.0, points: int = 12) -> List[Dict]:
    """Fixed orbit / racetrack around a point."""
    waypoints = []
    for i in range(points):
        angle = 2 * math.pi * i / points
        dlat = (radius_m * math.cos(angle)) / 111_320.0
        dlon = (radius_m * math.sin(angle)) / (111_320.0 * math.cos(math.radians(center["lat"])))
        waypoints.append({"lat": center["lat"] + dlat, "lon": center["lon"] + dlon,
                          "alt": alt_m, "radius_m": 10.0})
    return waypoints


def _expanding_square_waypoints(center: Dict, alt_m: float, legs: int = 5) -> List[Dict]:
    """Maritime expanding square search pattern from datum."""
    waypoints = []
    step_m = 100.0
    lat, lon = center["lat"], center["lon"]
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # N E S W
    length = step_m
    for leg in range(legs * 2):
        d = directions[leg % 4]
        dlat = (length * d[0]) / 111_320.0
        dlon = (length * d[1]) / (111_320.0 * math.cos(math.radians(lat)))
        lat += dlat
        lon += dlon
        waypoints.append({"lat": lat, "lon": lon, "alt": alt_m, "radius_m": 15.0})
        if leg % 2 == 1:
            length += step_m
    return waypoints


def _parallel_track_waypoints(area: List[Dict], alt_m: float, track_spacing_m: float = 200.0) -> List[Dict]:
    """Parallel track pattern for systematic maritime/aerial search."""
    if not area:
        return []
    lats = [p["lat"] for p in area]
    lons = [p["lon"] for p in area]
    min_lat, max_lat = min(lats), max(lats)
    min_lon, max_lon = min(lons), max(lons)
    lat_step = track_spacing_m / 111_320.0
    rows = max(2, int((max_lat - min_lat) / lat_step) + 1)
    waypoints = []
    for r in range(rows):
        lat = min_lat + r * lat_step
        if r % 2 == 0:
            waypoints.append({"lat": lat, "lon": min_lon, "alt": alt_m, "radius_m": 20.0})
            waypoints.append({"lat": lat, "lon": max_lon, "alt": alt_m, "radius_m": 20.0})
        else:
            waypoints.append({"lat": lat, "lon": max_lon, "alt": alt_m, "radius_m": 20.0})
            waypoints.append({"lat": lat, "lon": min_lon, "alt": alt_m, "radius_m": 20.0})
    return waypoints


def _generate_waypoints(pattern: str, area: List[Dict], alt_m: float) -> List[Dict]:
    pattern_lower = pattern.lower()
    center = {
        "lat": sum(p["lat"] for p in area) / max(len(area), 1),
        "lon": sum(p["lon"] for p in area) / max(len(area), 1),
    } if area else {"lat": 0.0, "lon": 0.0}

    if pattern_lower in ("grid", "lawnmower"):
        return _grid_waypoints(area, alt_m)
    if pattern_lower in ("spiral", "expanding_square"):
        return _expanding_square_waypoints(center, alt_m)
    if pattern_lower == "orbit":
        return _orbit_waypoints(center, alt_m)
    if pattern_lower == "parallel_track":
        return _parallel_track_waypoints(area, alt_m)
    if pattern_lower == "direct":
        # Single waypoint at area centroid — used for intercept / direct nav
        return [{"lat": center["lat"], "lon": center["lon"], "alt": alt_m, "radius_m": 20.0}]
    # default lawnmower
    return _grid_waypoints(area, alt_m)


# ---------------------------------------------------------------------------
# Natural language → mission intent
# ---------------------------------------------------------------------------

# Lazy-loaded intent classifier (trained by packages/training/train_intent.py)
_intent_clf = None
_intent_loaded = False

_MISSION_TYPES = [
    "SAR", "SURVEY", "PATROL", "RECON", "MONITOR", "ESCORT",
    "INSPECT", "HADR", "ACE", "FORCE_PROTECT", "CASEVAC_ESCORT",
    "MARITIME_SAR", "ANTI_POACH", "PRECISION_AG", "PIPELINE_PATROL",
]

# Keyword fallback — fast path when classifier isn't loaded
_KEYWORD_MAP: Dict[str, str] = {
    # SAR
    "find": "SAR", "search": "SAR", "rescue": "SAR", "missing": "SAR",
    "locate": "SAR", "survivor": "SAR", "hiker": "SAR", "sar": "SAR",
    # SURVEY
    "survey": "SURVEY", "map": "SURVEY", "assess": "SURVEY", "damage": "SURVEY",
    "document": "SURVEY", "photograph": "SURVEY", "imagery": "SURVEY",
    # PATROL
    "patrol": "PATROL", "perimeter": "PATROL", "security": "PATROL",
    "border": "PATROL", "sweep": "PATROL", "guard": "PATROL",
    # RECON
    "recon": "RECON", "reconnaissance": "RECON", "scout": "RECON",
    "intel": "RECON", "observe": "RECON", "eyes": "RECON",
    # MONITOR
    "monitor": "MONITOR", "watch": "MONITOR", "track": "MONITOR",
    "continuous": "MONITOR",
    # ESCORT
    "escort": "ESCORT", "convoy": "ESCORT", "accompany": "ESCORT",
    "overwatch": "ESCORT", "protect": "ESCORT",
    # INSPECT (utilities / infrastructure)
    "inspect": "INSPECT", "inspection": "INSPECT", "infrastructure": "INSPECT",
    "powerline": "INSPECT", "power line": "INSPECT", "bridge": "INSPECT",
    "pipeline inspection": "INSPECT", "corrosion": "INSPECT", "sag": "INSPECT",
    # HADR
    "hadr": "HADR", "humanitarian": "HADR", "disaster": "HADR",
    "flood": "HADR", "earthquake": "HADR", "relief": "HADR",
    # ACE (Agile Combat Employment)
    "ace": "ACE", "dispersal": "ACE", "forward base": "ACE",
    # FORCE_PROTECT
    "force protection": "FORCE_PROTECT", "fob": "FORCE_PROTECT",
    "base security": "FORCE_PROTECT", "perimeter defense": "FORCE_PROTECT",
    # CASEVAC_ESCORT
    "casevac": "CASEVAC_ESCORT", "medevac": "CASEVAC_ESCORT",
    "casualty": "CASEVAC_ESCORT", "9-line": "CASEVAC_ESCORT",
    # MARITIME_SAR
    "maritime": "MARITIME_SAR", "vessel": "MARITIME_SAR", "boat": "MARITIME_SAR",
    "coast": "MARITIME_SAR", "offshore": "MARITIME_SAR", "port": "MARITIME_SAR",
    # ANTI_POACH
    "poaching": "ANTI_POACH", "poacher": "ANTI_POACH", "anti-poach": "ANTI_POACH",
    "wildlife patrol": "ANTI_POACH", "snare": "ANTI_POACH", "reserve": "ANTI_POACH",
    # PRECISION_AG
    "crop": "PRECISION_AG", "ndvi": "PRECISION_AG", "agriculture": "PRECISION_AG",
    "precision ag": "PRECISION_AG", "spray": "PRECISION_AG", "field": "PRECISION_AG",
    "farm": "PRECISION_AG", "livestock": "PRECISION_AG",
    # PIPELINE_PATROL
    "pipeline": "PIPELINE_PATROL", "row": "PIPELINE_PATROL", "flare": "PIPELINE_PATROL",
    "compressor": "PIPELINE_PATROL", "wellpad": "PIPELINE_PATROL",
    "oil": "PIPELINE_PATROL", "gas": "PIPELINE_PATROL",
}

_SEARCH_PATTERN_KEYWORDS: Dict[str, str] = {
    "grid":       "grid",
    "lawnmower":  "lawnmower",
    "sweep":      "lawnmower",
    "spiral":     "spiral",
    "expanding":  "spiral",
    "orbit":      "spiral",
    "circle":     "spiral",
}


def _load_intent_classifier():
    global _intent_clf, _intent_loaded
    if _intent_loaded:
        return _intent_clf
    _intent_loaded = True

    try:
        import joblib
        from pathlib import Path as _Path
        model_path = _Path(__file__).resolve().parents[2] / "packages" / "c2_intel" / "models" / "intent_classifier.joblib"
        if model_path.exists():
            _intent_clf = joblib.load(model_path)
            logger.info("[MissionOrchestrator] Intent classifier loaded")
        else:
            logger.info("[MissionOrchestrator] Intent classifier not trained yet — using keyword fallback")
    except Exception as e:
        logger.warning("[MissionOrchestrator] Intent classifier load failed: %s", e)
    return _intent_clf


def parse_mission_nlp(text: str, area: Optional[List[Dict]] = None) -> Dict:
    """
    Parse operator free-text into a structured mission spec.

    Args:
        text: Natural language mission description (e.g. "Find the missing hiker")
        area: Optional polygon defining the search area (list of {lat, lon} dicts)

    Returns:
        Dict with:
          mission_type   (SAR | SURVEY | PATROL | RECON | MONITOR | ESCORT)
          pattern        (grid | lawnmower | spiral)
          altitude_m     (float)
          objectives     (list of str)
          confidence     (float, 0-1)
          raw_text       (str)
    """
    text_lower = text.lower().strip()
    mission_type = None
    confidence = 0.0

    # Try ML classifier first
    clf = _load_intent_classifier()
    if clf is not None:
        try:
            probs = clf.predict_proba([text_lower])[0]
            classes = clf.classes_
            best_idx = int(probs.argmax())
            mission_type = str(classes[best_idx])
            confidence = float(probs[best_idx])
        except Exception as e:
            logger.warning("Intent classifier inference failed: %s", e)

    # Keyword fallback
    if mission_type is None or confidence < 0.4:
        for kw, mtype in _KEYWORD_MAP.items():
            if kw in text_lower:
                mission_type = mtype
                confidence = max(confidence, 0.6)
                break

    if mission_type is None:
        mission_type = "RECON"  # conservative default
        confidence = 0.3

    # Infer search pattern from text
    pattern = "lawnmower"  # default for SAR/SURVEY
    for kw, pat in _SEARCH_PATTERN_KEYWORDS.items():
        if kw in text_lower:
            pattern = pat
            break
    if mission_type in ("PATROL", "ESCORT", "MONITOR") and pattern == "lawnmower":
        pattern = "grid"
    if mission_type == "SAR" and "expand" in text_lower:
        pattern = "spiral"

    # Infer altitude from context
    altitude_m = 50.0  # default
    if "low" in text_lower or "close" in text_lower:
        altitude_m = 25.0
    elif "high" in text_lower or "wide" in text_lower:
        altitude_m = 100.0
    elif mission_type == "SURVEY":
        altitude_m = 80.0
    elif mission_type == "MONITOR":
        altitude_m = 60.0

    # Extract objectives (simple: split on connectors)
    objectives = [text.strip()]
    for conj in (" and ", " then ", " after ", " while "):
        if conj in text_lower:
            parts = text.split(conj, 1)
            objectives = [p.strip() for p in parts if p.strip()]
            break

    return {
        "mission_type": mission_type,
        "pattern":      pattern,
        "altitude_m":   altitude_m,
        "objectives":   objectives,
        "confidence":   round(confidence, 3),
        "raw_text":     text,
    }


# ---------------------------------------------------------------------------
# Per-asset execution state
# ---------------------------------------------------------------------------

class AssetRunner:
    """
    Manages one asset's mission execution:
      - Maintains a waypoint queue assigned by SwarmPlanner
      - Issues GOTO commands when the asset advances
      - Monitors blackboard for RTB triggers
      - Reports detections back to the orchestrator
    """

    def __init__(
        self,
        asset_id: str,
        asset_type: str,
        waypoints: List[Dict],
        dispatch_fn: Callable,         # async (asset_id, command_dict) → None
        on_detection: Callable,        # async (asset_id, detection) → None
        on_complete: Callable,         # async (asset_id) → None
    ):
        self.asset_id = asset_id
        self.asset_type = asset_type
        self.waypoints = list(waypoints)
        self._dispatch = dispatch_fn
        self._on_detection = on_detection
        self._on_complete = on_complete

        self.current_wp_idx: int = 0
        self.battery_pct: float = 100.0
        self.lat: float = 0.0
        self.lon: float = 0.0
        self.status: str = "idle"       # idle | en_route | scanning | rtb | done
        self._rtb_triggered: bool = False

    def update_telemetry(self, entity: Dict) -> None:
        """Called each tick from WorldStore entity data."""
        meta = entity.get("metadata", {})
        self.battery_pct = float(
            meta.get("battery_remaining", meta.get("battery_pct", self.battery_pct))
        )
        self.lat = float(entity.get("lat", self.lat))
        self.lon = float(entity.get("lon", self.lon))

    async def tick(self) -> None:
        """One execution cycle for this asset."""
        if self.status in ("rtb", "done"):
            return

        # Battery safety — RTB below 20%
        if self.battery_pct < 20.0 and not self._rtb_triggered:
            logger.info("[%s] Battery %.0f%% — triggering RTB", self.asset_id, self.battery_pct)
            await self._rtb()
            return

        if self._rtb_triggered:
            return

        if self.current_wp_idx >= len(self.waypoints):
            logger.info("[%s] All waypoints visited — mission complete", self.asset_id)
            self.status = "done"
            await self._on_complete(self.asset_id)
            return

        target = self.waypoints[self.current_wp_idx]
        dist_m = _haversine_m(self.lat, self.lon, target["lat"], target["lon"])

        if dist_m < target.get("radius_m", 15.0):
            # Arrived — advance
            self.current_wp_idx += 1
            self.status = "scanning"
        else:
            if self.status != "en_route":
                self.status = "en_route"
                await self._goto(target)

    async def redirect(self, new_waypoints: List[Dict]) -> None:
        """Inject new waypoints (e.g. after replanning on detection)."""
        self.waypoints = new_waypoints
        self.current_wp_idx = 0
        self.status = "idle"
        logger.info("[%s] Redirected — %d new waypoints", self.asset_id, len(new_waypoints))

    async def _goto(self, wp: Dict) -> None:
        if self.asset_type == "onvif":
            # Cameras can't fly — issue PTZ toward target bearing instead
            bearing = math.degrees(math.atan2(
                wp["lon"] - self.lon, wp["lat"] - self.lat
            ))
            await self._dispatch(self.asset_id, {
                "type": "PTZ_MOVE",
                "pan": bearing / 180.0,
                "tilt": -0.1,
                "zoom": 0.5,
            })
        else:
            await self._dispatch(self.asset_id, {
                "type": "GOTO",
                "lat": wp["lat"],
                "lon": wp["lon"],
                "alt": wp.get("alt", 50.0),
            })

    async def _rtb(self) -> None:
        self._rtb_triggered = True
        self.status = "rtb"
        await self._dispatch(self.asset_id, {"type": "RTL"})


# ---------------------------------------------------------------------------
# Mission Orchestrator
# ---------------------------------------------------------------------------

class MissionOrchestrator:
    """
    Single active mission execution engine.

    Lifecycle:
      orch = MissionOrchestrator(mission_id, parse_result, area, world_store, dispatch_fn)
      await orch.start()         # launches background asyncio task
      await orch.stop()          # graceful shutdown
    """

    TICK_INTERVAL = 2.0          # seconds between execution ticks
    REPLAN_COOLDOWN = 15.0       # minimum seconds between replans

    def __init__(
        self,
        mission_id: str,
        nlp_result: Dict,          # output of parse_mission_nlp()
        area: List[Dict],          # [{lat, lon}, ...] search polygon
        world_store_fn: Callable,  # async () → List[entity_dict]
        dispatch_fn: Callable,     # async (entity_id, command_dict) → None
        broadcast_fn: Optional[Callable] = None,  # async (event_dict) → None
    ):
        self.mission_id = mission_id
        self.nlp = nlp_result
        self.area = area
        self._get_entities = world_store_fn
        self._dispatch = dispatch_fn
        self._broadcast = broadcast_fn or (lambda e: asyncio.sleep(0))

        self._runners: Dict[str, AssetRunner] = {}
        self._detections: List[Dict] = []
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self._last_replan: float = 0.0
        self._status = "initializing"

    # ── Public API ──────────────────────────────────────────────────────────

    async def start(self) -> None:
        self._running = True
        self._task = asyncio.create_task(self._run(), name=f"orch-{self.mission_id}")
        logger.info("Mission %s started: %s", self.mission_id, self.nlp.get("interpretation", ""))

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
        logger.info("Mission %s stopped", self.mission_id)

    def status_summary(self) -> Dict:
        return {
            "mission_id": self.mission_id,
            "status": self._status,
            "objective": self.nlp.get("interpretation", ""),
            "assets": {
                aid: {"status": r.status, "wp": r.current_wp_idx,
                      "total_wp": len(r.waypoints), "battery": r.battery_pct}
                for aid, r in self._runners.items()
            },
            "detections": len(self._detections),
        }

    # ── Core loop ───────────────────────────────────────────────────────────

    async def _run(self) -> None:
        try:
            await self._initialize()
            while self._running:
                await self._tick()
                await asyncio.sleep(self.TICK_INTERVAL)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error("Mission %s orchestrator error: %s", self.mission_id, e, exc_info=True)
            self._status = "error"

    async def _initialize(self) -> None:
        self._status = "planning"

        # 1. Discover assets from WorldStore
        entities = await self._get_entities()
        mission_type = self.nlp.get("mission_type", "SURVEY")
        required_cap = MISSION_CAPABILITY.get(mission_type, "goto")
        capable_assets = self._filter_capable_assets(entities, required_cap)

        if not capable_assets:
            logger.warning("Mission %s: no capable assets found", self.mission_id)
            self._status = "waiting_for_assets"
            return

        # 2. Generate waypoints
        pattern = self.nlp.get("pattern", "grid")
        alt_m = float(self.nlp.get("altitude_m", 50.0))
        all_waypoints = _generate_waypoints(pattern, self.area, alt_m)

        if not all_waypoints:
            logger.warning("Mission %s: no waypoints generated (empty area?)", self.mission_id)
            self._status = "error"
            return

        # 3. Allocate waypoints to assets via SwarmPlanner
        allocation = self._allocate(capable_assets, all_waypoints)

        # 4. Spin up an AssetRunner per assigned asset
        for asset_id, waypoints in allocation.items():
            entity = next((e for e in capable_assets if e["entity_id"] == asset_id), {})
            asset_type = entity.get("adapter_type", "mavlink")
            self._runners[asset_id] = AssetRunner(
                asset_id=asset_id,
                asset_type=asset_type,
                waypoints=waypoints,
                dispatch_fn=self._dispatch,
                on_detection=self._on_detection,
                on_complete=self._on_asset_complete,
            )

        self._status = "executing"
        logger.info(
            "Mission %s: %d assets, %d waypoints, pattern=%s",
            self.mission_id, len(self._runners), len(all_waypoints), pattern,
        )
        await self._broadcast({
            "type": "mission_started",
            "mission_id": self.mission_id,
            "assets": list(self._runners.keys()),
            "waypoint_count": len(all_waypoints),
            "objective": self.nlp.get("interpretation", ""),
        })

    async def _tick(self) -> None:
        if self._status not in ("executing",):
            return

        entities = await self._get_entities()
        entity_map = {e["entity_id"]: e for e in entities}

        for asset_id, runner in list(self._runners.items()):
            entity = entity_map.get(asset_id, {})
            if entity:
                runner.update_telemetry(entity)

            # Check c2_intel observations on this entity
            await self._check_c2intel(asset_id, entity)

            await runner.tick()

        # Broadcast status every tick
        await self._broadcast({
            "type": "mission_tick",
            "data": self.status_summary(),
        })

        # Check if all assets are done
        if self._runners and all(r.status in ("done", "rtb") for r in self._runners.values()):
            self._status = "complete"
            logger.info("Mission %s complete", self.mission_id)
            await self._broadcast({
                "type": "mission_complete",
                "mission_id": self.mission_id,
                "detections": self._detections,
            })
            self._running = False

    async def _check_c2intel(self, asset_id: str, entity: Dict) -> None:
        """
        Read latest c2_intel observations for this asset and react.

        ENTITY_DETECTED  → redirect nearest idle asset to investigate
        BATTERY_CRITICAL → preempt RTB (already handled in AssetRunner.tick)
        COMMS_DEGRADED   → mark asset degraded, skip to next
        ASSET_OFFLINE    → remove from pool, replan
        """
        observations = entity.get("_observations", [])
        for obs in observations:
            evt = obs.get("event_type", "")

            if evt == "entity_detected" and len(self._detections) == 0:
                detection = {
                    "asset_id": asset_id,
                    "lat": obs.get("lat", entity.get("lat")),
                    "lon": obs.get("lon", entity.get("lon")),
                    "confidence": obs.get("confidence", 0.5),
                    "ts": time.time(),
                }
                self._detections.append(detection)
                logger.info("[%s] ENTITY DETECTED — confidence %.2f", asset_id, detection["confidence"])
                await self._on_detection(asset_id, detection)

            elif evt == "asset_offline":
                if asset_id in self._runners:
                    logger.warning("[%s] ASSET OFFLINE — removing from mission", asset_id)
                    del self._runners[asset_id]
                    await self._replan()

    # ── Callbacks ───────────────────────────────────────────────────────────

    async def _on_detection(self, detecting_asset_id: str, detection: Dict) -> None:
        """Target detected — converge idle assets on detection point."""
        logger.info("Detection from %s — converging assets", detecting_asset_id)

        det_wp = {
            "lat": detection["lat"],
            "lon": detection["lon"],
            "alt": float(self.nlp.get("altitude_m", 50.0)),
            "radius_m": 20.0,
        }

        # Redirect idle assets toward detection
        redirected = 0
        for asset_id, runner in self._runners.items():
            if asset_id != detecting_asset_id and runner.status in ("scanning", "idle", "en_route"):
                await runner.redirect([det_wp])
                redirected += 1
                if redirected >= 2:  # don't send the whole fleet, just nearest 2
                    break

        await self._broadcast({
            "type": "detection",
            "mission_id": self.mission_id,
            "detection": detection,
            "redirected_assets": redirected,
        })

    async def _on_asset_complete(self, asset_id: str) -> None:
        logger.info("[%s] Asset finished waypoints", asset_id)

    # ── Planning helpers ────────────────────────────────────────────────────

    async def _replan(self) -> None:
        """Re-run allocation when asset pool changes."""
        now = time.time()
        if now - self._last_replan < self.REPLAN_COOLDOWN:
            return
        self._last_replan = now

        if not self._runners:
            self._status = "no_assets"
            return

        logger.info("Mission %s: replanning with %d assets", self.mission_id, len(self._runners))

        # Collect remaining unvisited waypoints across all runners
        remaining = []
        for runner in self._runners.values():
            remaining.extend(runner.waypoints[runner.current_wp_idx:])

        if not remaining:
            return

        # Re-allocate remaining waypoints to surviving assets
        surviving = [{"entity_id": aid, "lat": r.lat, "lon": r.lon,
                      "battery_pct": r.battery_pct, "adapter_type": r.asset_type}
                     for aid, r in self._runners.items()]
        allocation = self._allocate(surviving, remaining)

        for asset_id, waypoints in allocation.items():
            if asset_id in self._runners:
                await self._runners[asset_id].redirect(waypoints)

    def _filter_capable_assets(self, entities: List[Dict], required_cap: str) -> List[Dict]:
        capable = []
        for e in entities:
            adapter_type = e.get("adapter_type", "")
            caps = ASSET_CAPABILITIES.get(adapter_type, [])
            if required_cap in caps or not required_cap:
                capable.append(e)
        return capable

    def _allocate(self, assets: List[Dict], waypoints: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Divide waypoints across assets using distance-based greedy allocation.
        SwarmPlanner's Hungarian algorithm works per-task; here we chunk the
        waypoint list into N roughly-equal segments and assign each chunk to
        the closest asset geographically.
        """
        if not assets or not waypoints:
            return {}

        n = len(assets)
        chunk_size = max(1, math.ceil(len(waypoints) / n))
        chunks = [waypoints[i:i + chunk_size] for i in range(0, len(waypoints), chunk_size)]

        # Assign each chunk to the nearest asset (greedy, no repeat)
        assignment: Dict[str, List[Dict]] = {}
        assigned_assets = set()

        for chunk in chunks:
            if not chunk:
                continue
            target = chunk[0]  # first waypoint in chunk as anchor
            best_asset = None
            best_dist = float("inf")

            for asset in assets:
                aid = asset["entity_id"]
                if aid in assigned_assets:
                    continue
                dist = _haversine_m(
                    asset.get("lat", 0.0), asset.get("lon", 0.0),
                    target["lat"], target["lon"],
                )
                # Penalise low battery
                battery = float(asset.get("battery_pct", asset.get("battery_remaining", 100)))
                effective_dist = dist / max(0.1, battery / 100.0)

                if effective_dist < best_dist:
                    best_dist = effective_dist
                    best_asset = aid

            if best_asset:
                assignment[best_asset] = chunk
                assigned_assets.add(best_asset)

        return assignment


# ---------------------------------------------------------------------------
# Registry — one orchestrator per active mission
# ---------------------------------------------------------------------------

_active: Dict[str, MissionOrchestrator] = {}


async def launch_mission(
    nlp_result: Dict,
    area: List[Dict],
    world_store_fn: Callable,
    dispatch_fn: Callable,
    broadcast_fn: Optional[Callable] = None,
    mission_id: Optional[str] = None,
) -> str:
    """
    Parse, plan, and execute a mission. Returns the mission_id.

    Args:
        nlp_result:     Output of parse_mission_nlp() — mission_type, pattern, altitude_m, etc.
        area:           List of {lat, lon} dicts defining the search polygon.
        world_store_fn: async callable returning current entity list from WorldStore.
        dispatch_fn:    async callable (entity_id, command_dict) → None.
        broadcast_fn:   optional async callable (event_dict) → None for WebSocket push.
        mission_id:     optional; generated if not provided.
    """
    mid = mission_id or f"mission-{uuid.uuid4().hex[:8]}"

    # Stop any existing mission with same id
    if mid in _active:
        await _active[mid].stop()

    orch = MissionOrchestrator(
        mission_id=mid,
        nlp_result=nlp_result,
        area=area,
        world_store_fn=world_store_fn,
        dispatch_fn=dispatch_fn,
        broadcast_fn=broadcast_fn,
    )
    _active[mid] = orch
    await orch.start()
    return mid


async def stop_mission(mission_id: str) -> bool:
    if mission_id not in _active:
        return False
    await _active[mission_id].stop()
    del _active[mission_id]
    return True


def get_mission_status(mission_id: str) -> Optional[Dict]:
    orch = _active.get(mission_id)
    return orch.status_summary() if orch else None


def list_active_missions() -> List[Dict]:
    return [o.status_summary() for o in _active.values()]
