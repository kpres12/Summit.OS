"""
apps/tasking/role_decomposer.py — Mission role decomposition by asset domain.

This is the layer that answers: "Given a mixed fleet of drones, cameras,
ground robots, submarines, and mesh radios — and a mission intent like
'search and rescue' — what does each asset TYPE do?"

The decomposer:
  1. Classifies every available asset into a domain (AERIAL, GROUND, CAMERA,
     SUBSURFACE, MARITIME, MESH, SENSOR)
  2. Maps the mission intent to a per-domain role + behavior template
  3. Returns a RoleManifest: one RoleAssignment per domain group

The tasking planner then executes each RoleAssignment as an independent
sub-mission, all running in parallel under the same mission ID.

--- Example ---

Intent: "search_and_rescue"
Available assets:
  drone-1 (aerial, thermal)
  drone-2 (aerial, rgb_camera)
  spot-1  (ground robot)
  cam-1   (fixed RTSP camera)
  auv-1   (subsurface)
  mesh-1  (meshtastic relay)

Decomposed:
  AERIAL   → drones do expanding_square search of full area
  GROUND   → Spot does grid search of accessible terrain zones
  CAMERA   → Cameras orient toward area center, run inference, alert on detection
  SUBSURFACE → AUVs do grid search of water column beneath area
  MESH     → Mesh nodes position as comms relays between ground teams

Each role gets:
  - The right movement pattern for its domain
  - The right sensors activated for the intent
  - The right altitude/depth operating envelope
  - A role description the operator can read ("Aerial search — expanding square")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger("tasking.role_decomposer")


# ── Asset domain classification ────────────────────────────────────────────────

class AssetDomain:
    AERIAL     = "aerial"       # drones, UAVs, fixed-wing
    GROUND     = "ground"       # wheeled/legged robots, ground vehicles
    CAMERA     = "camera"       # fixed cameras (RTSP, ONVIF) — can't move
    SUBSURFACE = "subsurface"   # AUVs, submarines
    MARITIME   = "maritime"     # surface vessels, ASVs
    MESH       = "mesh"         # Meshtastic radio nodes (comms/relay)
    SENSOR     = "sensor"       # fixed IoT sensors — environmental, perimeter


# Keywords that identify an asset's domain from its type/capabilities fields
_DOMAIN_KEYWORDS: Dict[str, List[str]] = {
    AssetDomain.AERIAL: [
        "drone", "uav", "quadcopter", "hexacopter", "fixed_wing", "aerial",
        "mavlink", "skydio", "autel", "parrot", "dji", "vtol", "multirotor",
    ],
    AssetDomain.GROUND: [
        "spot", "robot", "ground", "rover", "ugv", "wheeled", "tracked",
        "legged", "ros2", "clearpath", "agile", "boston_dynamics",
    ],
    AssetDomain.CAMERA: [
        "camera", "rtsp", "onvif", "cam", "ptz", "fixed_camera", "cctv",
        "ip_camera", "surveillance_cam",
    ],
    AssetDomain.SUBSURFACE: [
        "submarine", "auv", "rov", "underwater", "subsurface", "torpedo",
        "glider", "bluerobotics", "fathom",
    ],
    AssetDomain.MARITIME: [
        "vessel", "asv", "boat", "ship", "maritime", "surface_vessel",
        "usv", "ais", "nmea",
    ],
    AssetDomain.MESH: [
        "meshtastic", "mesh", "lora", "tbeam", "rak", "heltec", "radio",
    ],
    AssetDomain.SENSOR: [
        "sensor", "weather_station", "lorawan", "zigbee", "iot",
        "environment", "perimeter_sensor",
    ],
}


def classify_asset_domain(asset: Dict[str, Any]) -> str:
    """
    Classify an asset into a domain based on its type and capabilities.
    Returns an AssetDomain constant. Defaults to AERIAL (most common).
    """
    asset_type = str(asset.get("type", "")).lower()
    caps = asset.get("capabilities") or {}
    if isinstance(caps, str):
        import json
        try:
            caps = json.loads(caps)
        except Exception:
            caps = {}

    # Build a searchable string from type + capability keys/values
    caps_str = " ".join(
        str(k) + " " + str(v)
        for k, v in (caps.items() if isinstance(caps, dict) else {}.items())
    ).lower()
    search_str = f"{asset_type} {caps_str}"

    for domain, keywords in _DOMAIN_KEYWORDS.items():
        for kw in keywords:
            if kw in search_str:
                return domain

    # Default: if it has a mavlink_conn or waypoints capability, it's aerial
    if isinstance(caps, dict) and caps.get("mavlink_conn"):
        return AssetDomain.AERIAL

    return AssetDomain.AERIAL  # safest default


# ── Role templates per (intent × domain) ──────────────────────────────────────

@dataclass
class RoleTemplate:
    """Defines what an asset domain DOES in a given mission intent."""
    role_name: str          # human-readable: "Aerial search — expanding square"
    pattern: str            # movement pattern key from planning.py
    altitude_m: float       # operating altitude/depth in metres
    speed_mps: float        # operating speed in m/s
    sensors_active: List[str]  # sensor types to activate
    behavior: str           # "search" | "monitor" | "relay" | "fixed_watch" | "orbit"
    priority: int           # execution order (lower = runs first)
    description: str        # what the operator sees in the console


# (intent, domain) → RoleTemplate
# Intents: search_and_rescue, surveillance, inspection, mapping,
#          disaster_response, wildfire_ops, monitoring, survey, delivery
_ROLE_TEMPLATES: Dict[tuple, RoleTemplate] = {

    # ── SEARCH AND RESCUE ──────────────────────────────────────────────────────
    ("search_and_rescue", AssetDomain.AERIAL): RoleTemplate(
        role_name="Aerial search",
        pattern="expanding_square",
        altitude_m=60,
        speed_mps=8.0,
        sensors_active=["thermal", "rgb_camera"],
        behavior="search",
        priority=1,
        description="Drones fly expanding square search pattern with thermal imaging",
    ),
    ("search_and_rescue", AssetDomain.GROUND): RoleTemplate(
        role_name="Ground search",
        pattern="grid",
        altitude_m=0,
        speed_mps=1.5,
        sensors_active=["rgb_camera", "thermal", "gas_sensor"],
        behavior="search",
        priority=2,
        description="Ground robots systematically cover accessible terrain",
    ),
    ("search_and_rescue", AssetDomain.CAMERA): RoleTemplate(
        role_name="Overwatch",
        pattern="fixed_watch",
        altitude_m=0,
        speed_mps=0,
        sensors_active=["rgb_camera", "thermal"],
        behavior="fixed_watch",
        priority=1,
        description="Fixed cameras orient toward search area and alert on human detection",
    ),
    ("search_and_rescue", AssetDomain.SUBSURFACE): RoleTemplate(
        role_name="Underwater search",
        pattern="grid",
        altitude_m=-15,  # depth in metres (negative = below surface)
        speed_mps=1.5,
        sensors_active=["sonar", "rgb_camera"],
        behavior="search",
        priority=2,
        description="AUVs run underwater grid search of submerged area",
    ),
    ("search_and_rescue", AssetDomain.MARITIME): RoleTemplate(
        role_name="Surface sweep",
        pattern="expanding_square",
        altitude_m=0,
        speed_mps=3.0,
        sensors_active=["rgb_camera", "thermal", "sonar"],
        behavior="search",
        priority=2,
        description="Surface vessels sweep water area for persons in water",
    ),
    ("search_and_rescue", AssetDomain.MESH): RoleTemplate(
        role_name="Comms relay",
        pattern="relay_positioning",
        altitude_m=0,
        speed_mps=0,
        sensors_active=[],
        behavior="relay",
        priority=0,
        description="Mesh nodes position as comms relay between field teams and base",
    ),

    # ── SURVEILLANCE ──────────────────────────────────────────────────────────
    ("surveillance", AssetDomain.AERIAL): RoleTemplate(
        role_name="Aerial overwatch",
        pattern="orbit",
        altitude_m=80,
        speed_mps=6.0,
        sensors_active=["rgb_camera", "thermal"],
        behavior="monitor",
        priority=1,
        description="Drones orbit area perimeter with continuous camera coverage",
    ),
    ("surveillance", AssetDomain.GROUND): RoleTemplate(
        role_name="Ground patrol",
        pattern="perimeter",
        altitude_m=0,
        speed_mps=1.2,
        sensors_active=["rgb_camera", "thermal"],
        behavior="monitor",
        priority=2,
        description="Ground robots patrol perimeter boundary",
    ),
    ("surveillance", AssetDomain.CAMERA): RoleTemplate(
        role_name="Fixed surveillance",
        pattern="fixed_watch",
        altitude_m=0,
        speed_mps=0,
        sensors_active=["rgb_camera", "thermal"],
        behavior="fixed_watch",
        priority=1,
        description="Fixed cameras provide continuous wide-area monitoring",
    ),
    ("surveillance", AssetDomain.SUBSURFACE): RoleTemplate(
        role_name="Underwater perimeter",
        pattern="perimeter",
        altitude_m=-10,
        speed_mps=1.0,
        sensors_active=["sonar", "rgb_camera"],
        behavior="monitor",
        priority=2,
        description="AUVs patrol underwater perimeter of monitored area",
    ),
    ("surveillance", AssetDomain.MARITIME): RoleTemplate(
        role_name="Surface patrol",
        pattern="perimeter",
        altitude_m=0,
        speed_mps=2.5,
        sensors_active=["rgb_camera", "radar"],
        behavior="monitor",
        priority=2,
        description="Surface vessels patrol waterway boundary",
    ),
    ("surveillance", AssetDomain.MESH): RoleTemplate(
        role_name="Comms relay",
        pattern="relay_positioning",
        altitude_m=0,
        speed_mps=0,
        sensors_active=[],
        behavior="relay",
        priority=0,
        description="Mesh radio relay for surveillance team coordination",
    ),

    # ── MAPPING ───────────────────────────────────────────────────────────────
    ("mapping", AssetDomain.AERIAL): RoleTemplate(
        role_name="Aerial mapping",
        pattern="grid",
        altitude_m=100,
        speed_mps=10.0,
        sensors_active=["rgb_camera", "lidar"],
        behavior="search",
        priority=1,
        description="Drones fly systematic grid for photogrammetry and LiDAR mapping",
    ),
    ("mapping", AssetDomain.GROUND): RoleTemplate(
        role_name="Ground truth survey",
        pattern="grid",
        altitude_m=0,
        speed_mps=0.8,
        sensors_active=["rgb_camera", "lidar", "gps"],
        behavior="search",
        priority=2,
        description="Ground robots collect ground-truth reference points for map accuracy",
    ),
    ("mapping", AssetDomain.SUBSURFACE): RoleTemplate(
        role_name="Bathymetric survey",
        pattern="grid",
        altitude_m=-5,
        speed_mps=1.2,
        sensors_active=["sonar", "depth_sensor"],
        behavior="search",
        priority=1,
        description="AUVs produce bathymetric depth map of underwater terrain",
    ),
    ("mapping", AssetDomain.MARITIME): RoleTemplate(
        role_name="Surface survey",
        pattern="grid",
        altitude_m=0,
        speed_mps=3.0,
        sensors_active=["sonar", "rgb_camera"],
        behavior="search",
        priority=1,
        description="Surface vessels run multibeam sonar survey",
    ),

    # ── DISASTER RESPONSE ─────────────────────────────────────────────────────
    ("disaster_response", AssetDomain.AERIAL): RoleTemplate(
        role_name="Damage assessment",
        pattern="grid",
        altitude_m=50,
        speed_mps=7.0,
        sensors_active=["rgb_camera", "thermal", "lidar"],
        behavior="search",
        priority=1,
        description="Drones systematically image affected area for damage assessment",
    ),
    ("disaster_response", AssetDomain.GROUND): RoleTemplate(
        role_name="Ground recon",
        pattern="grid",
        altitude_m=0,
        speed_mps=1.0,
        sensors_active=["rgb_camera", "gas_sensor", "thermal"],
        behavior="search",
        priority=2,
        description="Ground robots assess structural hazards and locate survivors",
    ),
    ("disaster_response", AssetDomain.CAMERA): RoleTemplate(
        role_name="Situation monitoring",
        pattern="fixed_watch",
        altitude_m=0,
        speed_mps=0,
        sensors_active=["rgb_camera"],
        behavior="fixed_watch",
        priority=1,
        description="Cameras monitor evacuation routes and hazard zones",
    ),
    ("disaster_response", AssetDomain.MESH): RoleTemplate(
        role_name="Emergency comms mesh",
        pattern="relay_positioning",
        altitude_m=0,
        speed_mps=0,
        sensors_active=[],
        behavior="relay",
        priority=0,
        description="Meshtastic mesh replaces downed cellular infrastructure",
    ),

    # ── WILDFIRE OPS ──────────────────────────────────────────────────────────
    ("wildfire_ops", AssetDomain.AERIAL): RoleTemplate(
        role_name="Fire perimeter mapping",
        pattern="perimeter",
        altitude_m=120,
        speed_mps=12.0,
        sensors_active=["thermal", "rgb_camera"],
        behavior="monitor",
        priority=1,
        description="Drones track active fire perimeter with thermal imaging",
    ),
    ("wildfire_ops", AssetDomain.GROUND): RoleTemplate(
        role_name="Ground sensor deployment",
        pattern="grid",
        altitude_m=0,
        speed_mps=0.8,
        sensors_active=["gas_sensor", "thermal", "rgb_camera"],
        behavior="search",
        priority=2,
        description="Ground robots deploy to measure fire behavior at ground level",
    ),
    ("wildfire_ops", AssetDomain.CAMERA): RoleTemplate(
        role_name="Fire watch",
        pattern="fixed_watch",
        altitude_m=0,
        speed_mps=0,
        sensors_active=["thermal", "rgb_camera"],
        behavior="fixed_watch",
        priority=1,
        description="Fixed cameras provide real-time fire front monitoring",
    ),
    ("wildfire_ops", AssetDomain.MESH): RoleTemplate(
        role_name="Firefighter comms relay",
        pattern="relay_positioning",
        altitude_m=0,
        speed_mps=0,
        sensors_active=[],
        behavior="relay",
        priority=0,
        description="Mesh relay for ground crew coordination in dead zones",
    ),

    # ── INSPECTION ────────────────────────────────────────────────────────────
    ("inspection", AssetDomain.AERIAL): RoleTemplate(
        role_name="Aerial inspection",
        pattern="orbit",
        altitude_m=30,
        speed_mps=3.0,
        sensors_active=["rgb_camera", "thermal", "lidar"],
        behavior="monitor",
        priority=1,
        description="Drones orbit structure with close-range imaging",
    ),
    ("inspection", AssetDomain.GROUND): RoleTemplate(
        role_name="Ground inspection",
        pattern="grid",
        altitude_m=0,
        speed_mps=0.5,
        sensors_active=["rgb_camera", "lidar", "thermal"],
        behavior="search",
        priority=2,
        description="Ground robots inspect base structure, foundation, equipment",
    ),
    ("inspection", AssetDomain.SUBSURFACE): RoleTemplate(
        role_name="Hull/infrastructure inspection",
        pattern="perimeter",
        altitude_m=-3,
        speed_mps=0.5,
        sensors_active=["rgb_camera", "sonar"],
        behavior="monitor",
        priority=1,
        description="AUVs inspect submerged hull, piles, or underwater infrastructure",
    ),
    ("inspection", AssetDomain.CAMERA): RoleTemplate(
        role_name="Fixed inspection camera",
        pattern="fixed_watch",
        altitude_m=0,
        speed_mps=0,
        sensors_active=["rgb_camera"],
        behavior="fixed_watch",
        priority=1,
        description="Fixed cameras monitor structure for ongoing change detection",
    ),
}

# Fallback role for any (intent, domain) pair not in the table
def _default_role(intent: str, domain: str) -> RoleTemplate:
    return RoleTemplate(
        role_name=f"{domain.title()} — {intent}",
        pattern="loiter" if domain != AssetDomain.CAMERA else "fixed_watch",
        altitude_m=50 if domain == AssetDomain.AERIAL else 0,
        speed_mps=5.0 if domain == AssetDomain.AERIAL else 1.0,
        sensors_active=["rgb_camera"],
        behavior="monitor",
        priority=3,
        description=f"{domain.title()} assets supporting {intent} mission",
    )


# ── Role manifest ──────────────────────────────────────────────────────────────

@dataclass
class RoleAssignment:
    """One role group within a decomposed mission."""
    domain: str
    role_name: str
    description: str
    pattern: str
    altitude_m: float
    speed_mps: float
    sensors_active: List[str]
    behavior: str
    priority: int
    assets: List[Dict[str, Any]]       # asset dicts assigned to this role
    asset_ids: List[str]               # convenience list of IDs
    planning_params: Dict[str, Any]    # ready-to-use params for _plan_assignments()


@dataclass
class RoleManifest:
    """Full decomposed mission — one RoleAssignment per domain group."""
    intent: str
    roles: List[RoleAssignment]
    unassigned_assets: List[Dict[str, Any]]  # assets that didn't fit any role

    def summary(self) -> str:
        parts = []
        for r in sorted(self.roles, key=lambda x: x.priority):
            parts.append(
                f"{r.role_name} ({len(r.assets)} asset{'s' if len(r.assets)!=1 else ''})"
            )
        return " | ".join(parts) if parts else "No roles assigned"

    def to_console_brief(self) -> List[Dict[str, Any]]:
        """Returns operator-readable role brief for the console."""
        return [
            {
                "role": r.role_name,
                "domain": r.domain,
                "description": r.description,
                "assets": r.asset_ids,
                "pattern": r.pattern,
                "priority": r.priority,
            }
            for r in sorted(self.roles, key=lambda x: x.priority)
        ]


# ── Decomposer ────────────────────────────────────────────────────────────────

class RoleDecomposer:
    """
    Decomposes a mission intent + available asset pool into per-domain roles.

    Usage:
        decomposer = RoleDecomposer()
        manifest = decomposer.decompose(
            intent="search_and_rescue",
            available_assets=assets_from_db,
            area=req.area,
        )
        # manifest.roles → one RoleAssignment per hardware domain
        # manifest.summary() → "Aerial search (2 assets) | Ground search (1 asset) | ..."
    """

    # Intents that map to a SAR-style search (expanding from center)
    _SEARCH_INTENTS = {"search_and_rescue", "sar", "search", "disaster_response"}

    # Intents that are perimeter/monitoring focused
    _MONITOR_INTENTS = {"surveillance", "monitoring", "wildfire_ops", "containment"}

    # Intents that need systematic coverage
    _COVERAGE_INTENTS = {"mapping", "survey", "inspection"}

    def decompose(
        self,
        intent: str,
        available_assets: List[Dict[str, Any]],
        area: Optional[Dict[str, Any]] = None,
        planning_params: Optional[Dict[str, Any]] = None,
    ) -> RoleManifest:
        """
        Decompose mission into per-domain role assignments.

        Args:
            intent:           Mission intent string
            available_assets: List of asset dicts from the tasking DB
            area:             Mission area dict (center/radius/polygon)
            planning_params:  Base planning params to merge with role defaults

        Returns RoleManifest with one RoleAssignment per detected domain.
        """
        intent_norm = self._normalize_intent(intent)
        base_params = planning_params or {}

        # 1. Group assets by domain
        domain_groups: Dict[str, List[Dict[str, Any]]] = {}
        unassigned = []

        for asset in available_assets:
            domain = classify_asset_domain(asset)
            domain_groups.setdefault(domain, []).append(asset)

        logger.info(
            "Role decomposition: intent=%s domains=%s",
            intent_norm,
            {d: len(a) for d, a in domain_groups.items()},
        )

        # 2. Build RoleAssignment for each domain
        roles: List[RoleAssignment] = []

        for domain, assets in domain_groups.items():
            template = _ROLE_TEMPLATES.get(
                (intent_norm, domain),
                _ROLE_TEMPLATES.get((intent_norm.replace("_", ""), domain))
                or _default_role(intent_norm, domain),
            )

            # Merge base planning params with role defaults
            role_params = {
                "altitude": template.altitude_m,
                "speed": template.speed_mps,
                "pattern": template.pattern,
                "sensors_active": template.sensors_active,
                "behavior": template.behavior,
                "intent": intent_norm,
                **base_params,  # operator-specified params override defaults
            }

            # Special handling for fixed assets (cameras, mesh nodes)
            if template.behavior in ("fixed_watch", "relay"):
                role_params["pattern"] = template.pattern
                # Fixed assets don't need waypoint planning — just activate
                role_params["fixed"] = True

            roles.append(RoleAssignment(
                domain=domain,
                role_name=template.role_name,
                description=template.description,
                pattern=template.pattern,
                altitude_m=template.altitude_m,
                speed_mps=template.speed_mps,
                sensors_active=template.sensors_active,
                behavior=template.behavior,
                priority=template.priority,
                assets=assets,
                asset_ids=[a["asset_id"] for a in assets],
                planning_params=role_params,
            ))

        # Sort by priority
        roles.sort(key=lambda r: r.priority)

        manifest = RoleManifest(
            intent=intent_norm,
            roles=roles,
            unassigned_assets=unassigned,
        )

        logger.info("Role manifest: %s", manifest.summary())
        return manifest

    def _normalize_intent(self, intent: str) -> str:
        """Normalize intent string to a known key."""
        s = intent.lower().strip().replace("-", "_").replace(" ", "_")
        # Common aliases
        aliases = {
            "sar": "search_and_rescue",
            "rescue": "search_and_rescue",
            "recon": "surveillance",
            "photo": "mapping",
            "photography": "mapping",
            "fire": "wildfire_ops",
            "wildfire": "wildfire_ops",
            "disaster": "disaster_response",
            "monitor": "surveillance",
            "patrol": "surveillance",
            "survey": "mapping",
        }
        return aliases.get(s, s)
