import os
import json
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
import paho.mqtt.client as mqtt
from sqlalchemy import (
    Column,
    DateTime,
    Integer,
    MetaData,
    String,
    Table,
    text,
    Boolean,
    Float,
    JSON,
)
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

# Optional metrics/tracing/auth imports
try:
    from prometheus_client import Counter, Gauge, CONTENT_TYPE_LATEST, generate_latest
    PROM_AVAILABLE = True
except Exception:
    PROM_AVAILABLE = False

try:
    from jose import jwt
    OIDC_AVAILABLE = True
except Exception:
    OIDC_AVAILABLE = False

# Globals
engine: Optional[AsyncEngine] = None
SessionLocal: Optional[sessionmaker] = None
mqtt_client: Optional[mqtt.Client] = None

# Optional direct autopilot control
DIRECT_AUTOPILOT = os.getenv("TASKING_DIRECT_AUTOPILOT", "false").lower() == "true"
_direct_queue: Optional[__import__('asyncio').Queue] = None
_autopilots: Dict[str, Any] = {}

# Configuration via env
OIDC_ENFORCE = os.getenv("OIDC_ENFORCE", "false").lower() == "true"
OIDC_ISSUER = os.getenv("OIDC_ISSUER")
OIDC_AUDIENCE = os.getenv("OIDC_AUDIENCE")

# DB metadata and tables
metadata = MetaData()

# Legacy task table (kept for backward compatibility)
tasks = Table(
    "tasks",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("task_id", String(128), nullable=False, unique=True),
    Column("asset_id", String(128)),
    Column("action", String(256)),
    Column("status", String(32)),  # PENDING, ACTIVE, COMPLETED, FAILED
    Column("created_at", DateTime(timezone=True)),
    Column("started_at", DateTime(timezone=True)),
    Column("completed_at", DateTime(timezone=True)),
)

assets = Table(
    "assets",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("asset_id", String(128), nullable=False, unique=True),
    Column("type", String(64), nullable=True),
    Column("capabilities", JSON, nullable=True),
    Column("battery", Float, nullable=True),
    Column("link", String(32), nullable=True),
    Column("constraints", JSON, nullable=True),
    Column("updated_at", DateTime(timezone=True)),
)

missions = Table(
    "missions",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("mission_id", String(128), nullable=False, unique=True),
    Column("name", String(256), nullable=True),
    Column("objectives", JSON, nullable=True),
    Column("area", JSON, nullable=True),
    Column("policy_ok", Boolean, nullable=False, default=False),
    Column("status", String(32), nullable=False),  # PLANNING, ACTIVE, COMPLETED, FAILED
    Column("created_at", DateTime(timezone=True)),
    Column("started_at", DateTime(timezone=True)),
    Column("completed_at", DateTime(timezone=True)),
)

mission_assignments = Table(
    "mission_assignments",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("mission_id", String(128), nullable=False),
    Column("asset_id", String(128), nullable=False),
    Column("plan", JSON, nullable=True),  # waypoints/patterns
    Column("status", String(32), nullable=False),  # ASSIGNED, DISPATCHED, ACTIVE, COMPLETED, FAILED
)

# Metrics
if PROM_AVAILABLE:
    METRIC_MISSIONS_CREATED = Counter("missions_created_total", "Number of missions created")
    METRIC_MISSIONS_ACTIVE = Gauge("missions_active", "Active missions")
    METRIC_ASSETS_REGISTERED = Counter("assets_registered_total", "Assets registered")
else:
    METRIC_MISSIONS_CREATED = None
    METRIC_MISSIONS_ACTIVE = None
    METRIC_ASSETS_REGISTERED = None


def _to_asyncpg_url(url: str) -> str:
    if url.startswith("postgresql+asyncpg://"):
        return url
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+asyncpg://", 1)
    return url


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine, SessionLocal, mqtt_client

    # DB setup
    pg_url = _to_asyncpg_url(
        os.getenv("POSTGRES_URL", "postgresql://summit:summit_password@localhost:5432/summit_os")
    )
    engine = create_async_engine(pg_url, echo=False, future=True)
    SessionLocal = sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

    async with engine.begin() as conn:
        await conn.run_sync(metadata.create_all)

    # MQTT setup
    broker = os.getenv("MQTT_BROKER", "localhost")
    port = int(os.getenv("MQTT_PORT", "1883"))
    mqtt_client = mqtt.Client()
    mqtt_user = os.getenv("MQTT_USERNAME")
    mqtt_pass = os.getenv("MQTT_PASSWORD")
    if mqtt_user and mqtt_pass:
        mqtt_client.username_pw_set(mqtt_user, mqtt_pass)
    mqtt_client.connect(broker, port, 60)
    mqtt_client.loop_start()

    # Optional: direct autopilot worker subscribes to dispatches
    if DIRECT_AUTOPILOT:
        await _init_direct_autopilot()

    try:
        yield
    finally:
        if mqtt_client:
            mqtt_client.loop_stop()
            mqtt_client.disconnect()
        if engine:
            await engine.dispose()


app = FastAPI(title="Summit Tasking", version="0.3.0", lifespan=lifespan)


# -------------------------
# Auth helper (optional OIDC)
# -------------------------
async def _require_auth(request: Request):
    if not OIDC_ENFORCE:
        return
    auth = request.headers.get("authorization") or request.headers.get("Authorization")
    if not auth or not auth.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = auth.split(" ", 1)[1]
    if not OIDC_AVAILABLE or not OIDC_ISSUER:
        # If enforcement is enabled but libs/config missing, deny
        raise HTTPException(status_code=401, detail="OIDC unavailable")
    try:
        # NOTE: In production, fetch JWKS and verify signature and claims properly
        # This is a placeholder decode without verification context
        jwt.get_unverified_claims(token)
        return
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")


# -------------------------
# Pydantic models
# -------------------------
class TaskDispatchRequest(BaseModel):
    task_id: str
    asset_id: str
    action: str
    waypoints: list = []


class Task(BaseModel):
    task_id: str
    asset_id: str
    action: str
    status: str
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None


class AssetIn(BaseModel):
    asset_id: str
    type: Optional[str] = None
    capabilities: Optional[Dict[str, Any]] = None
    battery: Optional[float] = Field(default=None, ge=0, le=100)
    link: Optional[str] = None
    constraints: Optional[Dict[str, Any]] = None


class AssetOut(AssetIn):
    updated_at: Optional[datetime] = None


class MissionCreateRequest(BaseModel):
    name: Optional[str] = None
    objectives: List[str] = Field(default_factory=list)
    area: Optional[Dict[str, Any]] = None  # e.g., {center: {lat, lon}, radius_m: 500} or {polygon: [[lat,lon], ...]}
    num_drones: Optional[int] = Field(default=None, ge=1)
    policy_context: Optional[Dict[str, Any]] = None  # airspace/geofence/weather/NOTAMs/operator
    # planning_params:
    #   pattern: "loiter" | "grid"
    #   altitude: meters
    #   speed: m/s
    #   grid_spacing_m: for pattern=grid
    #   heading_deg: lane orientation for grid
    planning_params: Optional[Dict[str, Any]] = None  # pattern: loiter|grid, altitude, speed, grid_spacing_m, heading_deg


class MissionAssignment(BaseModel):
    asset_id: str
    plan: Dict[str, Any]
    status: str


class MissionResponse(BaseModel):
    mission_id: str
    name: Optional[str]
    objectives: List[str]
    status: str
    policy_ok: bool
    assignments: List[MissionAssignment]
    created_at: datetime
    started_at: Optional[datetime] = None


# -------------------------
# Helpers
# -------------------------
async def _publish_mission_update(mission_id: str, event: Dict[str, Any]):
    if not mqtt_client:
        return
    payload = json.dumps({"mission_id": mission_id, **event, "ts_iso": datetime.now(timezone.utc).isoformat()})
    mqtt_client.publish("missions/updates", payload, qos=1)
    mqtt_client.publish(f"missions/{mission_id}", payload, qos=1)


async def _init_direct_autopilot():
    """If enabled, subscribe to tasks/+/dispatch and handle via FireFlyAutopilot when possible."""
    global _direct_queue
    assert mqtt_client is not None
    _direct_queue = __import__("asyncio").Queue()

    def _on_message(_client, _userdata, msg):
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
        except Exception:
            payload = {"raw": msg.payload.decode("utf-8", errors="ignore")}
        # thread-safe put into asyncio queue
        loop = __import__("asyncio").get_event_loop()
        loop.call_soon_threadsafe(_direct_queue.put_nowait, {"topic": msg.topic, "payload": payload})

    mqtt_client.on_message = _on_message
    mqtt_client.subscribe("tasks/+/dispatch", qos=1)
    __import__("asyncio").create_task(_direct_autopilot_worker())


async def _direct_autopilot_worker():
    """Consume dispatches and drive autopilot directly if MAVLink and connection info are available."""
    try:
        from drone_autopilot import FireFlyAutopilot, Waypoint  # type: ignore
        MAV_OK = True
    except Exception:
        MAV_OK = False
    if not MAV_OK:
        return
    assert _direct_queue is not None
    while True:
        item = await _direct_queue.get()
        topic = item.get("topic")
        payload = item.get("payload", {})
        asset_id = None
        try:
            # topic format tasks/{asset_id}/dispatch
            if topic and topic.startswith("tasks/"):
                asset_id = topic.split("/")[1]
            asset_id = payload.get("asset_id") or asset_id
            waypoints = payload.get("waypoints") or (payload.get("plan") or {}).get("waypoints") or []
            if not asset_id or not waypoints:
                continue
            # Ensure we have/establish autopilot
            if asset_id not in _autopilots:
                # lookup mavlink_conn from assets table
                assert SessionLocal is not None
                async with SessionLocal() as session:
                    res = await session.execute(
                        text("SELECT capabilities FROM assets WHERE asset_id = :aid"), {"aid": asset_id}
                    )
                    row = res.first()
                    conn_str = None
                    if row and row.capabilities and isinstance(row.capabilities, dict):
                        conn_str = row.capabilities.get("mavlink_conn")
                if not conn_str:
                    continue
                ap = FireFlyAutopilot(device_id=asset_id, connection_string=conn_str)
                ok = await ap.connect()
                if not ok:
                    continue
                _autopilots[asset_id] = ap
            ap = _autopilots[asset_id]
            # Build waypoint objects
            wps = []
            for wp in waypoints:
                try:
                    wps.append(
                        Waypoint(
                            lat=float(wp["lat"]),
                            lon=float(wp["lon"]),
                            alt=float(wp.get("alt") or wp.get("altitude") or 50.0),
                            speed=float(wp.get("speed") or 5.0),
                            action=str(wp.get("action") or "WAYPOINT"),
                            params=None,
                        )
                    )
                except Exception:
                    continue
            if not wps:
                continue
            ok = await ap.set_mission(wps)
            if ok:
                await ap.start_mission()
        except Exception:
            # swallow and continue
            pass


async def _validate_policies(req: MissionCreateRequest) -> List[str]:
    # TODO: integrate with real policy engines (airspace, geofence, weather, NOTAMs, operator gates)
    # For now, always OK
    return []


async def _plan_assignments(req: MissionCreateRequest, available_assets: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Planner: 'loiter' and 'grid' patterns, polygon support, heading rotation, deconfliction, and per-asset constraints."""
    num = req.num_drones or max(1, min(1, len(available_assets)))
    chosen = available_assets[:num]
    plans: Dict[str, Any] = {}

    p = req.planning_params or {}
    altitude_default = float(p.get("altitude", 60))
    speed_default = float(p.get("speed", 5.0))
    pattern = str(p.get("pattern", "loiter")).lower()
    spacing = float(p.get("grid_spacing_m", 75.0))
    heading = float(p.get("heading_deg", 0.0))
    min_sep_m = float(p.get("min_sep_m", 0.0))
    alt_offset_step = float(p.get("altitude_offset_step_m", 10.0))
    start_delay_step = float(p.get("start_delay_step_s", 2.0))

    # Enforce min separation by spacing
    if min_sep_m > spacing:
        spacing = min_sep_m

    # Helper conversions around a latitude
    from math import cos, radians, sin, cos as mcos, sin as msin

    def meters_to_deg_lat(m: float) -> float:
        return m / 111_111.0

    def meters_to_deg_lon(m: float, lat_deg: float) -> float:
        return m / (111_111.0 * max(0.1, cos(radians(lat_deg))))

    def deg_lon_per_meter(lat_deg: float) -> float:
        return 1.0 / (111_111.0 * max(0.1, cos(radians(lat_deg))))

    def point_in_polygon(lat: float, lon: float, poly: List[List[float]]) -> bool:
        # Ray casting algorithm (lat,lon order in poly)
        x = lon
        y = lat
        inside = False
        n = len(poly)
        for i in range(n):
            x1, y1 = poly[i][1], poly[i][0]
            x2, y2 = poly[(i + 1) % n][1], poly[(i + 1) % n][0]
            intersect = ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-9) + x1)
            if intersect:
                inside = not inside
        return inside

    def _segment_intersections(p1: Dict[str, float], p2: Dict[str, float], poly: List[List[float]]) -> List[float]:
        # Return t values (0..1) where segment p1->p2 crosses polygon edges
        ts: List[float] = []
        x1, y1 = p1["lon"], p1["lat"]
        x2, y2 = p2["lon"], p2["lat"]
        dx = x2 - x1
        dy = y2 - y1
        n = len(poly)
        for i in range(n):
            ex1, ey1 = poly[i][1], poly[i][0]
            ex2, ey2 = poly[(i + 1) % n][1], poly[(i + 1) % n][0]
            edx = ex2 - ex1
            edy = ey2 - ey1
            denom = dx * edy - dy * edx
            if abs(denom) < 1e-12:
                continue
            t = ((x1 - ex1) * edy - (y1 - ey1) * edx) / denom
            u = ((x1 - ex1) * dy - (y1 - ey1) * dx) / denom
            if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
                ts.append(max(0.0, min(1.0, t)))
        ts = sorted(set(ts))
        return ts

    def _clip_segment_to_polygon(p1: Dict[str, float], p2: Dict[str, float], poly: List[List[float]]) -> List[Dict[str, float]]:
        # Return list of waypoints representing inside intervals (pairs of points) after clipping
        if not poly or point_in_polygon(p1["lat"], p1["lon"], poly) and point_in_polygon(p2["lat"], p2["lon"], poly):
            return [p1, p2]
        ts = [0.0] + _segment_intersections(p1, p2, poly) + [1.0]
        inside = point_in_polygon(p1["lat"], p1["lon"], poly)
        result: List[Dict[str, float]] = []
        for i in range(len(ts) - 1):
            t0, t1 = ts[i], ts[i + 1]
            if inside:
                # keep this interval
                q0 = {"lat": p1["lat"] + (p2["lat"] - p1["lat"]) * t0, "lon": p1["lon"] + (p2["lon"] - p1["lon"]) * t0}
                q1 = {"lat": p1["lat"] + (p2["lat"] - p1["lat"]) * t1, "lon": p1["lon"] + (p2["lon"] - p1["lon"]) * t1}
                if not result or (result and (result[-1]["lat"] != q0["lat"] or result[-1]["lon"] != q0["lon"])):
                    result.append(q0)
                result.append(q1)
            inside = not inside
        return result

    # Determine area
    center = {"lat": 0.0, "lon": 0.0}
    radius = 100.0
    polygon = None
    if req.area:
        if "center" in req.area:
            center = req.area["center"]
        if "radius_m" in req.area:
            radius = float(req.area["radius_m"])
        if "polygon" in req.area and isinstance(req.area["polygon"], list) and req.area["polygon"]:
            polygon = req.area["polygon"]  # [[lat, lon], ...]

    # Extract per-asset constraints for clamping (min/max altitude/speed)
    asset_constraints: Dict[str, Dict[str, Any]] = {}
    for a in chosen:
        c = a.get("constraints") or {}
        asset_constraints[a["asset_id"]] = c if isinstance(c, dict) else {}

    # Optional OR-Tools lane assignment
    assigner = str(p.get("assigner", "round_robin")).lower()

    if pattern == "grid":
        # Work in local meters frame (x east, y north), then rotate by heading and convert to lat/lon
        lat0 = center["lat"]
        lon0 = center["lon"]
        # Determine bounds in meters
        if polygon:
            # Compute bbox of polygon
            lats = [pt[0] for pt in polygon]
            lons = [pt[1] for pt in polygon]
            min_lat = min(lats)
            max_lat = max(lats)
            min_lon = min(lons)
            max_lon = max(lons)
        else:
            # Square bounds from center/radius
            min_lat = lat0 - meters_to_deg_lat(radius)
            max_lat = lat0 + meters_to_deg_lat(radius)
            min_lon = lon0 - meters_to_deg_lon(radius, lat0)
            max_lon = lon0 + meters_to_deg_lon(radius, lat0)
        # Convert bounds to local meter offsets relative to center
        half_height_m = abs((max_lat - min_lat) / 2.0) * 111_111.0
        half_width_m = abs((max_lon - min_lon) / 2.0) / deg_lon_per_meter(lat0)

        # Build lane centerlines across height in meters
        total_height_m = 2.0 * half_height_m
        lanes = max(1, int(total_height_m // spacing) + 1)
        # For numeric stability, recompute spacing from lanes
        if lanes > 1:
            spacing_m_effective = total_height_m / (lanes - 1)
        else:
            spacing_m_effective = total_height_m

        # Generate lanes at y positions; endpoints from -half_width_m to +half_width_m
        # Apply rotation by heading (degrees) around origin
        th = radians(heading)
        cos_th = mcos(th)
        sin_th = msin(th)

        # Build lane endpoints sequence per lane
        lane_endpoints: List[List[Dict[str, float]]] = []
        for lane_idx in range(lanes):
            # y coordinate (north) for this lane before rotation
            y = -half_height_m + lane_idx * spacing_m_effective
            # Two endpoints along x east
            x1, x2 = -half_width_m, half_width_m
            # Rotate both points
            rx1 = x1 * cos_th - y * sin_th
            ry1 = x1 * sin_th + y * cos_th
            rx2 = x2 * cos_th - y * sin_th
            ry2 = x2 * sin_th + y * cos_th
            # Convert to lat/lon
            p1_lat = lat0 + meters_to_deg_lat(ry1)
            p1_lon = lon0 + meters_to_deg_lon(rx1, lat0)
            p2_lat = lat0 + meters_to_deg_lat(ry2)
            p2_lon = lon0 + meters_to_deg_lon(rx2, lat0)
            # Alternate direction for lawnmower effect
            if lane_idx % 2 == 1:
                p1_lat, p1_lon, p2_lat, p2_lon = p2_lat, p2_lon, p1_lat, p1_lon
            # Clip segment to polygon if provided
            clipped_points: List[Dict[str, float]]
            if polygon is not None:
                clipped_points = _clip_segment_to_polygon(
                    {"lat": p1_lat, "lon": p1_lon},
                    {"lat": p2_lat, "lon": p2_lon},
                    polygon,
                )
            else:
                clipped_points = [{"lat": p1_lat, "lon": p1_lon}, {"lat": p2_lat, "lon": p2_lon}]
            # Build waypoints from clipped points in order
            waypoints_seq: List[Dict[str, Any]] = []
            for qp in clipped_points:
                waypoints_seq.append({"lat": qp["lat"], "lon": qp["lon"], "alt": altitude_default, "speed": speed_default, "action": "WAYPOINT"})
            if not waypoints_seq:
                continue
            lane_endpoints.append(waypoints_seq)

        # Assign lanes to assets
        asset_lane_wpts: Dict[str, List[Dict[str, Any]]] = {a["asset_id"]: [] for a in chosen}
        if assigner == "ortools":
            try:
                from ortools.graph.python import min_cost_flow  # type: ignore
                # Simple assignment as min-cost flow: lanes -> assets with unit demand
                n_lanes = len(lane_endpoints)
                n_assets = len(chosen)
                # Create a bipartite graph with source->lanes, lanes->assets, assets->sink
                mcf = min_cost_flow.SimpleMinCostFlow()
                source = n_lanes + n_assets
                sink = source + 1
                # Node indexing: lanes [0..n_lanes-1], assets [n_lanes..n_lanes+n_assets-1]
                def asset_node(i: int) -> int:
                    return n_lanes + i
                # Add arcs from source to lanes
                for i in range(n_lanes):
                    mcf.add_arc_with_capacity_and_unit_cost(source, i, 1, 0)
                # Add arcs from assets to sink (capacity large)
                for j in range(n_assets):
                    mcf.add_arc_with_capacity_and_unit_cost(asset_node(j), sink, n_lanes, 0)
                # Costs: distance from lane midpoint to asset nominal position (use index distance)
                for i in range(n_lanes):
                    # Midpoint of lane i
                    mid = lane_endpoints[i][len(lane_endpoints[i]) // 2]
                    for j in range(n_assets):
                        # crude cost: prefer balancing by index distance, could use distance to asset last known pos
                        cost = abs(j - (i % n_assets))
                        mcf.add_arc_with_capacity_and_unit_cost(i, asset_node(j), 1, int(cost))
                # Supplies
                supplies = [0] * (sink + 1)
                supplies[source] = n_lanes
                supplies[sink] = -n_lanes
                for node, supply in enumerate(supplies):
                    mcf.set_node_supply(node, supply)
                status = mcf.solve()
                if status == mcf.OPTIMAL:
                    for arc in range(mcf.num_arcs()):
                        if mcf.flow(arc) > 0:
                            tail = mcf.tail(arc)
                            head = mcf.head(arc)
                            # lane to asset arcs fall in lanes->assets range (ignore source/ sink arcs)
                            if 0 <= tail < n_lanes and n_lanes <= head < n_lanes + n_assets:
                                asset_idx = head - n_lanes
                                asset_id = chosen[asset_idx]["asset_id"]
                                asset_lane_wpts[asset_id].extend(lane_endpoints[tail])
                else:
                    # fallback to round robin
                    for i, lane in enumerate(lane_endpoints):
                        asset_id = chosen[i % len(chosen)]["asset_id"]
                        asset_lane_wpts[asset_id].extend(lane)
            except Exception:
                # OR-Tools not available or error
                for i, lane in enumerate(lane_endpoints):
                    asset_id = chosen[i % len(chosen)]["asset_id"]
                    asset_lane_wpts[asset_id].extend(lane)
        else:
            # Round-robin
            for i, lane in enumerate(lane_endpoints):
                asset_id = chosen[i % len(chosen)]["asset_id"]
                asset_lane_wpts[asset_id].extend(lane)

        # Build per-asset plans with constraints & offsets
        for idx, a in enumerate(chosen):
            aid = a["asset_id"]
            c = asset_constraints.get(aid, {})
            # Clamp altitude and speed
            min_alt = float(c.get("min_altitude", -1e9))
            max_alt = float(c.get("max_altitude", 1e9))
            min_spd = float(c.get("min_speed", 0.0))
            max_spd = float(c.get("max_speed", 1000.0))
            alt = max(min_alt, min(max_alt, altitude_default + idx * alt_offset_step))
            spd = max(min_spd, min(max_spd, speed_default))
            # Apply clamped values to waypoints
            wps = []
            prev = None
            dist_total = 0.0
            max_time = float(c.get("max_flight_time_s", 1e12))
            for wp in asset_lane_wpts.get(aid, []):
                # distance estimate (flat earth approx)
                if prev is not None:
                    dy_m = (wp["lat"] - prev["lat"]) * 111_111.0
                    dx_m = (wp["lon"] - prev["lon"]) / deg_lon_per_meter(center["lat"])  # invert per-meter
                    seg = (dy_m**2 + dx_m**2) ** 0.5
                    if (dist_total + seg) / max(0.1, spd) > max_time:
                        break
                    dist_total += seg
                wps.append({"lat": wp["lat"], "lon": wp["lon"], "alt": alt, "speed": spd, "action": wp.get("action", "WAYPOINT")})
                prev = wp
            plans[aid] = {
                "pattern": "grid",
                "altitude": alt,
                "speed": spd,
                "grid": {
                    "spacing_m": spacing_m_effective,
                    "heading_deg": heading,
                    "bounds": {"center": center, "half_width_m": half_width_m, "half_height_m": half_height_m},
                },
                "start_delay_sec": round(idx * start_delay_step, 2),
                "waypoints": wps,
            }
        return plans

    if pattern == "spiral":
        # Generate a simple outward spiral of waypoints
        turns = max(3, int((radius or 100.0) / max(20.0, spacing)))
        spiral_wps: List[Dict[str, Any]] = []
        lat0, lon0 = center["lat"], center["lon"]
        for i in range(turns * 20):
            r_m = (i / (turns * 20)) * max(radius, 100.0)
            ang = i * (6.28318 / 20.0)
            lat = lat0 + meters_to_deg_lat(r_m * msin(ang))
            lon = lon0 + meters_to_deg_lon(r_m * mcos(ang), lat0)
            if (polygon is None) or point_in_polygon(lat, lon, polygon):
                spiral_wps.append({"lat": lat, "lon": lon})
        # Assign same path to all, but offset start delay/alt
        for idx, a in enumerate(chosen):
            c = asset_constraints.get(a["asset_id"], {})
            min_alt = float(c.get("min_altitude", -1e9))
            max_alt = float(c.get("max_altitude", 1e9))
            min_spd = float(c.get("min_speed", 0.0))
            max_spd = float(c.get("max_speed", 1000.0))
            alt = max(min_alt, min(max_alt, altitude_default + idx * alt_offset_step))
            spd = max(min_spd, min(max_spd, speed_default))
            wps = [
                {"lat": wp["lat"], "lon": wp["lon"], "alt": alt, "speed": spd, "action": "WAYPOINT"}
                for wp in spiral_wps
            ]
            plans[a["asset_id"]] = {
                "pattern": "spiral",
                "altitude": alt,
                "speed": spd,
                "start_delay_sec": round(idx * start_delay_step, 2),
                "waypoints": wps,
            }
        return plans

    if pattern == "expanding_square":
        # Build outward square loop waypoints around center
        lat0, lon0 = center["lat"], center["lon"]
        side = max(spacing * 2, 100.0)
        loops = max(1, int((radius or 100.0) // (side / 2)))
        sq_wps: List[Dict[str, Any]] = []
        for k in range(1, loops + 1):
            half = (side * k) / 2.0
            corners = [
                (lat0 + meters_to_deg_lat(+half), lon0 + meters_to_deg_lon(+half, lat0)),
                (lat0 + meters_to_deg_lat(+half), lon0 + meters_to_deg_lon(-half, lat0)),
                (lat0 + meters_to_deg_lat(-half), lon0 + meters_to_deg_lon(-half, lat0)),
                (lat0 + meters_to_deg_lat(-half), lon0 + meters_to_deg_lon(+half, lat0)),
                (lat0 + meters_to_deg_lat(+half), lon0 + meters_to_deg_lon(+half, lat0)),
            ]
            for lat, lon in corners:
                if (polygon is None) or point_in_polygon(lat, lon, polygon):
                    sq_wps.append({"lat": lat, "lon": lon})
        for idx, a in enumerate(chosen):
            c = asset_constraints.get(a["asset_id"], {})
            min_alt = float(c.get("min_altitude", -1e9))
            max_alt = float(c.get("max_altitude", 1e9))
            min_spd = float(c.get("min_speed", 0.0))
            max_spd = float(c.get("max_speed", 1000.0))
            alt = max(min_alt, min(max_alt, altitude_default + idx * alt_offset_step))
            spd = max(min_spd, min(max_spd, speed_default))
            wps = [
                {"lat": wp["lat"], "lon": wp["lon"], "alt": alt, "speed": spd, "action": "WAYPOINT"}
                for wp in sq_wps
            ]
            plans[a["asset_id"]] = {
                "pattern": "expanding_square",
                "altitude": alt,
                "speed": spd,
                "start_delay_sec": round(idx * start_delay_step, 2),
                "waypoints": wps,
            }
        return plans

    # Default: loiter (previous behavior)
    for idx, a in enumerate(chosen):
        # offset circle for each asset around center at radius
        angle = (idx / max(1, len(chosen))) * 6.28318
        lat = center["lat"] + (radius / 111111.0) * float(__import__("math").sin(angle))
        lon = center["lon"] + (radius / 111111.0) * float(__import__("math").cos(angle))
        # Clamp per-asset
        c = asset_constraints.get(a["asset_id"], {})
        min_alt = float(c.get("min_altitude", -1e9))
        max_alt = float(c.get("max_altitude", 1e9))
        min_spd = float(c.get("min_speed", 0.0))
        max_spd = float(c.get("max_speed", 1000.0))
        alt = max(min_alt, min(max_alt, altitude_default + idx * alt_offset_step))
        spd = max(min_spd, min(max_spd, speed_default))
        plan = {
            "pattern": "loiter",
            "altitude": alt,
            "speed": spd,
            "start_delay_sec": round(idx * start_delay_step, 2),
            "waypoints": [
                {"lat": lat, "lon": lon, "alt": alt, "speed": spd, "action": "WAYPOINT"}
            ],
        }
        plans[a["asset_id"]] = plan
    return plans


# -------------------------
# Endpoints
# -------------------------
@app.get("/health")
async def health():
    return {"status": "ok", "service": "tasking"}


# Metrics endpoint
if PROM_AVAILABLE:
    @app.get("/metrics")
    async def metrics():
        return __import__("fastapi").responses.Response(
            content=generate_latest(), media_type=CONTENT_TYPE_LATEST
        )


# Legacy dispatch endpoints (kept)
@app.post("/dispatch")
async def dispatch_task(req: TaskDispatchRequest, request: Request):
    await _require_auth(request)
    assert SessionLocal is not None
    assert mqtt_client is not None

    created_at = datetime.now(timezone.utc)

    # Store task in DB
    async with SessionLocal() as session:
        await session.execute(
            tasks.insert().values(
                task_id=req.task_id,
                asset_id=req.asset_id,
                action=req.action,
                status="ACTIVE",
                created_at=created_at,
                started_at=created_at,
            )
        )
        await session.commit()

    # Publish task to MQTT for edge agent to consume
    task_message = {
        "task_id": req.task_id,
        "action": req.action,
        "waypoints": req.waypoints,
        "ts_iso": created_at.isoformat(),
    }

    topic = f"tasks/{req.asset_id}/dispatch"
    mqtt_client.publish(topic, json.dumps(task_message), qos=1)

    return {
        "task_id": req.task_id,
        "status": "ACTIVE",
        "message": f"Task dispatched to {req.asset_id}",
    }


@app.get("/tasks/{task_id}", response_model=Task)
async def get_task(task_id: str):
    """Get task status by ID."""
    assert SessionLocal is not None

    async with SessionLocal() as session:
        result = await session.execute(
            text(
                "SELECT task_id, asset_id, action, status, created_at, started_at, completed_at FROM tasks WHERE task_id = :tid"
            ),
            {"tid": task_id},
        )
        row = result.first()

        if not row:
            raise HTTPException(status_code=404, detail="Task not found")

        return Task(**dict(row._mapping))


@app.get("/tasks", response_model=List[Task])
async def list_tasks(
    asset_id: Optional[str] = None, status: Optional[str] = None, limit: int = 50
):
    """List tasks, optionally filtered by asset_id or status."""
    assert SessionLocal is not None

    query = (
        "SELECT task_id, asset_id, action, status, created_at, started_at, completed_at FROM tasks"
    )
    params = {"lim": limit}
    where_clauses = []

    if asset_id:
        where_clauses.append("asset_id = :aid")
        params["aid"] = asset_id

    if status:
        where_clauses.append("status = :st")
        params["st"] = status

    if where_clauses:
        query += " WHERE " + " AND ".join(where_clauses)

    query += " ORDER BY id DESC LIMIT :lim"

    async with SessionLocal() as session:
        result = await session.execute(text(query), params)
        rows = result.all()

        return [Task(**dict(r._mapping)) for r in rows]


@app.post("/tasks/{task_id}/complete")
async def complete_task(task_id: str):
    """Mark a task as completed."""
    assert SessionLocal is not None

    async with SessionLocal() as session:
        await session.execute(
            tasks.update().where(tasks.c.task_id == task_id).values(
                status="COMPLETED", completed_at=datetime.now(timezone.utc)
            )
        )
        await session.commit()

    return {"task_id": task_id, "status": "COMPLETED"}


@app.post("/tasks/{task_id}/fail")
async def fail_task(task_id: str, reason: str = "Unknown error"):
    """Mark a task as failed."""
    assert SessionLocal is not None

    async with SessionLocal() as session:
        await session.execute(
            tasks.update().where(tasks.c.task_id == task_id).values(
                status="FAILED", completed_at=datetime.now(timezone.utc)
            )
        )
        await session.commit()

    return {"task_id": task_id, "status": "FAILED", "reason": reason}


# -------------------------
# Asset Registry API
# -------------------------
@app.post("/api/v1/assets", response_model=AssetOut)
async def register_asset(asset: AssetIn, request: Request):
    await _require_auth(request)
    assert SessionLocal is not None
    now = datetime.now(timezone.utc)
    async with SessionLocal() as session:
        # Upsert-like behavior: try update, if 0 rows then insert
        result = await session.execute(
            assets.update()
            .where(assets.c.asset_id == asset.asset_id)
            .values(
                type=asset.type,
                capabilities=asset.capabilities,
                battery=asset.battery,
                link=asset.link,
                constraints=asset.constraints,
                updated_at=now,
            )
        )
        if result.rowcount == 0:
            await session.execute(
                assets.insert().values(
                    asset_id=asset.asset_id,
                    type=asset.type,
                    capabilities=asset.capabilities,
                    battery=asset.battery,
                    link=asset.link,
                    constraints=asset.constraints,
                    updated_at=now,
                )
            )
        await session.commit()

    if METRIC_ASSETS_REGISTERED:
        METRIC_ASSETS_REGISTERED.inc()

    return AssetOut(**asset.model_dump(), updated_at=now)


@app.get("/api/v1/assets", response_model=List[AssetOut])
async def list_assets():
    assert SessionLocal is not None
    async with SessionLocal() as session:
        result = await session.execute(
            text(
                "SELECT asset_id, type, capabilities, battery, link, constraints, updated_at FROM assets ORDER BY updated_at DESC NULLS LAST"
            )
        )
        rows = result.all()
        out: List[AssetOut] = []
        for r in rows:
            m = dict(r._mapping)
            out.append(
                AssetOut(
                    asset_id=m["asset_id"],
                    type=m.get("type"),
                    capabilities=m.get("capabilities"),
                    battery=m.get("battery"),
                    link=m.get("link"),
                    constraints=m.get("constraints"),
                    updated_at=m.get("updated_at"),
                )
            )
        return out


@app.get("/api/v1/assets/{asset_id}", response_model=AssetOut)
async def get_asset(asset_id: str):
    assert SessionLocal is not None
    async with SessionLocal() as session:
        result = await session.execute(
            text(
                "SELECT asset_id, type, capabilities, battery, link, constraints, updated_at FROM assets WHERE asset_id = :aid"
            ),
            {"aid": asset_id},
        )
        row = result.first()
        if not row:
            raise HTTPException(status_code=404, detail="Asset not found")
        m = dict(row._mapping)
        return AssetOut(
            asset_id=m["asset_id"],
            type=m.get("type"),
            capabilities=m.get("capabilities"),
            battery=m.get("battery"),
            link=m.get("link"),
            constraints=m.get("constraints"),
            updated_at=m.get("updated_at"),
        )


# -------------------------
# Missions API (authoritative)
# -------------------------
@app.post("/api/v1/missions", response_model=MissionResponse)
async def create_mission(req: MissionCreateRequest, request: Request):
    await _require_auth(request)
    assert SessionLocal is not None

    mission_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc)

    # Policy validation
    violations = await _validate_policies(req)
    policy_ok = len(violations) == 0
    if not policy_ok:
        raise HTTPException(status_code=400, detail={"policy_violations": violations})

    # Fetch available assets
    async with SessionLocal() as session:
        result = await session.execute(
            text(
                "SELECT asset_id, type, capabilities, battery, link, constraints FROM assets ORDER BY updated_at DESC NULLS LAST"
            )
        )
        assets_rows = [dict(r._mapping) for r in result.all()]

    # Simple availability filter: battery >= 30 and link == 'OK'
    available_assets = [
        a
        for a in assets_rows
        if (a.get("battery") or 0) >= 30 and (a.get("link") in ("OK", "GOOD", "CONNECTED", None))
    ]
    if not available_assets:
        raise HTTPException(status_code=409, detail="No available assets to plan mission")

    # Plan
    assignments_map = await _plan_assignments(req, available_assets)

    # Persist mission and assignments
    async with SessionLocal() as session:
        await session.execute(
            missions.insert().values(
                mission_id=mission_id,
                name=req.name,
                objectives=req.objectives,
                area=req.area,
                policy_ok=policy_ok,
                status="ACTIVE",
                created_at=created_at,
                started_at=created_at,
            )
        )
        for asset_id, plan in assignments_map.items():
            await session.execute(
                mission_assignments.insert().values(
                    mission_id=mission_id,
                    asset_id=asset_id,
                    plan=plan,
                    status="ASSIGNED",
                )
            )
        await session.commit()

    # Emit MQTT events and dispatch to each asset
    await _publish_mission_update(
        mission_id,
        {"event": "MISSION_CREATED", "name": req.name, "objectives": req.objectives},
    )

    if METRIC_MISSIONS_CREATED:
        METRIC_MISSIONS_CREATED.inc()
    if METRIC_MISSIONS_ACTIVE:
        METRIC_MISSIONS_ACTIVE.inc()

    # Dispatch plans to per-asset task topics
    if mqtt_client:
        for asset_id, plan in assignments_map.items():
            topic = f"tasks/{asset_id}/dispatch"
            message = {
                "task_id": f"mission:{mission_id}",
                "action": "MISSION_EXECUTE",
                "waypoints": plan.get("waypoints", []),
                "plan": plan,
                "ts_iso": datetime.now(timezone.utc).isoformat(),
            }
            mqtt_client.publish(topic, json.dumps(message), qos=1)
            # Emit assignment update
            await _publish_mission_update(
                mission_id,
                {"event": "ASSIGNED", "asset_id": asset_id, "plan": plan},
            )

    # Build response
    assignments = [
        MissionAssignment(asset_id=k, plan=v, status="ASSIGNED")
        for k, v in assignments_map.items()
    ]
    return MissionResponse(
        mission_id=mission_id,
        name=req.name,
        objectives=req.objectives,
        status="ACTIVE",
        policy_ok=policy_ok,
        assignments=assignments,
        created_at=created_at,
        started_at=created_at,
    )


@app.get("/api/v1/missions/{mission_id}")
async def get_mission(mission_id: str):
    assert SessionLocal is not None
    async with SessionLocal() as session:
        res = await session.execute(
            text(
                "SELECT mission_id, name, objectives, area, policy_ok, status, created_at, started_at, completed_at FROM missions WHERE mission_id = :mid"
            ),
            {"mid": mission_id},
        )
        mrow = res.first()
        if not mrow:
            raise HTTPException(status_code=404, detail="Mission not found")
        ares = await session.execute(
            text(
                "SELECT asset_id, plan, status FROM mission_assignments WHERE mission_id = :mid"
            ),
            {"mid": mission_id},
        )
        assignments = [
            MissionAssignment(
                asset_id=r.asset_id, plan=r.plan or {}, status=r.status
            )
            for r in ares.all()
        ]
        m = dict(mrow._mapping)
        return {
            "mission_id": m["mission_id"],
            "name": m.get("name"),
            "objectives": m.get("objectives") or [],
            "status": m.get("status"),
            "policy_ok": bool(m.get("policy_ok")),
            "assignments": [a.model_dump() for a in assignments],
            "created_at": (m.get("created_at") or datetime.now(timezone.utc)).isoformat(),
            "started_at": (m.get("started_at") or datetime.now(timezone.utc)).isoformat() if m.get("started_at") else None,
            "completed_at": m.get("completed_at").isoformat() if m.get("completed_at") else None,
        }


@app.get("/api/v1/missions")
async def list_missions(limit: int = 50, status: Optional[str] = None):
    assert SessionLocal is not None
    query = (
        "SELECT mission_id, name, objectives, status, created_at, started_at, completed_at FROM missions"
    )
    params: Dict[str, Any] = {"lim": limit}
    if status:
        query += " WHERE status = :st"
        params["st"] = status
    query += " ORDER BY id DESC LIMIT :lim"
    async with SessionLocal() as session:
        res = await session.execute(text(query), params)
        rows = res.all()
        return [
            {
                "mission_id": r.mission_id,
                "name": r.name,
                "objectives": r.objectives or [],
                "status": r.status,
                "created_at": r.created_at.isoformat() if r.created_at else None,
                "started_at": r.started_at.isoformat() if r.started_at else None,
                "completed_at": r.completed_at.isoformat() if r.completed_at else None,
            }
            for r in rows
        ]
