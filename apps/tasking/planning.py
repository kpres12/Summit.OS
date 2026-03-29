"""Planning, assignment, and autopilot logic for the tasking service."""
import json
import logging
import os
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

import state

logger = logging.getLogger("tasking")


async def _init_direct_autopilot():
    """If enabled, subscribe to tasks/+/dispatch and handle via FireFlyAutopilot when possible."""
    import asyncio

    global _direct_queue
    assert state.mqtt_client is not None
    state._direct_queue = asyncio.Queue()

    def _on_message(_client, _userdata, msg):
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
        except Exception:
            payload = {"raw": msg.payload.decode("utf-8", errors="ignore")}
        # thread-safe put into asyncio queue
        loop = asyncio.get_event_loop()
        loop.call_soon_threadsafe(
            state._direct_queue.put_nowait, {"topic": msg.topic, "payload": payload}
        )

    state.mqtt_client.on_message = _on_message
    state.mqtt_client.subscribe("tasks/+/dispatch", qos=1)
    asyncio.create_task(_direct_autopilot_worker())


async def _direct_autopilot_worker():
    """Consume dispatches and drive autopilot directly if MAVLink and connection info are available."""
    from sqlalchemy import text

    try:
        from drone_autopilot import FireFlyAutopilot, Waypoint  # type: ignore

        MAV_OK = True
    except Exception:
        MAV_OK = False
    if not MAV_OK:
        return
    assert state._direct_queue is not None
    while True:
        item = await state._direct_queue.get()
        topic = item.get("topic")
        payload = item.get("payload", {})
        asset_id = None
        try:
            # topic format tasks/{asset_id}/dispatch
            if topic and topic.startswith("tasks/"):
                asset_id = topic.split("/")[1]
            asset_id = payload.get("asset_id") or asset_id
            waypoints = (
                payload.get("waypoints")
                or (payload.get("plan") or {}).get("waypoints")
                or []
            )
            if not asset_id or not waypoints:
                continue
            # Ensure we have/establish autopilot
            if asset_id not in state._autopilots:
                # lookup mavlink_conn from assets table
                assert state.SessionLocal is not None
                async with state.SessionLocal() as session:
                    res = await session.execute(
                        text("SELECT capabilities FROM assets WHERE asset_id = :aid"),
                        {"aid": asset_id},
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
                state._autopilots[asset_id] = ap
            ap = state._autopilots[asset_id]
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
                            params=(
                                wp.get("params")
                                if isinstance(wp.get("params"), dict)
                                else None
                            ),
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


async def _validate_policies(
    req, org_id: Optional[str] = None
) -> List[str]:
    """Validate mission against policy engine (OPA) and return violation reasons.

    Fails open (no violations) if OPA is unreachable to avoid blocking local dev.
    """
    try:
        from apps.tasking.opa import OPAClient
    except Exception:
        # If client not available, allow
        return []

    opa = OPAClient()
    # Build minimal input; extend as schemas mature
    input_data = {
        "mission": req.model_dump(),
        "org_id": org_id or "dev",
        "context": {
            "time": datetime.now(timezone.utc).isoformat(),
        },
    }
    try:
        result = await opa.evaluate("missions/allow", input_data)
        allow = bool(result.get("allow", True))
        if allow:
            return []
        # Collect reasons if provided
        reasons = result.get("deny_reasons") or result.get("reasons") or []
        if isinstance(reasons, list):
            return [str(r) for r in reasons]
        return [str(reasons)] if reasons else ["Policy denied"]
    except Exception:
        # Fail open
        return []


async def _assess_threat(
    location: Dict[str, float],
    verification_data: Optional[Dict[str, Any]] = None,
    domain: str = "generic",
):
    """Assess threat level using pluggable domain-specific assessors."""
    from packages.threat_assessment import threat_registry

    # Get appropriate threat assessor for domain
    assessor = threat_registry.get_assessor(domain)

    # Default sensor data if none provided
    sensor_data = verification_data or {
        "confidence": 0.5,
        "intensity": 0.0,
        "size": 0.0,
        "spread_rate": 0.0,
    }

    # Run threat assessment
    result = await assessor.assess_threat(
        location=location,
        sensor_data=sensor_data,
        environmental_data=(
            verification_data.get("environmental_data") if verification_data else None
        ),
    )

    return result


async def _select_tiered_assets(
    location: Dict[str, float],
    tier,
    available_assets: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Select appropriate assets for tiered mission based on tier and location."""
    from packages.schemas.drones import MissionTier

    selected = []

    if tier == MissionTier.TIER_1_VERIFY:
        # Select fastest Scout drone for reconnaissance
        scouts = [
            a
            for a in available_assets
            if a.get("capabilities", {}).get("drone_type") == "scout"
        ]
        if scouts:
            # Sort by dash speed, take fastest
            scouts.sort(
                key=lambda x: x.get("capabilities", {}).get("dash_speed", 0),
                reverse=True,
            )
            selected.append(scouts[0])

    elif tier == MissionTier.TIER_2_SUPPRESS:
        # Select Interceptor drone with intervention payload
        interceptors = [
            a
            for a in available_assets
            if a.get("capabilities", {}).get("drone_type") == "interceptor"
            and a.get("capabilities", {}).get("payload_capacity", 0) > 5
        ]
        if interceptors:
            # Sort by payload capacity, take largest
            interceptors.sort(
                key=lambda x: x.get("capabilities", {}).get("payload_capacity", 0),
                reverse=True,
            )
            selected.append(interceptors[0])

    elif tier == MissionTier.TIER_3_CONTAIN:
        # Select multiple drones for containment pattern
        containment_drones = [
            a
            for a in available_assets
            if a.get("capabilities", {}).get("drone_type")
            in ["interceptor", "scout", "relay"]
        ]
        # Take up to 4 drones for containment ring
        selected = containment_drones[:4]

    return selected


async def _plan_tiered_mission(
    req, assets: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Plan tiered mission with escalation logic."""
    from packages.schemas.drones import MissionTier

    plans = {}

    # Start with Tier 1 verification
    tier_1_assets = await _select_tiered_assets(
        req.initial_location, MissionTier.TIER_1_VERIFY, assets
    )

    for asset in tier_1_assets:
        asset_id = asset["asset_id"]
        capabilities = asset.get("capabilities", {})

        # Create verification mission plan
        plans[asset_id] = {
            "tier": "tier_1_verify",
            "role": "verification",
            "target_location": req.initial_location,
            "altitude": capabilities.get("operating_altitude", {}).get("optimal", 60),
            "speed": capabilities.get("dash_speed", capabilities.get("max_speed", 50)),
            "pattern": "approach_and_hover",
            "sensors_active": ["thermal", "rgb", "gas"],
            "verification_duration": 120,  # 2 minutes
            "waypoints": [
                {
                    "lat": req.initial_location["lat"],
                    "lon": req.initial_location["lon"],
                    "alt": capabilities.get("operating_altitude", {}).get(
                        "optimal", 60
                    ),
                    "speed": capabilities.get("dash_speed", 100),
                    "action": "VERIFY_TARGET",
                }
            ],
        }

    return plans


async def _create_containment_pattern(
    center: Dict[str, float], assets: List[Dict[str, Any]], radius: float = 100.0
) -> Dict[str, Dict[str, Any]]:
    """Create containment ring pattern for multiple assets."""
    import math

    plans = {}
    num_assets = len(assets)

    if num_assets == 0:
        return plans

    # Distribute assets in a ring around the target
    for i, asset in enumerate(assets):
        angle = (2 * math.pi * i) / num_assets

        # Calculate position on the ring
        lat_offset = (radius / 111111.0) * math.sin(angle)  # degrees lat per meter
        lon_offset = (
            radius / (111111.0 * math.cos(math.radians(center["lat"])))
        ) * math.cos(angle)

        position_lat = center["lat"] + lat_offset
        position_lon = center["lon"] + lon_offset

        asset_id = asset["asset_id"]
        capabilities = asset.get("capabilities", {})

        plans[asset_id] = {
            "tier": "tier_3_contain",
            "role": "containment",
            "formation_position": i,
            "containment_radius": radius,
            "target_location": center,
            "patrol_position": {"lat": position_lat, "lon": position_lon},
            "altitude": capabilities.get("operating_altitude", {}).get("optimal", 60),
            "pattern": "orbit",
            "orbit_radius": 25.0,  # orbit around assigned position
            "coordination": {
                "formation": "containment_ring",
                "separation_distance": radius
                * 2
                * math.pi
                / num_assets,  # arc length between drones
                "anti_collision": True,
            },
            "waypoints": [
                {
                    "lat": position_lat,
                    "lon": position_lon,
                    "alt": capabilities.get("operating_altitude", {}).get(
                        "optimal", 60
                    ),
                    "speed": capabilities.get("max_speed", 30),
                    "action": "ORBIT",
                }
            ],
        }

    return plans


async def _plan_assignments(
    req, available_assets: List[Dict[str, Any]]
) -> Dict[str, Any]:
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
            intersect = ((y1 > y) != (y2 > y)) and (
                x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-9) + x1
            )
            if intersect:
                inside = not inside
        return inside

    def _segment_intersections(
        p1: Dict[str, float], p2: Dict[str, float], poly: List[List[float]]
    ) -> List[float]:
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

    def _clip_segment_to_polygon(
        p1: Dict[str, float], p2: Dict[str, float], poly: List[List[float]]
    ) -> List[Dict[str, float]]:
        # Return list of waypoints representing inside intervals (pairs of points) after clipping
        if (
            not poly
            or point_in_polygon(p1["lat"], p1["lon"], poly)
            and point_in_polygon(p2["lat"], p2["lon"], poly)
        ):
            return [p1, p2]
        ts = [0.0] + _segment_intersections(p1, p2, poly) + [1.0]
        inside = point_in_polygon(p1["lat"], p1["lon"], poly)
        result: List[Dict[str, float]] = []
        for i in range(len(ts) - 1):
            t0, t1 = ts[i], ts[i + 1]
            if inside:
                # keep this interval
                q0 = {
                    "lat": p1["lat"] + (p2["lat"] - p1["lat"]) * t0,
                    "lon": p1["lon"] + (p2["lon"] - p1["lon"]) * t0,
                }
                q1 = {
                    "lat": p1["lat"] + (p2["lat"] - p1["lat"]) * t1,
                    "lon": p1["lon"] + (p2["lon"] - p1["lon"]) * t1,
                }
                if not result or (
                    result
                    and (
                        result[-1]["lat"] != q0["lat"] or result[-1]["lon"] != q0["lon"]
                    )
                ):
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
        if (
            "polygon" in req.area
            and isinstance(req.area["polygon"], list)
            and req.area["polygon"]
        ):
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
                clipped_points = [
                    {"lat": p1_lat, "lon": p1_lon},
                    {"lat": p2_lat, "lon": p2_lon},
                ]
            # Build waypoints from clipped points in order
            waypoints_seq: List[Dict[str, Any]] = []
            for qp in clipped_points:
                waypoints_seq.append(
                    {
                        "lat": qp["lat"],
                        "lon": qp["lon"],
                        "alt": altitude_default,
                        "speed": speed_default,
                        "action": "WAYPOINT",
                    }
                )
            if not waypoints_seq:
                continue
            lane_endpoints.append(waypoints_seq)

        # Assign lanes to assets
        asset_lane_wpts: Dict[str, List[Dict[str, Any]]] = {
            a["asset_id"]: [] for a in chosen
        }
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
                    mcf.add_arc_with_capacity_and_unit_cost(
                        asset_node(j), sink, n_lanes, 0
                    )
                # Costs: distance from lane midpoint to asset nominal position (use index distance)
                for i in range(n_lanes):
                    # Midpoint of lane i
                    mid = lane_endpoints[i][len(lane_endpoints[i]) // 2]
                    for j in range(n_assets):
                        # crude cost: prefer balancing by index distance, could use distance to asset last known pos
                        cost = abs(j - (i % n_assets))
                        mcf.add_arc_with_capacity_and_unit_cost(
                            i, asset_node(j), 1, int(cost)
                        )
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
                            if (
                                0 <= tail < n_lanes
                                and n_lanes <= head < n_lanes + n_assets
                            ):
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
                    dx_m = (wp["lon"] - prev["lon"]) / deg_lon_per_meter(
                        center["lat"]
                    )  # invert per-meter
                    seg = (dy_m**2 + dx_m**2) ** 0.5
                    if (dist_total + seg) / max(0.1, spd) > max_time:
                        break
                    dist_total += seg
                # DEM terrain-following: add ground elevation so alt is truly AGL (Gap 7)
                wp_alt = alt
                try:
                    import sys as _sys

                    _sys.path.insert(
                        0, os.path.join(os.path.dirname(__file__), "../..")
                    )
                    from packages.geo.dem import get_provider as _get_dem

                    terrain_m = _get_dem().get_elevation(wp["lat"], wp["lon"])
                    wp_alt = alt + terrain_m
                except Exception:
                    pass
                wps.append(
                    {
                        "lat": wp["lat"],
                        "lon": wp["lon"],
                        "alt": wp_alt,
                        "speed": spd,
                        "action": wp.get("action", "WAYPOINT"),
                    }
                )
                prev = wp
            plans[aid] = {
                "pattern": "grid",
                "altitude": alt,
                "speed": spd,
                "grid": {
                    "spacing_m": spacing_m_effective,
                    "heading_deg": heading,
                    "bounds": {
                        "center": center,
                        "half_width_m": half_width_m,
                        "half_height_m": half_height_m,
                    },
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
                {
                    "lat": wp["lat"],
                    "lon": wp["lon"],
                    "alt": alt,
                    "speed": spd,
                    "action": "WAYPOINT",
                }
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
                (
                    lat0 + meters_to_deg_lat(+half),
                    lon0 + meters_to_deg_lon(+half, lat0),
                ),
                (
                    lat0 + meters_to_deg_lat(+half),
                    lon0 + meters_to_deg_lon(-half, lat0),
                ),
                (
                    lat0 + meters_to_deg_lat(-half),
                    lon0 + meters_to_deg_lon(-half, lat0),
                ),
                (
                    lat0 + meters_to_deg_lat(-half),
                    lon0 + meters_to_deg_lon(+half, lat0),
                ),
                (
                    lat0 + meters_to_deg_lat(+half),
                    lon0 + meters_to_deg_lon(+half, lat0),
                ),
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
                {
                    "lat": wp["lat"],
                    "lon": wp["lon"],
                    "alt": alt,
                    "speed": spd,
                    "action": "WAYPOINT",
                }
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
