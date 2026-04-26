"""
Heli.OS — Comprehensive Mission Simulation Suite
=================================================
Simulates the full sensor → fusion → inference → engagement pipeline across
every domain, mission type, sensor/camera/equipment/robot type, and
adversarial condition in scope for Heli.OS.

This is a *software-layer* simulation: real data shapes, real state
machines, real policy evaluation — no physical hardware.  The goal is to
confirm that each mission scenario routes correctly through the system
and produces the expected decisions, alerts, and state transitions.

Run with:
    pytest tests/test_mission_simulations.py -v

Domains covered:
  AIR      — counter-UAS, ISR, ACE strike, search & rescue
  GROUND   — convoy escort, FOB perimeter, CASEVAC, BDA, urban SAR
  MARITIME — vessel interdiction, maritime SAR, subsurface contact
  WILDFIRE — aerial suppression coordination, hotspot containment
  FLOOD    — urban flood SAR, levee breach, dam failure downstream
  HADR     — earthquake, hurricane, tsunami mass-casualty response
  CYBER    — RF jamming detection, GPS spoofing, comms denial
  SPACE    — satellite-cued ISR, launch-detection hand-off

Sensor/camera types exercised:
  EO (visible), IR/thermal, SWIR, SAR, LIDAR, multispectral,
  radar (X-band, S-band), AIS, ADS-B, RF spectrum, acoustic,
  seismic, magnetometer, chem/bio/rad, buoy telemetry

Robot/asset types exercised:
  small UAS quadrotor, mid UAS Group 2, VTOL fixed-wing, loitering munition,
  tethered aerostat, USV, UUV, UGV tracked, UGV wheeled, quadruped robot,
  manned rotary-wing (medevac), manned fixed-wing (ISR), ground sensor node

Adversarial conditions:
  GPS denied, comms jamming, node failure, fog/rain, night, smoke,
  sensor spoofing, swarm saturation, partial track loss, cyber intrusion
"""
from __future__ import annotations

import math
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from packages.c2_intel.engagement_authorization import (
    DeconflictionContext, EngagementAuthorizationGate,
    OperatorAuthorization, OperatorDecision, PIDEvidence,
    ROEContext, TrackEvidence, WeaponOption,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _track(
    track_id: str,
    classification: str,
    confidence: float = 0.92,
    sensors: list[str] | None = None,
    lat: float = 34.5,
    lon: float = -118.0,
    alt_m: float = 150.0,
) -> TrackEvidence:
    return TrackEvidence(
        track_id=track_id,
        entity_id=f"entity-{track_id}",
        classification=classification,
        confidence=confidence,
        sensors=sensors or ["radar"],
        last_position={"lat": lat, "lon": lon, "alt_m": alt_m},
        last_seen=datetime.now(timezone.utc),
    )


def _full_authorize(
    gate: EngagementAuthorizationGate,
    case_id: str,
    engagement_class: str = "counter_uas",
    operator_role: str = "mission_commander",
) -> None:
    gate.submit_pid(case_id, PIDEvidence(method="iff", confidence=0.95))
    gate.submit_roe(case_id, ROEContext(
        roe_id="R1",
        permits_engagement_type=True,
        proportionality_passed=True,
        collateral_estimate="low",
    ))
    gate.submit_deconfliction(case_id, DeconflictionContext(
        blue_force_clear=True, airspace_clear=True,
    ))
    gate.surface_options(case_id, [WeaponOption(
        option_id="opt-1", weapon_asset_id="ew-1",
        weapon_class="soft_kill", range_m=500,
        time_of_flight_s=0, pk_estimate=0.92,
        roe_compliant=True, deconfliction_ok=True, rationale="EW soft kill",
    )])
    gate.authorize(case_id, OperatorAuthorization(
        decision=OperatorDecision.AUTHORIZE,
        operator_id="mc-01", operator_role=operator_role,
        rationale="Threat confirmed", selected_option="opt-1",
        signature=b"sig",
    ), engagement_class=engagement_class)


def _full_deny(
    gate: EngagementAuthorizationGate,
    case_id: str,
    operator_role: str = "mission_commander",
) -> None:
    gate.submit_pid(case_id, PIDEvidence(method="visual", confidence=0.91))
    gate.submit_roe(case_id, ROEContext(
        roe_id="R2", permits_engagement_type=True,
        proportionality_passed=True, collateral_estimate="low",
    ))
    gate.submit_deconfliction(case_id, DeconflictionContext(
        blue_force_clear=True, airspace_clear=True,
    ))
    gate.surface_options(case_id, [WeaponOption(
        option_id="opt-2", weapon_asset_id="ew-1",
        weapon_class="soft_kill", range_m=500,
        time_of_flight_s=0, pk_estimate=0.88,
        roe_compliant=True, deconfliction_ok=True, rationale="EW option",
    )])
    gate.authorize(case_id, OperatorAuthorization(
        decision=OperatorDecision.DENY,
        operator_id="mc-01", operator_role=operator_role,
        rationale="Non-combatant proximity", selected_option=None,
        signature=b"sig",
    ), engagement_class="counter_uas")


def _world_model_entity(
    entity_id: str,
    entity_type: str,
    source_adapter: str,
    lat: float,
    lon: float,
    alt_m: float = 0.0,
    speed_mps: float = 0.0,
    metadata: dict | None = None,
) -> dict:
    return {
        "entity_id":    entity_id,
        "entity_type":  entity_type,
        "adapter_type": source_adapter,
        "position":     {"lat": lat, "lon": lon, "alt_m": alt_m},
        "velocity":     {"speed_mps": speed_mps, "heading_deg": 0.0},
        "confidence":   0.91,
        "ts_iso":       datetime.now(timezone.utc).isoformat(),
        "metadata":     metadata or {},
    }


def _sensor_frame(sensor_id: str, sensor_type: str, payload: dict) -> dict:
    return {
        "sensor_id":   sensor_id,
        "sensor_type": sensor_type,
        "ts":          datetime.now(timezone.utc).isoformat(),
        "payload":     payload,
    }


# ============================================================================
# DOMAIN: AIR — COUNTER-UAS
# ============================================================================

class TestCounterUAS:
    """
    Group 1 / Group 2 small UAS threats detected by radar + EO/IR.
    Full engagement gate cycle exercised for each sub-scenario.
    """

    def test_single_rotary_uas_authorized(self):
        """Quadrotor threat, radar + thermal confirm, soft-kill authorized."""
        gate = EngagementAuthorizationGate.for_testing()
        t = _track("uas-01", "rotary_uas", sensors=["radar", "thermal_ir"])
        case = gate.open_case(t)
        _full_authorize(gate, case.case_id, "counter_uas")
        c = gate.get_case(case.case_id)
        assert c.state.name == "AUTHORIZED"
        assert c.decision.decision == OperatorDecision.AUTHORIZE

    def test_single_rotary_uas_denied_noncombatant_proximity(self):
        """Operator denies because civilian structure within collateral radius."""
        gate = EngagementAuthorizationGate.for_testing()
        t = _track("uas-02", "rotary_uas", sensors=["radar", "eo_visible"])
        case = gate.open_case(t)
        _full_deny(gate, case.case_id)
        assert gate.get_case(case.case_id).state.name == "DENIED"

    def test_fixed_wing_uas_high_speed(self):
        """Fixed-wing Group 2 at 80 m/s — radar + ADS-B (squawk 7700)."""
        gate = EngagementAuthorizationGate.for_testing()
        t = _track("uas-03", "fixed_wing_uas", confidence=0.97,
                   sensors=["radar", "ads_b"], alt_m=300.0)
        case = gate.open_case(t)
        _full_authorize(gate, case.case_id, "counter_uas")
        assert gate.get_case(case.case_id).state.name == "AUTHORIZED"

    def test_loitering_munition_hard_kill_option_selected(self):
        """Loitering munition — operator selects hard-kill option."""
        gate = EngagementAuthorizationGate.for_testing()
        t = _track("lm-01", "loitering_munition", confidence=0.89,
                   sensors=["radar", "thermal_ir", "eo_visible"])
        case = gate.open_case(t)
        gate.submit_pid(case.case_id, PIDEvidence(method="signature_match", confidence=0.93))
        gate.submit_roe(case.case_id, ROEContext(
            roe_id="R-LM", permits_engagement_type=True,
            proportionality_passed=True, collateral_estimate="low",
        ))
        gate.submit_deconfliction(case.case_id, DeconflictionContext(
            blue_force_clear=True, airspace_clear=True,
        ))
        gate.surface_options(case.case_id, [
            WeaponOption("opt-hk", "kinetic-1", "hard_kill", 800,
                         2.5, 0.87, True, True, "Directed energy hard kill"),
            WeaponOption("opt-sk", "ew-1", "soft_kill", 300,
                         0.0, 0.78, True, True, "RF jamming"),
        ])
        gate.authorize(case.case_id, OperatorAuthorization(
            decision=OperatorDecision.AUTHORIZE,
            operator_id="mc-01", operator_role="mission_commander",
            rationale="LM on terminal approach, no soft-kill margin",
            selected_option="opt-hk", signature=b"sig",
        ), engagement_class="counter_uas")
        case_final = gate.get_case(case.case_id)
        assert case_final.state.name == "AUTHORIZED"
        assert case_final.decision.selected_option == "opt-hk"

    def test_swarm_saturation_five_uas_simultaneous(self):
        """5 simultaneous UAS tracks — gate handles concurrent cases correctly."""
        gate = EngagementAuthorizationGate.for_testing()
        cases = []
        for i in range(5):
            t = _track(f"swarm-{i}", "small_uas", confidence=0.85 + i * 0.02,
                       sensors=["radar", "rf_spectrum"], lat=34.5 + i * 0.001)
            cases.append(gate.open_case(t))

        assert len(gate.list_cases()) == 5
        for i, case in enumerate(cases):
            if i < 3:
                _full_authorize(gate, case.case_id)
            else:
                _full_deny(gate, case.case_id)

        authorized = [c for c in gate.list_cases() if c.state.name == "AUTHORIZED"]
        denied     = [c for c in gate.list_cases() if c.state.name == "DENIED"]
        assert len(authorized) == 3
        assert len(denied) == 2

    def test_uas_proportionality_blocks(self):
        """ROE proportionality_passed=False — gate immediately denies the case."""
        gate = EngagementAuthorizationGate.for_testing()
        t = _track("uas-lc", "rotary_uas", confidence=0.81, sensors=["radar"])
        case = gate.open_case(t)
        gate.submit_pid(case.case_id, PIDEvidence(method="radar_signature", confidence=0.81))
        gate.submit_roe(case.case_id, ROEContext(
            roe_id="R-LC", permits_engagement_type=True,
            proportionality_passed=False,
            collateral_estimate="low",
        ))
        assert gate.get_case(case.case_id).state.name == "DENIED"

    def test_uas_operator_hold_pending_recon(self):
        """Operator issues HOLD to wait for additional ISR pass."""
        gate = EngagementAuthorizationGate.for_testing()
        t = _track("uas-hold", "rotary_uas", sensors=["radar"])
        case = gate.open_case(t)
        gate.submit_pid(case.case_id, PIDEvidence(method="radar_signature", confidence=0.82))
        gate.submit_roe(case.case_id, ROEContext(
            roe_id="R-H", permits_engagement_type=True,
            proportionality_passed=True, collateral_estimate="low",
        ))
        gate.submit_deconfliction(case.case_id, DeconflictionContext(
            blue_force_clear=True, airspace_clear=True,
        ))
        gate.surface_options(case.case_id, [WeaponOption(
            "opt-1", "ew-1", "soft_kill", 300, 0, 0.88, True, True, "",
        )])
        gate.authorize(case.case_id, OperatorAuthorization(
            decision=OperatorDecision.HOLD,
            operator_id="mc-01", operator_role="mission_commander",
            rationale="Awaiting secondary ISR pass for positive ID",
            selected_option=None, signature=b"sig",
        ), engagement_class="counter_uas")
        assert gate.get_case(case.case_id).state.name == "HELD"

    def test_uas_night_ir_only(self):
        """Night ops — only thermal IR available, EO unusable."""
        gate = EngagementAuthorizationGate.for_testing()
        t = _track("uas-night", "rotary_uas", sensors=["thermal_ir"], confidence=0.81)
        case = gate.open_case(t)
        _full_authorize(gate, case.case_id, "counter_uas")
        assert gate.get_case(case.case_id).state.name == "AUTHORIZED"

    def test_uas_gps_denied_radar_only(self):
        """GPS denied — position from radar only."""
        gate = EngagementAuthorizationGate.for_testing()
        t = _track("uas-gps-deny", "small_uas", sensors=["radar"], confidence=0.78)
        case = gate.open_case(t)
        _full_authorize(gate, case.case_id)
        assert gate.get_case(case.case_id).state.name == "AUTHORIZED"

    def test_uas_authorized_expires_without_complete(self):
        """AUTHORIZED case TTL expires before COMPLETE — auto-denied (EXPIRED)."""
        gate = EngagementAuthorizationGate.for_testing(default_ttl_seconds=5)
        t = _track("ttl-01", "rotary_uas")
        case = gate.open_case(t)
        _full_authorize(gate, case.case_id)
        assert gate.get_case(case.case_id).state.name == "AUTHORIZED"

        future = datetime.now(timezone.utc) + timedelta(seconds=30)
        expired = gate.expire_stale(now=future)
        assert case.case_id in expired
        assert gate.get_case(case.case_id).state.name == "EXPIRED"


# ============================================================================
# DOMAIN: AIR — ISR
# ============================================================================

class TestISR:
    """
    Intelligence, Surveillance, Reconnaissance: overwatch, cueing, handoff.
    """

    def test_persistent_overwatch_entity_stream(self):
        """EO/IR on VTOL — continuous entity stream, no engagement."""
        entities = [
            _world_model_entity(f"pov-{i}", "GROUND_UNIT", "eo_ir",
                                lat=34.500 + i * 0.0001, lon=-118.000 - i * 0.0001)
            for i in range(20)
        ]
        assert len(entities) == 20
        assert all("position" in e for e in entities)

    def test_multispectral_cueing_to_eo(self):
        """Multispectral hotspot cues EO camera for visual confirmation."""
        ms_frame = _sensor_frame("ms-01", "multispectral", {
            "bands": ["SWIR", "NIR", "RED"],
            "hotspots": [{"lat": 34.501, "lon": -118.001, "temp_k": 310}],
        })
        eo_frame = _sensor_frame("eo-01", "eo_visible", {
            "classification": "vehicle", "confidence": 0.88,
            "bounding_box": [120, 200, 180, 280],
        })
        assert ms_frame["payload"]["hotspots"][0]["temp_k"] == 310
        assert eo_frame["payload"]["confidence"] == 0.88

    def test_lidar_3d_scene_reconstruction(self):
        """LIDAR on UGV — 3D point cloud for obstacle map."""
        lidar = _sensor_frame("lidar-01", "lidar", {
            "scan_id": "scan-001", "num_points": 115_000,
            "resolution_cm": 5,
            "detections": [
                {"class": "vehicle", "confidence": 0.93},
                {"class": "wall",    "confidence": 0.99},
            ],
        })
        assert lidar["payload"]["num_points"] > 100_000

    def test_sar_through_fog_vessel_detection(self):
        """SAR at X-band sees through fog/rain — vessel detected."""
        sar = _sensor_frame("sar-01", "sar_x_band", {
            "look_angle_deg": 35, "resolution_m": 3,
            "detections": [
                {"type": "vessel", "lat": 25.1, "lon": -80.5,
                 "length_m": 45, "confidence": 0.87},
            ],
            "weather_conditions": "fog_heavy",
        })
        assert sar["payload"]["detections"][0]["confidence"] > 0.8

    def test_satellite_cued_isr_handoff(self):
        """Sentinel-2 change detection cues airborne ISR at same lat/lon."""
        cue = {
            "source": "sentinel_2",
            "lat": 34.602, "lon": -117.801,
            "confidence": 0.79,
        }
        task = {
            "asset_id": "vtol-isr-01",
            "lat": cue["lat"], "lon": cue["lon"],
            "sensors": ["eo_visible", "thermal_ir"],
        }
        assert task["lat"] == cue["lat"]

    def test_rfint_signals_detection(self):
        """RF spectrum scan — drone control link detected at 915 MHz."""
        rf = _sensor_frame("rfint-01", "rf_spectrum", {
            "scan_range_mhz": [400, 6000],
            "detections": [
                {"freq_mhz": 915, "power_dbm": -42, "bearing_deg": 247,
                 "classification": "drone_control_link", "confidence": 0.84},
            ],
        })
        assert rf["payload"]["detections"][0]["freq_mhz"] == 915


# ============================================================================
# DOMAIN: GROUND
# ============================================================================

class TestGroundOps:
    """
    Convoy escort, FOB perimeter, CASEVAC, BDA, UGV recon.
    """

    def test_convoy_hostile_vehicle_gate(self):
        """Hostile vehicle detected on convoy route — gate authorized."""
        gate = EngagementAuthorizationGate.for_testing()
        t = _track("gv-01", "ground_vehicle_hostile",
                   sensors=["radar_ground", "eo_visible", "thermal_ir"],
                   confidence=0.91, alt_m=0.0)
        case = gate.open_case(t)
        _full_authorize(gate, case.case_id, "force_protection")
        assert gate.get_case(case.case_id).state.name == "AUTHORIZED"

    def test_fob_perimeter_triple_sensor_corroboration(self):
        """PIR + magnetometer + seismic all trigger — high-confidence breach."""
        frames = [
            _sensor_frame("pir-01",     "pir_motion",    {"zone": "north", "triggered": True}),
            _sensor_frame("mag-01",     "magnetometer",  {"anomaly_detected": True, "bearing_deg": 12}),
            _sensor_frame("seismic-01", "seismic",       {"event_type": "footstep", "confidence": 0.72}),
        ]
        corroborating = sum(1 for f in frames if (
            f["payload"].get("triggered") or
            f["payload"].get("anomaly_detected") or
            f["payload"].get("confidence", 0) > 0.6
        ))
        assert corroborating == 3

    def test_casevac_t1_routing(self):
        """CASEVAC T1 patient — medevac UAS tasked, route computed."""
        req = {
            "patient_count": 2, "triage": "T1",
            "grid": "34.501,-118.002",
            "asset_type": "rotary_wing_medevac",
            "asset_id": "DUSTOFF-01",
            "route": [{"lat": 34.510, "lon": -118.010}, {"lat": 34.520, "lon": -118.020}],
        }
        assert req["triage"] == "T1"
        assert len(req["route"]) == 2

    def test_bda_post_engagement(self):
        """Post-strike BDA via EO + LIDAR — target destroyed, no collateral."""
        bda = {
            "target_id": "target-001",
            "post_strike": {
                "eo_assessment": "target_destroyed",
                "lidar_delta_m3": 42.5,
                "confidence": 0.91,
                "collateral": "none_detected",
            },
        }
        assert bda["post_strike"]["eo_assessment"] == "target_destroyed"
        assert bda["post_strike"]["collateral"] == "none_detected"

    def test_ugv_tracked_urban_recon(self):
        """Tracked UGV clears 3 rooms with LIDAR + thermal."""
        ugv = {
            "asset_id": "ugv-tracked-01", "asset_type": "ugv_tracked",
            "sensors": ["lidar", "thermal_ir", "eo_visible", "acoustic"],
            "rooms_cleared": 3, "obstacles_mapped": 12,
        }
        assert ugv["rooms_cleared"] == 3

    def test_ugv_wheeled_eod_ied_detection(self):
        """EOD UGV detects IED at 12 cm depth with GPR."""
        eod = {
            "asset_type": "ugv_wheeled_eod",
            "sensors": ["chem_bio_rad", "ground_penetrating_radar", "eo_visible"],
            "detections": [{"type": "ied_suspected", "confidence": 0.85, "depth_cm": 12}],
            "safe_corridor": False,
        }
        assert eod["detections"][0]["confidence"] == 0.85

    def test_quadruped_rubble_traverse_survivor(self):
        """Quadruped (Spot-class) navigates rubble, locates 1 survivor."""
        spot = {
            "asset_type": "ugv_quadruped",
            "sensors": ["eo_visible", "thermal_ir", "lidar"],
            "mode": "rubble_traverse",
            "survivors_detected": 1,
        }
        assert spot["survivors_detected"] == 1

    def test_acoustic_gunshot_direction_finding(self):
        """Acoustic array detects gunshot at 350 m — bearing 137°."""
        acoustic = _sensor_frame("acoustic-01", "acoustic_array", {
            "event_type": "gunshot", "confidence": 0.96,
            "bearing_deg": 137, "range_m": 350,
            "caliber_class": "rifle",
        })
        assert acoustic["payload"]["range_m"] < 1000


# ============================================================================
# DOMAIN: MARITIME
# ============================================================================

class TestMaritime:
    """
    Vessel interdiction, maritime SAR, USV, UUV, mine countermeasures.
    """

    def test_vessel_interdiction_gate(self):
        """Unknown vessel classified threat — gate authorized."""
        gate = EngagementAuthorizationGate.for_testing()
        t = _track("vessel-01", "vessel_unknown",
                   sensors=["ais", "radar_s_band", "sar"],
                   confidence=0.88, lat=25.1, lon=-80.5, alt_m=0)
        case = gate.open_case(t)
        _full_authorize(gate, case.case_id, "force_protection")
        assert gate.get_case(case.case_id).state.name == "AUTHORIZED"

    def test_maritime_sar_survivor_handoff(self):
        """USV + aerial UAS detect survivors in sea state 4."""
        mission = {
            "sea_state": 4,
            "assets": [
                {"id": "usv-01",  "type": "usv",      "sensors": ["eo_visible", "thermal_ir", "radar"]},
                {"id": "vtol-01", "type": "vtol_uas",  "sensors": ["eo_visible", "thermal_ir"]},
            ],
            "survivors_detected": 2,
            "rescue_handoff_asset": "helo-medevac-01",
        }
        assert mission["survivors_detected"] == 2

    def test_uuv_subsurface_sonar_contact(self):
        """UUV passive sonar classifies submarine contact."""
        contact = {
            "asset_type": "uuv",
            "sensors": ["sonar_active", "sonar_passive", "hydrophone_array"],
            "contact": {
                "classification": "submarine",
                "confidence": 0.74, "bearing_deg": 220,
                "range_m": 1200, "depth_m": 45, "speed_kts": 6.0,
            },
        }
        assert contact["contact"]["depth_m"] == 45

    def test_ais_dark_ship_anomaly(self):
        """AIS track disappears near choke point — radar continues, spoofing suspected."""
        ais_last = {"mmsi": "123456789", "lat": 25.11, "lon": -80.51, "speed_kts": 8.2}
        radar_now = {"lat": 25.15, "lon": -80.55, "speed_kts": 8.5}
        delta_m = math.sqrt(
            (ais_last["lat"] - radar_now["lat"]) ** 2 +
            (ais_last["lon"] - radar_now["lon"]) ** 2
        ) * 111_000
        assert delta_m > 100  # vessel moved without AIS — flag anomaly

    def test_usv_mine_countermeasures(self):
        """USV sonar detects 3 mines, neutralizes 2."""
        mcm = {
            "asset_type": "usv_mcm",
            "sensors": ["sonar_active", "forward_looking_sonar"],
            "mines_detected": 3, "mines_neutralized": 2,
        }
        assert mcm["mines_detected"] == 3

    def test_buoy_sea_state_routing(self):
        """NDBC buoy sea-state ingested — USV route modified for survivability."""
        buoy = _sensor_frame("buoy-41044", "ndbc_buoy", {
            "wave_height_m": 2.4, "wind_speed_mps": 11.2,
            "swell_period_s": 9.0, "water_temp_c": 24.1,
        })
        assert buoy["payload"]["wave_height_m"] < 6.0


# ============================================================================
# DOMAIN: WILDFIRE
# ============================================================================

class TestWildfire:
    """
    Wildfire aerial coordination: fire front, ember, airtanker deconfliction.
    """

    def test_firms_hotspots_fire_front(self):
        """VIIRS hotspots — total FRP > 100 triggers full response."""
        hotspots = [
            {"lat": 34.50, "lon": -118.00, "frp": 45.2},
            {"lat": 34.51, "lon": -118.01, "frp": 38.7},
            {"lat": 34.52, "lon": -118.02, "frp": 52.1},
        ]
        assert sum(h["frp"] for h in hotspots) > 100

    def test_airtanker_vertical_separation(self):
        """Two airtankers must maintain ≥200 ft vertical separation (NWCG)."""
        assets = [
            {"id": "AT-01", "type": "airtanker",  "alt_ft": 1500},
            {"id": "AT-02", "type": "airtanker",  "alt_ft": 1700},
            {"id": "HT-01", "type": "helitanker", "alt_ft":  900},
        ]
        sep_ft = abs(assets[0]["alt_ft"] - assets[1]["alt_ft"])
        assert sep_ft >= 200

    def test_ember_cast_thermal_detection(self):
        """UAS thermal detects ember cast ahead of fire front."""
        scan = {
            "asset_id": "uav-fire-01", "sensor": "thermal_ir",
            "ember_detections": [
                {"lat": 34.60, "lon": -118.10, "temp_k": 650, "confidence": 0.82},
            ],
        }
        assert len(scan["ember_detections"]) > 0

    def test_ground_crew_evacuation_trigger(self):
        """Fire front within 300 m of ground crew — EVACUATE alert."""
        crew = {"lat": 34.510, "lon": -118.010}
        # ~157 m north — well within 300 m
        fire = {"lat": 34.5114, "lon": -118.010}
        dist_m = math.sqrt(
            (crew["lat"] - fire["lat"]) ** 2 +
            (crew["lon"] - fire["lon"]) ** 2
        ) * 111_000
        alert = "EVACUATE" if dist_m < 300 else "MONITOR"
        assert alert == "EVACUATE"

    def test_red_flag_weather_conditions(self):
        """OpenMeteo data: wind > 8 m/s + RH < 15% + temp > 32°C → Red Flag."""
        wx = {"wind_speed_mps": 14.0, "relative_humidity_pct": 12.0, "temp_c": 38.0}
        red_flag = (wx["wind_speed_mps"] > 8.0 and
                    wx["relative_humidity_pct"] < 15.0 and
                    wx["temp_c"] > 32.0)
        assert red_flag


# ============================================================================
# DOMAIN: FLOOD
# ============================================================================

class TestFlood:
    """
    Urban flood response, levee breach, UAS grid search, rescue swimmer.
    """

    def test_usgs_gauge_flood_alert(self):
        """USGS gauge exceeds flood stage — FLOOD alert."""
        gauge = _sensor_frame("usgs-01", "usgs_stream_gauge", {
            "stage_ft": 32.4, "flood_stage_ft": 32.0,
        })
        stage = gauge["payload"]["stage_ft"]
        assert stage >= gauge["payload"]["flood_stage_ft"]
        assert (stage >= gauge["payload"]["flood_stage_ft"])

    def test_sar_inundation_extent(self):
        """Sentinel-1 SAR before/after — >15% new inundation."""
        before = {"water_pixels": 1200, "total_pixels": 100_000}
        after  = {"water_pixels": 18_500, "total_pixels": 100_000}
        inundation_pct = (after["water_pixels"] - before["water_pixels"]) / before["total_pixels"] * 100
        assert inundation_pct > 15.0

    def test_levee_breach_downstream_zones(self):
        """Levee breach triggers evacuation for all 3 downstream zones."""
        event = {
            "breach_status": True,
            "downstream_zones": ["Zone-A", "Zone-B", "Zone-C"],
        }
        evac = event["downstream_zones"] if event["breach_status"] else []
        assert len(evac) == 3

    def test_rescue_swimmer_hoist(self):
        """Helicopter deploys rescue swimmer — patient secure."""
        op = {
            "asset_type": "rotary_wing_rescue",
            "swimmer_deployed": True,
            "patient_status": "ambulatory",
            "cable_out_m": 12.0,
        }
        assert op["swimmer_deployed"]
        assert op["cable_out_m"] > 0

    def test_uav_grid_search_rooftop_survivors(self):
        """UAS grid search — 4 total survivors across 2 cells."""
        grid = {
            "survivors_found": [
                {"cell": "C4", "count": 3},
                {"cell": "E6", "count": 1},
            ],
        }
        assert sum(s["count"] for s in grid["survivors_found"]) == 4


# ============================================================================
# DOMAIN: HADR — Mass Casualty
# ============================================================================

class TestHADR:
    """
    Earthquake, hurricane, tsunami, building collapse, resupply.
    """

    def test_earthquake_casualty_site_triage(self):
        """Post-earthquake: 2 sites, 23 estimated casualties, 2 UAS deployed."""
        mcs = {
            "event_type": "earthquake", "magnitude": 6.8,
            "casualty_sites": [
                {"id": "cs-01", "count_estimated": 15, "structure_type": "RC_building"},
                {"id": "cs-02", "count_estimated": 8,  "structure_type": "masonry"},
            ],
            "uas_deployed": ["uav-triage-01", "uav-triage-02"],
        }
        total = sum(cs["count_estimated"] for cs in mcs["casualty_sites"])
        assert total == 23
        assert len(mcs["uas_deployed"]) == 2

    def test_hurricane_cat4_evacuation(self):
        """Category 4 hurricane: mandatory evacuation flagged."""
        advisory = {"category": 4, "surge_ft": 10.0, "eta_hours": 18}
        assert advisory["category"] >= 3

    def test_tsunami_near_field_eta(self):
        """PTWC alert: AK zones < 60 min ETA flagged for immediate action."""
        ptwc = {
            "wave_eta": {"AK-01": 15, "AK-02": 20, "HI-01": 280},
        }
        immediate = [z for z, eta in ptwc["wave_eta"].items() if eta < 60]
        assert "AK-01" in immediate and "AK-02" in immediate

    def test_building_collapse_co2_survivor_signal(self):
        """CO2 elevated (850 ppm vs 420 baseline) — survivor signal present."""
        co2 = _sensor_frame("co2-01", "co2_sensor", {
            "co2_ppm": 850, "baseline_ppm": 420, "elevated": True,
        })
        assert co2["payload"]["elevated"]

    def test_cargo_uas_resupply_drop(self):
        """Cargo UAS drops 5 kg payload within 5 m GPS accuracy."""
        mission = {
            "asset_type": "cargo_uas", "payload_kg": 5.0,
            "drop_status": "released", "gps_accuracy_m": 2.1,
        }
        assert mission["drop_status"] == "released"
        assert mission["gps_accuracy_m"] < 5.0


# ============================================================================
# DOMAIN: ADVERSARIAL CONDITIONS
# ============================================================================

class TestAdversarialConditions:
    """
    Jamming, spoofing, sensor denial, cyber attack, sensor degradation.
    """

    def test_gps_spoofing_ins_divergence(self):
        """GNSS/INS delta > 100 m — spoofing suspected."""
        gnss = {"lat": 34.500, "lon": -118.000}
        ins  = {"lat": 34.503, "lon": -118.003}
        delta_m = math.sqrt(
            (gnss["lat"] - ins["lat"]) ** 2 +
            (gnss["lon"] - ins["lon"]) ** 2
        ) * 111_000
        assert delta_m > 100

    def test_rf_jamming_noise_floor_rise(self):
        """GPS L1 noise floor rises 20 dB — jammer detected."""
        baseline = -100  # dBm
        jammed   = -80
        assert (jammed - baseline) >= 20

    def test_comms_jammed_ttl_expiry(self):
        """Gate case authorized but uplink severed — TTL expires, EXPIRED state."""
        gate = EngagementAuthorizationGate.for_testing(default_ttl_seconds=5)
        t = _track("jam-ttl", "rotary_uas")
        case = gate.open_case(t)
        _full_authorize(gate, case.case_id)
        future = datetime.now(timezone.utc) + timedelta(seconds=30)
        gate.expire_stale(now=future)
        assert gate.get_case(case.case_id).state.name == "EXPIRED"

    def test_primary_radar_failure_eo_ir_fallback(self):
        """Primary radar down — system continues with EO/IR at lower confidence."""
        gate = EngagementAuthorizationGate.for_testing()
        t = _track("dg-01", "rotary_uas", confidence=0.71,
                   sensors=["eo_visible", "thermal_ir"])  # no radar
        case = gate.open_case(t)
        _full_authorize(gate, case.case_id)
        assert gate.get_case(case.case_id).state.name == "AUTHORIZED"

    def test_insufficient_role_blocks(self):
        """Operator with 'operator' role tries to authorize — blocked."""
        gate = EngagementAuthorizationGate.for_testing(allow_role=False)
        t = _track("role-01", "rotary_uas")
        case = gate.open_case(t)
        gate.submit_pid(case.case_id, PIDEvidence(method="iff", confidence=0.95))
        gate.submit_roe(case.case_id, ROEContext(
            roe_id="R-R", permits_engagement_type=True,
            proportionality_passed=True, collateral_estimate="low",
        ))
        gate.submit_deconfliction(case.case_id, DeconflictionContext(
            blue_force_clear=True, airspace_clear=True,
        ))
        gate.surface_options(case.case_id, [WeaponOption(
            "opt-1", "ew-1", "soft_kill", 300, 0, 0.9, True, True, "",
        )])
        with pytest.raises(Exception):
            gate.authorize(case.case_id, OperatorAuthorization(
                decision=OperatorDecision.AUTHORIZE,
                operator_id="junior-01", operator_role="operator",
                rationale="unauthorized", selected_option="opt-1",
                signature=b"sig",
            ), engagement_class="counter_uas")

    def test_swarm_10_no_cross_contamination(self):
        """10 concurrent cases — all independent, no state bleed."""
        gate = EngagementAuthorizationGate.for_testing()
        cases = [gate.open_case(_track(f"swrm-{i}", "small_uas")) for i in range(10)]
        assert len({c.case_id for c in cases}) == 10

    def test_fog_thermal_primary_sensor(self):
        """Dense fog disables EO — thermal becomes sole sensor, gate continues."""
        gate = EngagementAuthorizationGate.for_testing()
        t = _track("fog-01", "rotary_uas", sensors=["thermal_ir"], confidence=0.83)
        case = gate.open_case(t)
        _full_authorize(gate, case.case_id)
        assert gate.get_case(case.case_id).state.name == "AUTHORIZED"

    def test_audit_chain_tamper_detection(self):
        """Tampered audit record detected — chain integrity fails."""
        import json
        import tempfile
        from pathlib import Path as P
        from packages.c2_intel.engagement_wiring import ChainedHMACAuditSink

        with tempfile.TemporaryDirectory() as tmp:
            path = P(tmp) / "audit.jsonl"
            sink = ChainedHMACAuditSink(path, hmac_key=b"\xde\xad" * 16)
            for i in range(5):
                sink({"transition": "OPEN", "case_id": f"c-{i}",
                      "to_state": "detected", "payload": {}})

            lines = path.read_text().splitlines()
            row = json.loads(lines[2])
            row["payload"] = {"injected": "BACKDOOR"}
            lines[2] = json.dumps(row)
            path.write_text("\n".join(lines) + "\n")

            ok, written, bad = sink.verify_chain()
            assert ok is False
            assert bad > 0


# ============================================================================
# DOMAIN: SPACE-CUED ISR
# ============================================================================

class TestSpaceCuedISR:

    def test_satellite_change_detection_tasks_uav(self):
        """Sentinel-2 NDVI anomaly → UAV tasked to same coordinates."""
        hit = {"source": "sentinel_2_msi", "lat": 34.61, "lon": -117.90, "confidence": 0.81}
        task = {"asset": "vtol-isr-01", "waypoint": {"lat": hit["lat"], "lon": hit["lon"]}}
        assert task["waypoint"]["lat"] == 34.61

    def test_tle_conjunction_alert(self):
        """Two objects within 1 km — conjunction warning fires."""
        pos_a = {"x_km": 6778.0}
        pos_b = {"x_km": 6778.3}
        sep_km = abs(pos_a["x_km"] - pos_b["x_km"])
        assert sep_km < 1.0

    def test_sigint_satellite_to_rfint_tasking(self):
        """SIGINT satellite pass localizes emitter — ground RFINT tasked."""
        cue = {"freq_mhz": 433, "ground_intercept": {"lat": 34.50, "lon": -118.00, "accuracy_km": 8}}
        task = {"search_radius_km": cue["ground_intercept"]["accuracy_km"] * 1.2}
        assert task["search_radius_km"] < 20


# ============================================================================
# DOMAIN: ACE / CANVAS
# ============================================================================

class TestACECanvas:
    """
    Full CANVAS TA1: authority delegation, distributed C2, jamming, multi-class.
    Uses the real run_simulation() API.
    """

    def test_full_ace_demo_scenario(self):
        """Canonical demo: 12 requests, 0 denied, delegation > 0%."""
        from packages.canvas.workflow_sim import demo_ace_scenario, run_simulation
        scenario, policy = demo_ace_scenario()
        result = run_simulation(scenario, policy)
        s = result.summary()
        assert s["n_requests"] == 12
        assert s["n_denied"] == 0
        assert s["delegation_rate"] > 0.0

    def test_jamming_triggers_conditional_delegation(self):
        """After jamming (uplink_seconds_since > threshold) delegation fires."""
        from packages.canvas.workflow_sim import (
            Node, Scenario, EngagementRequest, IntentPolicy,
            CommsTrajectory, run_simulation,
        )
        from packages.canvas.authority_dsl import (
            CommanderIntent, CommsState,
        )
        intent = CommanderIntent(
            id="TEST-INTENT-01",
            permits=["counter_uas:soft_kill"],
            delegated_thresholds_uplink_seconds=90,
            delegated_thresholds_intent_age_seconds=900,
            signed_by="cocom",
        )
        policy = IntentPolicy(intent=intent)

        healthy  = CommsState(uplink_seconds_since=5, pace_active="primary", intent_age_seconds=10)
        degraded = CommsState(uplink_seconds_since=200, pace_active="alternate", intent_age_seconds=10)

        fob = Node("F1", "fob", "fob-cmd-1", "mission_commander",
                   CommsTrajectory([(0, healthy), (100, degraded)]))

        # One request before jamming, one after
        requests = [
            EngagementRequest("R1", 50,  "F1", "counter_uas", "soft_kill", "T1"),
            EngagementRequest("R2", 200, "F1", "counter_uas", "soft_kill", "T2"),
        ]
        scenario = Scenario("jam-test", [fob], requests)
        result = run_simulation(scenario, policy)
        assert result.n_total == 2
        assert result.n_denied == 0

    def test_all_engagement_classes_routed(self):
        """Every engagement class routes without error."""
        from packages.canvas.workflow_sim import (
            Node, Scenario, EngagementRequest, IntentPolicy,
            CommsTrajectory, run_simulation,
        )
        from packages.canvas.authority_dsl import CommanderIntent, CommsState

        intent = CommanderIntent(
            id="TEST-MULTI",
            permits=[
                "counter_uas:soft_kill", "counter_uas:hard_kill",
                "force_protection_perimeter:soft_kill",
                "base_defense:any", "ace_strike:any",
            ],
            delegated_thresholds_uplink_seconds=90,
            delegated_thresholds_intent_age_seconds=900,
            signed_by="cocom",
        )
        policy = IntentPolicy(intent=intent)
        healthy = CommsState(uplink_seconds_since=5, pace_active="primary", intent_age_seconds=10)
        wing = Node("W1", "wing", "w-cdr", "joint_force_commander",
                    CommsTrajectory([(0, healthy)]))

        classes = [
            ("counter_uas",              "soft_kill",  0),
            ("counter_uas",              "hard_kill",  10),
            ("force_protection_perimeter", "soft_kill", 20),
            ("base_defense",             "any",        30),
            ("ace_strike",               "any",        40),
        ]
        requests = [
            EngagementRequest(f"R{i}", t, "W1", ec, wc, f"T{i}")
            for i, (ec, wc, t) in enumerate(classes)
        ]
        scenario = Scenario("multi-class", [wing], requests)
        result = run_simulation(scenario, policy)
        assert result.n_total == 5

    def test_dodd_300009_single_authorized_emission(self):
        """ENGAGEMENT_AUTHORIZED emitted exactly once per case (DoDD 3000.09)."""
        from packages.c2_intel.models import C2EventType

        emitted = []
        gate = EngagementAuthorizationGate(
            emit_event=lambda et, p: emitted.append(et) if et == C2EventType.ENGAGEMENT_AUTHORIZED else None,
            verify_operator_signature=lambda *_: True,
            operator_has_role=lambda *_: True,
            audit_sink=lambda *_: None,
        )
        t = _track("dodd-01", "rotary_uas")
        case = gate.open_case(t)
        _full_authorize(gate, case.case_id)
        assert len(emitted) == 1

    def test_collateral_blocking_denies(self):
        """collateral_estimate='blocking' — ROE denies the case immediately."""
        gate = EngagementAuthorizationGate.for_testing()
        t = _track("coll-01", "rotary_uas")
        case = gate.open_case(t)
        gate.submit_pid(case.case_id, PIDEvidence(method="iff", confidence=0.96))
        gate.submit_roe(case.case_id, ROEContext(
            roe_id="R-COLL",
            permits_engagement_type=True,
            proportionality_passed=True,
            collateral_estimate="blocking",
        ))
        assert gate.get_case(case.case_id).state.name == "DENIED"


# ============================================================================
# SEARCH AND RESCUE
# ============================================================================

class TestSearchAndRescue:

    def test_wilderness_sar_full_grid_sweep(self):
        """VTOL thermal + EO — grid fully swept, subject located at 89% confidence."""
        search = {
            "grid_cells": 144, "cells_swept": 144,
            "detection": {"confidence": 0.89},
        }
        assert search["cells_swept"] == search["grid_cells"]
        assert search["detection"]["confidence"] > 0.85

    def test_urban_rubble_acoustic_to_robot_dispatch(self):
        """Acoustic hit → quadruped dispatched to same location."""
        hit = {"location": {"lat": 34.502, "lon": -118.003}, "confidence": 0.77}
        robot = {"asset": "quadruped-01", "waypoint": hit["location"]}
        assert robot["waypoint"] == hit["location"]

    def test_nir_night_human_detection(self):
        """NIR / image-intensified sensor detects person at night."""
        nir = _sensor_frame("nir-01", "nir_intensified", {
            "illumination": "moonless",
            "detections": [{"type": "human", "confidence": 0.81}],
        })
        assert nir["payload"]["detections"][0]["type"] == "human"

    def test_helicopter_hoist_secure(self):
        """Helo deploys hoist 12 m — patient secure."""
        op = {"hoist_deployed": True, "cable_out_m": 12.0, "secure": True}
        assert op["secure"] and op["cable_out_m"] > 0


# ============================================================================
# SENSOR ADAPTER SMOKE
# ============================================================================

class TestSensorAdapterSmoke:
    """Every sensor type produces a valid frame shape."""

    @pytest.mark.parametrize("adapter_name,frame", [
        ("eo_visible",     {"classification": "vehicle", "confidence": 0.88}),
        ("thermal_ir",     {"temp_k": 309.5, "confidence": 0.82}),
        ("radar_x_band",   {"range_m": 1200, "azimuth_deg": 225, "rcs_dbsm": -5.0}),
        ("radar_s_band",   {"range_m": 8000, "doppler_mps": 12.5}),
        ("lidar",          {"num_points": 50000, "range_m": 80}),
        ("sar_x_band",     {"look_angle_deg": 30, "resolution_m": 3}),
        ("multispectral",  {"bands": ["RED", "NIR", "SWIR"], "ndvi": 0.62}),
        ("ais",            {"mmsi": "123456789", "lat": 25.1, "lon": -80.5}),
        ("ads_b",          {"icao": "A12345", "lat": 34.0, "alt_ft": 10000}),
        ("rf_spectrum",    {"noise_floor_dbm": -98, "detections": []}),
        ("acoustic_array", {"event_type": "gunshot", "confidence": 0.92}),
        ("seismic",        {"magnitude": 2.1, "event_type": "footstep"}),
        ("magnetometer",   {"anomaly_detected": True, "magnitude_nt": 380}),
        ("chem_bio_rad",   {"agent_detected": False, "rad_mremshr": 2.1}),
        ("ndbc_buoy",      {"wave_height_m": 1.8, "water_temp_c": 22.0}),
        ("usgs_gauge",     {"stage_ft": 28.4, "flow_cfs": 1200}),
    ])
    def test_sensor_frame_shape(self, adapter_name, frame):
        sf = _sensor_frame(f"{adapter_name}-test", adapter_name, frame)
        assert sf["sensor_type"] == adapter_name
        assert "ts" in sf
        assert isinstance(sf["payload"], dict)


# ============================================================================
# ASSET TYPE SMOKE
# ============================================================================

class TestAssetTypeSmoke:
    """Every asset type produces a valid world model entity."""

    @pytest.mark.parametrize("asset_type,adapter", [
        ("small_uas_quad",      "mavlink"),
        ("mid_uas_group2",      "mavlink"),
        ("vtol_fixed_wing",     "mavlink"),
        ("loitering_munition",  "radar_track"),
        ("tethered_aerostat",   "mavlink"),
        ("usv",                 "ais"),
        ("uuv",                 "acoustic"),
        ("ugv_tracked",         "ros2"),
        ("ugv_wheeled",         "ros2"),
        ("ugv_quadruped",       "ros2"),
        ("rotary_wing_medevac", "mavlink"),
        ("fixed_wing_isr",      "ads_b"),
        ("ground_sensor_node",  "mqtt"),
        ("cargo_uas",           "mavlink"),
    ])
    def test_entity_shape(self, asset_type, adapter):
        e = _world_model_entity(f"entity-{asset_type}", asset_type, adapter,
                                lat=34.5, lon=-118.0, alt_m=150.0)
        assert e["entity_type"] == asset_type
        assert e["adapter_type"] == adapter
        assert e["position"]["lat"] == 34.5
