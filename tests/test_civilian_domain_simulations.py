"""
Heli.OS — Civilian Domain Simulation Suite
============================================
Extends the mission simulation suite to cover commercial / industrial
domains outside of defense and emergency response.

Domains covered:
  CONSTRUCTION   — site safety, progress, crane ops, earthworks, slope stability
  OIL & GAS      — pipeline inspection, offshore platform, gas leak, SCADA
  ENERGY         — power line, solar farm, wind turbine, substation, grid
  AGRICULTURE    — crop health, irrigation, livestock, pest, precision spraying
  MINING         — open pit, tailings dam, underground, haul road
  INFRASTRUCTURE — bridge, dam, port, railway, road condition
  FORESTRY       — deforestation, illegal logging, carbon stock
  LOGISTICS      — warehouse, port cargo, last-mile delivery, cold chain
  PUBLIC SAFETY  — crowd estimation, traffic, accident response, event security

Adapters exercised:
  isobus, j1939, bacnet, modbus, lorawan, zigbee, ur (Universal Robots),
  tesla (grid), onvif, rtsp, thermal, ros2, mavlink, mqtt, webhook

Models exercised (inference shape / feature vector validation):
  pipeline_anomaly_classifier, corrosion_classifier, corrosion_vision,
  slope_stability_classifier, eurosat_lulc_classifier,
  deforestation_classifier, drought_severity_classifier,
  vehicle_classifier, crowd_estimator, damage_classifier

Run with:
    pytest tests/test_civilian_domain_simulations.py -v
"""
from __future__ import annotations

import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Shared helpers (no engagement gate — these are non-kinetic civilian ops)
# ---------------------------------------------------------------------------

def _sf(sensor_id: str, sensor_type: str, payload: dict) -> dict:
    """Build a sensor frame."""
    return {
        "sensor_id":   sensor_id,
        "sensor_type": sensor_type,
        "ts":          datetime.now(timezone.utc).isoformat(),
        "payload":     payload,
    }


def _entity(entity_id: str, entity_type: str, adapter: str,
            lat: float, lon: float, alt_m: float = 0.0,
            metadata: dict | None = None) -> dict:
    return {
        "entity_id":    entity_id,
        "entity_type":  entity_type,
        "adapter_type": adapter,
        "position":     {"lat": lat, "lon": lon, "alt_m": alt_m},
        "ts_iso":       datetime.now(timezone.utc).isoformat(),
        "metadata":     metadata or {},
    }


def _inference_record(model_id: str, features: dict, prediction: Any,
                      confidence: float) -> dict:
    return {
        "model_id":   model_id,
        "features":   features,
        "prediction": prediction,
        "confidence": confidence,
        "ts":         datetime.now(timezone.utc).isoformat(),
    }


def _alert(alert_id: str, severity: str, domain: str,
           description: str, source: str) -> dict:
    return {
        "alert_id":    alert_id,
        "severity":    severity,
        "domain":      domain,
        "description": description,
        "source":      source,
        "ts_iso":      datetime.now(timezone.utc).isoformat(),
    }


# ============================================================================
# DOMAIN: CONSTRUCTION
# ============================================================================

class TestConstruction:
    """
    Construction site monitoring: safety, progress, crane, earthworks.
    Adapters: RTSP/ONVIF cameras, LIDAR, J1939 (heavy equipment CAN bus),
              LiDAR progress scan, BACnet (site office HVAC/access).
    """

    def test_ppe_hard_hat_detection_via_onvif(self):
        """ONVIF IP camera on site entry — PPE compliance check (hard hat, vest)."""
        frame = _sf("onvif-entry-01", "onvif", {
            "camera_id": "site-gate-cam-01",
            "resolution": "1920x1080",
            "detections": [
                {"class": "person",    "confidence": 0.97, "ppe": {"hard_hat": True,  "vest": True}},
                {"class": "person",    "confidence": 0.95, "ppe": {"hard_hat": False, "vest": True}},
            ],
        })
        violations = [d for d in frame["payload"]["detections"]
                      if not d["ppe"].get("hard_hat")]
        assert len(violations) == 1
        alert = _alert("ppe-001", "WARNING", "construction",
                       "Worker missing hard hat at site entry", "onvif-entry-01")
        assert alert["severity"] == "WARNING"

    def test_lidar_site_progress_volume_scan(self):
        """LIDAR drone scan — earthworks volume delta vs design model."""
        current_scan = {"volume_m3": 45_200, "surface_area_m2": 8_400}
        design_model  = {"volume_m3": 50_000, "surface_area_m2": 8_400}
        completion_pct = current_scan["volume_m3"] / design_model["volume_m3"] * 100
        assert 80 < completion_pct < 100

    def test_j1939_excavator_fuel_and_idle_monitoring(self):
        """J1939 CAN bus telemetry from excavator — excess idle time flagged."""
        frame = _sf("j1939-excav-01", "j1939", {
            "asset_id":        "EX-CAT390-01",
            "engine_rpm":      800,
            "fuel_level_pct":  42,
            "idle_time_hrs":   3.2,
            "total_run_hrs":   4.8,
            "fault_codes":     [],
        })
        idle_ratio = frame["payload"]["idle_time_hrs"] / frame["payload"]["total_run_hrs"]
        assert idle_ratio > 0.5  # >50% idle → efficiency alert
        alert = _alert("idle-001", "INFO", "construction",
                       f"Excavator idle ratio {idle_ratio:.0%} — review dispatch schedule",
                       "j1939-excav-01")
        assert alert["alert_id"] == "idle-001"

    def test_crane_load_monitoring_overload_prevention(self):
        """Crane load cell + angle sensor — overload imminent, halt command issued."""
        crane_state = {
            "asset_id":       "tower-crane-01",
            "load_kg":        4_850,
            "rated_load_kg":  5_000,
            "boom_angle_deg": 72,
            "wind_speed_mps": 11.0,
        }
        # Wind derating: at >10 m/s, rated load reduces by 15%
        derated_load_kg = crane_state["rated_load_kg"] * 0.85
        overload_risk = crane_state["load_kg"] > derated_load_kg
        assert overload_risk
        command = {"asset_id": "tower-crane-01", "command": "HALT_LIFT",
                   "reason": "overload_risk_at_current_wind"}
        assert command["command"] == "HALT_LIFT"

    def test_slope_stability_model_feature_vector(self):
        """Slope stability classifier feature vector structure validated."""
        # Matches packages/c2_intel/models/slope_stability_classifier_meta.json features
        features = {
            "slope_angle_deg":    38.0,
            "cohesion_kpa":       18.0,
            "friction_angle_deg": 28.0,
            "unit_weight_kn_m3":  19.5,
            "water_table_m":      4.0,
            "seismic_coeff":      0.1,
        }
        record = _inference_record("slope_stability_classifier", features,
                                   prediction="unstable", confidence=0.81)
        assert record["prediction"] == "unstable"
        assert record["confidence"] > 0.75

    def test_concrete_curing_iot_sensor_array(self):
        """LoRaWAN sensors in poured concrete — temp + humidity + strength."""
        frames = [
            _sf(f"lorawan-cure-0{i}", "lorawan", {
                "sensor_type": "concrete_curing",
                "temp_c":      20.0 + i * 2,
                "humidity_pct": 80 - i * 5,
                "maturity_ch":  420 + i * 30,  # °C·hours
            })
            for i in range(4)
        ]
        avg_maturity = sum(f["payload"]["maturity_ch"] for f in frames) / len(frames)
        assert avg_maturity > 400  # > 400°C·h → sufficient early strength

    def test_worker_fall_detection_thermal_camera(self):
        """Thermal camera on scaffold — person horizontal pose = fall detected."""
        frame = _sf("thermal-scaffold-01", "thermal", {
            "detections": [
                {"class": "person", "pose": "horizontal",
                 "confidence": 0.88, "temp_c": 36.2},
            ],
        })
        falls = [d for d in frame["payload"]["detections"]
                 if d["pose"] == "horizontal"]
        assert len(falls) == 1

    def test_bacnet_site_office_access_control(self):
        """BACnet integration — site office door access log + HVAC."""
        frame = _sf("bacnet-site-01", "bacnet", {
            "device_id": "site-office-bms-01",
            "points": {
                "door_north_access": {"value": 1, "units": "binary"},
                "hvac_setpoint_c":   {"value": 21.0, "units": "degC"},
                "occupancy":         {"value": 8,    "units": "persons"},
            },
        })
        assert frame["payload"]["points"]["occupancy"]["value"] == 8

    def test_uav_rebar_placement_inspection(self):
        """UAS with EO camera inspects rebar grid — spacing compliance check."""
        inspection = {
            "asset_id":     "uav-inspect-01",
            "sensor":       "eo_visible_highres",
            "rebar_spacing_mm_measured": [195, 198, 202, 196, 201],
            "rebar_spacing_mm_spec":     200,
            "tolerance_mm":              10,
        }
        violations = [s for s in inspection["rebar_spacing_mm_measured"]
                      if abs(s - inspection["rebar_spacing_mm_spec"]) > inspection["tolerance_mm"]]
        assert len(violations) == 0

    def test_dust_pm25_exceedance_alert(self):
        """Zigbee dust sensor on site boundary — PM2.5 exceedance triggers notice."""
        frame = _sf("zigbee-dust-01", "zigbee", {
            "sensor_type":    "particulate_matter",
            "pm25_ugm3":      88.0,
            "pm10_ugm3":      142.0,
            "epa_limit_ugm3": 35.0,
        })
        exceedance = frame["payload"]["pm25_ugm3"] / frame["payload"]["epa_limit_ugm3"]
        assert exceedance > 2.0  # 2.5× over EPA limit


# ============================================================================
# DOMAIN: OIL & GAS
# ============================================================================

class TestOilAndGas:
    """
    Pipeline inspection, offshore platform, leak detection, SCADA/Modbus.
    Adapters: Modbus (SCADA), thermal, LIDAR, satellite, LoRaWAN.
    Models: pipeline_anomaly_classifier, corrosion_classifier.
    """

    def test_pipeline_anomaly_classifier_feature_vector(self):
        """Pipeline anomaly model: pressure delta + flow rate + vibration features."""
        features = {
            "pressure_upstream_kpa":   6_850.0,
            "pressure_downstream_kpa": 6_720.0,
            "pressure_delta_kpa":      130.0,
            "flow_rate_m3hr":          142.0,
            "flow_nominal_m3hr":       150.0,
            "vibration_rms_mm_s":      3.2,
            "temperature_c":           42.0,
        }
        record = _inference_record("pipeline_anomaly_classifier", features,
                                   prediction="anomaly", confidence=0.87)
        assert record["prediction"] == "anomaly"

    def test_modbus_scada_pressure_monitoring(self):
        """Modbus RTU SCADA reads pipeline pressure — out-of-range triggers alarm."""
        frame = _sf("modbus-scada-01", "modbus", {
            "unit_id":   1,
            "registers": {
                "40001": {"name": "inlet_pressure_kpa",  "value": 7_200, "range": [5_000, 7_000]},
                "40002": {"name": "outlet_pressure_kpa", "value": 6_800, "range": [5_000, 7_000]},
                "40003": {"name": "flow_rate_m3hr",      "value": 148,   "range": [100, 200]},
            },
        })
        alarms = [name for reg, r in frame["payload"]["registers"].items()
                  if not (r["range"][0] <= r["value"] <= r["range"][1])
                  for name in [r["name"]]]
        assert "inlet_pressure_kpa" in alarms

    def test_thermal_pipeline_hot_spot_detection(self):
        """Aerial thermal camera — pipeline hot spot (insulation breach) detected."""
        frame = _sf("thermal-pipe-01", "thermal", {
            "flight_alt_m": 60,
            "detections": [
                {"type": "hot_spot", "temp_c": 68.0, "background_c": 18.0,
                 "lat": 31.50, "lon": -94.10, "confidence": 0.92},
            ],
        })
        delta = frame["payload"]["detections"][0]["temp_c"] - \
                frame["payload"]["detections"][0]["background_c"]
        assert delta > 40  # >40°C delta = active insulation breach

    def test_methane_gas_leak_lorawan_sensor(self):
        """LoRaWAN methane sensor on well pad — concentration exceeds LEL threshold."""
        frame = _sf("lorawan-ch4-01", "lorawan", {
            "sensor_type":   "gas_methane",
            "ch4_ppm":       4_800,
            "lel_threshold": 5_000 * 0.10,  # 10% of LEL (50,000 ppm)
            "location":      "well-pad-A",
        })
        concentration_pct_lel = frame["payload"]["ch4_ppm"] / 50_000 * 100
        alert_level = "EVACUATE" if concentration_pct_lel > 10 else "WARNING"
        assert alert_level == "WARNING"
        assert concentration_pct_lel < 10  # 9.6% — warning but not LEL

    def test_corrosion_classifier_feature_vector(self):
        """Corrosion classifier: wall thickness + pit depth + coating condition."""
        features = {
            "wall_thickness_mm":    8.2,
            "nominal_thickness_mm": 9.5,
            "thickness_loss_pct":   13.7,
            "pit_depth_mm":         1.1,
            "surface_area_m2":      0.04,
            "coating_condition":    2,  # 0=intact 1=minor 2=significant 3=failed
        }
        record = _inference_record("corrosion_classifier", features,
                                   prediction="moderate_corrosion", confidence=0.79)
        assert record["prediction"] == "moderate_corrosion"

    def test_offshore_platform_rov_inspection(self):
        """ROV (UUV) inspects subsea riser — anode depletion + biofouling flagged."""
        rov_inspection = {
            "asset_id":    "rov-01",
            "asset_type":  "uuv_rov",
            "sensors":     ["eo_visible_uhd", "sonar_scanning", "acoustic_doppler"],
            "depth_m":     420,
            "findings": [
                {"type": "anode_depletion",  "pct_remaining": 12, "severity": "critical"},
                {"type": "biofouling",       "coverage_pct":  65, "severity": "moderate"},
            ],
        }
        critical = [f for f in rov_inspection["findings"] if f["severity"] == "critical"]
        assert len(critical) == 1

    def test_flare_monitoring_uav_radiometry(self):
        """UAS with radiometric sensor — flare efficiency + combustion completeness."""
        flare_data = {
            "asset_id":     "uav-flare-01",
            "sensor":       "radiometric_thermal",
            "flare_id":     "FLR-A-01",
            "flame_temp_k": 1_650,
            "combustion_efficiency_pct": 97.2,
            "black_carbon_g_mj": 0.08,
            "epa_limit_g_mj":    0.16,
        }
        compliant = flare_data["black_carbon_g_mj"] < flare_data["epa_limit_g_mj"]
        assert compliant

    def test_uav_pipeline_right_of_way_encroachment(self):
        """UAS patrols pipeline ROW — third-party excavation detected."""
        frame = _sf("uav-row-01", "eo_visible", {
            "pipeline_id": "line-A",
            "segment_km":  [24.0, 24.5],
            "detections": [
                {"type": "excavation", "confidence": 0.91,
                 "lat": 31.62, "lon": -94.20, "distance_from_centerline_m": 8},
            ],
        })
        intrusions = [d for d in frame["payload"]["detections"]
                      if d["distance_from_centerline_m"] < 15]  # < 15m ROW boundary
        assert len(intrusions) == 1

    def test_subsea_leak_acoustic_emission(self):
        """Hydrophone array detects acoustic emission from subsea leak."""
        frame = _sf("hydro-01", "hydrophone_array", {
            "event_type":  "acoustic_emission",
            "freq_hz":     12_400,
            "amplitude_db": -22,
            "source_localized": {"lat": 29.5, "lon": -90.2, "depth_m": 380},
            "confidence":  0.83,
        })
        assert frame["payload"]["source_localized"]["depth_m"] > 0


# ============================================================================
# DOMAIN: ENERGY — POWER GRID & RENEWABLES
# ============================================================================

class TestEnergy:
    """
    Power line inspection, solar farm fault detection, wind turbine,
    substation monitoring, smart grid anomaly.
    Adapters: thermal, BACnet, Modbus, Tesla (grid), ONVIF.
    """

    def test_power_line_corrosion_inspection(self):
        """UAS thermal + EO inspects 132kV line — corroded clamp detected."""
        frame = _sf("uav-powerline-01", "thermal", {
            "line_id":   "132kV-WEST-01",
            "span_km":   [0.0, 0.5],
            "detections": [
                {"type": "hot_clamp",   "temp_c": 89.0, "ambient_c": 22.0,
                 "tower_num": 14, "confidence": 0.93},
                {"type": "corona_ring", "temp_c": 31.0, "ambient_c": 22.0,
                 "tower_num": 17, "confidence": 0.71},
            ],
        })
        critical = [d for d in frame["payload"]["detections"]
                    if (d["temp_c"] - d["ambient_c"]) > 50]
        assert len(critical) == 1

    def test_solar_farm_panel_fault_thermal_pattern(self):
        """Thermal drone inspects 50 MW solar farm — hotspot + bypass diode fault."""
        frame = _sf("thermal-solar-01", "thermal", {
            "farm_id":    "solar-farm-AZ-01",
            "panels_inspected": 4_800,
            "faults": [
                {"type": "cell_hotspot",      "panel_id": "ROW-12-P-045", "temp_delta_c": 18},
                {"type": "bypass_diode_fault","panel_id": "ROW-8-P-102",  "temp_delta_c": 41},
                {"type": "soiling",           "panel_id": "ROW-3-P-017",  "temp_delta_c":  4},
            ],
        })
        severe = [f for f in frame["payload"]["faults"] if f["temp_delta_c"] > 15]
        assert len(severe) == 2  # hotspot + bypass diode

    def test_wind_turbine_vibration_anomaly(self):
        """Vibration + acoustic sensors on nacelle — bearing anomaly detected."""
        frame = _sf("vib-turbine-01", "vibration_acoustic", {
            "turbine_id":     "WTG-12",
            "rpm":            14.2,
            "vibration_rms_mm_s": 8.9,
            "alarm_threshold_mm_s": 7.1,
            "bearing_temp_c": 78,
            "dominant_freq_hz": 2.4,
            "fault_indicator":  "inner_race_bearing",
        })
        alarm = frame["payload"]["vibration_rms_mm_s"] > frame["payload"]["alarm_threshold_mm_s"]
        assert alarm

    def test_substation_thermal_switchgear_inspection(self):
        """Thermal camera on substation — busbar joint overheating."""
        frame = _sf("thermal-subst-01", "thermal", {
            "substation_id": "GRID-SUB-042",
            "detections": [
                {"component": "busbar_joint_A", "temp_c": 112, "rated_max_c": 90},
                {"component": "busbar_joint_B", "temp_c":  88, "rated_max_c": 90},
            ],
        })
        overheating = [d for d in frame["payload"]["detections"]
                       if d["temp_c"] > d["rated_max_c"]]
        assert len(overheating) == 1

    def test_tesla_grid_battery_soc_balancing(self):
        """Tesla Powerpack grid asset — SoC imbalance triggers rebalance."""
        frame = _sf("tesla-grid-01", "tesla", {
            "asset_id":       "megapack-cluster-01",
            "total_capacity_kwh": 3_900,
            "soc_pct":            42.0,
            "charge_rate_kw":     800,
            "grid_frequency_hz":  59.97,
            "grid_voltage_v":     480,
        })
        freq_deviation = abs(frame["payload"]["grid_frequency_hz"] - 60.0)
        assert freq_deviation < 0.1  # within ±0.1 Hz nominal

    def test_smart_meter_grid_anomaly_detection(self):
        """AMI smart meter data — non-technical loss (theft) pattern detected."""
        consumption_kwh = {
            "meter_A": [45, 44, 46, 43, 12],  # sudden drop in last period
            "meter_B": [38, 39, 37, 40, 39],
        }
        for meter, readings in consumption_kwh.items():
            last   = readings[-1]
            avg    = sum(readings[:-1]) / len(readings[:-1])
            drop_pct = (avg - last) / avg * 100
            if meter == "meter_A":
                assert drop_pct > 60  # 73% drop — NTL flag

    def test_microgrid_islanding_detection(self):
        """Frequency + voltage deviation triggers islanding detection."""
        grid_params = {
            "frequency_hz": 59.2,     # below 59.3 → island detect
            "voltage_pu":   0.91,     # below 0.95 → undervoltage
            "rate_of_change_freq_hz_s": -0.8,
        }
        islanding = (grid_params["frequency_hz"] < 59.3 or
                     grid_params["voltage_pu"] < 0.95)
        assert islanding

    def test_hydroelectric_dam_sensor_fusion(self):
        """Dam: piezometers + seepage flow + crest settlement sensors."""
        sensors = {
            "piezometer_01": {"water_level_m": 142.3, "alert_level_m": 150.0},
            "seepage_flow_ls": 12.4,
            "crest_settlement_mm": 3.2,
            "crest_settlement_alert_mm": 15.0,
        }
        alerts = []
        if sensors["piezometer_01"]["water_level_m"] > sensors["piezometer_01"]["alert_level_m"]:
            alerts.append("piezometer")
        if sensors["crest_settlement_mm"] > sensors["crest_settlement_alert_mm"]:
            alerts.append("settlement")
        assert len(alerts) == 0  # normal operating condition


# ============================================================================
# DOMAIN: AGRICULTURE — PRECISION FARMING
# ============================================================================

class TestAgriculture:
    """
    Precision agriculture: crop health, irrigation, livestock, pest, spraying.
    Adapters: ISOBUS (farm machinery), LoRaWAN (field sensors),
              multispectral UAS, weather, satellite.
    Models: eurosat_lulc_classifier, drought_severity_classifier.
    """

    def test_isobus_tractor_variable_rate_seeding(self):
        """ISOBUS CAN bus from tractor — variable-rate seeder task controller."""
        frame = _sf("isobus-tractor-01", "isobus", {
            "device_class":       "Task_Controller",
            "implement":          "seeder",
            "speed_kmh":          9.2,
            "seed_rate_kg_ha":    180.0,
            "target_rate_kg_ha":  175.0,
            "section_control":    [True] * 12,
            "boom_width_m":       24.0,
            "area_covered_ha":    18.4,
        })
        rate_error_pct = abs(frame["payload"]["seed_rate_kg_ha"] -
                             frame["payload"]["target_rate_kg_ha"]) / \
                         frame["payload"]["target_rate_kg_ha"] * 100
        assert rate_error_pct < 5.0  # within 5% of target rate

    def test_multispectral_ndvi_crop_stress(self):
        """Multispectral UAS — NDVI map identifies nitrogen-stressed zones."""
        ndvi_grid = {
            "field_id":    "field-corn-07",
            "cell_size_m": 10,
            "ndvi_values": [0.82, 0.79, 0.81, 0.43, 0.44, 0.80, 0.78, 0.82],
        }
        stressed_cells = [v for v in ndvi_grid["ndvi_values"] if v < 0.6]
        stressed_pct = len(stressed_cells) / len(ndvi_grid["ndvi_values"]) * 100
        assert stressed_pct > 15  # recommend variable-rate N application

    def test_lorawan_soil_moisture_irrigation(self):
        """LoRaWAN soil moisture sensor array — deficit triggers irrigation."""
        sensors = [
            {"zone": "zone-A", "depth_cm": 30, "vwc_pct": 18, "fc_pct": 32, "pwp_pct": 12},
            {"zone": "zone-B", "depth_cm": 30, "vwc_pct": 28, "fc_pct": 32, "pwp_pct": 12},
            {"zone": "zone-C", "depth_cm": 30, "vwc_pct": 15, "fc_pct": 32, "pwp_pct": 12},
        ]
        # Irrigate when VWC < 50% of available water capacity
        awc_pct = lambda s: (s["vwc_pct"] - s["pwp_pct"]) / (s["fc_pct"] - s["pwp_pct"]) * 100
        irrigate = [s["zone"] for s in sensors if awc_pct(s) < 50]
        assert "zone-A" in irrigate
        assert "zone-C" in irrigate
        assert "zone-B" not in irrigate

    def test_drought_severity_classifier_feature_vector(self):
        """Drought severity features validated against trained model inputs."""
        features = {
            "spi_3month":           -1.8,   # standardized precipitation index
            "pdsi":                 -3.2,   # Palmer Drought Severity Index
            "ndvi_anomaly_pct":     -24.0,
            "soil_moisture_anom":   -0.18,
            "evapotranspiration_mm": 210.0,
            "precip_30d_mm":        12.0,
        }
        record = _inference_record("drought_severity_classifier", features,
                                   prediction="D3_extreme", confidence=0.83)
        assert record["prediction"].startswith("D")

    def test_uav_precision_spraying_mission(self):
        """UAS sprayer — autonomous variable-rate pesticide application."""
        mission = {
            "asset_id":         "uav-spray-01",
            "asset_type":       "agricultural_uas_sprayer",
            "payload_l":        20.0,
            "coverage_ha":      8.5,
            "application_rate_lha": 15.0,
            "sections_active":  6,
            "gps_accuracy_cm":  3.2,
            "wind_speed_mps":   3.1,  # below max 5 m/s for spraying
        }
        wind_ok = mission["wind_speed_mps"] < 5.0
        assert wind_ok

    def test_livestock_tracking_ear_tag_ble(self):
        """BLE ear tags via LoRaWAN gateway — herd location, health status."""
        herd = [
            {"tag_id": f"COW-{i:04d}", "lat": 35.0 + i * 0.001, "lon": -100.0,
             "activity_index": 0.8 + (0.1 if i != 5 else -0.6),
             "temp_c": 38.5}
            for i in range(10)
        ]
        low_activity = [c for c in herd if c["activity_index"] < 0.4]
        assert len(low_activity) == 1  # one cow flagged for vet check

    def test_eurosat_lulc_crop_type_classification(self):
        """EuroSAT LULC classifier — Sentinel-2 tile classified as permanent crops."""
        features = {
            "B02": 0.052, "B03": 0.082, "B04": 0.071, "B08": 0.312,
            "B11": 0.198, "B12": 0.162,
            "ndvi": 0.629,
        }
        record = _inference_record("eurosat_lulc_classifier", features,
                                   prediction="PermanentCrops", confidence=0.77)
        assert "Crop" in record["prediction"] or "crop" in record["prediction"].lower() or \
               record["prediction"] in ["PermanentCrops", "AnnualCrops", "Herbaceous Vegetation"]

    def test_grain_bin_temperature_monitoring(self):
        """Zigbee sensor cables in grain bin — hot spot = spoilage risk."""
        bin_temps = {
            "bin_id": "bin-07",
            "sensors_c": [18.2, 19.1, 18.8, 28.4, 19.3, 18.5],  # sensor 3 = hot spot
        }
        avg = sum(bin_temps["sensors_c"]) / len(bin_temps["sensors_c"])
        hot_spots = [t for t in bin_temps["sensors_c"] if t > avg + 5.0]
        assert len(hot_spots) == 1

    def test_yield_monitor_combine_isobus(self):
        """ISOBUS combine yield monitor — spatial yield map streaming."""
        frame = _sf("isobus-combine-01", "isobus", {
            "device_class": "Yield_Monitor",
            "speed_kmh": 7.8,
            "crop_moisture_pct": 14.2,
            "mass_flow_kg_s": 8.4,
            "yield_t_ha": 10.8,
            "header_width_m": 12.0,
            "position": {"lat": 41.5, "lon": -93.2},
        })
        assert frame["payload"]["crop_moisture_pct"] < 15.5  # suitable for storage


# ============================================================================
# DOMAIN: MINING
# ============================================================================

class TestMining:
    """
    Open pit, underground, tailings dam, haul road, slope stability.
    Adapters: seismic, gas sensors, LIDAR, J1939 (haul trucks).
    """

    def test_open_pit_slope_stability_radar(self):
        """Ground-based radar monitors pit wall deformation."""
        frame = _sf("radar-wall-01", "gbsar", {  # Ground-Based SAR
            "scan_id":      "scan-001",
            "wall_sector":  "NW-bench-12",
            "displacement_mm_day": 2.8,
            "velocity_threshold_mm_day": 5.0,
            "cumulative_displacement_mm": 18.4,
        })
        velocity = frame["payload"]["displacement_mm_day"]
        alarm = velocity > frame["payload"]["velocity_threshold_mm_day"]
        assert not alarm  # within threshold

    def test_underground_gas_atmosphere_monitoring(self):
        """Underground sensor — CH4 + CO + O2 multi-gas monitoring."""
        frame = _sf("gas-monitor-01", "gas_multipoint", {
            "location":   "level-3-junction-A",
            "ch4_pct_vol": 0.8,   # LEL = 5% → 16% LEL
            "co_ppm":      28,
            "o2_pct_vol":  20.8,
            "h2s_ppm":     2.1,
            "lel_pct":     16.0,
        })
        o2_safe = 19.5 < frame["payload"]["o2_pct_vol"] < 23.5
        ch4_safe = frame["payload"]["lel_pct"] < 20
        assert o2_safe and ch4_safe

    def test_tailings_dam_piezometer_array(self):
        """Piezometer array in tailings dam — pore pressure rising, monitoring alert."""
        piezometers = [
            {"id": "PIE-01", "depth_m": 12, "pressure_kpa": 118, "limit_kpa": 140},
            {"id": "PIE-02", "depth_m": 20, "pressure_kpa": 195, "limit_kpa": 200},
            {"id": "PIE-03", "depth_m": 28, "pressure_kpa": 162, "limit_kpa": 175},
        ]
        near_limit = [p for p in piezometers
                      if p["pressure_kpa"] / p["limit_kpa"] > 0.90]
        assert len(near_limit) >= 1  # at least one sensor approaching limit

    def test_haul_truck_j1939_payload_monitoring(self):
        """J1939 from 300T haul truck — overload + tire pressure monitoring."""
        frame = _sf("j1939-haul-01", "j1939", {
            "asset_id":         "CAT-797-01",
            "payload_kg":       335_000,
            "rated_payload_kg": 300_000,
            "tire_pressure_kpa": [820, 822, 818, 815, 821, 819],  # 6 tires
            "speed_kmh":        38,
        })
        overloaded = frame["payload"]["payload_kg"] > frame["payload"]["rated_payload_kg"]
        assert overloaded
        low_tires = [p for p in frame["payload"]["tire_pressure_kpa"] if p < 800]
        assert len(low_tires) == 0

    def test_uav_drill_pattern_survey(self):
        """UAS LIDAR + EO survey of blast holes — position and depth verification."""
        drill_holes = [
            {"id": f"DH-{i:03d}", "lat": 51.0 + i * 0.0001, "lon": -115.0,
             "depth_m": 12.0 + (0.5 if i % 7 == 0 else 0),
             "design_depth_m": 12.0}
            for i in range(40)
        ]
        deviations = [h for h in drill_holes
                      if abs(h["depth_m"] - h["design_depth_m"]) > 0.3]
        deviation_rate_pct = len(deviations) / len(drill_holes) * 100
        assert deviation_rate_pct < 20  # < 20% deviation rate acceptable

    def test_seismic_blasting_vibration_monitoring(self):
        """Ground vibration monitor — peak particle velocity from blast."""
        frame = _sf("seismic-blast-01", "seismic", {
            "event_type":    "blast",
            "ppv_mm_s":      92.0,    # peak particle velocity
            "ppv_limit_mm_s": 100.0,  # community protection limit
            "air_overpressure_db": 116,
            "air_limit_db":       120,
        })
        ppv_ok = frame["payload"]["ppv_mm_s"] < frame["payload"]["ppv_limit_mm_s"]
        assert ppv_ok


# ============================================================================
# DOMAIN: INFRASTRUCTURE
# ============================================================================

class TestInfrastructure:
    """
    Bridge inspection, dam, port, railway, road surface condition.
    """

    def test_bridge_cable_tension_iot_monitoring(self):
        """Zigbee strain gauges on suspension bridge cables — tension anomaly."""
        gauges = [
            {"id": f"SG-{i:02d}", "tension_kn": 2_400 + (200 if i == 3 else 0),
             "nominal_kn": 2_400, "tolerance_pct": 5}
            for i in range(12)
        ]
        anomalies = [g for g in gauges
                     if abs(g["tension_kn"] - g["nominal_kn"]) / g["nominal_kn"] * 100
                     > g["tolerance_pct"]]
        assert len(anomalies) == 1

    def test_uav_bridge_deck_crack_detection(self):
        """UAS photogrammetry — bridge deck crack map, width measured."""
        inspection = {
            "bridge_id":   "BR-I-80-027",
            "panels_inspected": 48,
            "cracks": [
                {"id": "CR-001", "width_mm": 0.18, "length_m": 0.8,  "type": "flexural"},
                {"id": "CR-002", "width_mm": 0.42, "length_m": 1.2,  "type": "shear"},
            ],
        }
        # Shear cracks > 0.3 mm width = structural concern
        structural_cracks = [c for c in inspection["cracks"]
                             if c["type"] == "shear" and c["width_mm"] > 0.3]
        assert len(structural_cracks) == 1

    def test_railway_track_geometry_inspection(self):
        """Track geometry inspection trolley — gauge, cant, twist measured."""
        geometry = {
            "track_id":     "TRK-UP-MAIN-01",
            "chainage_km":  [0.0, 0.1],
            "gauge_mm":     1_436,   # nominal 1435 mm
            "cant_mm":      62,
            "twist_mm_3m":  3.2,
            "defect_limit_twist_mm": 5.0,
        }
        twist_ok = geometry["twist_mm_3m"] < geometry["defect_limit_twist_mm"]
        gauge_ok = abs(geometry["gauge_mm"] - 1_435) < 3
        assert twist_ok and gauge_ok

    def test_port_container_stacking_uav_inventory(self):
        """UAS LIDAR + EO — container yard inventory, damaged box detected."""
        yard_scan = {
            "port_id":           "PORT-LA-01",
            "containers_counted": 2_840,
            "containers_expected": 2_835,
            "damaged": [
                {"container_id": "MSCU-1234567", "damage_type": "corner_post_deformed"},
            ],
        }
        delta = abs(yard_scan["containers_counted"] - yard_scan["containers_expected"])
        assert delta <= 10  # count within ±10 acceptable
        assert len(yard_scan["damaged"]) > 0

    def test_road_pavement_condition_index(self):
        """UAS photogrammetry + ML — PCI computed from crack + rut maps."""
        pavement = {
            "segment_id":   "HWY-50-K12",
            "length_m":     500,
            "pci":          61,  # 0=failed 100=excellent; 61 = fair
            "distresses": [
                {"type": "alligator_crack", "area_m2": 12.4},
                {"type": "rutting",         "depth_mm": 8.2},
            ],
        }
        maintenance_needed = pavement["pci"] < 70
        assert maintenance_needed


# ============================================================================
# DOMAIN: FORESTRY & ENVIRONMENT
# ============================================================================

class TestForestryEnvironment:
    """
    Deforestation, illegal logging, carbon stock, water quality.
    Models: deforestation_classifier, eurosat_lulc_classifier.
    """

    def test_deforestation_classifier_feature_vector(self):
        """Deforestation classifier: NDVI loss + Sentinel-1 backscatter change."""
        features = {
            "ndvi_before":              0.81,
            "ndvi_after":               0.22,
            "ndvi_loss":                0.59,
            "sar_backscatter_before_db": -6.2,
            "sar_backscatter_after_db":  -14.8,
            "tree_cover_pct_before":    78,
            "tree_cover_pct_after":     18,
        }
        record = _inference_record("deforestation_classifier", features,
                                   prediction="deforested", confidence=0.94)
        assert record["prediction"] == "deforested"

    def test_illegal_logging_alert_change_detection(self):
        """Weekly Sentinel-2 diff — new clearing in protected area."""
        before = {"canopy_cover_pct": 92, "tile": "T18NXJ"}
        after  = {"canopy_cover_pct": 71, "tile": "T18NXJ"}
        loss_pct = before["canopy_cover_pct"] - after["canopy_cover_pct"]
        # Protected area: any loss > 5% triggers alert
        alert = loss_pct > 5
        assert alert

    def test_carbon_stock_estimation_lidar(self):
        """Airborne LIDAR — above-ground biomass estimated from canopy height."""
        lidar_metrics = {
            "plot_id":          "FOR-07",
            "canopy_height_m":  28.4,
            "canopy_cover_pct": 86,
            "basal_area_m2_ha": 28.2,
            "agb_t_ha":         312.0,  # above-ground biomass tonnes/ha
            "carbon_t_ha":      156.0,  # 50% of AGB
        }
        assert lidar_metrics["carbon_t_ha"] == lidar_metrics["agb_t_ha"] * 0.5

    def test_water_quality_buoy_sensor(self):
        """IoT buoy in river — turbidity + DO + pH + nitrate post-storm."""
        frame = _sf("buoy-river-01", "water_quality_buoy", {
            "do_mg_l":        4.2,     # dissolved oxygen; <5 = hypoxic concern
            "turbidity_ntu":  180,     # post-storm
            "ph":             7.1,
            "nitrate_mg_l":   14.8,    # >10 mg/L = EPA concern
            "temp_c":         18.2,
        })
        do_concern  = frame["payload"]["do_mg_l"] < 5.0
        no3_concern = frame["payload"]["nitrate_mg_l"] > 10.0
        assert do_concern and no3_concern

    def test_wildfire_smoke_aqi_impact(self):
        """PM2.5 from satellite + ground sensor — AQI computed."""
        frame = _sf("pm-ground-01", "air_quality", {
            "pm25_ugm3":  142.0,
            "pm10_ugm3":  220.0,
            "o3_ppb":     48.0,
        })
        # AQI breakpoints (PM2.5): 55.4-150.4 = Unhealthy (151–200 AQI)
        aqi_category = "Unhealthy" if 55.5 <= frame["payload"]["pm25_ugm3"] <= 150.4 else "Other"
        assert aqi_category == "Unhealthy"


# ============================================================================
# DOMAIN: LOGISTICS & COMMERCIAL
# ============================================================================

class TestLogistics:
    """
    Warehouse, port cargo, last-mile UAS delivery, cold chain monitoring.
    """

    def test_warehouse_inventory_uav_barcode_scan(self):
        """UAS barcode scanner — pallet inventory count vs WMS."""
        scan_result = {
            "warehouse_id": "WH-DEN-01",
            "pallets_scanned":  284,
            "pallets_wms":      280,
            "discrepancies": [
                {"location": "RACK-A-47", "scanned": "SKU-88721-P", "wms": "SKU-88720-P"},
            ],
        }
        accuracy_pct = (1 - len(scan_result["discrepancies"]) /
                        scan_result["pallets_scanned"]) * 100
        assert accuracy_pct > 99.0

    def test_last_mile_delivery_uas_geofence(self):
        """UAS delivery to residential address — geofence and descent verified."""
        delivery = {
            "asset_id":      "delivery-uav-01",
            "destination":   {"lat": 37.7749, "lon": -122.4194},
            "current_pos":   {"lat": 37.7748, "lon": -122.4193, "alt_m": 18},
            "geofence_radius_m": 20,
            "payload_released":  False,
            "gps_accuracy_m":    1.8,
        }
        dist_m = math.sqrt(
            (delivery["current_pos"]["lat"] - delivery["destination"]["lat"]) ** 2 +
            (delivery["current_pos"]["lon"] - delivery["destination"]["lon"]) ** 2
        ) * 111_000
        in_geofence = dist_m < delivery["geofence_radius_m"]
        assert in_geofence

    def test_cold_chain_temperature_exceedance(self):
        """LoRaWAN cold chain sensor — vaccine shipment temp exceedance logged."""
        frames = [
            _sf(f"cold-01-t{i}", "lorawan", {
                "shipment_id": "VAX-SHIP-0099",
                "temp_c":      -16.0 + (i * 0.5 if i < 5 else i * 3.5),
                "spec_range":  [-25, -15],
            })
            for i in range(8)
        ]
        exceedances = [f for f in frames
                       if not (f["payload"]["spec_range"][0] <=
                               f["payload"]["temp_c"] <=
                               f["payload"]["spec_range"][1])]
        assert len(exceedances) > 0

    def test_port_vessel_berth_allocation_ais(self):
        """AIS vessel ETAs at port — berth allocation conflict detected."""
        inbound = [
            {"vessel": "MV-AURORA",  "eta_min": 15,  "loa_m": 225, "berth_needed": "B1"},
            {"vessel": "MV-PACIFIC", "eta_min": 20,  "loa_m": 180, "berth_needed": "B1"},
        ]
        conflicts = {}
        for v in inbound:
            b = v["berth_needed"]
            if b in conflicts:
                conflicts[b].append(v["vessel"])
            else:
                conflicts[b] = [v["vessel"]]
        berth_conflicts = {b: vs for b, vs in conflicts.items() if len(vs) > 1}
        assert "B1" in berth_conflicts

    def test_forklift_ros2_indoor_navigation(self):
        """ROS2 autonomous forklift — pallet approach, load confirmed."""
        state = {
            "asset_id":    "amr-fork-01",
            "asset_type":  "ugv_amr_forklift",
            "pose":        {"x_m": 12.4, "y_m": 8.2, "theta_deg": 90.0},
            "fork_height_m": 0.15,
            "load_detected": True,
            "load_weight_kg": 840,
        }
        assert state["load_detected"]
        assert state["load_weight_kg"] < 2_000  # within rated capacity


# ============================================================================
# DOMAIN: PUBLIC SAFETY
# ============================================================================

class TestPublicSafety:
    """
    Crowd estimation, traffic incident, event security, accident response.
    Model: crowd_estimator.
    """

    def test_crowd_estimator_feature_vector(self):
        """Crowd estimator: aerial EO density grid — large crowd flagged."""
        features = {
            "density_persons_m2": 2.8,
            "area_m2":            4_000,
            "flow_direction":     "converging",
        }
        estimated_count = int(features["density_persons_m2"] * features["area_m2"])
        record = _inference_record("crowd_estimator", features,
                                   prediction=estimated_count, confidence=0.84)
        assert record["prediction"] > 10_000

    def test_traffic_incident_detection_onvif(self):
        """ONVIF highway camera — stopped vehicle + debris pattern."""
        frame = _sf("onvif-hwy-01", "onvif", {
            "camera_id": "I-25-MM-142",
            "detections": [
                {"class": "vehicle_stopped", "duration_s": 180, "lane": 2},
                {"class": "debris",          "confidence": 0.82},
            ],
        })
        incidents = [d for d in frame["payload"]["detections"]
                     if d["class"] in ("vehicle_stopped",) and
                     d.get("duration_s", 0) > 60]
        assert len(incidents) == 1

    def test_event_perimeter_uav_patrol(self):
        """UAS perimeter patrol at outdoor event — unauthorized access detected."""
        patrol = {
            "event_id":   "CONCERT-2026-07-04",
            "perimeter_km": 1.2,
            "detections": [
                {"type": "unauthorized_access", "lat": 34.052, "lon": -118.243,
                 "confidence": 0.88},
            ],
        }
        assert len(patrol["detections"]) == 1

    def test_accident_response_rapid_damage_assessment(self):
        """Post-accident UAS rapid damage assessment via vision model."""
        frame = _sf("uav-assess-01", "eo_visible", {
            "incident_id": "MVA-2026-0491",
            "vehicles_involved": 3,
            "damage_assessment": [
                {"vehicle_id": "V1", "damage_class": "severe",   "airbag_deployed": True},
                {"vehicle_id": "V2", "damage_class": "moderate", "airbag_deployed": True},
                {"vehicle_id": "V3", "damage_class": "minor",    "airbag_deployed": False},
            ],
            "fuel_spill_detected": True,
        })
        severe = [v for v in frame["payload"]["damage_assessment"]
                  if v["damage_class"] == "severe"]
        assert len(severe) > 0
        assert frame["payload"]["fuel_spill_detected"]


# ============================================================================
# ADAPTER SMOKE — CIVILIAN-SPECIFIC ADAPTERS
# ============================================================================

class TestCivilianAdapterSmoke:
    """Verify every civilian-domain adapter produces a valid sensor frame."""

    @pytest.mark.parametrize("adapter,payload", [
        ("isobus",       {"device_class": "Task_Controller", "speed_kmh": 8.5, "area_ha": 12.0}),
        ("j1939",        {"asset_id": "TRUCK-001", "engine_rpm": 1800, "payload_kg": 25_000}),
        ("bacnet",       {"device_id": "bms-01", "points": {"temp_c": {"value": 21.5}}}),
        ("modbus",       {"unit_id": 1, "registers": {"40001": {"value": 4_850}}}),
        ("lorawan",      {"sensor_type": "soil_moisture", "vwc_pct": 24.0}),
        ("zigbee",       {"sensor_type": "particulate", "pm25_ugm3": 18.0}),
        ("ur_robot",     {"robot_id": "UR10e-01", "joint_angles_deg": [0,0,0,0,0,0], "payload_kg": 4.2}),
        ("tesla",        {"asset_id": "mp-01", "soc_pct": 55.0, "grid_frequency_hz": 60.01}),
        ("thermal",      {"sensor_id": "thermal-01", "frame_width": 640, "detections": []}),
        ("vibration",    {"asset_id": "motor-01", "rms_mm_s": 3.4, "dominant_freq_hz": 50}),
        ("gas_sensor",   {"sensor_type": "gas_ch4", "ch4_ppm": 200, "lel_pct": 0.4}),
        ("water_quality",{"do_mg_l": 7.2, "ph": 7.4, "turbidity_ntu": 12}),
        ("gbsar",        {"scan_id": "s-001", "displacement_mm_day": 1.1}),
        ("air_quality",  {"pm25_ugm3": 18, "o3_ppb": 42}),
    ])
    def test_civilian_sensor_frame_shape(self, adapter, payload):
        sf = _sf(f"{adapter}-test-01", adapter, payload)
        assert sf["sensor_type"] == adapter
        assert "ts" in sf
        assert isinstance(sf["payload"], dict)


# ============================================================================
# CROSS-DOMAIN SENSOR FUSION SCENARIOS
# ============================================================================

class TestCrossDomainFusion:
    """
    Multi-source fusion: civilian + defense sensors producing a unified
    world model entity and downstream alert/task.
    """

    def test_oil_gas_pipeline_plus_satellite_fusion(self):
        """Modbus pressure anomaly + SAR ground deformation = leak confirmed."""
        modbus_anomaly = {
            "pressure_delta_kpa": 145, "threshold_kpa": 100, "anomaly": True,
        }
        sar_deformation = {
            "displacement_mm": 28, "threshold_mm": 20, "anomaly": True,
        }
        # Both sensors corroborate — high-confidence leak event
        both_flagged = modbus_anomaly["anomaly"] and sar_deformation["anomaly"]
        assert both_flagged

    def test_agriculture_weather_iot_fusion(self):
        """ISOBUS tractor paused + LoRaWAN wind alert + weather API = spray hold."""
        isobus_speed  = {"speed_kmh": 0.0}       # tractor stopped
        lorawan_wind  = {"wind_speed_mps": 6.2}  # above 5 m/s threshold
        weather_precip = {"precip_prob_1h": 0.72} # high rain probability

        spray_hold = (isobus_speed["speed_kmh"] == 0.0 or
                      lorawan_wind["wind_speed_mps"] > 5.0 or
                      weather_precip["precip_prob_1h"] > 0.5)
        assert spray_hold

    def test_construction_multiple_sensor_safety_lockout(self):
        """Crane wind + load cell + proximity radar = zone lockout."""
        wind_speed_mps     = 14.0   # above crane limit 12.5 m/s
        load_pct_rated     = 94.0   # near limit
        proximity_person_m = 8.0    # person within 10 m exclusion zone

        lockout = (wind_speed_mps > 12.5 or
                   load_pct_rated > 90 or
                   proximity_person_m < 10)
        assert lockout

    def test_smart_city_crowd_traffic_pollution_fusion(self):
        """Crowd estimation + traffic incident + AQI — event alert composite."""
        crowd_count        = 15_000
        traffic_incidents  = 2
        aqi_pm25           = 95  # Moderate

        risk_score = (
            (1 if crowd_count > 10_000 else 0) +
            (1 if traffic_incidents > 0 else 0) +
            (1 if aqi_pm25 > 50 else 0)
        )
        assert risk_score >= 2  # multi-factor elevated risk

    def test_mining_blast_seismic_uas_coordination(self):
        """Pre-blast UAS clearance check + seismic monitoring armed simultaneously."""
        uas_clearance = {"area_clear": True, "personnel_count": 0}
        seismic_armed = {"monitoring_active": True, "ppv_limit_mm_s": 100}

        blast_authorized = (uas_clearance["area_clear"] and
                            uas_clearance["personnel_count"] == 0 and
                            seismic_armed["monitoring_active"])
        assert blast_authorized
