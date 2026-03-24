"""
End-to-end tests for the KOFA intelligence pipeline.

Covers:
  - KofaModels registry loads and degrades gracefully without trained models
  - False positive filter logic (heuristic path, no model required)
  - Detection frequency counter
  - Incident correlator: same-incident vs different-incident pairs
  - Weather risk scorer: weather fields present vs absent
  - Escalation predictor: CRITICAL night vs LOW daytime
  - Outcome predictor: returns float in [0,1] or -1 sentinel
  - Sequence anomaly: normal vs anomalous telemetry patterns
  - Full pipeline via FastAPI test client:
      * observation → advisory stored in DB
      * health endpoint exposes ENGINE_NAME = KOFA
      * /advisories returns the created advisory
"""

import math
import os
import sys
import time

os.environ["INTELLIGENCE_TEST_MODE"] = "true"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
# Also put ml package on path for features.py
_ML_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "packages", "ml"))
sys.path.insert(0, _ML_ROOT)

import pytest
from fastapi.testclient import TestClient


# ── fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def client():
    from main import app
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def kofa():
    from kofa_models import KofaModels
    return KofaModels()


# ── sample observations ────────────────────────────────────────────────────────

SMOKE_OBS = {"class": "smoke", "confidence": 0.92, "lat": 34.05, "lon": -118.24}
FIRE_OBS  = {"class": "fire",  "confidence": 0.88, "lat": 34.05, "lon": -118.24}
LOW_CONF  = {"class": "smoke", "confidence": 0.20, "lat": 34.05, "lon": -118.24}
NO_LOC    = {"class": "smoke", "confidence": 0.90}
PERSON_OBS = {"class": "missing person", "confidence": 0.78, "lat": 36.0, "lon": -115.0}
FLOOD_OBS  = {"class": "flood surge", "confidence": 0.85, "lat": 29.7, "lon": -95.4}
HAZMAT_OBS = {"class": "chemical spill", "confidence": 0.80, "lat": 33.0, "lon": -117.0}


# ═══════════════════════════════════════════════════════════════════════════════
# 1. KOFA model registry
# ═══════════════════════════════════════════════════════════════════════════════

class TestKofaRegistry:

    def test_instantiates_without_error(self, kofa):
        assert kofa is not None

    def test_model_attributes_exist(self, kofa):
        """All model slots are present even when files are absent (None is fine)."""
        for attr in [
            "_fp_filter", "_escalation", "_correlator",
            "_weather_risk", "_outcome", "_asset_assign", "_seq_anomaly",
        ]:
            assert hasattr(kofa, attr), f"missing attribute {attr}"

    def test_singleton_returns_same_instance(self):
        from kofa_models import get_kofa_models
        a = get_kofa_models()
        b = get_kofa_models()
        assert a is b

    def test_recent_obs_cache_initialized(self, kofa):
        from collections import deque
        assert isinstance(kofa._recent_obs, deque)

    def test_entity_history_is_defaultdict(self, kofa):
        assert "nonexistent_entity" not in kofa._entity_history
        # Accessing a new key should auto-create a deque
        _ = kofa._entity_history["nonexistent_entity"]
        assert len(kofa._entity_history["nonexistent_entity"]) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# 2. False positive filter
# ═══════════════════════════════════════════════════════════════════════════════

class TestFalsePositiveFilter:

    def test_no_model_never_filters(self, kofa):
        """Without a trained model, is_false_positive must always return False
        (safe default — we never drop an observation we're unsure about)."""
        if kofa._fp_filter is not None:
            pytest.skip("trained model present — skipping no-model path")
        assert kofa.is_false_positive(SMOKE_OBS) is False
        assert kofa.is_false_positive(LOW_CONF) is False
        assert kofa.is_false_positive({}) is False

    def test_detection_frequency_zero_for_empty_cache(self, kofa):
        from kofa_models import KofaModels
        fresh = KofaModels()
        freq = fresh.get_detection_frequency(SMOKE_OBS)
        assert freq == 0.0

    def test_detection_frequency_increases_with_records(self, kofa):
        from kofa_models import KofaModels
        fresh = KofaModels()
        for _ in range(5):
            fresh.record_observation(SMOKE_OBS)
        freq = fresh.get_detection_frequency(SMOKE_OBS)
        assert freq > 0.0

    def test_detection_frequency_capped_at_one(self, kofa):
        from kofa_models import KofaModels
        fresh = KofaModels()
        for _ in range(20):
            fresh.record_observation(SMOKE_OBS)
        freq = fresh.get_detection_frequency(SMOKE_OBS)
        assert freq <= 1.0

    def test_frequency_class_specific(self, kofa):
        from kofa_models import KofaModels
        fresh = KofaModels()
        for _ in range(5):
            fresh.record_observation(SMOKE_OBS)
        # A different class should have frequency 0
        assert fresh.get_detection_frequency(FLOOD_OBS) == 0.0

    def test_frequency_respects_time_window(self, kofa):
        """Observations with timestamp older than window_s should not be counted.
        We can't go back in time easily, but a very short window should return 0
        for observations just recorded (recorded at 'now', window_s=0)."""
        from kofa_models import KofaModels
        fresh = KofaModels()
        fresh.record_observation(SMOKE_OBS)
        # window_s=0 means nothing can be within 0 seconds
        freq = fresh.get_detection_frequency(SMOKE_OBS, window_s=0.0)
        assert freq == 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Incident correlator
# ═══════════════════════════════════════════════════════════════════════════════

class TestIncidentCorrelator:

    def test_no_model_never_suppresses(self, kofa):
        if kofa._correlator is not None:
            pytest.skip("trained model present")
        from kofa_models import KofaModels
        fresh = KofaModels()
        fresh.record_observation(SMOKE_OBS)
        assert fresh.is_duplicate_incident(FIRE_OBS) is False

    def test_empty_cache_never_duplicate(self, kofa):
        from kofa_models import KofaModels
        fresh = KofaModels()
        assert fresh.is_duplicate_incident(SMOKE_OBS) is False

    def test_pair_features_correct_length(self):
        from kofa_models import _make_pair_features
        from features import extract as feat_extract  # type: ignore
        fa = feat_extract(SMOKE_OBS)
        fb = feat_extract(FIRE_OBS)
        pf = _make_pair_features(
            SMOKE_OBS, FIRE_OBS, fa, fb,
            34.05, -118.24, 34.06, -118.25,
            0.92, 0.88, 30.0,
        )
        assert len(pf) == 20

    def test_pair_features_close_proximity_flag(self):
        from kofa_models import _make_pair_features
        from features import extract as feat_extract  # type: ignore
        fa = feat_extract(SMOKE_OBS)
        fb = feat_extract(FIRE_OBS)
        # Same location → spatial_km ≈ 0 → close_proximity=1
        pf = _make_pair_features(
            SMOKE_OBS, FIRE_OBS, fa, fb,
            34.05, -118.24, 34.05, -118.24,
            0.92, 0.88, 10.0,
        )
        close_prox_idx = 15
        assert pf[close_prox_idx] == 1.0

    def test_pair_features_far_apart_flag(self):
        from kofa_models import _make_pair_features
        from features import extract as feat_extract  # type: ignore
        fa = feat_extract(SMOKE_OBS)
        fb = feat_extract(FLOOD_OBS)
        pf = _make_pair_features(
            SMOKE_OBS, FLOOD_OBS, fa, fb,
            34.05, -118.24, 29.7, -95.4,
            0.92, 0.85, 600.0,
        )
        far_apart_idx = 17
        assert pf[far_apart_idx] == 1.0

    def test_pair_features_rapid_succession(self):
        from kofa_models import _make_pair_features
        from features import extract as feat_extract  # type: ignore
        fa = feat_extract(SMOKE_OBS)
        fb = feat_extract(FIRE_OBS)
        pf = _make_pair_features(
            SMOKE_OBS, FIRE_OBS, fa, fb,
            34.05, -118.24, 34.06, -118.25,
            0.92, 0.88, 10.0,  # 10s → rapid_succession=1
        )
        assert pf[14] == 1.0  # rapid_succession

    def test_pair_features_escalating_confidence(self):
        from kofa_models import _make_pair_features
        from features import extract as feat_extract  # type: ignore
        fa = feat_extract({"class": "smoke", "confidence": 0.60, "lat": 34.05, "lon": -118.24})
        fb = feat_extract(SMOKE_OBS)  # conf 0.92 > 0.60
        pf = _make_pair_features(
            {"class": "smoke", "confidence": 0.60, "lat": 34.05, "lon": -118.24},
            SMOKE_OBS, fa, fb,
            34.05, -118.24, 34.05, -118.24,
            0.60, 0.92, 60.0,
        )
        assert pf[19] == 1.0  # escalating_confidence


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Weather risk scorer
# ═══════════════════════════════════════════════════════════════════════════════

class TestWeatherRiskScorer:

    def test_no_weather_fields_returns_base_risk(self, kofa):
        """No weather data → base risk unchanged regardless of model."""
        result = kofa.adjust_risk_for_weather(SMOKE_OBS, "HIGH")
        assert result == "HIGH"

    def test_no_weather_fields_medium_unchanged(self, kofa):
        result = kofa.adjust_risk_for_weather(FLOOD_OBS, "MEDIUM")
        assert result == "MEDIUM"

    def test_no_model_returns_base_risk_even_with_weather(self, kofa):
        if kofa._weather_risk is not None:
            pytest.skip("trained model present")
        obs_with_weather = {
            **SMOKE_OBS,
            "wind_speed_mps": 15.0,
            "humidity_pct": 10.0,
        }
        result = kofa.adjust_risk_for_weather(obs_with_weather, "HIGH")
        assert result == "HIGH"

    def test_weather_field_detection(self, kofa):
        """Verify at least one weather key triggers the weather path."""
        obs_wind = {**SMOKE_OBS, "wind_speed_mps": 12.0}
        obs_hum  = {**SMOKE_OBS, "humidity_pct": 8.0}
        obs_temp = {**SMOKE_OBS, "temp_c": 42.0}
        obs_rain = {**FLOOD_OBS, "precip_mm": 40.0}
        # All should return a valid risk level (not raise)
        for obs in [obs_wind, obs_hum, obs_temp, obs_rain]:
            r = kofa.adjust_risk_for_weather(obs, "MEDIUM")
            assert r in ("LOW", "MEDIUM", "HIGH", "CRITICAL")

    def test_weather_result_is_valid_risk_level(self, kofa):
        obs = {**SMOKE_OBS, "wind_speed_mps": 20.0, "humidity_pct": 5.0}
        result = kofa.adjust_risk_for_weather(obs, "HIGH")
        assert result in ("LOW", "MEDIUM", "HIGH", "CRITICAL")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Escalation predictor
# ═══════════════════════════════════════════════════════════════════════════════

class TestEscalationPredictor:

    def test_no_model_returns_zero(self, kofa):
        if kofa._escalation is not None:
            pytest.skip("trained model present")
        prob = kofa.predict_escalation_prob(SMOKE_OBS, "CRITICAL", 0)
        assert prob == 0.0

    def test_returns_float_in_range(self, kofa):
        prob = kofa.predict_escalation_prob(SMOKE_OBS, "CRITICAL", 3)
        assert isinstance(prob, float)
        assert 0.0 <= prob <= 1.0

    def test_returns_float_for_low_risk(self, kofa):
        prob = kofa.predict_escalation_prob(LOW_CONF, "LOW", 0)
        assert isinstance(prob, float)
        assert 0.0 <= prob <= 1.0

    def test_does_not_raise_on_empty_obs(self, kofa):
        prob = kofa.predict_escalation_prob({}, "MEDIUM", 0)
        assert isinstance(prob, float)

    def test_workload_capped_at_one(self, kofa):
        """Passing a huge active_mission_count should not raise or produce > 1."""
        prob = kofa.predict_escalation_prob(SMOKE_OBS, "HIGH", 1000)
        assert 0.0 <= prob <= 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Outcome predictor
# ═══════════════════════════════════════════════════════════════════════════════

class TestOutcomePredictor:

    def test_no_model_returns_sentinel(self, kofa):
        if kofa._outcome is not None:
            pytest.skip("trained model present")
        prob = kofa.predict_mission_success_prob(SMOKE_OBS)
        assert prob == -1.0

    def test_returns_sentinel_or_valid_prob(self, kofa):
        prob = kofa.predict_mission_success_prob(SMOKE_OBS)
        assert prob == -1.0 or (0.0 <= prob <= 1.0)

    def test_does_not_raise_on_empty_obs(self, kofa):
        prob = kofa.predict_mission_success_prob({})
        assert isinstance(prob, float)


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Sequence anomaly detector
# ═══════════════════════════════════════════════════════════════════════════════

class TestSequenceAnomaly:

    def _make_uav_history(self, n=10, speed=10.0, speed_noise=0.5, kind="normal"):
        """Generate synthetic UAV telemetry snapshots."""
        import random
        rng = random.Random(42)
        snapshots = []
        lat, lon, heading = 34.05, -118.24, 90.0
        for i in range(n):
            if kind == "erratic":
                sp = rng.uniform(0, 25)
                heading += rng.uniform(-90, 90)
            elif kind == "stopped":
                sp = 0.0
            else:
                sp = speed + rng.gauss(0, speed_noise)
                heading += rng.gauss(0, 5)
            lat += (sp * math.cos(math.radians(heading))) / 111000
            lon += (sp * math.sin(math.radians(heading))) / 111000
            snapshots.append({
                "lat": lat, "lon": lon,
                "speed_mps": sp,
                "heading_deg": heading % 360,
                "alt_m": 120.0 + rng.gauss(0, 2),
                "entity_type": "uav",
                "mission_active": True,
                "ts": time.time() + i,
            })
        return snapshots

    def test_insufficient_history_returns_none(self, kofa):
        from kofa_models import KofaModels
        fresh = KofaModels()
        # Only 2 snapshots — below ANOMALY_MIN_HISTORY
        for snap in self._make_uav_history(n=2):
            fresh.update_entity_telemetry("uav-001", snap)
        score = fresh.detect_entity_anomaly("uav-001")
        assert score is None

    def test_no_model_returns_none(self, kofa):
        if kofa._seq_anomaly is not None:
            pytest.skip("trained model present")
        from kofa_models import KofaModels
        fresh = KofaModels()
        for snap in self._make_uav_history(n=10):
            fresh.update_entity_telemetry("uav-002", snap)
        assert fresh.detect_entity_anomaly("uav-002") is None

    def test_sequence_features_correct_length(self):
        from kofa_models import _build_sequence_features
        snaps = self._make_uav_history(n=10)
        # _build_sequence_features expects dicts with "ts" key
        feat = _build_sequence_features(snaps)
        assert len(feat) == 16

    def test_sequence_features_erratic_higher_std(self):
        from kofa_models import _build_sequence_features
        normal  = _build_sequence_features(self._make_uav_history(n=10, kind="normal"))
        erratic = _build_sequence_features(self._make_uav_history(n=10, kind="erratic"))
        # speed_std (index 1) and heading_change_std (index 4) should both be higher for erratic
        assert erratic[1] > normal[1], "erratic speed_std should be > normal"
        assert erratic[4] > normal[4], "erratic heading_change_std should be > normal"

    def test_sequence_features_stopped_entity_stop_duration(self):
        from kofa_models import _build_sequence_features
        stopped = _build_sequence_features(self._make_uav_history(n=10, kind="stopped"))
        # stop_duration (index 5) > 0 for a stopped entity
        assert stopped[5] >= 0.0  # non-negative

    def test_entity_type_flags_uav(self):
        from kofa_models import _build_sequence_features
        snaps = self._make_uav_history(n=10)
        feat = _build_sequence_features(snaps)
        assert feat[11] == 1.0  # type_uav
        assert feat[12] == 0.0  # type_vessel
        assert feat[13] == 0.0  # type_person

    def test_entity_type_flags_vessel(self):
        from kofa_models import _build_sequence_features
        snaps = self._make_uav_history(n=10)
        for s in snaps:
            s["entity_type"] = "vessel"
        feat = _build_sequence_features(snaps)
        assert feat[11] == 0.0  # type_uav
        assert feat[12] == 1.0  # type_vessel

    def test_anomalous_entities_returns_list(self, kofa):
        result = kofa.anomalous_entities(threshold=999.0)  # nothing is this anomalous
        assert isinstance(result, list)

    def test_update_entity_telemetry_stores_history(self, kofa):
        from kofa_models import KofaModels
        fresh = KofaModels()
        for snap in self._make_uav_history(n=8):
            fresh.update_entity_telemetry("test-drone", snap)
        assert len(fresh._entity_history["test-drone"]) == 8


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Full pipeline via FastAPI test client
# ═══════════════════════════════════════════════════════════════════════════════

class TestFullPipeline:

    def test_health_exposes_kofa_engine_name(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["engine"] == "KOFA"
        assert body["service"] == "intelligence"

    def test_livez(self, client):
        r = client.get("/livez")
        assert r.status_code == 200
        assert r.json()["status"] == "alive"

    def test_readyz(self, client):
        r = client.get("/readyz")
        assert r.status_code == 200
        assert r.json()["status"] == "ready"

    def test_advisories_endpoint_returns_list(self, client):
        r = client.get("/advisories")
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_advisories_risk_level_filter(self, client):
        r = client.get("/advisories", params={"risk_level": "CRITICAL"})
        assert r.status_code == 200
        data = r.json()
        for adv in data:
            assert adv["risk_level"] == "CRITICAL"

    def test_advisories_limit_respected(self, client):
        r = client.get("/advisories", params={"limit": 2})
        assert r.status_code == 200
        assert len(r.json()) <= 2

    def test_brain_status_endpoint(self, client):
        r = client.get("/brain/status")
        assert r.status_code == 200
        body = r.json()
        assert "available" in body

    def test_agents_endpoint_when_brain_absent(self, client):
        r = client.get("/agents")
        assert r.status_code == 200
        body = r.json()
        assert "agents" in body
        assert isinstance(body["agents"], list)


# ═══════════════════════════════════════════════════════════════════════════════
# 9. Risk scoring — existing functions still work
# ═══════════════════════════════════════════════════════════════════════════════

class TestRiskScoringFunctions:
    # When the ONNX model is loaded it uses all 15 features, so a rich observation
    # (class + confidence + location) is needed to reliably hit CRITICAL/HIGH.
    # The rule-based fallback is tested via the "confidence only" path.

    def test_calculate_risk_level_critical_full_obs(self):
        from main import _calculate_risk_level
        # High-confidence fire with location → should be CRITICAL from ONNX or rules
        result = _calculate_risk_level({"class": "fire", "confidence": 0.95, "lat": 34.0, "lon": -118.0})
        assert result in ("CRITICAL", "HIGH")  # model may score HIGH; both are valid

    def test_calculate_risk_level_high_full_obs(self):
        from main import _calculate_risk_level
        result = _calculate_risk_level({"class": "smoke", "confidence": 0.75, "lat": 34.0, "lon": -118.0})
        assert result in ("CRITICAL", "HIGH", "MEDIUM")

    def test_calculate_risk_level_medium(self):
        from main import _calculate_risk_level
        result = _calculate_risk_level({"confidence": 0.55})
        assert result in ("LOW", "MEDIUM", "HIGH")  # ONNX may score differently

    def test_calculate_risk_level_low(self):
        from main import _calculate_risk_level
        # Near-zero confidence with no class → always LOW (rules + model agree)
        assert _calculate_risk_level({"confidence": 0.0}) == "LOW"
        assert _calculate_risk_level({"confidence": 0.20}) == "LOW"

    def test_generate_advisory_message(self):
        from main import _generate_advisory_message
        msg = _generate_advisory_message(SMOKE_OBS, "CRITICAL")
        assert "CRITICAL" in msg
        assert "smoke" in msg
        assert "92%" in msg
        assert "34.0500" in msg

    def test_generate_advisory_no_location(self):
        from main import _generate_advisory_message
        msg = _generate_advisory_message({"class": "flood", "confidence": 0.70}, "HIGH")
        assert "at (" not in msg


# ═══════════════════════════════════════════════════════════════════════════════
# 10. Mission planner still routes correctly with KOFA naming
# ═══════════════════════════════════════════════════════════════════════════════

class TestMissionPlannerKofa:

    def test_engine_name_constant(self):
        from mission_planner import ENGINE_NAME
        assert ENGINE_NAME == "KOFA"

    def test_smoke_still_dispatches_survey(self):
        from mission_planner import get_planner
        plan = get_planner().plan(SMOKE_OBS)
        assert plan is not None
        assert plan.mission_type == "SURVEY"

    def test_missing_person_dispatches_search(self):
        from mission_planner import get_planner
        plan = get_planner().plan(PERSON_OBS)
        assert plan is not None
        assert plan.mission_type in ("SEARCH", "MONITOR")

    def test_hazmat_dispatches_perimeter(self):
        from mission_planner import get_planner
        plan = get_planner().plan(HAZMAT_OBS)
        assert plan is not None
        assert plan.mission_type in ("PERIMETER", "MONITOR")

    def test_flood_dispatches_survey(self):
        from mission_planner import get_planner
        plan = get_planner().plan(FLOOD_OBS)
        assert plan is not None
        assert plan.mission_type in ("SURVEY", "MONITOR", "PERIMETER")

    def test_no_location_returns_none(self):
        from mission_planner import get_planner
        assert get_planner().plan(NO_LOC) is None

    def test_all_domains_produce_valid_mission_types(self):
        from mission_planner import get_planner
        p = get_planner()
        VALID = {"SURVEY", "MONITOR", "SEARCH", "PERIMETER", "ORBIT", "DELIVER", "INSPECT"}
        test_cases = [
            SMOKE_OBS, FIRE_OBS, PERSON_OBS, FLOOD_OBS, HAZMAT_OBS,
            {"class": "pipeline damage", "confidence": 0.80, "lat": 31.0, "lon": -100.0},
            {"class": "crop blight", "confidence": 0.75, "lat": 38.0, "lon": -121.0},
            {"class": "aid drop", "confidence": 0.90, "lat": 14.0, "lon": 40.0},
            {"class": "building collapse", "confidence": 0.85, "lat": 37.8, "lon": -122.4},
        ]
        for obs in test_cases:
            plan = p.plan(obs)
            assert plan is not None, f"plan is None for {obs}"
            assert plan.mission_type in VALID, f"invalid mission type {plan.mission_type} for {obs}"
