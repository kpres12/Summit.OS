"""
Tests for swarm_planner.py and sitrep.py.
"""

import os
import sys

os.environ["INTELLIGENCE_TEST_MODE"] = "true"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from fastapi.testclient import TestClient

CENTER_LAT = 34.05
CENTER_LON = -118.24
RADIUS_M = 600.0


# ── fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def client():
    from main import app

    with TestClient(app) as c:
        yield c


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Geo helpers
# ═══════════════════════════════════════════════════════════════════════════════


class TestGeoHelpers:

    def test_offset_north(self):
        from swarm_planner import _offset

        lat2, lon2 = _offset(34.0, -118.0, 0.0, 1000.0)  # 1km north
        assert lat2 > 34.0
        assert abs(lon2 - (-118.0)) < 0.001

    def test_offset_east(self):
        from swarm_planner import _offset

        lat2, lon2 = _offset(34.0, -118.0, 90.0, 1000.0)  # 1km east
        assert lon2 > -118.0
        assert abs(lat2 - 34.0) < 0.001

    def test_haversine_zero(self):
        from swarm_planner import _haversine_m

        assert _haversine_m(34.0, -118.0, 34.0, -118.0) == 0.0

    def test_haversine_known_distance(self):
        from swarm_planner import _haversine_m

        # 1 degree latitude ≈ 111 km
        d = _haversine_m(34.0, -118.0, 35.0, -118.0)
        assert 110_000 < d < 113_000

    def test_offset_roundtrip(self):
        from swarm_planner import _offset, _haversine_m

        lat2, lon2 = _offset(CENTER_LAT, CENTER_LON, 45.0, 500.0)
        dist = _haversine_m(CENTER_LAT, CENTER_LON, lat2, lon2)
        assert abs(dist - 500.0) < 5.0  # within 5m


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Lawnmower grid (SEARCH)
# ═══════════════════════════════════════════════════════════════════════════════


class TestLawnmowerGrid:

    def _make(self, n):
        from swarm_planner import _lawnmower_grid

        return _lawnmower_grid(CENTER_LAT, CENTER_LON, RADIUS_M, n, 80.0)

    def test_returns_n_sectors(self):
        for n in [2, 3, 4, 6]:
            sectors = self._make(n)
            assert len(sectors) == n, f"expected {n} sectors, got {len(sectors)}"

    def test_each_sector_has_waypoints(self):
        for sector in self._make(4):
            assert len(sector.waypoints) > 0

    def test_sector_ids_unique(self):
        sectors = self._make(4)
        ids = [s.sector_id for s in sectors]
        assert len(set(ids)) == len(ids)

    def test_swarm_ids_consistent(self):
        sectors = self._make(4)
        swarm_ids = {s.swarm_id for s in sectors}
        assert len(swarm_ids) == 1  # all same swarm

    def test_waypoints_within_radius_approx(self):
        from swarm_planner import _haversine_m

        sectors = self._make(3)
        for sec in sectors:
            for wp in sec.waypoints:
                d = _haversine_m(CENTER_LAT, CENTER_LON, wp.lat, wp.lon)
                # Lawnmower strips are rectangular, not circular — strip corners can
                # reach up to sqrt(2) * radius ≈ 1.42x. Allow 1.5x as safe margin.
                assert (
                    d <= RADIUS_M * 1.5
                ), f"waypoint {wp} is {d:.0f}m from center (limit {RADIUS_M * 1.5:.0f}m)"

    def test_strips_have_correct_alt(self):
        sectors = self._make(3)
        for sec in sectors:
            for wp in sec.waypoints:
                assert wp.alt_m == 80.0

    def test_drone_indices_sequential(self):
        sectors = self._make(4)
        indices = sorted(s.drone_index for s in sectors)
        assert indices == [0, 1, 2, 3]


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Radial sectors (SURVEY)
# ═══════════════════════════════════════════════════════════════════════════════


class TestRadialSectors:

    def _make(self, n):
        from swarm_planner import _radial_sectors

        return _radial_sectors(CENTER_LAT, CENTER_LON, RADIUS_M, n, 120.0)

    def test_returns_n_sectors(self):
        for n in [2, 3, 4, 8]:
            assert len(self._make(n)) == n

    def test_sector_angular_coverage(self):
        """Each sector covers 360/n degrees."""
        n = 4
        sectors = self._make(n)
        for sec in sectors:
            span = sec.coverage_deg_end - sec.coverage_deg_start
            assert abs(span - 360.0 / n) < 0.01

    def test_sectors_cover_full_360(self):
        n = 6
        sectors = self._make(n)
        total = sum(s.coverage_deg_end - s.coverage_deg_start for s in sectors)
        assert abs(total - 360.0) < 0.01

    def test_first_waypoint_at_center(self):
        from swarm_planner import _haversine_m

        sectors = self._make(3)
        for sec in sectors:
            first = sec.waypoints[0]
            d = _haversine_m(CENTER_LAT, CENTER_LON, first.lat, first.lon)
            assert d < 10.0, f"first waypoint should be at center, is {d:.1f}m away"

    def test_all_waypoints_within_radius(self):
        from swarm_planner import _haversine_m

        sectors = self._make(4)
        for sec in sectors:
            for wp in sec.waypoints:
                d = _haversine_m(CENTER_LAT, CENTER_LON, wp.lat, wp.lon)
                assert d <= RADIUS_M + 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Perimeter arcs (PERIMETER/MONITOR)
# ═══════════════════════════════════════════════════════════════════════════════


class TestPerimeterArcs:

    def _make(self, n):
        from swarm_planner import _perimeter_arcs

        return _perimeter_arcs(CENTER_LAT, CENTER_LON, RADIUS_M, n, 100.0)

    def test_returns_n_sectors(self):
        for n in [2, 3, 4, 6]:
            assert len(self._make(n)) == n

    def test_waypoints_at_correct_radius(self):
        from swarm_planner import _haversine_m

        sectors = self._make(4)
        for sec in sectors:
            for wp in sec.waypoints:
                d = _haversine_m(CENTER_LAT, CENTER_LON, wp.lat, wp.lon)
                # Perimeter waypoints should be close to the target radius
                assert (
                    abs(d - RADIUS_M) < RADIUS_M * 0.05
                ), f"perimeter waypoint is {d:.0f}m from center, expected {RADIUS_M}m"

    def test_arc_coverage_sums_to_360(self):
        n = 4
        sectors = self._make(n)
        total = sum(s.coverage_deg_end - s.coverage_deg_start for s in sectors)
        assert abs(total - 360.0) < 0.01


# ═══════════════════════════════════════════════════════════════════════════════
# 5. SwarmPlanner.expand
# ═══════════════════════════════════════════════════════════════════════════════


class TestSwarmPlannerExpand:

    def _base_plan(self, mission_type="SURVEY"):
        from mission_planner import MissionPlan

        return MissionPlan(
            mission_type=mission_type,
            lat=CENTER_LAT,
            lon=CENTER_LON,
            alt_m=100.0,
            priority="CRITICAL",
            asset_class="UAV",
            loiter=False,
            rationale="test observation",
            raw_observation={"class": "smoke", "confidence": 0.9},
        )

    def test_expand_survey_returns_n_plans(self):
        from swarm_planner import SwarmPlanner

        sp = SwarmPlanner()
        plans = sp.expand(self._base_plan("SURVEY"), n_assets=4)
        assert len(plans) == 4

    def test_expand_search_returns_n_plans(self):
        from swarm_planner import SwarmPlanner

        sp = SwarmPlanner()
        plans = sp.expand(self._base_plan("SEARCH"), n_assets=3)
        assert len(plans) == 3

    def test_expand_perimeter_returns_n_plans(self):
        from swarm_planner import SwarmPlanner

        sp = SwarmPlanner()
        plans = sp.expand(self._base_plan("PERIMETER"), n_assets=4)
        assert len(plans) == 4

    def test_expand_preserves_mission_type(self):
        from swarm_planner import SwarmPlanner

        sp = SwarmPlanner()
        for mt in ("SURVEY", "SEARCH", "PERIMETER", "MONITOR"):
            plans = sp.expand(self._base_plan(mt), n_assets=3)
            for p in plans:
                assert p.mission_type == mt

    def test_expand_preserves_priority(self):
        from swarm_planner import SwarmPlanner

        sp = SwarmPlanner()
        plans = sp.expand(self._base_plan("SURVEY"), n_assets=3)
        for p in plans:
            assert p.priority == "CRITICAL"

    def test_expand_injects_swarm_metadata(self):
        from swarm_planner import SwarmPlanner

        sp = SwarmPlanner()
        plans = sp.expand(self._base_plan("SURVEY"), n_assets=4)
        swarm_ids = {p.raw_observation.get("_swarm_id") for p in plans}
        assert len(swarm_ids) == 1  # all same swarm
        assert None not in swarm_ids

    def test_expand_injects_waypoints(self):
        from swarm_planner import SwarmPlanner

        sp = SwarmPlanner()
        plans = sp.expand(self._base_plan("SEARCH"), n_assets=3)
        for p in plans:
            wps = p.raw_observation.get("_waypoints", [])
            assert isinstance(wps, list)
            assert len(wps) > 0
            for wp in wps:
                assert "lat" in wp and "lon" in wp and "alt_m" in wp

    def test_rationale_contains_sector_info(self):
        from swarm_planner import SwarmPlanner

        sp = SwarmPlanner()
        plans = sp.expand(self._base_plan("SURVEY"), n_assets=3)
        for p in plans:
            assert "sector" in p.rationale.lower() or "swarm" in p.rationale.lower()

    def test_n_clamped_to_max(self):
        from swarm_planner import SwarmPlanner, MAX_SWARM

        sp = SwarmPlanner()
        plans = sp.expand(self._base_plan("SURVEY"), n_assets=999)
        assert len(plans) <= MAX_SWARM

    def test_n_clamped_to_min(self):
        from swarm_planner import SwarmPlanner, MIN_SWARM

        sp = SwarmPlanner()
        plans = sp.expand(self._base_plan("SURVEY"), n_assets=1)
        assert len(plans) >= MIN_SWARM

    def test_should_swarm_true_for_search(self):
        from swarm_planner import SwarmPlanner

        sp = SwarmPlanner()
        assert sp.should_swarm("SEARCH", 3) is True

    def test_should_swarm_true_for_survey(self):
        from swarm_planner import SwarmPlanner

        sp = SwarmPlanner()
        assert sp.should_swarm("SURVEY", 4) is True

    def test_should_swarm_false_single_asset(self):
        from swarm_planner import SwarmPlanner

        sp = SwarmPlanner()
        assert sp.should_swarm("SEARCH", 1) is False

    def test_should_swarm_false_for_deliver(self):
        from swarm_planner import SwarmPlanner

        sp = SwarmPlanner()
        assert sp.should_swarm("DELIVER", 5) is False

    def test_singleton(self):
        from swarm_planner import get_swarm_planner

        a = get_swarm_planner()
        b = get_swarm_planner()
        assert a is b


# ═══════════════════════════════════════════════════════════════════════════════
# 6. SitRepGenerator — template path
# ═══════════════════════════════════════════════════════════════════════════════


class TestSitRepGenerator:

    def _make_advisories(self, n=5, risk="CRITICAL", cls="smoke"):
        return [
            {
                "advisory_id": f"adv-{i}",
                "risk_level": risk,
                "message": f"{risk} risk: {cls} detected with 90% confidence at (34.05, -118.24)",
                "confidence": 0.9,
                "ts": "2026-03-24T12:00:00Z",
            }
            for i in range(n)
        ]

    def test_empty_advisories(self):
        from sitrep import SitRepGenerator

        gen = SitRepGenerator()
        sr = gen.from_advisories([])
        assert sr.advisory_count == 0
        assert sr.highest_risk == "LOW"
        assert sr.generated_by == "kofa-template"

    def test_basic_sitrep_structure(self):
        from sitrep import SitRepGenerator, SitRep

        gen = SitRepGenerator()
        sr = gen.from_advisories(self._make_advisories(5))
        assert isinstance(sr, SitRep)
        assert sr.advisory_count == 5
        assert sr.sitrep_id != ""
        assert sr.generated_at != ""

    def test_highest_risk_critical(self):
        from sitrep import SitRepGenerator

        gen = SitRepGenerator()
        sr = gen.from_advisories(self._make_advisories(3, "CRITICAL"))
        assert sr.highest_risk == "CRITICAL"

    def test_highest_risk_from_mix(self):
        from sitrep import SitRepGenerator

        gen = SitRepGenerator()
        advs = (
            self._make_advisories(2, "LOW", "crop survey")
            + self._make_advisories(1, "HIGH", "flood surge")
            + self._make_advisories(1, "CRITICAL", "fire")
        )
        sr = gen.from_advisories(advs)
        assert sr.highest_risk == "CRITICAL"

    def test_findings_not_empty(self):
        from sitrep import SitRepGenerator

        gen = SitRepGenerator()
        sr = gen.from_advisories(self._make_advisories(4))
        assert len(sr.findings) > 0

    def test_finding_domain_classified(self):
        from sitrep import SitRepGenerator

        gen = SitRepGenerator()
        sr = gen.from_advisories(self._make_advisories(3, "HIGH", "smoke"))
        assert any(f.domain == "fire_smoke" for f in sr.findings)

    def test_finding_count_correct(self):
        from sitrep import SitRepGenerator

        gen = SitRepGenerator()
        sr = gen.from_advisories(self._make_advisories(7))
        assert sr.findings[0].count == 7

    def test_findings_sorted_by_severity(self):
        from sitrep import SitRepGenerator, _SEV_ORDER

        gen = SitRepGenerator()
        advs = self._make_advisories(2, "LOW", "crop survey") + self._make_advisories(
            3, "CRITICAL", "fire"
        )
        sr = gen.from_advisories(advs)
        sevs = [_SEV_ORDER.get(f.max_severity, 0) for f in sr.findings]
        assert sevs == sorted(sevs, reverse=True)

    def test_summary_non_empty(self):
        from sitrep import SitRepGenerator

        gen = SitRepGenerator()
        sr = gen.from_advisories(self._make_advisories(3))
        assert len(sr.summary) > 10

    def test_recommended_action_non_empty(self):
        from sitrep import SitRepGenerator

        gen = SitRepGenerator()
        sr = gen.from_advisories(self._make_advisories(3))
        assert len(sr.recommended_action) > 10

    def test_to_dict_serializable(self):
        from sitrep import SitRepGenerator
        import json

        gen = SitRepGenerator()
        sr = gen.from_advisories(self._make_advisories(3))
        d = sr.to_dict()
        # Should serialize to JSON without error
        json.dumps(d)
        assert d["highest_risk"] == "CRITICAL"
        assert isinstance(d["findings"], list)

    def test_sitrep_ids_unique(self):
        from sitrep import SitRepGenerator

        gen = SitRepGenerator()
        ids = {
            gen.from_advisories(self._make_advisories(2)).sitrep_id for _ in range(5)
        }
        assert len(ids) == 5

    def test_singleton(self):
        from sitrep import get_sitrep_generator

        a = get_sitrep_generator()
        b = get_sitrep_generator()
        assert a is b

    def test_domain_classification(self):
        from sitrep import _classify_domain

        assert _classify_domain("smoke") == "fire_smoke"
        assert _classify_domain("missing person") == "person_sar"
        assert _classify_domain("flood surge") == "flood_water"
        assert _classify_domain("chemical spill") == "hazmat"
        assert _classify_domain("building collapse") == "structural"
        assert _classify_domain("power line damage") == "infrastructure"
        assert _classify_domain("crop blight") == "agricultural"
        assert _classify_domain("completely unknown") == "other"

    def test_multi_domain_sitrep(self):
        from sitrep import SitRepGenerator

        gen = SitRepGenerator()
        advs = (
            self._make_advisories(3, "CRITICAL", "fire")
            + self._make_advisories(2, "HIGH", "missing person")
            + self._make_advisories(1, "MEDIUM", "flood surge")
        )
        sr = gen.from_advisories(advs)
        domains = {f.domain for f in sr.findings}
        assert "fire_smoke" in domains
        assert "person_sar" in domains
        assert "flood_water" in domains


# ═══════════════════════════════════════════════════════════════════════════════
# 7. SITREP API endpoints
# ═══════════════════════════════════════════════════════════════════════════════


class TestSitRepEndpoints:

    def test_get_sitrep_returns_200(self, client):
        r = client.get("/sitrep")
        assert r.status_code == 200

    def test_get_sitrep_structure(self, client):
        r = client.get("/sitrep")
        body = r.json()
        assert "sitrep_id" in body
        assert "summary" in body
        assert "findings" in body
        assert "recommended_action" in body
        assert "highest_risk" in body
        assert "generated_by" in body

    def test_post_sitrep_returns_200(self, client):
        r = client.post("/sitrep", json={"time_window_s": 60})
        assert r.status_code == 200

    def test_post_sitrep_respects_window(self, client):
        r = client.post("/sitrep", json={"time_window_s": 120})
        body = r.json()
        assert body["time_window_s"] == 120

    def test_post_sitrep_with_risk_filter(self, client):
        r = client.post("/sitrep", json={"risk_level": "HIGH"})
        assert r.status_code == 200

    def test_get_sitrep_with_risk_level_param(self, client):
        r = client.get("/sitrep", params={"risk_level": "CRITICAL"})
        assert r.status_code == 200

    def test_sitrep_generated_by_template(self, client):
        r = client.get("/sitrep")
        body = r.json()
        # Without Ollama, always template
        assert body["generated_by"] in ("kofa-template", "kofa-llm")

    def test_sitrep_findings_is_list(self, client):
        r = client.get("/sitrep")
        assert isinstance(r.json()["findings"], list)


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Swarm status endpoint
# ═══════════════════════════════════════════════════════════════════════════════


class TestSwarmEndpoint:

    def test_swarm_status_returns_200(self, client):
        r = client.get("/swarm/test-swarm-123")
        assert r.status_code == 200

    def test_swarm_status_structure(self, client):
        r = client.get("/swarm/test-swarm-abc")
        body = r.json()
        assert "swarm_id" in body
        assert "missions" in body
        assert isinstance(body["missions"], list)

    def test_swarm_status_echoes_id(self, client):
        r = client.get("/swarm/my-swarm-xyz")
        assert r.json()["swarm_id"] == "my-swarm-xyz"
