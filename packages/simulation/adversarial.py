"""
Adversarial Scenario Generator — Heli.OS

Generates adversarial test scenarios by mutating baseline mission scenarios.
Used to stress-test the replanning engine, deconfliction, and fusion pipeline.

Mutation strategies:
  - ASSET_FAIL: randomly kill an asset mid-mission
  - SENSOR_DRIFT: inject GPS drift into entity positions
  - COMMS_LOSS: simulate link outage for a random duration
  - THREAT_INJECT: add unexpected threat entities near mission area
  - WEATHER_DEGRADE: ramp weather score from 1.0 to 0.0 over time
  - FLOOD_MQTT: burst 10x normal message rate

Fitness function: scenario complexity = failures_triggered / assets_count
"""

from __future__ import annotations

import asyncio
import copy
import logging
import random
import time
from typing import Callable, List, Optional

logger = logging.getLogger("simulation.adversarial")

ALL_STRATEGIES = [
    "ASSET_FAIL",
    "SENSOR_DRIFT",
    "COMMS_LOSS",
    "THREAT_INJECT",
    "WEATHER_DEGRADE",
    "FLOOD_MQTT",
]


class ScenarioMutator:
    """Applies random mutations to a baseline scenario dict."""

    def __init__(self, seed: int = None):
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------
    def mutate(self, baseline: dict, strategies: List[str] = None) -> dict:
        """
        Apply one or more random mutations to a copy of baseline.
        If strategies is None, a random subset (1–3) is chosen.
        """
        scenario = copy.deepcopy(baseline)
        chosen = strategies or self._rng.sample(
            ALL_STRATEGIES, k=self._rng.randint(1, 3)
        )
        for strategy in chosen:
            fn = getattr(self, f"_{strategy.lower()}", None)
            if fn is None:
                logger.warning("Unknown mutation strategy: %s", strategy)
                continue
            scenario = fn(scenario)
        scenario.setdefault("mutations_applied", []).extend(chosen)
        return scenario

    # ------------------------------------------------------------------
    def _asset_fail(self, scenario: dict) -> dict:
        """Kill a random asset mid-mission."""
        assets = scenario.get("assets", [])
        if not assets:
            return scenario
        target = self._rng.choice(assets)
        target["status"] = "FAILED"
        target["fail_at_pct"] = round(self._rng.uniform(0.1, 0.9), 2)
        logger.debug("ASSET_FAIL: %s fails at %.0f%%", target.get("asset_id", "?"), target["fail_at_pct"] * 100)
        return scenario

    def _sensor_drift(self, scenario: dict) -> dict:
        """Inject GPS drift into entity positions."""
        entities = scenario.get("entities", [])
        for ent in entities:
            drift_m = self._rng.uniform(10, 200)
            ent["gps_drift_m"] = round(drift_m, 1)
        scenario["sensor_drift_injected"] = True
        return scenario

    def _comms_loss(self, scenario: dict) -> dict:
        """Simulate a comms link outage for a random duration."""
        duration_s = self._rng.randint(15, 180)
        start_pct  = round(self._rng.uniform(0.05, 0.7), 2)
        scenario["comms_loss"] = {
            "start_pct":  start_pct,
            "duration_s": duration_s,
        }
        return scenario

    def _threat_inject(self, scenario: dict) -> dict:
        """Add unexpected threat entities near the mission area."""
        n_threats = self._rng.randint(1, 4)
        threats = []
        area = scenario.get("area", {})
        center_lat = area.get("lat", 37.0)
        center_lon = area.get("lon", -122.0)
        for i in range(n_threats):
            lat = center_lat + self._rng.uniform(-0.05, 0.05)
            lon = center_lon + self._rng.uniform(-0.05, 0.05)
            threats.append({
                "entity_id": f"threat_{int(time.time())}_{i}",
                "type":      "unknown_vehicle",
                "lat":       round(lat, 6),
                "lon":       round(lon, 6),
            })
        scenario.setdefault("threat_entities", []).extend(threats)
        return scenario

    def _weather_degrade(self, scenario: dict) -> dict:
        """Ramp weather score from 1.0 to 0.0 over the mission duration."""
        scenario["weather_degrade"] = {
            "start_score": 1.0,
            "end_score":   0.0,
            "ramp_start_pct": round(self._rng.uniform(0.0, 0.5), 2),
        }
        return scenario

    def _flood_mqtt(self, scenario: dict) -> dict:
        """Burst 10x normal MQTT message rate for a period."""
        scenario["mqtt_flood"] = {
            "multiplier":  10,
            "duration_s":  self._rng.randint(5, 30),
            "start_pct":   round(self._rng.uniform(0.1, 0.8), 2),
        }
        return scenario

    # ------------------------------------------------------------------
    def fitness(self, results: dict) -> float:
        """
        Score a scenario based on failures and replans triggered.
        Higher = more adversarial.
        fitness = failures_triggered / max(assets_count, 1)
        """
        failures  = results.get("failures_triggered", 0)
        replans   = results.get("replan_count", 0)
        assets    = max(results.get("assets_count", 1), 1)
        return (failures + 0.5 * replans) / assets


# ── Runner ────────────────────────────────────────────────────────────────────

class AdversarialRunner:
    """Run a population of adversarial scenarios and rank them by fitness."""

    def __init__(self, mutator: ScenarioMutator, run_fn: Callable):
        self.mutator = mutator
        self.run_fn  = run_fn  # async callable: (scenario) -> results dict

    async def run_population(self, baseline: dict, n: int = 20) -> List[dict]:
        """
        Generate n mutations of baseline, run each through run_fn,
        and return list of {scenario, results, fitness} dicts.
        """
        population = [self.mutator.mutate(baseline) for _ in range(n)]
        output: List[dict] = []

        for scenario in population:
            try:
                results = await self.run_fn(scenario)
            except Exception as exc:
                logger.error("run_fn failed: %s", exc)
                results = {"error": str(exc)}
            fitness = self.mutator.fitness(results)
            output.append({
                "scenario": scenario,
                "results":  results,
                "fitness":  fitness,
            })

        return output

    def top_k(self, results: List[dict], k: int = 5) -> List[dict]:
        """Return the k highest-fitness scenarios."""
        return sorted(results, key=lambda r: r.get("fitness", 0.0), reverse=True)[:k]
