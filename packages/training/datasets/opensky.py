"""
OpenSky Network State Vector Fetcher
======================================
Pulls real aircraft state-change data from the OpenSky Network REST API.
Used to train the C2 timing predictor — state transitions (online→offline,
normal→degraded) provide realistic timing priors for the C2EventType taxonomy.

No auth required for anonymous access (rate-limited to 100 req/day).
Historical data requires registered account — we use live + synthetic priors.

Source: https://opensky-network.org/apidoc/
License: ODbL (open)
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import requests

logger = logging.getLogger(__name__)

_OPENSKY_STATES_URL = "https://opensky-network.org/api/states/all"

# C2EventType → typical timing distribution (minutes) seeded from operational doctrine.
# OpenSky live data augments these priors with real observed intervals.
_DOCTRINE_PRIORS: dict[str, dict] = {
    "COMMS_DEGRADED":       {"p25": 1,  "median": 3,  "p75": 8,  "context": "other"},
    "COMMS_RESTORED":       {"p25": 2,  "median": 5,  "p75": 15, "context": "other"},
    "ASSET_OFFLINE":        {"p25": 2,  "median": 6,  "p75": 20, "context": "other"},
    "ASSET_ONLINE":         {"p25": 1,  "median": 2,  "p75": 5,  "context": "other"},
    "BATTERY_CRITICAL":     {"p25": 1,  "median": 2,  "p75": 4,  "context": "other"},
    "BATTERY_LOW":          {"p25": 5,  "median": 10, "p75": 20, "context": "other"},
    "THREAT_IDENTIFIED":    {"p25": 1,  "median": 2,  "p75": 5,  "context": "urban_sar"},
    "THREAT_NEUTRALIZED":   {"p25": 2,  "median": 4,  "p75": 10, "context": "other"},
    "ENTITY_DETECTED":      {"p25": 1,  "median": 3,  "p75": 7,  "context": "urban_sar"},
    "ENTITY_LOST":          {"p25": 3,  "median": 8,  "p75": 20, "context": "other"},
    "GEOFENCE_BREACH":      {"p25": 1,  "median": 2,  "p75": 5,  "context": "other"},
    "GEOFENCE_CLEARED":     {"p25": 2,  "median": 5,  "p75": 12, "context": "other"},
    "SENSOR_LOSS":          {"p25": 2,  "median": 5,  "p75": 15, "context": "other"},
    "SENSOR_RESTORED":      {"p25": 3,  "median": 7,  "p75": 18, "context": "other"},
    "WEATHER_ALERT":        {"p25": 5,  "median": 15, "p75": 45, "context": "wildfire"},
    "AIRSPACE_CONFLICT":    {"p25": 1,  "median": 3,  "p75": 8,  "context": "other"},
    "MISSION_STARTED":      {"p25": 5,  "median": 15, "p75": 40, "context": "other"},
    "MISSION_COMPLETED":    {"p25": 2,  "median": 5,  "p75": 12, "context": "other"},
    "MISSION_ABORTED":      {"p25": 1,  "median": 3,  "p75": 7,  "context": "other"},
    "HANDOFF_INITIATED":    {"p25": 2,  "median": 4,  "p75": 10, "context": "disaster_response"},
    "HANDOFF_COMPLETE":     {"p25": 3,  "median": 6,  "p75": 15, "context": "disaster_response"},
    "AUTHORITY_DELEGATED":  {"p25": 3,  "median": 8,  "p75": 20, "context": "other"},
    "AUTHORITY_REVOKED":    {"p25": 1,  "median": 3,  "p75": 8,  "context": "other"},
    "NODE_DEGRADED":        {"p25": 2,  "median": 5,  "p75": 15, "context": "other"},
    "NODE_FAILED":          {"p25": 1,  "median": 3,  "p75": 8,  "context": "other"},
    "NODE_RECOVERED":       {"p25": 3,  "median": 8,  "p75": 20, "context": "other"},
    "LINK_DEGRADED":        {"p25": 2,  "median": 5,  "p75": 12, "context": "other"},
    "LINK_LOST":            {"p25": 1,  "median": 4,  "p75": 10, "context": "other"},
    "ENGAGEMENT_AUTHORIZED":{"p25": 1,  "median": 2,  "p75": 5,  "context": "other"},
    "ENGAGEMENT_DENIED":    {"p25": 1,  "median": 3,  "p75": 7,  "context": "other"},
    "ENGAGEMENT_COMPLETE":  {"p25": 2,  "median": 5,  "p75": 12, "context": "other"},
    "PEER_OBSERVATION":     {"p25": 5,  "median": 12, "p75": 30, "context": "other"},
}

_C2_CONTEXTS = ["urban_sar", "wildfire", "disaster_response", "military_ace", "border_patrol", "other"]


def _fetch_opensky_states() -> list[dict]:
    """Fetch current state vectors from OpenSky (anonymous, rate-limited)."""
    try:
        r = requests.get(_OPENSKY_STATES_URL, timeout=30)
        r.raise_for_status()
        data = r.json()
        states = data.get("states", []) or []
        logger.info("OpenSky: fetched %d state vectors", len(states))
        return states
    except Exception as e:
        logger.warning("OpenSky fetch failed: %s", e)
        return []


def _build_training_records(states: list[dict], rng: np.random.Generator) -> list[dict]:
    """
    Map OpenSky state vectors to C2 training examples.

    Each state vector: [icao24, callsign, origin_country, time_position,
                        last_contact, longitude, latitude, baro_altitude,
                        on_ground, velocity, true_track, vertical_rate,
                        sensors, geo_altitude, squawk, spi, position_source]
    """
    records = []
    for state in states:
        if not state or len(state) < 12:
            continue
        try:
            on_ground      = bool(state[8])
            velocity       = float(state[9] or 0)
            vertical_rate  = float(state[11] or 0)
            baro_alt       = float(state[7] or 0)
            last_contact   = float(state[5] or 0)
            time_position  = float(state[3] or last_contact)
            gap_secs       = max(0, last_contact - time_position)

            # Classify event type from state
            if on_ground:
                evt = "ASSET_ONLINE"
                score = 40
            elif velocity < 10:
                evt = "ASSET_OFFLINE"
                score = 60
            elif baro_alt < 300 and not on_ground:
                evt = "BATTERY_LOW"   # low altitude proxy
                score = 65
            elif abs(vertical_rate) > 5:
                evt = "AIRSPACE_CONFLICT"
                score = 55
            elif gap_secs > 60:
                evt = "COMMS_DEGRADED"
                score = 70
            else:
                evt = "PEER_OBSERVATION"
                score = 35

            # Derive timing target from gap_secs (proxy for operator response time)
            minutes_to_action = gap_secs / 60.0 + rng.normal(0, 0.5)
            minutes_to_action = max(0.5, minutes_to_action)

            context = rng.choice(_C2_CONTEXTS, p=[0.25, 0.15, 0.20, 0.10, 0.10, 0.20])
            n_obs = rng.integers(1, 6)
            urgency_tier = min(4, max(1, int(score / 25)))

            records.append({
                "event_type":      evt,
                "context":         str(context),
                "score":           score,
                "n_obs":           int(n_obs),
                "urgency_tier":    urgency_tier,
                "minutes_to_action": minutes_to_action,
            })
        except Exception:
            continue

    return records


def _generate_from_priors(rng: np.random.Generator, n_per_event: int = 200) -> list[dict]:
    """
    Generate training records from operational doctrine timing priors.
    Each event type gets n_per_event synthetic samples drawn from the
    prior distribution, varied by context and score.
    """
    records = []
    for evt, prior in _DOCTRINE_PRIORS.items():
        p25, med, p75 = prior["p25"], prior["median"], prior["p75"]
        base_ctx = prior["context"]

        for _ in range(n_per_event):
            # Sample from a log-normal approximation of the prior distribution
            log_med = np.log(max(0.5, med))
            log_std = (np.log(max(0.5, p75)) - np.log(max(0.5, p25))) / 2.7  # IQR → sigma
            minutes = float(np.exp(rng.normal(log_med, log_std)))
            minutes = max(0.5, minutes)

            # Vary context and score
            ctx   = base_ctx if rng.random() < 0.6 else rng.choice(_C2_CONTEXTS)
            score = int(rng.integers(30, 90))
            n_obs = int(rng.integers(1, 8))
            urgency_tier = min(4, max(1, int(score / 25)))

            records.append({
                "event_type":      evt,
                "context":         str(ctx),
                "score":           score,
                "n_obs":           n_obs,
                "urgency_tier":    urgency_tier,
                "minutes_to_action": minutes,
            })

    return records


def download(output_dir: str = "/tmp/heli-training-data/opensky") -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    # Start with doctrine priors (always available)
    records = _generate_from_priors(rng, n_per_event=300)
    logger.info("Generated %d doctrine-prior timing records", len(records))

    # Augment with live OpenSky data if available
    states = _fetch_opensky_states()
    if states:
        live_records = _build_training_records(states, rng)
        # Repeat live records to augment (limited live data)
        augmented = live_records * max(1, 50 // max(1, len(live_records)))
        records.extend(augmented)
        logger.info("Augmented with %d OpenSky records (×%d)", len(live_records),
                    max(1, 50 // max(1, len(live_records))))

    import pandas as pd
    df = pd.DataFrame(records)
    out_file = out / "timing_training.parquet"
    df.to_parquet(out_file, index=False)
    logger.info("Timing dataset: %d records → %s", len(df), out_file)
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="/tmp/heli-training-data/opensky")
    args = p.parse_args()
    download(args.output)
