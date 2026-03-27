"""
KOFA model registry — loads all trained ONNX models and exposes inference methods.

All methods degrade gracefully: if a model isn't trained yet or onnxruntime is
absent, the method returns a safe default and logs at DEBUG level. The core
dispatch pipeline is never blocked by a missing optional model.

Models managed here:
  false_positive_filter   — noise filter before dispatch
  escalation_predictor    — will this alert go unacknowledged?
  incident_correlator     — are two detections the same event?
  weather_risk_scorer     — risk score adjusted for weather conditions
  outcome_predictor       — pre-dispatch mission success probability
  asset_assignment        — UAV capability scoring (used by tasking)
  sequence_anomaly        — anomalous entity behavior in telemetry windows
"""

import json
import logging
import math
import os
import sys
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger("kofa.models")

# ── paths ──────────────────────────────────────────────────────────────────────
_ML_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "packages", "ml")
)
_MODELS_DIR = os.path.join(_ML_ROOT, "models")

if _ML_ROOT not in sys.path:
    sys.path.insert(0, _ML_ROOT)

try:
    from features import extract as _feat_extract, FEATURE_DIM  # type: ignore

    _FEATURES_AVAILABLE = True
except ImportError:
    _FEATURES_AVAILABLE = False
    logger.debug("kofa_models: features.py not on path — all model calls will no-op")


# ── ONNX helpers ───────────────────────────────────────────────────────────────


def _load_onnx(name: str) -> Optional[Any]:
    try:
        import onnxruntime as ort  # type: ignore

        path = os.path.join(_MODELS_DIR, f"{name}.onnx")
        if os.path.exists(path):
            sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
            logger.info("KOFA: loaded %s.onnx", name)
            return sess
        logger.debug("KOFA: %s.onnx not found at %s", name, path)
    except Exception as exc:
        logger.debug("KOFA: %s.onnx load failed: %s", name, exc)
    return None


def _load_json(name: str) -> dict:
    path = os.path.join(_MODELS_DIR, f"{name}.json")
    try:
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def _infer_proba(session: Any, feat_vec: List[float], np: Any) -> float:
    """
    Run a binary classifier session and return the probability of class 1.
    Handles both (class_output,) and (class_output, proba_output) ONNX graphs.
    """
    x = np.array([feat_vec], dtype=np.float32)
    inp = session.get_inputs()[0].name
    result = session.run(None, {inp: x})
    if len(result) > 1:
        proba = result[1]
        if hasattr(proba[0], "__len__"):
            return float(proba[0][1])
        return float(proba[0])
    return float(result[0][0])


# ── KofaModels ─────────────────────────────────────────────────────────────────


class KofaModels:
    """
    Central registry for all KOFA intelligence models.
    Instantiate once at startup via get_kofa_models().
    """

    # Configurable thresholds
    FP_THRESHOLD = float(os.getenv("KOFA_FP_THRESHOLD", "0.50"))
    CORRELATE_THRESHOLD = float(os.getenv("KOFA_CORRELATE_THRESHOLD", "0.65"))
    ESCALATION_THRESHOLD = float(os.getenv("KOFA_ESCALATION_THRESHOLD", "0.55"))
    INCIDENT_WINDOW_S = int(os.getenv("KOFA_INCIDENT_WINDOW_S", "600"))
    RECENT_OBS_MAXLEN = int(os.getenv("KOFA_RECENT_OBS_MAXLEN", "200"))
    ANOMALY_MIN_HISTORY = int(os.getenv("KOFA_ANOMALY_MIN_HISTORY", "5"))

    def __init__(self) -> None:
        try:
            import numpy as np  # type: ignore

            self._np: Any = np
        except ImportError:
            self._np = None
            logger.warning("kofa_models: numpy not available — all models disabled")

        self._fp_filter = _load_onnx("false_positive_filter")
        self._escalation = _load_onnx("escalation_predictor")
        self._correlator = _load_onnx("incident_correlator")
        self._weather_risk = _load_onnx("weather_risk_scorer")
        self._outcome = _load_onnx("outcome_predictor")
        self._asset_assign = _load_onnx("asset_assignment")
        self._seq_anomaly = _load_onnx("sequence_anomaly")

        # Weather risk label map (0→LOW … 3→CRITICAL)
        _wrm = _load_json("weather_risk_scorer_feature_names")
        self._weather_labels: Dict[str, str] = _wrm.get(
            "risk_label_map",
            {"0": "LOW", "1": "MEDIUM", "2": "HIGH", "3": "CRITICAL"},
        )

        # Recent observations cache — used for incident correlation + FP frequency
        # Each entry: {ts, obs, feat, lat, lon}
        self._recent_obs: deque = deque(maxlen=self.RECENT_OBS_MAXLEN)

        # Per-entity telemetry history — used for sequence anomaly detection
        # entity_id → deque of telemetry snapshots
        self._entity_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=20))

        n = sum(
            1
            for s in [
                self._fp_filter,
                self._escalation,
                self._correlator,
                self._weather_risk,
                self._outcome,
                self._asset_assign,
                self._seq_anomaly,
            ]
            if s is not None
        )
        logger.info("KOFA model registry ready: %d/7 models loaded", n)

    # ── Observation cache ──────────────────────────────────────────────────────

    def record_observation(self, obs: dict) -> None:
        """Store observation in the rolling cache (call before is_duplicate_incident)."""
        if not _FEATURES_AVAILABLE or self._np is None:
            return
        try:
            self._recent_obs.append(
                {
                    "ts": datetime.now(timezone.utc).timestamp(),
                    "obs": obs,
                    "feat": _feat_extract(obs),
                    "lat": float(obs.get("lat") or 0),
                    "lon": float(obs.get("lon") or 0),
                }
            )
        except Exception:
            pass

    def get_detection_frequency(self, obs: dict, window_s: float = 300.0) -> float:
        """
        Return detection frequency 0–1 for this class in the last window_s seconds.
        Used as a feature for the FP filter (repeated detection = more likely real).
        """
        try:
            cls = str(obs.get("class") or "").lower()
            now = datetime.now(timezone.utc).timestamp()
            count = sum(
                1
                for r in self._recent_obs
                if (now - r["ts"]) <= window_s
                and str(r["obs"].get("class") or "").lower() == cls
            )
            return min(count / 10.0, 1.0)
        except Exception:
            return 0.0

    # ── False positive filter ──────────────────────────────────────────────────

    def is_false_positive(self, obs: dict) -> bool:
        """
        Return True if this observation looks like sensor noise and should be dropped.
        Computes detection_frequency from the recent cache — call record_observation first.
        """
        if self._fp_filter is None or not _FEATURES_AVAILABLE or self._np is None:
            return False
        try:
            freq = self.get_detection_frequency(obs)
            now = datetime.now(timezone.utc)
            h = now.hour + now.minute / 60.0

            base = _feat_extract(obs)
            feat = base + [
                freq,
                math.sin(2 * math.pi * h / 24),
                math.cos(2 * math.pi * h / 24),
            ]
            real_prob = _infer_proba(self._fp_filter, feat, self._np)
            return real_prob < self.FP_THRESHOLD
        except Exception as exc:
            logger.debug("FP filter inference error: %s", exc)
            return False

    # ── Incident correlator ────────────────────────────────────────────────────

    def is_duplicate_incident(self, obs: dict) -> bool:
        """
        Return True if obs likely belongs to an already-dispatched incident.
        Compares obs against all recent observations within INCIDENT_WINDOW_S.
        Call record_observation *after* this check so obs doesn't correlate with itself.
        """
        if self._correlator is None or not _FEATURES_AVAILABLE or self._np is None:
            return False
        try:
            now_ts = datetime.now(timezone.utc).timestamp()
            lat_a = float(obs.get("lat") or 0)
            lon_a = float(obs.get("lon") or 0)
            feat_a = _feat_extract(obs)
            conf_a = float(obs.get("confidence") or 0)

            for recent in reversed(list(self._recent_obs)):
                age = now_ts - recent["ts"]
                if age > self.INCIDENT_WINDOW_S:
                    continue

                pair_feat = _make_pair_features(
                    obs,
                    recent["obs"],
                    feat_a,
                    recent["feat"],
                    lat_a,
                    lon_a,
                    recent["lat"],
                    recent["lon"],
                    conf_a,
                    float(recent["obs"].get("confidence") or 0),
                    age,
                )
                same_prob = _infer_proba(self._correlator, pair_feat, self._np)
                if same_prob >= self.CORRELATE_THRESHOLD:
                    logger.info(
                        "KOFA correlator: duplicate suppressed "
                        "(p=%.0f%%, %.0fs ago, class=%s→%s)",
                        same_prob * 100,
                        age,
                        obs.get("class"),
                        recent["obs"].get("class"),
                    )
                    return True
            return False
        except Exception as exc:
            logger.debug("Incident correlator error: %s", exc)
            return False

    # ── Weather risk scorer ────────────────────────────────────────────────────

    def adjust_risk_for_weather(self, obs: dict, base_risk: str) -> str:
        """
        Re-score risk level using weather data embedded in obs.
        Weather fields (optional): wind_speed_mps, wind_gust_mps, humidity_pct,
                                   temp_c, precip_mm, visibility_km
        Returns base_risk unchanged if no weather data present or model unavailable.
        """
        if self._weather_risk is None or not _FEATURES_AVAILABLE or self._np is None:
            return base_risk

        _WEATHER_FIELDS = [
            "wind_speed_mps",
            "wind_gust_mps",
            "humidity_pct",
            "temp_c",
            "precip_mm",
            "visibility_km",
        ]
        if not any(obs.get(f) is not None for f in _WEATHER_FIELDS):
            return base_risk

        try:
            base = _feat_extract(obs)
            feat = base + [
                float(obs.get("wind_speed_mps") or 0) / 30.0,
                float(obs.get("wind_gust_mps") or 0) / 40.0,
                (100 - float(obs.get("humidity_pct") or 50)) / 100.0,
                (float(obs.get("temp_c") or 20) + 20) / 70.0,
                min(float(obs.get("precip_mm") or 0) / 50.0, 1.0),
                min(float(obs.get("visibility_km") or 20) / 20.0, 1.0),
            ]
            x = self._np.array([feat], dtype=self._np.float32)
            inp = self._weather_risk.get_inputs()[0].name
            out = self._weather_risk.run(None, {inp: x})
            label = self._weather_labels.get(str(int(out[0][0])), base_risk)
            if label != base_risk:
                logger.debug(
                    "KOFA weather risk: %s → %s (wind=%.1f, hum=%.0f%%)",
                    base_risk,
                    label,
                    obs.get("wind_speed_mps", 0),
                    obs.get("humidity_pct", 50),
                )
            return label
        except Exception as exc:
            logger.debug("Weather risk scorer error: %s", exc)
            return base_risk

    # ── Escalation predictor ──────────────────────────────────────────────────

    def predict_escalation_prob(
        self, obs: dict, risk_level: str, active_mission_count: int = 0
    ) -> float:
        """
        Return probability [0–1] that this alert will escalate without acknowledgement.
        Above ESCALATION_THRESHOLD, pre-escalate immediately rather than waiting for timeout.
        """
        if self._escalation is None or not _FEATURES_AVAILABLE or self._np is None:
            return 0.0
        try:
            _SEV = {"CRITICAL": 1.0, "HIGH": 0.75, "MEDIUM": 0.5, "LOW": 0.25}
            now = datetime.now(timezone.utc)
            h = now.hour + now.minute / 60.0
            dow = now.weekday()

            feat = _feat_extract(obs) + [
                _SEV.get(risk_level, 0.5),
                math.sin(2 * math.pi * h / 24),
                math.cos(2 * math.pi * h / 24),
                math.sin(2 * math.pi * dow / 7),
                math.cos(2 * math.pi * dow / 7),
                min(active_mission_count / 10.0, 1.0),
            ]
            return _infer_proba(self._escalation, feat, self._np)
        except Exception as exc:
            logger.debug("Escalation predictor error: %s", exc)
            return 0.0

    # ── Outcome predictor ─────────────────────────────────────────────────────

    def predict_mission_success_prob(self, obs: dict) -> float:
        """
        Return estimated mission success probability [0–1], or -1.0 if unavailable.
        Included in auto-dispatched mission metadata for operator situational awareness.
        """
        if self._outcome is None or not _FEATURES_AVAILABLE or self._np is None:
            return -1.0
        try:
            feat = _feat_extract(obs)
            return _infer_proba(self._outcome, feat, self._np)
        except Exception as exc:
            logger.debug("Outcome predictor error: %s", exc)
            return -1.0

    # ── Sequence anomaly ──────────────────────────────────────────────────────

    def update_entity_telemetry(self, entity_id: str, snapshot: dict) -> None:
        """
        Record a telemetry snapshot for anomaly detection.
        Expected keys (all optional): lat, lon, speed_mps, heading_deg,
                                      alt_m, entity_type, mission_active
        """
        self._entity_history[entity_id].append(
            {
                "ts": datetime.now(timezone.utc).timestamp(),
                **snapshot,
            }
        )

    def detect_entity_anomaly(self, entity_id: str) -> Optional[float]:
        """
        Return anomaly score for entity (IsolationForest convention: more negative = more anomalous).
        Returns None if insufficient history or model unavailable.
        Scores below -0.1 are suspicious; below -0.3 should generate an advisory.
        """
        if self._seq_anomaly is None or self._np is None:
            return None

        history = list(self._entity_history.get(entity_id, []))
        if len(history) < self.ANOMALY_MIN_HISTORY:
            return None

        try:
            feat = _build_sequence_features(history)
            x = self._np.array([feat], dtype=self._np.float32)
            inp = self._seq_anomaly.get_inputs()[0].name
            out = self._seq_anomaly.run(None, {inp: x})
            return float(out[0][0])
        except Exception as exc:
            logger.debug("Sequence anomaly error (entity=%s): %s", entity_id, exc)
            return None

    def anomalous_entities(self, threshold: float = -0.15) -> List[Dict]:
        """
        Return list of {entity_id, score} for all entities with anomaly score below threshold.
        Intended for the background anomaly scan loop.
        """
        results = []
        for eid in list(self._entity_history.keys()):
            score = self.detect_entity_anomaly(eid)
            if score is not None and score < threshold:
                results.append({"entity_id": eid, "anomaly_score": round(score, 4)})
        return results


# ── Feature helpers (module-level, no self) ────────────────────────────────────


def _make_pair_features(
    obs_a: dict,
    obs_b: dict,
    feat_a: List[float],
    feat_b: List[float],
    lat_a: float,
    lon_a: float,
    lat_b: float,
    lon_b: float,
    conf_a: float,
    conf_b: float,
    temporal_s: float,
) -> List[float]:
    """Build the 20-float pairwise feature vector for the incident correlator."""

    # Spatial distance (km, haversine)
    if lat_a and lon_a and lat_b and lon_b:
        dlat = math.radians(lat_b - lat_a)
        dlon = math.radians(lon_b - lon_a)
        a_ = (
            math.sin(dlat / 2) ** 2
            + math.cos(math.radians(lat_a))
            * math.cos(math.radians(lat_b))
            * math.sin(dlon / 2) ** 2
        )
        spatial_km = 6371.0 * 2 * math.atan2(math.sqrt(a_), math.sqrt(1 - a_))
        bearing = (math.degrees(math.atan2(lat_b - lat_a, lon_b - lon_a)) % 360) / 360.0
    else:
        spatial_km = 0.0
        bearing = 0.0

    cls_a = str(obs_a.get("class") or "").lower()
    cls_b = str(obs_b.get("class") or "").lower()

    _DOMAIN_MAP = {
        "fire": 0,
        "smoke": 0,
        "flame": 0,
        "wildfire": 0,
        "ember": 0,
        "flood": 1,
        "surge": 1,
        "inundation": 1,
        "tsunami": 1,
        "person": 2,
        "missing": 2,
        "survivor": 2,
        "casualty": 2,
        "hazmat": 3,
        "chemical": 3,
        "spill": 3,
        "radiation": 3,
        "collapse": 4,
        "rubble": 4,
        "earthquake": 4,
        "landslide": 4,
    }
    _COPAIRS = [
        ("fire", "smoke"),
        ("flood", "surge"),
        ("flood", "inundation"),
        ("person", "missing"),
        ("earthquake", "collapse"),
        ("chemical", "plume"),
    ]

    def _domain(c: str) -> int:
        for kw, d in _DOMAIN_MAP.items():
            if kw in c:
                return d
        return -1

    dom_a, dom_b = _domain(cls_a), _domain(cls_b)
    same_domain = 1.0 if dom_a == dom_b and dom_a != -1 else 0.0
    same_class = 1.0 if cls_a == cls_b else 0.0
    complementary = 0.0
    for ca, cb in _COPAIRS:
        if (ca in cls_a and cb in cls_b) or (cb in cls_a and ca in cls_b):
            complementary = 1.0
            break

    dot = sum(x * y for x, y in zip(feat_a, feat_b))
    l2 = math.sqrt(sum((x - y) ** 2 for x, y in zip(feat_a, feat_b)))
    both_loc = 1.0 if (lat_a and lon_a and lat_b and lon_b) else 0.0

    return [
        spatial_km,  # [0]
        float(temporal_s),  # [1]
        conf_a,  # [2]
        conf_b,  # [3]
        abs(conf_a - conf_b),  # [4]
        same_domain,  # [5]
        same_class,  # [6]
        complementary,  # [7]
        dot,  # [8]
        l2,  # [9]
        both_loc,  # [10]
        bearing,  # [11]
        0.5,  # [12] wind_alignment stub
        0.0,  # [13] sensor_same_type stub
        1.0 if temporal_s < 30 else 0.0,  # [14] rapid_succession
        1.0 if spatial_km < 0.5 else 0.0,  # [15] close_proximity
        1.0 if 0.5 <= spatial_km < 2 else 0.0,  # [16] medium_proximity
        1.0 if spatial_km > 10 else 0.0,  # [17] far_apart
        1.0 if temporal_s > 600 else 0.0,  # [18] long_time_gap
        1.0 if conf_b > conf_a else 0.0,  # [19] escalating_confidence
    ]


def _build_sequence_features(history: List[dict]) -> List[float]:
    """Build the 16-float sequence feature vector from entity telemetry history."""
    speeds = [float(h.get("speed_mps", 0)) for h in history]
    headings = [float(h.get("heading_deg", 0)) for h in history]
    alts = [float(h.get("alt_m", 0)) for h in history]
    lats = [float(h.get("lat", 0)) for h in history]
    lons = [float(h.get("lon", 0)) for h in history]
    tss = [float(h["ts"]) for h in history]

    n = len(history)

    mean_sp = sum(speeds) / n
    sp_std = math.sqrt(sum((s - mean_sp) ** 2 for s in speeds) / n)
    max_sp = max(speeds)

    h_changes = [
        min(
            abs(headings[i + 1] - headings[i]), 360 - abs(headings[i + 1] - headings[i])
        )
        for i in range(n - 1)
    ]
    mean_hc = sum(h_changes) / len(h_changes) if h_changes else 0.0
    hc_std = (
        math.sqrt(sum((c - mean_hc) ** 2 for c in h_changes) / len(h_changes))
        if h_changes
        else 0.0
    )

    stop_dur = 0.0
    for i in range(n - 1, -1, -1):
        if speeds[i] < 0.5:
            stop_dur += tss[min(i + 1, n - 1)] - tss[i]
        else:
            break

    mean_lat = sum(lats) / n
    mean_lon = sum(lons) / n
    pos_var = math.sqrt(
        sum(
            ((la - mean_lat) * 111000) ** 2 + ((lo - mean_lon) * 111000) ** 2
            for la, lo in zip(lats, lons)
        )
        / n
    )

    mean_alt = sum(alts) / n
    alt_var = math.sqrt(sum((a - mean_alt) ** 2 for a in alts) / n)
    alt_ch_mean = sum(abs(alts[i + 1] - alts[i]) for i in range(n - 1)) / max(n - 1, 1)

    gaps = [tss[i + 1] - tss[i] for i in range(n - 1)]
    gap_mean = sum(gaps) / len(gaps) if gaps else 0.0
    gap_max = max(gaps) if gaps else 0.0

    et = history[-1].get("entity_type", "").lower()
    return [
        mean_sp,
        sp_std,
        max_sp,
        mean_hc,
        hc_std,
        stop_dur,
        pos_var,
        alt_ch_mean,
        alt_var,
        gap_mean,
        gap_max,
        1.0 if ("uav" in et or "drone" in et) else 0.0,
        1.0 if ("vessel" in et or "boat" in et or "ship" in et) else 0.0,
        1.0 if ("person" in et or "survivor" in et) else 0.0,
        1.0 if ("vehicle" in et or "truck" in et or "car" in et) else 0.0,
        float(bool(history[-1].get("mission_active", False))),
    ]


# ── Singleton ──────────────────────────────────────────────────────────────────

_instance: Optional[KofaModels] = None


def get_kofa_models() -> KofaModels:
    global _instance
    if _instance is None:
        _instance = KofaModels()
    return _instance
