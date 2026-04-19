"""
KOFA — Heli.OS autonomous dispatch engine.

MissionPlanner — deterministic dispatch with trained ML upgrade path.

Primary path: keyword rule table → always works, sub-ms, no dependencies.
ML path:      trained GradientBoosting ONNX (packages/ml/models/) loaded
              when PLANNER_MODEL_PATH env var points at the .onnx file.
              Uses the same feature extractor as training (no train/serve skew).

Domains: wildfire, flood, SAR, infrastructure, hazmat, agricultural,
         wildlife, maritime, industrial, medical, security, logistics.
         NOT limited to any single use case.
"""

ENGINE_NAME = "KOFA"

import json
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("kofa.mission_planner")

# Shared feature extractor (same code used during training)
_ML_PKG = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "packages", "ml")
)
if _ML_PKG not in sys.path:
    sys.path.insert(0, _ML_PKG)

try:
    from features import extract as _feat_extract, FEATURE_DIM  # type: ignore

    _FEATURES_AVAILABLE = True
except ImportError:
    _FEATURES_AVAILABLE = False
    logger.debug("packages/ml/features.py not on path — ONNX model disabled")


@dataclass
class MissionPlan:
    mission_type: (
        str  # SURVEY | MONITOR | SEARCH | PERIMETER | ORBIT | DELIVER | INSPECT
    )
    lat: float
    lon: float
    alt_m: float  # AGL — terrain offset applied downstream by Tasking
    priority: str  # CRITICAL | HIGH | MEDIUM | LOW
    asset_class: str  # UAV | UGV | FIXED_WING
    loiter: bool
    rationale: str
    raw_observation: dict = field(default_factory=dict)


# ── Rule table ────────────────────────────────────────────────────────────────
# Evaluated in order; first match wins.
# Rules intentionally broad — the ML model handles the fine-grained splits.
_RULES: List[Dict] = [
    # Fire / smoke
    {
        "keywords": [
            "smoke",
            "fire",
            "flame",
            "wildfire",
            "ember",
            "hotspot",
            "burning",
            "blaze",
        ],
        "confidence_min": 0.55,
        "mission_type": "SURVEY",
        "alt_m": 120,
        "asset_class": "UAV",
        "loiter": True,
        "rationale_template": "{cls} detected — aerial survey to confirm extent",
    },
    # Person / survivor
    {
        "keywords": [
            "person",
            "human",
            "survivor",
            "victim",
            "pedestrian",
            "casualty",
            "injured",
            "missing",
            "stranded",
            "civilian",
            "hiker",
            "swimmer",
        ],
        "confidence_min": 0.60,
        "mission_type": "MONITOR",
        "alt_m": 60,
        "asset_class": "UAV",
        "loiter": True,
        "rationale_template": "{cls} detected — continuous monitoring for response coordination",
    },
    # Search targets (lost/downed/mayday)
    {
        "keywords": [
            "missing",
            "lost",
            "overdue",
            "distress",
            "sos",
            "mayday",
            "downed",
            "epirb",
            "plb",
        ],
        "confidence_min": 0.45,
        "mission_type": "SEARCH",
        "alt_m": 80,
        "asset_class": "UAV",
        "loiter": False,
        "rationale_template": "{cls} — systematic search pattern",
    },
    # Flood / water
    {
        "keywords": [
            "flood",
            "inundation",
            "surge",
            "submerged",
            "tsunami",
            "levee breach",
            "rising water",
        ],
        "confidence_min": 0.60,
        "mission_type": "SURVEY",
        "alt_m": 200,
        "asset_class": "FIXED_WING",
        "loiter": False,
        "rationale_template": "{cls} event — wide-area survey for extent mapping",
    },
    # Structural damage
    {
        "keywords": [
            "collapse",
            "rubble",
            "debris",
            "landslide",
            "mudslide",
            "sinkhole",
            "destroyed",
        ],
        "confidence_min": 0.55,
        "mission_type": "SEARCH",
        "alt_m": 50,
        "asset_class": "UAV",
        "loiter": False,
        "rationale_template": "{cls} — search pattern for trapped survivors",
    },
    # Hazmat / chemical
    {
        "keywords": [
            "hazmat",
            "chemical",
            "spill",
            "leak",
            "toxic",
            "gas",
            "radiation",
            "explosion",
            "contamination",
            "plume",
            "biological",
        ],
        "confidence_min": 0.50,
        "mission_type": "PERIMETER",
        "alt_m": 150,
        "asset_class": "UAV",
        "loiter": False,
        "rationale_template": "{cls} — perimeter for exclusion zone",
    },
    # Security / intrusion
    {
        "keywords": [
            "intrusion",
            "unauthorized",
            "trespass",
            "breach",
            "armed",
            "hostile",
            "perimeter violation",
        ],
        "confidence_min": 0.55,
        "mission_type": "PERIMETER",
        "alt_m": 100,
        "asset_class": "UAV",
        "loiter": True,
        "rationale_template": "{cls} — perimeter enforcement and monitoring",
    },
    # Rogue / suspicious drone → orbit
    {
        "keywords": ["unauthorized uav", "rogue drone", "suspicious drone", "uav"],
        "confidence_min": 0.60,
        "mission_type": "ORBIT",
        "alt_m": 120,
        "asset_class": "UAV",
        "loiter": True,
        "rationale_template": "{cls} — orbit and track",
    },
    # Infrastructure inspection
    {
        "keywords": [
            "power line",
            "pipeline",
            "bridge",
            "tower",
            "dam",
            "rail",
            "pylon",
            "transformer",
            "substation",
            "antenna",
            "solar",
            "turbine",
        ],
        "confidence_min": 0.55,
        "mission_type": "INSPECT",
        "alt_m": 30,
        "asset_class": "UAV",
        "loiter": False,
        "rationale_template": "{cls} — close-proximity inspection",
    },
    # Delivery / logistics
    {
        "keywords": [
            "delivery",
            "drop zone",
            "supply drop",
            "aid delivery",
            "package",
            "cargo",
            "payload",
        ],
        "confidence_min": 0.70,
        "mission_type": "DELIVER",
        "alt_m": 80,
        "asset_class": "UAV",
        "loiter": False,
        "rationale_template": "{cls} — logistics delivery mission",
    },
    # Agricultural
    {
        "keywords": [
            "crop",
            "field",
            "farm",
            "orchard",
            "irrigation",
            "pest",
            "harvest",
            "blight",
        ],
        "confidence_min": 0.55,
        "mission_type": "SURVEY",
        "alt_m": 60,
        "asset_class": "UAV",
        "loiter": False,
        "rationale_template": "{cls} — agricultural survey",
    },
]

_DEFAULT_RULE: Dict = {
    "mission_type": "SURVEY",
    "alt_m": 100,
    "asset_class": "UAV",
    "loiter": False,
    "rationale_template": "{cls} anomaly — general survey",
}

# Mission type → default alt if ML picks a type the matched rule doesn't cover
_TYPE_ALTS: Dict[str, float] = {
    "SURVEY": 100,
    "MONITOR": 60,
    "SEARCH": 80,
    "PERIMETER": 150,
    "ORBIT": 120,
    "DELIVER": 80,
    "INSPECT": 30,
}
_TYPE_ASSET: Dict[str, str] = {
    "SURVEY": "UAV",
    "MONITOR": "UAV",
    "SEARCH": "UAV",
    "PERIMETER": "UAV",
    "ORBIT": "UAV",
    "DELIVER": "UAV",
    "INSPECT": "UAV",
}
_LABEL_MAP_INV: Dict[str, int] = {
    "SURVEY": 0,
    "MONITOR": 1,
    "SEARCH": 2,
    "PERIMETER": 3,
    "ORBIT": 4,
    "DELIVER": 5,
    "INSPECT": 6,
}


def _match_rule(obs_class: str, confidence: float) -> Optional[Dict]:
    lower = obs_class.lower()
    for rule in _RULES:
        if confidence < rule["confidence_min"]:
            continue
        if any(kw in lower for kw in rule["keywords"]):
            return rule
    return None


# ── MissionPlanner ─────────────────────────────────────────────────────────────


class MissionPlanner:
    """
    Two-layer dispatch: rule table (always active) + trained ONNX (optional upgrade).

    The ML model is loaded automatically if PLANNER_MODEL_PATH is set and the
    file exists.  Default path: packages/ml/models/mission_classifier.onnx.

    As operator-approved mission data accumulates, retrain with:
        python packages/ml/train_mission_classifier.py --real-data <pg_url>
    and redeploy the new .onnx — no code change required.
    """

    _DEFAULT_MODEL = os.path.normpath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "packages",
            "ml",
            "models",
            "mission_classifier.onnx",
        )
    )

    def __init__(self, model_path: Optional[str] = None):
        self._session = None
        self._label_map: Dict[str, str] = {}

        path = model_path or os.getenv("PLANNER_MODEL_PATH", self._DEFAULT_MODEL)
        if path and os.path.exists(path) and _FEATURES_AVAILABLE:
            try:
                import onnxruntime as ort  # type: ignore

                self._session = ort.InferenceSession(
                    path, providers=["CPUExecutionProvider"]
                )
                labels_path = path.replace(".onnx", "_labels.json")
                if os.path.exists(labels_path):
                    with open(labels_path) as f:
                        self._label_map = json.load(f)  # {"0": "SURVEY", ...}
                logger.info(
                    "MissionPlanner: ML model loaded (%s)", os.path.basename(path)
                )
            except Exception as exc:
                logger.warning(
                    "MissionPlanner: ML model load failed (%s) — rules only", exc
                )
                self._session = None

    def plan(self, observation: Dict[str, Any]) -> Optional[MissionPlan]:
        obs_class = str(observation.get("class") or "unknown")
        confidence = float(observation.get("confidence") or 0.0)
        lat = observation.get("lat")
        lon = observation.get("lon")

        if lat is None or lon is None:
            return None
        lat, lon = float(lat), float(lon)

        # ML prediction — only consult model when confidence is high enough
        # to be worth dispatching; very-low-confidence unknowns always return None below
        ml_type: Optional[str] = (
            self._predict(observation)
            if (self._session and confidence >= 0.55)
            else None
        )

        # Rule matching (keyword-based, always active as safety net)
        rule = _match_rule(obs_class, confidence)

        if rule is None and ml_type is None:
            if confidence < 0.85:
                return None
            rule = _DEFAULT_RULE

        if rule is None:
            rule = _DEFAULT_RULE

        # ML overrides rule on mission type; rule supplies alt/asset/loiter defaults
        mission_type = ml_type or rule["mission_type"]
        alt_m = _TYPE_ALTS.get(mission_type, float(rule.get("alt_m", 100)))
        asset_class = rule.get("asset_class", _TYPE_ASSET.get(mission_type, "UAV"))
        loiter = rule.get("loiter", False)
        # If ML changed the mission type away from the matched rule, adjust loiter
        if ml_type and ml_type != rule["mission_type"]:
            loiter = mission_type in ("MONITOR", "ORBIT", "PERIMETER")

        rationale = rule["rationale_template"].format(cls=obs_class)
        priority = "CRITICAL" if confidence >= 0.85 else "HIGH"

        return MissionPlan(
            mission_type=mission_type,
            lat=lat,
            lon=lon,
            alt_m=alt_m,
            priority=priority,
            asset_class=asset_class,
            loiter=loiter,
            rationale=rationale,
            raw_observation=observation,
        )

    def _predict(self, observation: Dict[str, Any]) -> Optional[str]:
        if not _FEATURES_AVAILABLE:
            return None
        try:
            import numpy as np  # type: ignore

            feat = np.array([_feat_extract(observation)], dtype=np.float32)
            input_name = self._session.get_inputs()[0].name  # type: ignore
            pred = self._session.run(None, {input_name: feat})[0][0]  # type: ignore
            return self._label_map.get(str(int(pred)))
        except Exception:
            return None


# Module-level singleton
_planner: Optional[MissionPlanner] = None


def get_planner() -> MissionPlanner:
    global _planner
    if _planner is None:
        _planner = MissionPlanner()
    return _planner
