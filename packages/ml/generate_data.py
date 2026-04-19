"""
Synthetic training data generator for Heli.OS ML models.

Strategy: domain-expert rules → labeled synthetic samples → trained model.
The model then generalizes beyond the exact rules (fuzzy matches, compound
classes, confidence interpolation, unseen class strings).

When real operator-approved mission data accumulates in the tasking DB,
run: python train_mission_classifier.py --real-data  to blend synthetic
with real, progressively replacing synthetic over time.

Domains covered (NOT just wildfire — Heli.OS is general-purpose):
  Disaster response: wildfire, flood, earthquake, hurricane, tornado
  Search & rescue:   wilderness SAR, maritime SAR, avalanche, urban rescue
  Infrastructure:    power grid, pipeline, bridge, dam, rail inspection
  Agricultural:      crop monitoring, pest detection, irrigation audit
  Environmental:     wildlife monitoring, poaching detection, erosion
  Industrial:        hazmat, chemical plant, oil spill, mine safety
  Commercial fleet:  delivery, inspection, survey missions
  Maritime:          vessel tracking, port security, ocean survey
  Medical:           mass casualty, triage support, emergency landing
  Security:          perimeter monitoring, intrusion detection
"""

import random
from typing import List, Tuple

import numpy as np

from features import extract, FEATURE_DIM

# ── Mission type taxonomy ─────────────────────────────────────────────────────
#  0 SURVEY   — broad area coverage, mapping, situational awareness
#  1 MONITOR  — sustained watch on a known, stationary target
#  2 SEARCH   — systematic pattern to locate unknown / lost target
#  3 PERIMETER— establish/maintain boundary around incident
#  4 ORBIT    — continuous circular orbit for persistent ISR
#  5 DELIVER  — logistics / payload delivery to a waypoint
#  6 INSPECT  — close-proximity detailed inspection of structure/asset

MISSION_LABELS = {
    0: "SURVEY",
    1: "MONITOR",
    2: "SEARCH",
    3: "PERIMETER",
    4: "ORBIT",
    5: "DELIVER",
    6: "INSPECT",
}

# ── Risk level taxonomy ───────────────────────────────────────────────────────
RISK_LABELS = {0: "LOW", 1: "MEDIUM", 2: "HIGH", 3: "CRITICAL"}

# ── Scenario catalogue ────────────────────────────────────────────────────────
# Each entry: (class_strings, mission_type_int, base_risk, min_conf, max_conf)
# base_risk: (low_weight, med_weight, high_weight, crit_weight) at conf midpoint

_SCENARIOS = [
    # ── Wildfire / fire ──────────────────────────────────────────────────────
    (
        ["smoke", "smoke plume", "smoke column", "wildfire smoke", "fire smoke"],
        0,
        (0, 0, 1, 3),
        0.5,
        1.0,
    ),
    (
        ["fire", "active fire", "fire front", "open flame", "wildfire"],
        0,
        (0, 0, 0, 4),
        0.6,
        1.0,
    ),
    (["ember shower", "ember cast", "fire ember"], 0, (0, 0, 1, 3), 0.55, 1.0),
    (["hotspot", "thermal hotspot", "heat anomaly"], 4, (0, 1, 2, 1), 0.5, 1.0),
    (
        ["burning vehicle", "burning structure", "burning building"],
        3,
        (0, 0, 1, 3),
        0.6,
        1.0,
    ),
    # ── Flood / water ────────────────────────────────────────────────────────
    (["flood", "flash flood", "flooding"], 0, (0, 0, 1, 3), 0.5, 1.0),
    (["rising water", "storm surge", "coastal inundation"], 3, (0, 0, 1, 3), 0.55, 1.0),
    (
        ["submerged vehicle", "submerged road", "submerged structure"],
        0,
        (0, 1, 2, 1),
        0.5,
        0.95,
    ),
    (["levee breach", "dam breach", "dam failure"], 3, (0, 0, 0, 4), 0.5, 1.0),
    (["tsunami wave", "tsunami surge"], 3, (0, 0, 0, 4), 0.6, 1.0),
    # ── Search & rescue (wilderness) ─────────────────────────────────────────
    (
        ["missing person", "lost hiker", "missing hiker", "overdue hiker"],
        2,
        (0, 0, 2, 2),
        0.4,
        0.95,
    ),
    (
        ["stranded person", "stranded hiker", "stranded climber"],
        2,
        (0, 0, 1, 3),
        0.5,
        1.0,
    ),
    (
        ["distress signal", "sos signal", "signal mirror", "emergency beacon"],
        2,
        (0, 0, 1, 3),
        0.45,
        1.0,
    ),
    (["survivor", "injured survivor", "avalanche survivor"], 1, (0, 0, 1, 3), 0.5, 1.0),
    (["body", "human remains", "casualty"], 1, (0, 1, 2, 1), 0.4, 0.95),
    # ── Maritime SAR ─────────────────────────────────────────────────────────
    (
        ["overboard person", "man overboard", "swimmer in distress"],
        2,
        (0, 0, 1, 3),
        0.5,
        1.0,
    ),
    (
        ["disabled vessel", "sinking vessel", "sinking boat", "capsized boat"],
        2,
        (0, 0, 1, 3),
        0.5,
        1.0,
    ),
    (["mayday signal", "epirb signal", "plb activation"], 2, (0, 0, 0, 4), 0.4, 1.0),
    (["life raft", "life ring", "survival equipment"], 1, (0, 1, 2, 1), 0.4, 0.95),
    (["oil spill", "fuel spill", "marine pollution"], 3, (0, 1, 2, 1), 0.5, 1.0),
    # ── Structural / earthquake ──────────────────────────────────────────────
    (
        ["collapsed building", "building collapse", "structure collapse"],
        2,
        (0, 0, 1, 3),
        0.5,
        1.0,
    ),
    (
        ["earthquake damage", "seismic damage", "rubble field"],
        0,
        (0, 1, 2, 1),
        0.5,
        1.0,
    ),
    (["landslide", "mudslide", "rockslide", "debris flow"], 3, (0, 0, 1, 3), 0.5, 1.0),
    (["sinkhole", "ground subsidence", "road crack"], 1, (0, 1, 2, 1), 0.45, 0.95),
    (["damaged bridge", "bridge crack", "bridge damage"], 6, (0, 1, 2, 1), 0.5, 1.0),
    # ── Hazmat / industrial ──────────────────────────────────────────────────
    (["chemical spill", "hazmat spill", "toxic spill"], 3, (0, 0, 1, 3), 0.5, 1.0),
    (["gas leak", "natural gas leak", "methane plume"], 3, (0, 0, 1, 3), 0.45, 1.0),
    (["chemical cloud", "toxic plume", "vapor cloud"], 3, (0, 0, 0, 4), 0.5, 1.0),
    (["explosion", "blast", "detonation"], 0, (0, 0, 1, 3), 0.5, 1.0),
    (
        ["radiation source", "radioactive leak", "nuclear incident"],
        3,
        (0, 0, 0, 4),
        0.4,
        1.0,
    ),
    # ── Infrastructure inspection ─────────────────────────────────────────────
    (
        ["power line damage", "downed power line", "damaged pylon"],
        6,
        (0, 1, 2, 1),
        0.5,
        1.0,
    ),
    (
        ["pipeline leak", "pipeline damage", "pipe corrosion"],
        6,
        (0, 1, 2, 1),
        0.45,
        1.0,
    ),
    (
        ["solar panel", "solar farm inspection", "solar array"],
        6,
        (1, 2, 1, 0),
        0.5,
        1.0,
    ),
    (
        ["wind turbine", "wind turbine inspection", "blade damage"],
        6,
        (1, 2, 1, 0),
        0.5,
        1.0,
    ),
    (["transmission tower", "antenna tower", "cell tower"], 6, (1, 2, 1, 0), 0.5, 1.0),
    (["road damage", "road crack", "pothole"], 6, (1, 2, 1, 0), 0.5, 0.95),
    (["rail damage", "track damage", "railway inspection"], 6, (0, 1, 2, 1), 0.5, 1.0),
    # ── Agricultural ─────────────────────────────────────────────────────────
    (["crop disease", "crop blight", "plant disease"], 0, (1, 2, 1, 0), 0.5, 0.95),
    (["pest infestation", "locust swarm", "crop pest"], 0, (0, 1, 2, 1), 0.5, 1.0),
    (
        ["irrigation leak", "irrigation failure", "dry field"],
        6,
        (1, 2, 1, 0),
        0.5,
        0.95,
    ),
    (["field survey", "crop survey", "farmland survey"], 0, (2, 2, 0, 0), 0.5, 1.0),
    (["livestock", "cattle herd", "sheep flock"], 1, (2, 1, 1, 0), 0.5, 0.95),
    (
        ["illegal crop", "illicit cultivation", "illegal farm"],
        0,
        (0, 1, 2, 1),
        0.5,
        1.0,
    ),
    # ── Wildlife / environmental ──────────────────────────────────────────────
    (["wildlife", "wild animal", "animal"], 1, (2, 2, 0, 0), 0.5, 0.95),
    (
        ["poaching activity", "illegal poaching", "wildlife poaching"],
        3,
        (0, 1, 2, 1),
        0.5,
        1.0,
    ),
    (["large predator", "bear", "wolf", "mountain lion"], 3, (0, 1, 2, 1), 0.5, 0.95),
    (["whale", "marine mammal", "dolphin"], 1, (2, 2, 0, 0), 0.5, 0.95),
    (["coral bleaching", "reef damage", "algae bloom"], 0, (1, 2, 1, 0), 0.5, 0.95),
    (["deforestation", "illegal logging", "clear cutting"], 0, (0, 1, 2, 1), 0.5, 1.0),
    (["erosion", "soil erosion", "coastal erosion"], 0, (1, 2, 1, 0), 0.5, 0.95),
    # ── Medical / mass casualty ───────────────────────────────────────────────
    (["mass casualty", "mci", "multiple casualties"], 0, (0, 0, 0, 4), 0.5, 1.0),
    (
        ["medical emergency", "person down", "unresponsive person"],
        1,
        (0, 0, 1, 3),
        0.5,
        1.0,
    ),
    (["trauma scene", "accident scene", "crash scene"], 0, (0, 0, 1, 3), 0.5, 1.0),
    (["triage area", "casualty collection", "aid station"], 1, (0, 1, 2, 1), 0.5, 0.95),
    # ── Security / intrusion ──────────────────────────────────────────────────
    (
        ["perimeter breach", "fence breach", "intrusion detected"],
        3,
        (0, 1, 2, 1),
        0.5,
        1.0,
    ),
    (
        ["unauthorized uav", "rogue drone", "suspicious drone"],
        4,
        (0, 0, 1, 3),
        0.5,
        1.0,
    ),
    (
        ["armed individual", "weapon detected", "hostile individual"],
        3,
        (0, 0, 0, 4),
        0.5,
        1.0,
    ),
    (["suspicious vehicle", "suspicious activity"], 4, (0, 1, 2, 1), 0.45, 1.0),
    (["crowd gathering", "protest", "civil unrest"], 1, (0, 1, 2, 1), 0.5, 0.95),
    # ── Additional ORBIT scenarios ────────────────────────────────────────────
    (
        ["orbit target", "loiter pattern", "persistent surveillance"],
        4,
        (0, 1, 2, 1),
        0.6,
        1.0,
    ),
    (["thermal orbit", "isr orbit", "continuous watch"], 4, (0, 1, 2, 1), 0.6, 1.0),
    (
        ["vessel tracking", "ship tracking", "maritime surveillance"],
        4,
        (1, 2, 1, 0),
        0.5,
        1.0,
    ),
    # ── Vehicle / vessel tracking ─────────────────────────────────────────────
    (["vehicle", "car", "truck", "van"], 1, (2, 2, 0, 0), 0.5, 0.95),
    (["vessel", "ship", "cargo ship", "container vessel"], 1, (2, 2, 0, 0), 0.5, 0.95),
    (["aircraft", "fixed wing", "helicopter"], 1, (1, 2, 1, 0), 0.5, 0.95),
    (["abandoned vehicle", "derelict vessel"], 6, (1, 2, 1, 0), 0.5, 0.95),
    (["convoy", "vehicle convoy", "truck convoy"], 1, (1, 2, 1, 0), 0.5, 0.95),
    # ── Delivery / logistics ──────────────────────────────────────────────────
    (["delivery waypoint", "drop zone", "supply drop"], 5, (2, 2, 0, 0), 0.7, 1.0),
    (
        ["aid delivery", "medical supply delivery", "food supply delivery"],
        5,
        (0, 1, 2, 1),
        0.7,
        1.0,
    ),
    (["package delivery", "cargo drop", "payload delivery"], 5, (2, 2, 0, 0), 0.7, 1.0),
    (
        ["logistics mission", "resupply run", "transport mission"],
        5,
        (2, 2, 0, 0),
        0.7,
        1.0,
    ),
    (
        ["humanitarian delivery", "emergency supply drop", "relief delivery"],
        5,
        (0, 1, 2, 1),
        0.7,
        1.0,
    ),
    (
        ["courier mission", "last-mile delivery", "package drop"],
        5,
        (2, 2, 0, 0),
        0.7,
        1.0,
    ),
    # ── Additional INSPECT scenarios ─────────────────────────────────────────
    (
        ["infrastructure inspection", "asset inspection", "close inspection"],
        6,
        (1, 2, 1, 0),
        0.6,
        1.0,
    ),
    (
        ["roof inspection", "facade inspection", "structure inspection"],
        6,
        (1, 2, 1, 0),
        0.55,
        1.0,
    ),
    (
        ["bridge inspection", "road inspection", "rail inspection"],
        6,
        (0, 1, 2, 1),
        0.55,
        1.0,
    ),
    (
        ["photovoltaic inspection", "panel inspection", "cable inspection"],
        6,
        (1, 2, 1, 0),
        0.6,
        1.0,
    ),
    # ── Hurricane / severe weather ────────────────────────────────────────────
    (["tornado", "tornado touchdown", "twister"], 3, (0, 0, 0, 4), 0.5, 1.0),
    (
        ["hurricane damage", "cyclone damage", "typhoon damage"],
        0,
        (0, 1, 2, 1),
        0.5,
        1.0,
    ),
    (["hail damage", "wind damage", "storm damage"], 0, (1, 2, 1, 0), 0.5, 0.95),
    # ── Generic anomaly (catch-all) ───────────────────────────────────────────
    (["anomaly", "unknown object", "unidentified"], 0, (1, 2, 1, 0), 0.7, 1.0),
    # ── Earthquake / seismic ──────────────────────────────────────────────────
    (
        ["major earthquake", "great earthquake", "m7 earthquake"],
        2,
        (0, 0, 0, 4),
        0.7,
        1.0,
    ),
    (
        ["significant earthquake", "m6 earthquake", "strong earthquake"],
        2,
        (0, 0, 1, 3),
        0.6,
        1.0,
    ),
    (
        ["moderate earthquake", "earthquake", "seismic event"],
        0,
        (0, 1, 2, 1),
        0.5,
        0.95,
    ),
    (["aftershock sequence", "tremor", "seismic swarm"], 3, (0, 1, 2, 1), 0.5, 0.95),
    (["liquefaction", "ground failure", "fissure"], 0, (0, 0, 1, 3), 0.55, 1.0),
    # ── Volcanic ─────────────────────────────────────────────────────────────
    (["volcanic eruption", "lava flow", "pyroclastic flow"], 3, (0, 0, 0, 4), 0.6, 1.0),
    (["volcanic ash cloud", "ash fall", "tephra"], 3, (0, 0, 1, 3), 0.55, 1.0),
    (["volcanic alert", "volcanic activity", "lava dome"], 3, (0, 0, 2, 2), 0.5, 1.0),
    (["lahar", "volcanic mudflow", "debris avalanche"], 3, (0, 0, 1, 3), 0.6, 1.0),
    # ── Tsunami / coastal ─────────────────────────────────────────────────────
    (
        ["tsunami warning", "tsunami wave", "seismic sea wave"],
        3,
        (0, 0, 0, 4),
        0.65,
        1.0,
    ),
    (
        ["coastal flooding", "storm surge flooding", "inundation zone"],
        3,
        (0, 0, 1, 3),
        0.55,
        1.0,
    ),
    # ── Epidemic / public health → DELIVER (medical supply) ──────────────────
    (["disease outbreak", "epidemic", "cholera outbreak"], 5, (0, 1, 2, 1), 0.6, 0.95),
    (
        ["medical supply needed", "medicine shortage", "vaccine delivery"],
        5,
        (0, 1, 2, 1),
        0.65,
        1.0,
    ),
    (
        ["mass vaccination site", "field hospital", "medical camp"],
        5,
        (0, 1, 2, 1),
        0.60,
        1.0,
    ),
    (
        ["contaminated water", "water contamination", "toxic water"],
        5,
        (0, 0, 2, 2),
        0.55,
        1.0,
    ),
    (["food shortage", "aid required", "famine relief"], 5, (0, 0, 2, 2), 0.55, 1.0),
    # ── Chemical / nuclear (expanded) ────────────────────────────────────────
    (
        ["nuclear incident", "radiation release", "radiological hazard"],
        3,
        (0, 0, 0, 4),
        0.5,
        1.0,
    ),
    (
        ["chemical weapons", "nerve agent", "biological agent"],
        3,
        (0, 0, 0, 4),
        0.5,
        1.0,
    ),
    (
        ["industrial explosion", "refinery fire", "chemical plant fire"],
        3,
        (0, 0, 0, 4),
        0.55,
        1.0,
    ),
    (["oil spill", "fuel spill", "petroleum release"], 3, (0, 0, 2, 2), 0.5, 1.0),
    # ── Air quality / pollution ───────────────────────────────────────────────
    (
        ["hazardous air quality", "smoke haze", "wildfire smoke"],
        0,
        (0, 0, 2, 2),
        0.6,
        1.0,
    ),
    (["dust storm", "sandstorm", "haboob"], 0, (0, 1, 2, 1), 0.55, 1.0),
    (
        ["industrial smog", "pollution plume", "emission plume"],
        3,
        (0, 1, 2, 1),
        0.55,
        1.0,
    ),
    (
        ["severe air pollution", "pm25 hazardous", "air quality emergency"],
        3,
        (0, 0, 1, 3),
        0.65,
        1.0,
    ),
    # ── Tropical cyclone / hurricane (expanded) ───────────────────────────────
    (
        ["major hurricane", "category 4 hurricane", "category 5 hurricane"],
        3,
        (0, 0, 0, 4),
        0.7,
        1.0,
    ),
    (["tropical cyclone", "typhoon", "cyclone landfall"], 3, (0, 0, 1, 3), 0.6, 1.0),
    (
        ["hurricane evacuation zone", "mandatory evacuation", "hurricane shelter"],
        0,
        (0, 0, 2, 2),
        0.55,
        1.0,
    ),
    # ── Ice / avalanche / cold (expanded) ────────────────────────────────────
    (["avalanche", "snow avalanche", "slab avalanche"], 2, (0, 0, 1, 3), 0.55, 1.0),
    (["ice jam", "river ice", "ice hazard"], 3, (0, 1, 2, 1), 0.5, 0.95),
    (
        ["blizzard", "whiteout conditions", "severe snowstorm"],
        0,
        (0, 1, 2, 1),
        0.55,
        1.0,
    ),
    (["extreme cold", "hypothermia risk", "cold wave"], 0, (0, 1, 2, 1), 0.55, 0.95),
    # ── Drought / extreme heat ────────────────────────────────────────────────
    (
        ["extreme heat", "heat wave", "dangerous heat index"],
        0,
        (0, 1, 2, 1),
        0.60,
        0.95,
    ),
    (
        ["drought", "severe drought", "crop failure drought"],
        0,
        (0, 2, 1, 1),
        0.55,
        0.90,
    ),
    (
        ["wildfire risk", "red flag condition", "fire weather warning"],
        0,
        (0, 0, 2, 2),
        0.60,
        1.0,
    ),
    # ── Expanded wildlife (new species from iNaturalist/GBIF) ─────────────────
    (["jaguar sighting", "jaguar", "leopard sighting"], 3, (0, 1, 2, 1), 0.5, 0.95),
    (["tiger sighting", "tiger"], 3, (0, 0, 1, 3), 0.5, 0.95),
    (["crocodile", "alligator", "crocodile sighting"], 3, (0, 1, 2, 1), 0.5, 0.95),
    (["orca", "killer whale", "orca pod"], 1, (2, 2, 0, 0), 0.5, 0.95),
    (["humpback whale", "whale sighting", "whale breach"], 1, (2, 2, 0, 0), 0.5, 0.95),
    (["elephant herd", "elephant sighting", "elephant"], 1, (0, 1, 2, 1), 0.5, 0.95),
    (["rhino sighting", "rhino", "endangered species"], 1, (0, 1, 2, 1), 0.5, 0.95),
    (["gorilla sighting", "gorilla", "great ape"], 1, (0, 1, 2, 1), 0.5, 0.95),
    (["python", "invasive snake", "feral animal"], 3, (0, 1, 2, 1), 0.5, 0.90),
    (["bird strike risk", "raptor", "bald eagle"], 1, (1, 2, 1, 0), 0.5, 0.90),
    # ── Maritime / port security ──────────────────────────────────────────────
    (
        ["unidentified vessel", "suspicious vessel", "vessel of interest"],
        4,
        (0, 1, 2, 1),
        0.5,
        1.0,
    ),
    (
        ["port security breach", "unauthorized dock access", "maritime intrusion"],
        3,
        (0, 0, 2, 2),
        0.5,
        1.0,
    ),
    (
        ["search and rescue zone", "maritime search grid", "sar sector"],
        2,
        (0, 0, 1, 3),
        0.5,
        1.0,
    ),
    (["vessel aground", "grounded ship", "beached vessel"], 2, (0, 1, 2, 1), 0.5, 1.0),
    (
        ["diving emergency", "diver in distress", "diver missing"],
        2,
        (0, 0, 1, 3),
        0.5,
        1.0,
    ),
    # ── Urban SAR ─────────────────────────────────────────────────────────────
    (
        ["trapped person", "person trapped in rubble", "urban rescue"],
        2,
        (0, 0, 1, 3),
        0.5,
        1.0,
    ),
    (["building fire", "structure fire", "high-rise fire"], 3, (0, 0, 1, 3), 0.55, 1.0),
    (
        ["gas explosion", "building explosion", "blast damage"],
        2,
        (0, 0, 0, 4),
        0.55,
        1.0,
    ),
    (
        ["mineshaft collapse", "mine rescue", "underground collapse"],
        2,
        (0, 0, 1, 3),
        0.5,
        1.0,
    ),
]

_RNG = random.Random(42)
_NP_RNG = np.random.default_rng(42)


def _risk_from_conf(base_weights: tuple, conf: float) -> int:
    """Sample risk level: base_weights shift toward CRITICAL as confidence rises."""
    low_w, med_w, high_w, crit_w = base_weights
    # Confidence amplifies the higher risk weights
    scale = conf
    weights = [
        max(0.1, low_w * (1 - scale)),
        max(0.1, med_w),
        high_w * scale,
        crit_w * scale * scale,
    ]
    total = sum(weights)
    weights = [w / total for w in weights]
    return int(_NP_RNG.choice(4, p=weights))


def generate_mission_samples(n: int = 8000) -> Tuple[np.ndarray, np.ndarray]:
    """Generate (X, y) for mission type classification."""
    X, y = [], []
    per_scenario = max(1, n // len(_SCENARIOS))

    for class_strs, mission_type, risk_weights, min_conf, max_conf in _SCENARIOS:
        for _ in range(per_scenario):
            # Pick a class string and optionally add descriptors
            cls = _RNG.choice(class_strs)
            conf = _RNG.uniform(min_conf, max_conf)
            has_loc = _RNG.random() > 0.05  # 95% have a location
            lat = _RNG.uniform(-60, 70) if has_loc else None
            lon = _RNG.uniform(-180, 180) if has_loc else None

            obs = {"class": cls, "confidence": conf, "lat": lat, "lon": lon}
            X.append(extract(obs))
            y.append(mission_type)

            # Compound class: add a second domain word to same label
            if _RNG.random() < 0.3:
                extra = _RNG.choice(["near", "adjacent to", "with", "and"])
                other_cls, other_mission, *_ = _RNG.choice(_SCENARIOS)
                obs2 = {
                    "class": f"{cls} {extra} {_RNG.choice(other_cls)}",
                    "confidence": conf * _RNG.uniform(0.85, 1.0),
                    "lat": lat,
                    "lon": lon,
                }
                # Label stays primary scenario's mission type
                X.append(extract(obs2))
                y.append(mission_type)

    # Low-confidence unknowns → SURVEY (default safe action)
    for _ in range(n // 10):
        conf = _RNG.uniform(0.1, 0.5)
        obs = {
            "class": f"unknown_{_RNG.randint(0,999)}",
            "confidence": conf,
            "lat": _RNG.uniform(-60, 70),
            "lon": _RNG.uniform(-180, 180),
        }
        X.append(extract(obs))
        y.append(0)  # SURVEY

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


def generate_risk_samples(n: int = 8000) -> Tuple[np.ndarray, np.ndarray]:
    """Generate (X, y) for risk level classification."""
    X, y = [], []
    per_scenario = max(1, n // len(_SCENARIOS))

    for class_strs, _, risk_weights, min_conf, max_conf in _SCENARIOS:
        for _ in range(per_scenario):
            cls = _RNG.choice(class_strs)
            conf = _RNG.uniform(min_conf, max_conf)
            has_loc = _RNG.random() > 0.1
            lat = _RNG.uniform(-60, 70) if has_loc else None
            lon = _RNG.uniform(-180, 180) if has_loc else None

            obs = {"class": cls, "confidence": conf, "lat": lat, "lon": lon}
            X.append(extract(obs))
            risk = _risk_from_conf(risk_weights, conf)
            y.append(risk)

    # Very-low confidence → LOW risk
    for _ in range(n // 10):
        conf = _RNG.uniform(0.0, 0.35)
        obs = {
            "class": _RNG.choice(["object", "blob", "movement", "anomaly"]),
            "confidence": conf,
            "lat": _RNG.uniform(-60, 70),
            "lon": _RNG.uniform(-180, 180),
        }
        X.append(extract(obs))
        y.append(0)  # LOW

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)
