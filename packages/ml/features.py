"""
Feature extraction for Summit.OS ML models.

Shared between training and inference — the exact same function is called
in both contexts so there is no train/serve skew.

Feature vector (15 floats, index-stable):
  [0]  confidence
  [1]  has_location      (1 if lat+lon present, else 0)
  [2]  is_fire_smoke     fire, smoke, wildfire, ember, hotspot, burning, blaze,
                         ignition, char, ash, pyroclastic, lava (volcanic fire)
  [3]  is_person         person, human, survivor, victim, casualty, injured,
                         missing, stranded, body, hiker, swimmer, diver, trapped
  [4]  is_flood_water    flood, water, inundation, surge, submerged, tsunami,
                         flash flood, coastal flooding, cyclone flooding, lahar
  [5]  is_structural     collapse, structure, building, rubble, debris, damage,
                         earthquake, seismic, landslide, mudslide, avalanche,
                         sinkhole, volcanic eruption, crater, mine collapse
  [6]  is_vehicle        vehicle, truck, car, boat, vessel, aircraft, uav, drone,
                         ship, helicopter, submarine, kayak, raft
  [7]  is_hazmat         hazmat, chemical, spill, leak, toxic, gas, explosion,
                         radiation, biological, contamination, plume, nuclear,
                         radiological, nerve agent, volcanic ash, air quality,
                         pollution, pm25, smog, emission, ammonia, chlorine
  [8]  is_wildlife       animal, wildlife, livestock, bear, deer, wolf, cattle,
                         elephant, shark, crocodile, alligator, tiger, jaguar,
                         leopard, orca, whale, dolphin, rhino, gorilla, python,
                         moose, bison, eagle, raptor, poaching
  [9]  is_infrastructure power line, pipeline, bridge, road, tower, antenna, dam,
                         levee, rail, utility, cable, pylon, transformer, port,
                         substation, refinery, reactor
  [10] is_agricultural   crop, field, irrigation, pest, harvest, farm, orchard,
                         blight, infestation, locust, drought, famine, food shortage
  [11] is_medical        medical, injury, emergency, cardiac, trauma, triage,
                         wounded, mass casualty, epidemic, disease, outbreak,
                         cholera, contaminated water, vaccination, hospital,
                         heat stroke, hypothermia, seizure
  [12] is_security       intrusion, unauthorized, trespass, suspicious, threat,
                         armed, hostile, illegal, civil unrest, protest, poacher,
                         perimeter breach, maritime intrusion
  [13] is_search_target  missing, lost, stranded, downed, overdue, distress, sos,
                         mayday, beacon, epirb, plb, search grid, sar sector
  [14] is_logistics      delivery, supply, drop zone, aid drop, cargo, payload,
                         courier, resupply, package, logistics, humanitarian,
                         relief, medicine shortage, vaccine delivery, food aid
"""

from typing import Any, Dict, List, Optional

# Each group: (feature_index, keywords)
_GROUPS: List[tuple] = [
    (
        2,
        [
            "fire",
            "smoke",
            "flame",
            "wildfire",
            "ember",
            "hotspot",
            "burning",
            "blaze",
            "ignition",
            "char",
            "ash fall",
            "pyroclastic",
            "lava",
            "tephra",
        ],
    ),
    (
        3,
        [
            "person",
            "human",
            "survivor",
            "victim",
            "pedestrian",
            "casualty",
            "injured",
            "missing",
            "stranded",
            "body",
            "civilian",
            "child",
            "hiker",
            "swimmer",
            "diver",
            "trapped",
        ],
    ),
    (
        4,
        [
            "flood",
            "inundation",
            "surge",
            "submerged",
            "drowning",
            "overflow",
            "tsunami",
            "storm surge",
            "rising water",
            "flash flood",
            "levee breach",
            "coastal flood",
            "lahar",
            "mudflow",
        ],
    ),
    (
        5,
        [
            "collapse",
            "structure",
            "building",
            "rubble",
            "debris",
            "damage",
            "destroyed",
            "ruin",
            "wreckage",
            "crack",
            "sinkhole",
            "landslide",
            "mudslide",
            "earthquake",
            "seismic",
            "tremor",
            "aftershock",
            "liquefaction",
            "avalanche",
            "mine collapse",
            "crater",
            "eruption",
        ],
    ),
    (
        6,
        [
            "vehicle",
            "truck",
            "car",
            "boat",
            "vessel",
            "aircraft",
            "uav",
            "drone",
            "ship",
            "helicopter",
            "plane",
            "motorcycle",
            "atv",
            "submarine",
            "kayak",
            "raft",
            "life raft",
        ],
    ),
    (
        7,
        [
            "hazmat",
            "chemical",
            "spill",
            "leak",
            "toxic",
            "gas",
            "explosion",
            "radiation",
            "biological",
            "contamination",
            "plume",
            "cloud",
            "vapor",
            "nuclear",
            "radiological",
            "nerve agent",
            "ammonia",
            "chlorine",
            "volcanic ash",
            "ash cloud",
            "air quality",
            "pollution",
            "pm25",
            "smog",
            "emission",
            "industrial smog",
        ],
    ),
    (
        8,
        [
            "animal",
            "wildlife",
            "livestock",
            "poaching",
            "bear",
            "deer",
            "wolf",
            "cattle",
            "herd",
            "elephant",
            "shark",
            "crocodile",
            "alligator",
            "tiger",
            "jaguar",
            "leopard",
            "mountain lion",
            "orca",
            "whale",
            "dolphin",
            "rhino",
            "gorilla",
            "python",
            "moose",
            "bison",
            "eagle",
            "raptor",
            "feral",
        ],
    ),
    (
        9,
        [
            "power line",
            "pipeline",
            "bridge",
            "road",
            "tower",
            "antenna",
            "dam",
            "levee",
            "rail",
            "utility",
            "cable",
            "pylon",
            "transformer",
            "substation",
            "refinery",
            "reactor",
            "port infrastructure",
        ],
    ),
    (
        10,
        [
            "crop",
            "field",
            "irrigation",
            "pest",
            "harvest",
            "farm",
            "orchard",
            "vineyard",
            "soil",
            "erosion",
            "drought",
            "blight",
            "infestation",
            "locust",
            "famine",
            "food shortage",
            "food insecurity",
        ],
    ),
    (
        11,
        [
            "medical",
            "injury",
            "unconscious",
            "cardiac",
            "trauma",
            "triage",
            "wounded",
            "mass casualty",
            "overdose",
            "seizure",
            "epidemic",
            "disease outbreak",
            "outbreak",
            "cholera",
            "contaminated water",
            "vaccination",
            "field hospital",
            "heat stroke",
            "hypothermia",
            "heat wave",
            "cold wave",
        ],
    ),
    (
        12,
        [
            "intrusion",
            "unauthorized",
            "trespass",
            "suspicious",
            "threat",
            "breach",
            "perimeter violation",
            "armed",
            "hostile",
            "illegal",
            "poacher",
            "civil unrest",
            "protest",
            "maritime intrusion",
            "port breach",
        ],
    ),
    (
        13,
        [
            "missing",
            "lost",
            "stranded",
            "downed",
            "overdue",
            "distress",
            "sos",
            "mayday",
            "signal",
            "beacon",
            "epirb",
            "plb",
            "search grid",
            "sar sector",
            "man overboard",
            "diver missing",
        ],
    ),
    (
        14,
        [
            "delivery",
            "supply drop",
            "drop zone",
            "aid drop",
            "cargo",
            "payload",
            "transport",
            "courier",
            "resupply",
            "package",
            "logistics",
            "deliver",
            "humanitarian",
            "relief delivery",
            "medicine shortage",
            "vaccine delivery",
            "food aid",
            "aid required",
        ],
    ),
]

FEATURE_DIM = 15
FEATURE_NAMES = [
    "confidence",
    "has_location",
    "is_fire_smoke",
    "is_person",
    "is_flood_water",
    "is_structural",
    "is_vehicle",
    "is_hazmat",
    "is_wildlife",
    "is_infrastructure",
    "is_agricultural",
    "is_medical",
    "is_security",
    "is_search_target",
    "is_logistics",
]


def extract(observation: Dict[str, Any]) -> List[float]:
    """
    Return a 14-float feature vector for the given observation dict.
    Deterministic, no external dependencies, <0.01ms.

    Expected keys (all optional): class, confidence, lat, lon
    """
    cls = str(observation.get("class") or "").lower()
    conf = float(observation.get("confidence") or 0.0)
    lat = observation.get("lat")
    lon = observation.get("lon")
    has_loc = 1.0 if (lat is not None and lon is not None) else 0.0

    vec = [0.0] * FEATURE_DIM
    vec[0] = conf
    vec[1] = has_loc

    for idx, keywords in _GROUPS:
        if any(kw in cls for kw in keywords):
            vec[idx] = 1.0

    return vec
