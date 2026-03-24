"""
Feature extraction for Summit.OS ML models.

Shared between training and inference — the exact same function is called
in both contexts so there is no train/serve skew.

Feature vector (14 floats, index-stable):
  [0]  confidence
  [1]  has_location      (1 if lat+lon present, else 0)
  [2]  is_fire_smoke     fire, smoke, flame, wildfire, ember, hotspot, burning, blaze
  [3]  is_person         person, human, survivor, victim, pedestrian, casualty,
                         injured, missing, stranded, body, civilian, child
  [4]  is_flood_water    flood, water, inundation, surge, submerged, drowning,
                         overflow, tsunami, storm surge, rising water
  [5]  is_structural     collapse, structure, building, rubble, debris, damage,
                         destroyed, ruin, wreckage, infrastructure breach
  [6]  is_vehicle        vehicle, truck, car, boat, vessel, aircraft, uav, drone,
                         ship, helicopter, plane, motorcycle, atv
  [7]  is_hazmat         hazmat, chemical, spill, leak, toxic, gas, explosion,
                         radiation, biological, contamination, plume
  [8]  is_wildlife       animal, wildlife, livestock, poaching, bear, deer,
                         wolf, cattle, herd
  [9]  is_infrastructure power line, pipeline, bridge, road, tower, antenna,
                         dam, levee, rail, utility, cable, pylon
  [10] is_agricultural   crop, field, irrigation, pest, harvest, farm, orchard,
                         vineyard, soil, erosion, drought
  [11] is_medical        medical, injury, unconscious, emergency, cardiac,
                         trauma, triage, wounded, mass casualty
  [12] is_security       intrusion, unauthorized, trespass, suspicious, threat,
                         breach, perimeter violation, armed
  [13] is_search_target  missing, lost, stranded, downed, wreckage, overdue,
                         distress, sos, mayday, signal
  [14] is_logistics      delivery, supply, drop zone, aid drop, cargo, payload,
                         transport, courier, resupply, package, logistics
"""

from typing import Any, Dict, List, Optional

# Each group: (feature_index, keywords)
_GROUPS: List[tuple] = [
    (2,  ["fire", "smoke", "flame", "wildfire", "ember", "hotspot", "burning", "blaze", "ignition", "char"]),
    (3,  ["person", "human", "survivor", "victim", "pedestrian", "casualty", "injured", "missing",
           "stranded", "body", "civilian", "child", "hiker", "swimmer", "diver"]),
    (4,  ["flood", "water", "inundation", "surge", "submerged", "drowning", "overflow",
           "tsunami", "storm surge", "rising water", "flash flood", "levee breach"]),
    (5,  ["collapse", "structure", "building", "rubble", "debris", "damage", "destroyed",
           "ruin", "wreckage", "breach", "crack", "sinkhole", "landslide", "mudslide"]),
    (6,  ["vehicle", "truck", "car", "boat", "vessel", "aircraft", "uav", "drone",
           "ship", "helicopter", "plane", "motorcycle", "atv", "submarine"]),
    (7,  ["hazmat", "chemical", "spill", "leak", "toxic", "gas", "explosion",
           "radiation", "biological", "contamination", "plume", "cloud", "vapor", "ammonia", "chlorine"]),
    (8,  ["animal", "wildlife", "livestock", "poaching", "bear", "deer", "wolf",
           "cattle", "herd", "elephant", "shark", "crocodile"]),
    (9,  ["power line", "pipeline", "bridge", "road", "tower", "antenna", "dam",
           "levee", "rail", "utility", "cable", "pylon", "transformer", "substation"]),
    (10, ["crop", "field", "irrigation", "pest", "harvest", "farm", "orchard",
           "vineyard", "soil", "erosion", "drought", "blight", "infestation"]),
    (11, ["medical", "injury", "unconscious", "emergency", "cardiac", "trauma",
           "triage", "wounded", "mass casualty", "overdose", "seizure"]),
    (12, ["intrusion", "unauthorized", "trespass", "suspicious", "threat", "breach",
           "perimeter violation", "armed", "hostile", "illegal", "poacher"]),
    (13, ["missing", "lost", "stranded", "downed", "overdue", "distress", "sos",
           "mayday", "signal", "beacon", "epirb", "plb"]),
    (14, ["delivery", "supply drop", "drop zone", "aid drop", "cargo", "payload",
           "transport", "courier", "resupply", "package", "logistics", "deliver"]),
]

FEATURE_DIM = 15
FEATURE_NAMES = [
    "confidence", "has_location",
    "is_fire_smoke", "is_person", "is_flood_water", "is_structural",
    "is_vehicle", "is_hazmat", "is_wildlife", "is_infrastructure",
    "is_agricultural", "is_medical", "is_security", "is_search_target",
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
