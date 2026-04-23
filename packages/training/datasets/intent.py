"""
Mission Intent Training Data Builder
======================================
Builds a labeled dataset for mapping free-text operator input to
Heli.OS mission types (SAR, SURVEY, PATROL, RECON, MONITOR, ESCORT).

Sources:
  1. FEMA DisasterDeclarationsSummaries public API — real incident titles
  2. NIFC wildfire incident data — fire operation descriptions
  3. Template-augmented synthetic examples (ICS/NIMS terminology)

Output: intent_training.parquet with columns [text, mission_type]
"""
from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

_FEMA_API = "https://www.fema.gov/api/open/v2/DisasterDeclarationsSummaries"
_NIFC_URL = "https://www.nifc.gov/nicc/sitreprt.pdf"  # not machine-readable, skip

_MISSION_TYPES = ["SAR", "SURVEY", "PATROL", "RECON", "MONITOR", "ESCORT"]

# Template examples per class — ICS/NIMS/SAR operational language
_TEMPLATES: dict[str, list[str]] = {
    "SAR": [
        "Find the missing {subject}",
        "Search for missing {subject} in {area}",
        "Locate missing person reported in {area}",
        "Person overdue on {trail} trail — initiate search",
        "Missing hiker in {area} — begin SAR operations",
        "Find and rescue {subject} last seen near {landmark}",
        "Rescue operation for {subject} in {area}",
        "Life safety — locate missing {subject} in {area}",
        "Overdue party — search {area} for survivors",
        "Lost child near {landmark} — begin search",
        "Distress beacon activated in {area} — investigate",
        "Search and rescue {area}",
        "Find survivor in {area}",
        "Missing person operation {area}",
        "Scan {area} for missing {subject}",
        "{subject} reported missing — grid search {area}",
        "Emergency locator beacon in {area} — SAR response",
        "Conduct SAR sweep of {area}",
        "Urban search and rescue {landmark}",
        "Wilderness SAR {area}",
    ],
    "SURVEY": [
        "Survey {area} for damage assessment",
        "Map the {area}",
        "Aerial survey of {area}",
        "Grid survey {area} — document infrastructure status",
        "Survey flood damage in {area}",
        "Post-event survey {area}",
        "Photographic survey {landmark}",
        "Assess structural damage {area}",
        "Infrastructure survey {area}",
        "Survey road network in {area}",
        "Document conditions {area}",
        "Flyover and map {area}",
        "Survey the {landmark} area",
        "Rapid damage assessment {area}",
        "Post-fire survey {area}",
        "Ground truth survey {area}",
        "Imagery collection {area}",
        "Overhead survey {landmark}",
        "Survey pipeline corridor {area}",
        "Condition report {area}",
    ],
    "PATROL": [
        "Patrol {area}",
        "Perimeter patrol {area}",
        "Routine patrol of {area}",
        "Security patrol {landmark}",
        "Border patrol {area}",
        "Patrol the {area} perimeter",
        "Conduct patrol along {trail}",
        "Monitor {area} perimeter",
        "Patrol and report activity {area}",
        "Sweep {area} for unauthorized access",
        "Maintain patrol coverage {area}",
        "Night patrol {area}",
        "Patrol {landmark} sector",
        "Regular patrol {area}",
        "Hold patrol pattern {area}",
        "Conduct security sweep {area}",
        "Patrol corridor {area}",
        "Perimeter check {landmark}",
        "Surveillance patrol {area}",
        "Rolling patrol {area}",
    ],
    "RECON": [
        "Recon {area}",
        "Reconnaissance of {area}",
        "Intel gathering {area}",
        "Forward recon {landmark}",
        "Assess situation in {area}",
        "Get eyes on {landmark}",
        "Check conditions at {landmark}",
        "Recon the {area} approach",
        "Observe activity at {landmark}",
        "Conduct recon of {area}",
        "Forward observation {area}",
        "Situational assessment {area}",
        "Verify conditions {area}",
        "Scout {area} for access routes",
        "Recon fire perimeter",
        "Assess {landmark} approach",
        "Eyes on {area}",
        "Establish observation point {area}",
        "Check {area} for hazards",
        "Advance recon {landmark}",
    ],
    "MONITOR": [
        "Monitor {area}",
        "Watch {landmark}",
        "Keep eyes on {area}",
        "Monitor fire behavior {area}",
        "Track movement {area}",
        "Observe and report {area}",
        "Continuous monitoring {area}",
        "Watch for changes at {landmark}",
        "Monitor flood levels {area}",
        "Track crowd at {landmark}",
        "Observe {area} for activity",
        "Standby and monitor {landmark}",
        "Monitor weather {area}",
        "Watch the {landmark} sector",
        "Maintain observation {area}",
        "Monitor traffic {area}",
        "Observe {landmark} approach",
        "Track {subject} movement",
        "Monitor asset at {landmark}",
        "Continuous observation {area}",
    ],
    "ESCORT": [
        "Escort convoy through {area}",
        "Provide air escort to {subject}",
        "Escort {subject} to {landmark}",
        "Accompany {subject} through {area}",
        "VIP escort {area}",
        "Provide overwatch for convoy",
        "Escort medical team to {landmark}",
        "Accompany rescue team {area}",
        "Provide escort and overwatch {area}",
        "Escort ground team through {area}",
        "Air escort {subject}",
        "Overwatch {subject} movement",
        "Escort relief convoy {area}",
        "Provide coverage for {subject}",
        "Escort and protect {subject} to {landmark}",
        "Follow {subject} through {area}",
        "Cover {subject} extraction",
        "Escort hazmat team {area}",
        "Provide air cover for ground team",
        "Escort and report {subject} status",
    ],
}

_SUBJECTS  = ["hiker", "climber", "swimmer", "child", "patient", "evacuee",
               "worker", "pilot", "crew member", "researcher"]
_AREAS     = ["sector alpha", "grid 7", "the northern quadrant", "the valley",
               "zone 3", "the southern ridge", "the eastern sector", "area of operations",
               "the search area", "the incident area", "target area", "the designated zone"]
_LANDMARKS = ["the bridge", "the dam", "the summit", "the command post",
               "the staging area", "the LZ", "the fire line", "the road junction",
               "the river crossing", "the tower"]
_TRAILS    = ["main", "forest", "mountain", "coastal", "river"]


def _fill_template(tmpl: str) -> str:
    tmpl = tmpl.replace("{subject}", random.choice(_SUBJECTS))
    tmpl = tmpl.replace("{area}", random.choice(_AREAS))
    tmpl = tmpl.replace("{landmark}", random.choice(_LANDMARKS))
    tmpl = tmpl.replace("{trail}", random.choice(_TRAILS))
    return tmpl


def _fetch_fema_incidents() -> list[tuple[str, str]]:
    """Pull FEMA incident titles and map to mission types."""
    logger.info("Fetching FEMA disaster declarations ...")
    results = []
    try:
        params = {
            "$top": 1000,
            "$select": "incidentType,declarationTitle,state",
            "$orderby": "declarationDate desc",
        }
        r = requests.get(_FEMA_API, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        items = data.get("DisasterDeclarationsSummaries", [])

        incident_to_mission = {
            "Fire":          "SAR",
            "Flood":         "RECON",
            "Hurricane":     "SAR",
            "Tornado":       "SAR",
            "Earthquake":    "SAR",
            "Snow":          "MONITOR",
            "Drought":       "MONITOR",
            "Typhoon":       "SAR",
            "Landslide":     "SAR",
            "Dam/Levee":     "RECON",
            "Chemical":      "RECON",
            "Coastal Storm": "MONITOR",
            "Fishing Losses":"SURVEY",
        }

        for item in items:
            title = item.get("declarationTitle", "").strip()
            inc_type = item.get("incidentType", "")
            mission = incident_to_mission.get(inc_type)
            if not mission or not title:
                continue
            results.append((title.lower(), mission))

        logger.info("FEMA: %d labeled incident titles fetched", len(results))
    except Exception as e:
        logger.warning("FEMA API failed: %s", e)
    return results


def download(output_dir: str = "/tmp/heli-training-data/intent") -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    random.seed(42)

    records: list[dict] = []

    # 1. Template-based synthetic examples
    for mission_type, templates in _TEMPLATES.items():
        for tmpl in templates:
            for _ in range(15):  # augment each template 15×
                text = _fill_template(tmpl)
                records.append({"text": text.lower(), "mission_type": mission_type})

    logger.info("Template examples: %d", len(records))

    # 2. FEMA incident data
    fema = _fetch_fema_incidents()
    for text, mission in fema:
        records.append({"text": text, "mission_type": mission})
    logger.info("FEMA examples: %d", len(fema))

    # 3. Shuffle
    random.shuffle(records)

    import pandas as pd
    df = pd.DataFrame(records)
    counts = df["mission_type"].value_counts().to_dict()
    logger.info("Intent dataset class distribution: %s", counts)

    out_file = out / "intent_training.parquet"
    df.to_parquet(out_file, index=False)
    logger.info("Intent dataset: %d records → %s", len(df), out_file)
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="/tmp/heli-training-data/intent")
    args = p.parse_args()
    download(args.output)
