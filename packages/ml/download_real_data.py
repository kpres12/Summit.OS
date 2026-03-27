"""
Download and preprocess public real-world datasets for Summit.OS model training.

Sources (all public, no auth required):
  NASA FIRMS      — active fire detections, global, updated every 10 min
  NOAA Storm      — severe weather events 1950-present (flood, tornado, hurricane, etc.)
  PHMSA           — pipeline and hazmat incidents 2010-present
  GBIF            — wildlife occurrence observations (API)
  USGS            — earthquake catalog M2.5+, global, real-time feed
  NASA EONET      — natural events (volcanoes, floods, wildfires, landslides, dust)
  iNaturalist     — 100M+ wildlife observations, threatened/endangered species
  OpenAQ          — air quality readings (fire smoke + hazmat proxy)
  GDACS           — global disaster alerts (earthquakes, cyclones, floods, volcanoes)
  ReliefWeb       — humanitarian crisis events (floods, earthquakes, epidemics)

Output: data/real_observations.csv
  Columns: class, confidence, lat, lon, mission_type, risk_level, source

Usage:
  python download_real_data.py                        # download all sources
  python download_real_data.py --source firms usgs    # specific sources
  python download_real_data.py --years 2021 2022 2023 # NOAA multi-year
"""

import argparse
import csv
import io
import json
import os
import sys
import time
import urllib.request
import urllib.error
import zipfile
from datetime import datetime, timezone
from typing import List, Dict, Optional

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Helpers ───────────────────────────────────────────────────────────────────


def _fetch(url: str, timeout: int = 30, retries: int = 3) -> Optional[bytes]:
    headers = {"User-Agent": "Summit.OS/1.0 (public data research)"}
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            print(f"  HTTP {e.code} for {url}")
            return None
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2**attempt)
            else:
                print(f"  Failed: {e}")
                return None
    return None


def _write_csv(rows: List[Dict], path: str):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "class",
                "confidence",
                "lat",
                "lon",
                "mission_type",
                "risk_level",
                "source",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Wrote {len(rows)} rows → {path}")


# ── NASA FIRMS — Active fire detections ──────────────────────────────────────


def download_firms(max_rows: int = 50000) -> List[Dict]:
    """
    Download VIIRS active fire detections from NASA FIRMS.
    No API key required for the 7-day global CSV files.

    Confidence field semantics:
      VIIRS: 'l' (low ~30%), 'n' (nominal ~50%), 'h' (high ~80%)
      MODIS: integer 0-100

    Mission type mapping:
      All fire detections → SURVEY (assess extent of active fire)
      FRP > 100 MW or confidence=high → CRITICAL risk
    """
    print("Downloading NASA FIRMS fire data...")

    urls = [
        # VIIRS NOAA-20 (most current sensor)
        (
            "https://firms.modaps.eosdis.nasa.gov/data/active_fire/noaa-20-viirs-c2/csv/J1_VIIRS_C2_Global_7d.csv",
            "viirs_noaa20",
        ),
        # VIIRS SUOMI NPP
        (
            "https://firms.modaps.eosdis.nasa.gov/data/active_fire/suomi-npp-viirs-c2/csv/SUOMI_VIIRS_C2_Global_7d.csv",
            "viirs_suomi",
        ),
        # MODIS (older, broader coverage)
        (
            "https://firms.modaps.eosdis.nasa.gov/data/active_fire/modis-c6.1/csv/MODIS_C6_1_Global_7d.csv",
            "modis",
        ),
    ]

    # VIIRS confidence: l=low, n=nominal, h=high
    _viirs_conf = {"l": 0.30, "n": 0.55, "h": 0.85}

    rows = []
    for url, sensor in urls:
        data = _fetch(url, timeout=60)
        if not data:
            continue
        text = data.decode("utf-8", errors="replace")
        reader = csv.DictReader(io.StringIO(text))
        count = 0
        for rec in reader:
            if count >= max_rows // len(urls):
                break
            try:
                lat = float(rec.get("latitude") or rec.get("lat") or 0)
                lon = float(rec.get("longitude") or rec.get("lon") or 0)

                # Confidence
                raw_conf = rec.get("confidence", "n")
                if raw_conf in _viirs_conf:
                    conf = _viirs_conf[raw_conf]
                else:
                    try:
                        conf = float(raw_conf) / 100.0
                    except (ValueError, TypeError):
                        conf = 0.5

                # FRP (fire radiative power) → intensity proxy
                frp = float(rec.get("frp") or 0)
                if frp > 500:
                    obs_class = "large wildfire"
                    conf = max(conf, 0.85)
                elif frp > 100:
                    obs_class = "active fire"
                    conf = max(conf, 0.75)
                elif frp > 20:
                    obs_class = "fire"
                else:
                    obs_class = "smoke"

                risk = (
                    "CRITICAL"
                    if (conf >= 0.80 or frp > 100)
                    else ("HIGH" if conf >= 0.60 else "MEDIUM")
                )

                rows.append(
                    {
                        "class": obs_class,
                        "confidence": round(conf, 3),
                        "lat": round(lat, 5),
                        "lon": round(lon, 5),
                        "mission_type": "SURVEY",
                        "risk_level": risk,
                        "source": f"firms_{sensor}",
                    }
                )
                count += 1
            except (ValueError, KeyError):
                continue

        print(f"  {sensor}: {count} fire detections")

    return rows


# ── NOAA Storm Events ─────────────────────────────────────────────────────────

# Maps NOAA event types to (mission_type, risk_level)
_NOAA_EVENT_MAP = {
    # Flood events
    "flash flood": ("PERIMETER", "CRITICAL"),
    "flood": ("SURVEY", "HIGH"),
    "coastal flood": ("SURVEY", "HIGH"),
    "lakeshore flood": ("SURVEY", "MEDIUM"),
    "storm surge/tide": ("PERIMETER", "CRITICAL"),
    # Wind/tornado
    "tornado": ("PERIMETER", "CRITICAL"),
    "funnel cloud": ("MONITOR", "HIGH"),
    "waterspout": ("MONITOR", "HIGH"),
    "hurricane (typhoon)": ("SURVEY", "CRITICAL"),
    "tropical storm": ("SURVEY", "HIGH"),
    # Fire weather
    "wildfire": ("SURVEY", "CRITICAL"),
    "dense smoke": ("SURVEY", "HIGH"),
    # Winter/ice
    "avalanche": ("SEARCH", "CRITICAL"),
    "blizzard": ("SURVEY", "HIGH"),
    "ice storm": ("INSPECT", "MEDIUM"),
    # Hazmat-adjacent
    "dust storm": ("SURVEY", "MEDIUM"),
    "dense fog": ("MONITOR", "MEDIUM"),
    # Structural
    "landslide": ("SEARCH", "CRITICAL"),
    "debris flow": ("PERIMETER", "CRITICAL"),
    # Marine
    "marine thunderstorm wind": ("MONITOR", "HIGH"),
    "high surf": ("MONITOR", "HIGH"),
    "rip current": ("SEARCH", "CRITICAL"),
    "marine dense fog": ("MONITOR", "MEDIUM"),
}


# NOAA magnitude scale → confidence proxy
def _noaa_magnitude_to_conf(mag: str, mag_type: str) -> float:
    try:
        v = float(mag)
        if mag_type in ("EF", "F"):  # tornado scale 0-5
            return min(0.95, 0.50 + v * 0.09)
        if mag_type in ("MPH", "KTS"):  # wind speed
            return min(0.95, 0.40 + v / 200.0)
        if mag_type in ("FT",):  # hail/surge
            return min(0.90, 0.50 + v / 20.0)
        return 0.70
    except (ValueError, TypeError):
        return 0.70


def download_noaa_storm_events(year: int = 2023, max_rows: int = 30000) -> List[Dict]:
    """
    Download NOAA Storm Events database for a given year.
    Hosted on NOAA FTP as a zip of CSVs.

    Covers: tornadoes, floods, hurricanes, wildfires, avalanches, landslides,
            rip currents, storm surge, marine events.
    """
    print(f"Downloading NOAA Storm Events {year}...")

    # Details file has lat/lon and event type
    url = f"https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/StormEvents_details-ftp_v1.0_d{year}_c20240620.csv.gz"

    # Try to find the right filename (NOAA updates the _c date suffix)
    # Try a few known suffixes
    base = f"https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/"
    candidate_suffixes = ["20240620", "20240516", "20231116", "20240417"]

    data = None
    for suffix in candidate_suffixes:
        url = f"{base}StormEvents_details-ftp_v1.0_d{year}_c{suffix}.csv.gz"
        data = _fetch(url, timeout=60)
        if data:
            break

    if not data:
        print(f"  Could not fetch NOAA storm events for {year} — trying index")
        # Try index page to find actual filename
        idx = _fetch(base, timeout=30)
        if idx:
            text = idx.decode("utf-8", errors="replace")
            import re

            matches = re.findall(
                rf"StormEvents_details-ftp_v1\.0_d{year}_c\d+\.csv\.gz", text
            )
            if matches:
                url = base + matches[-1]
                data = _fetch(url, timeout=60)

    if not data:
        print(f"  NOAA storm events unavailable offline — skipping")
        return []

    import gzip

    try:
        csv_bytes = gzip.decompress(data)
    except Exception as e:
        print(f"  Decompress failed: {e}")
        return []

    text = csv_bytes.decode("utf-8", errors="replace")
    reader = csv.DictReader(io.StringIO(text))

    rows = []
    for rec in reader:
        if len(rows) >= max_rows:
            break
        event_type = (rec.get("EVENT_TYPE") or "").lower().strip()
        mapped = _NOAA_EVENT_MAP.get(event_type)
        if mapped is None:
            continue

        mission_type, risk_level = mapped

        # Location: BEGIN_LAT/BEGIN_LON (decimal degrees)
        try:
            lat = float(rec.get("BEGIN_LAT") or 0)
            lon = float(rec.get("BEGIN_LON") or 0)
            if lat == 0 and lon == 0:
                continue
        except (ValueError, TypeError):
            continue

        mag = rec.get("MAGNITUDE") or ""
        mag_type = rec.get("MAGNITUDE_TYPE") or ""
        conf = _noaa_magnitude_to_conf(mag, mag_type)

        rows.append(
            {
                "class": event_type,
                "confidence": round(conf, 3),
                "lat": round(lat, 5),
                "lon": round(lon, 5),
                "mission_type": mission_type,
                "risk_level": risk_level,
                "source": f"noaa_storm_{year}",
            }
        )

    print(f"  NOAA storm events: {len(rows)} labeled incidents")
    return rows


# ── PHMSA Pipeline & Hazmat Incidents ────────────────────────────────────────


def download_phmsa(max_rows: int = 20000) -> List[Dict]:
    """
    Download PHMSA pipeline incident data.
    Source: https://www.phmsa.dot.gov/data-and-statistics/pipeline/pipeline-incident-flagged-files
    """
    print("Downloading PHMSA hazmat/pipeline incidents...")

    # PHMSA provides annual CSVs; use gas transmission incidents
    urls = [
        "https://www.phmsa.dot.gov/sites/phmsa.dot.gov/files/2024-06/incident_gas_transmission_gathering_jan2024.csv",
        "https://www.phmsa.dot.gov/sites/phmsa.dot.gov/files/2024-06/incident_hazmat_air_jan2024.csv",
    ]

    rows = []
    for url in urls:
        data = _fetch(url, timeout=60)
        if not data:
            continue
        # PHMSA files sometimes have BOM
        text = data.decode("utf-8-sig", errors="replace")
        reader = csv.DictReader(io.StringIO(text))
        count = 0
        for rec in reader:
            if count >= max_rows // len(urls):
                break
            try:
                lat_str = rec.get("LOCATION_LATITUDE") or rec.get("LATITUDE") or ""
                lon_str = rec.get("LOCATION_LONGITUDE") or rec.get("LONGITUDE") or ""
                lat = float(lat_str)
                lon = float(lon_str)
                if abs(lat) < 0.1 and abs(lon) < 0.1:
                    continue

                # Determine severity
                fatalities = int(rec.get("FATAL") or rec.get("FATALITIES") or 0)
                injuries = int(rec.get("INJURE") or rec.get("INJURIES") or 0)
                ignite = str(rec.get("IGNITE_IND") or rec.get("IGNITION") or "").upper()

                if fatalities > 0 or ignite == "YES":
                    conf, risk = 0.92, "CRITICAL"
                elif injuries > 0:
                    conf, risk = 0.82, "HIGH"
                else:
                    conf, risk = 0.68, "MEDIUM"

                rows.append(
                    {
                        "class": "pipeline leak",
                        "confidence": conf,
                        "lat": round(lat, 5),
                        "lon": round(lon, 5),
                        "mission_type": "PERIMETER",
                        "risk_level": risk,
                        "source": "phmsa_pipeline",
                    }
                )
                count += 1
            except (ValueError, TypeError):
                continue
        print(f"  PHMSA: {count} incidents from {url.split('/')[-1]}")

    return rows


# ── GBIF Wildlife Occurrences ─────────────────────────────────────────────────


def download_gbif_wildlife(max_rows: int = 5000) -> List[Dict]:
    """
    Sample wildlife occurrence data from GBIF public API.
    Maps to MONITOR missions for animal observations.
    Large predator species → PERIMETER.
    """
    print("Downloading GBIF wildlife observations...")

    # Query large mammals likely relevant to field operations
    species_queries = [
        ("Ursus arctos", "brown bear", "PERIMETER", "HIGH"),
        ("Panthera leo", "lion", "PERIMETER", "HIGH"),
        ("Canis lupus", "wolf", "MONITOR", "MEDIUM"),
        ("Loxodonta africana", "elephant", "MONITOR", "MEDIUM"),
        ("Crocuta crocuta", "hyena", "MONITOR", "MEDIUM"),
    ]

    rows = []
    per_species = max_rows // len(species_queries)

    for scientific, common, mission_type, risk_level in species_queries:
        url = (
            f"https://api.gbif.org/v1/occurrence/search"
            f"?scientificName={urllib.parse.quote(scientific)}"
            f"&hasCoordinate=true&limit={min(per_species, 300)}"
            f"&fields=scientificName,decimalLatitude,decimalLongitude,coordinateUncertaintyInMeters"
        )
        data = _fetch(url, timeout=15)
        if not data:
            continue
        try:
            payload = json.loads(data)
            results = payload.get("results", [])
            for rec in results:
                lat = rec.get("decimalLatitude")
                lon = rec.get("decimalLongitude")
                uncertainty = rec.get("coordinateUncertaintyInMeters") or 10000
                if lat is None or lon is None:
                    continue
                # Lower confidence for high-uncertainty fixes
                conf = max(0.50, min(0.92, 1.0 - uncertainty / 50000.0))
                rows.append(
                    {
                        "class": f"{common} sighting",
                        "confidence": round(conf, 3),
                        "lat": round(float(lat), 5),
                        "lon": round(float(lon), 5),
                        "mission_type": mission_type,
                        "risk_level": risk_level,
                        "source": "gbif_wildlife",
                    }
                )
        except Exception as e:
            print(f"  GBIF {common}: {e}")
        time.sleep(0.5)  # be a polite API consumer

    print(f"  GBIF: {len(rows)} wildlife observations")
    return rows


# ── USGS Earthquake Catalog ───────────────────────────────────────────────────


def download_usgs_earthquakes(
    min_magnitude: float = 4.0, max_rows: int = 10000
) -> List[Dict]:
    """
    Download earthquake events from USGS ComCat API.
    No auth required. Real-time global catalog.

    Mission mapping:
      M4.0–5.5  → SURVEY (damage assessment)
      M5.5–6.5  → SEARCH (building collapse rescue)
      M6.5+     → SEARCH CRITICAL (mass casualty, structural collapse)
    """
    print("Downloading USGS earthquake catalog...")
    import urllib.parse

    # Pull last 10 years of significant quakes globally
    params = {
        "format": "geojson",
        "minmagnitude": str(min_magnitude),
        "orderby": "magnitude",
        "limit": str(min(max_rows, 20000)),
        "starttime": "2015-01-01",
    }
    url = "https://earthquake.usgs.gov/fdsnws/event/1/query?" + urllib.parse.urlencode(
        params
    )
    data = _fetch(url, timeout=60)
    if not data:
        print("  USGS earthquakes unavailable — skipping")
        return []

    try:
        payload = json.loads(data)
    except Exception as e:
        print(f"  USGS parse error: {e}")
        return []

    rows = []
    for feature in payload.get("features", []):
        props = feature.get("properties", {})
        geom = feature.get("geometry", {})
        coords = geom.get("coordinates", [None, None, None])
        try:
            lon = float(coords[0])
            lat = float(coords[1])
            mag = float(props.get("mag") or 0)
        except (TypeError, ValueError):
            continue

        if mag < min_magnitude:
            continue

        if mag >= 6.5:
            obs_class, mission, risk, conf = (
                "major earthquake",
                "SEARCH",
                "CRITICAL",
                0.97,
            )
        elif mag >= 5.5:
            obs_class, mission, risk, conf = (
                "significant earthquake",
                "SEARCH",
                "HIGH",
                0.90,
            )
        elif mag >= 5.0:
            obs_class, mission, risk, conf = (
                "moderate earthquake",
                "SURVEY",
                "HIGH",
                0.82,
            )
        else:
            obs_class, mission, risk, conf = "earthquake", "SURVEY", "MEDIUM", 0.70

        rows.append(
            {
                "class": obs_class,
                "confidence": round(conf, 3),
                "lat": round(lat, 5),
                "lon": round(lon, 5),
                "mission_type": mission,
                "risk_level": risk,
                "source": "usgs_earthquakes",
            }
        )

    print(f"  USGS earthquakes: {len(rows)} events (M{min_magnitude}+)")
    return rows


# ── NASA EONET — Natural Events ───────────────────────────────────────────────

_EONET_CATEGORY_MAP = {
    "Wildfires": ("active fire", "SURVEY", "CRITICAL", 0.90),
    "Volcanoes": ("volcanic eruption", "PERIMETER", "HIGH", 0.88),
    "Floods": ("flood", "SURVEY", "HIGH", 0.82),
    "Severe Storms": ("severe storm", "PERIMETER", "HIGH", 0.80),
    "Landslides": ("landslide", "SEARCH", "CRITICAL", 0.88),
    "Dust and Haze": ("dust storm", "SURVEY", "MEDIUM", 0.72),
    "Sea and Lake Ice": ("ice hazard", "MONITOR", "MEDIUM", 0.75),
    "Manmade": ("industrial incident", "PERIMETER", "HIGH", 0.80),
    "Snow": ("blizzard", "SURVEY", "MEDIUM", 0.68),
    "Temperature Extremes": ("extreme heat", "SURVEY", "MEDIUM", 0.70),
    "Earthquakes": ("earthquake", "SEARCH", "HIGH", 0.85),
    "Drought": ("drought", "SURVEY", "LOW", 0.60),
    "Water Color": ("water contamination", "MONITOR", "MEDIUM", 0.65),
}


def download_nasa_eonet(max_rows: int = 5000) -> List[Dict]:
    """
    Download natural event data from NASA EONET (Earth Observatory Natural Event Tracker).
    No auth required. Covers wildfires, volcanoes, floods, severe storms, landslides, etc.
    """
    print("Downloading NASA EONET natural events...")
    import urllib.parse

    url = f"https://eonet.gsfc.nasa.gov/api/v3/events?status=closed&limit={min(max_rows, 2500)}&days=1825"
    data = _fetch(url, timeout=30)
    if not data:
        print("  NASA EONET unavailable — skipping")
        return []

    try:
        payload = json.loads(data)
    except Exception as e:
        print(f"  EONET parse error: {e}")
        return []

    rows = []
    for event in payload.get("events", []):
        categories = event.get("categories", [])
        if not categories:
            continue
        cat_title = categories[0].get("title", "")
        mapping = _EONET_CATEGORY_MAP.get(cat_title)
        if not mapping:
            continue

        obs_class, mission, risk, base_conf = mapping

        # Extract coordinates from geometry
        for geom in event.get("geometry", []):
            coords = geom.get("coordinates")
            if not coords:
                continue
            try:
                # Point: [lon, lat] or Polygon: [[[lon,lat],...]]
                if geom.get("type") == "Point":
                    lon, lat = float(coords[0]), float(coords[1])
                elif geom.get("type") == "Polygon":
                    lon, lat = float(coords[0][0][0]), float(coords[0][0][1])
                else:
                    continue
                rows.append(
                    {
                        "class": obs_class,
                        "confidence": round(base_conf, 3),
                        "lat": round(lat, 5),
                        "lon": round(lon, 5),
                        "mission_type": mission,
                        "risk_level": risk,
                        "source": "nasa_eonet",
                    }
                )
                break  # one point per event is enough
            except (TypeError, ValueError, IndexError):
                continue

    print(f"  NASA EONET: {len(rows)} natural events")
    return rows


# ── iNaturalist — Expanded Wildlife ──────────────────────────────────────────

_INAT_SPECIES = [
    # Large predators / dangerous wildlife → PERIMETER
    ("Ursus arctos", "grizzly bear", "PERIMETER", "HIGH", 0.88),
    ("Ursus americanus", "black bear", "PERIMETER", "MEDIUM", 0.80),
    ("Panthera onca", "jaguar", "PERIMETER", "HIGH", 0.88),
    ("Panthera leo", "lion", "PERIMETER", "HIGH", 0.90),
    ("Panthera tigris", "tiger", "PERIMETER", "HIGH", 0.90),
    ("Puma concolor", "mountain lion", "PERIMETER", "HIGH", 0.85),
    ("Crocodylia", "crocodile", "PERIMETER", "HIGH", 0.87),
    ("Canis lupus", "wolf", "MONITOR", "MEDIUM", 0.78),
    ("Alces alces", "moose", "MONITOR", "MEDIUM", 0.75),
    ("Bison bison", "bison", "MONITOR", "MEDIUM", 0.75),
    # Marine mammals / vessel hazard → MONITOR
    ("Orcinus orca", "orca", "MONITOR", "MEDIUM", 0.85),
    ("Tursiops truncatus", "bottlenose dolphin", "MONITOR", "LOW", 0.72),
    ("Mirounga", "elephant seal", "MONITOR", "LOW", 0.70),
    ("Megaptera novaeangliae", "humpback whale", "MONITOR", "MEDIUM", 0.80),
    # Invasive / feral
    ("Python bivittatus", "burmese python", "PERIMETER", "MEDIUM", 0.78),
    ("Sus scrofa", "feral pig", "MONITOR", "LOW", 0.68),
    # Threatened / endangered → MONITOR (conservation ops)
    ("Ailuropoda melanoleuca", "giant panda", "MONITOR", "MEDIUM", 0.82),
    ("Gorilla gorilla", "western gorilla", "MONITOR", "HIGH", 0.85),
    ("Diceros bicornis", "black rhino", "MONITOR", "HIGH", 0.87),
    ("Elephas maximus", "asian elephant", "MONITOR", "MEDIUM", 0.82),
    # Birds relevant to UAV ops / wildlife strikes
    ("Haliaeetus leucocephalus", "bald eagle", "MONITOR", "LOW", 0.78),
    ("Aquila chrysaetos", "golden eagle", "MONITOR", "LOW", 0.75),
]


def download_inat_wildlife(max_rows: int = 8000) -> List[Dict]:
    """
    Download wildlife observations from iNaturalist API.
    Research-grade observations only (community-verified).
    ~100M observations available; much richer than GBIF.
    """
    print("Downloading iNaturalist wildlife observations...")
    import urllib.parse

    rows = []
    per_species = max(10, max_rows // len(_INAT_SPECIES))

    for sci_name, common, mission, risk, base_conf in _INAT_SPECIES:
        params = {
            "taxon_name": sci_name,
            "quality_grade": "research",
            "has_geo": "true",
            "per_page": str(min(per_species, 200)),
            "order_by": "observed_on",
        }
        url = "https://api.inaturalist.org/v1/observations?" + urllib.parse.urlencode(
            params
        )
        data = _fetch(url, timeout=20)
        if not data:
            time.sleep(1)
            continue

        try:
            payload = json.loads(data)
        except Exception:
            continue

        count = 0
        for obs in payload.get("results", []):
            loc = obs.get("location")
            if not loc:
                continue
            try:
                lat_s, lon_s = loc.split(",")
                lat, lon = float(lat_s), float(lon_s)
            except (ValueError, AttributeError):
                continue

            # Positional accuracy → confidence adjustment
            acc = obs.get("positional_accuracy") or 10000
            conf = round(max(0.50, min(base_conf, base_conf - acc / 200000.0)), 3)

            rows.append(
                {
                    "class": f"{common} sighting",
                    "confidence": conf,
                    "lat": round(lat, 5),
                    "lon": round(lon, 5),
                    "mission_type": mission,
                    "risk_level": risk,
                    "source": "inat_wildlife",
                }
            )
            count += 1

        time.sleep(0.75)  # iNaturalist rate limit: ~60 req/min

    print(
        f"  iNaturalist: {len(rows)} wildlife observations ({len(_INAT_SPECIES)} species)"
    )
    return rows


# ── OpenAQ — Air Quality (fire smoke + hazmat proxy) ─────────────────────────


def download_openaq(max_rows: int = 5000) -> List[Dict]:
    """
    Download air quality readings from OpenAQ.
    High PM2.5/PM10 → fire smoke proxy → SURVEY
    High NO2/SO2    → industrial/hazmat proxy → PERIMETER
    No auth required for public data.
    """
    print("Downloading OpenAQ air quality readings...")

    rows = []

    # PM2.5: fine particulate (wildfire smoke, industrial)
    for parameter, threshold_high, threshold_critical, obs_class, mission in [
        ("pm25", 75.0, 150.0, "smoke haze", "SURVEY"),
        ("pm10", 150.0, 350.0, "dust hazard", "SURVEY"),
        ("no2", 100.0, 200.0, "industrial emission", "PERIMETER"),
        ("so2", 50.0, 150.0, "chemical release", "PERIMETER"),
    ]:
        url = (
            f"https://api.openaq.org/v2/measurements"
            f"?parameter={parameter}&limit={min(max_rows // 4, 500)}"
            f"&order_by=value&sort=desc&has_geo=true"
        )
        data = _fetch(url, timeout=20)
        if not data:
            continue
        try:
            payload = json.loads(data)
        except Exception:
            continue

        for rec in payload.get("results", []):
            val = rec.get("value")
            coords = rec.get("coordinates") or {}
            lat = coords.get("latitude")
            lon = coords.get("longitude")
            if val is None or lat is None or lon is None:
                continue
            try:
                val = float(val)
                lat = float(lat)
                lon = float(lon)
            except (ValueError, TypeError):
                continue

            if val >= threshold_critical:
                risk, conf = "CRITICAL", 0.92
            elif val >= threshold_high:
                risk, conf = "HIGH", 0.80
            else:
                continue  # below threshold — not interesting for training

            rows.append(
                {
                    "class": obs_class,
                    "confidence": round(conf, 3),
                    "lat": round(lat, 5),
                    "lon": round(lon, 5),
                    "mission_type": mission,
                    "risk_level": risk,
                    "source": f"openaq_{parameter}",
                }
            )

        time.sleep(0.5)

    print(f"  OpenAQ: {len(rows)} high-pollution readings")
    return rows


# ── GDACS — Global Disaster Alert and Coordination System ────────────────────

_GDACS_ALERT_MAP = {
    "EQ": {  # earthquake
        "Red": ("major earthquake", "SEARCH", "CRITICAL", 0.95),
        "Orange": ("significant earthquake", "SEARCH", "HIGH", 0.88),
        "Green": ("earthquake", "SURVEY", "MEDIUM", 0.72),
    },
    "TC": {  # tropical cyclone
        "Red": ("major hurricane", "PERIMETER", "CRITICAL", 0.95),
        "Orange": ("tropical cyclone", "PERIMETER", "HIGH", 0.88),
        "Green": ("tropical storm", "SURVEY", "MEDIUM", 0.72),
    },
    "FL": {  # flood
        "Red": ("major flood", "PERIMETER", "CRITICAL", 0.92),
        "Orange": ("flood", "SURVEY", "HIGH", 0.82),
        "Green": ("minor flooding", "SURVEY", "MEDIUM", 0.65),
    },
    "VO": {  # volcano
        "Red": ("volcanic eruption", "PERIMETER", "CRITICAL", 0.93),
        "Orange": ("volcanic activity", "PERIMETER", "HIGH", 0.85),
        "Green": ("volcanic alert", "MONITOR", "MEDIUM", 0.68),
    },
    "WF": {  # wildfire
        "Red": ("large wildfire", "SURVEY", "CRITICAL", 0.94),
        "Orange": ("wildfire", "SURVEY", "HIGH", 0.85),
        "Green": ("fire alert", "SURVEY", "MEDIUM", 0.68),
    },
    "DR": {  # drought
        "Red": ("severe drought", "SURVEY", "HIGH", 0.80),
        "Orange": ("drought", "SURVEY", "MEDIUM", 0.65),
    },
}


def download_gdacs(max_rows: int = 3000) -> List[Dict]:
    """
    Download global disaster alerts from GDACS GeoJSON feed.
    Covers earthquakes, tropical cyclones, floods, volcanoes, wildfires.
    No auth required.
    """
    print("Downloading GDACS global disaster alerts...")

    # GDACS provides a GeoJSON feed of recent/historical events
    url = "https://www.gdacs.org/gdacsapi/api/events/geteventlist/EVENTS?eventlist=EQ,TC,FL,VO,WF,DR&alertlevel=Red,Orange,Green&fromDate=2018-01-01&toDate=2024-12-31&pagesize=500"
    data = _fetch(url, timeout=30)

    if not data:
        # Fallback: try the simpler RSS-based JSON
        url = "https://www.gdacs.org/xml/gdacs_rss_1_0_detail.xml"
        data = _fetch(url, timeout=30)
        if not data:
            print("  GDACS unavailable — skipping")
            return []
        # Parse RSS as text, extract basic info
        text = data.decode("utf-8", errors="replace")
        rows = _parse_gdacs_rss(text, max_rows)
        print(f"  GDACS RSS: {len(rows)} disaster alerts")
        return rows

    try:
        payload = json.loads(data)
    except Exception:
        print("  GDACS parse error — skipping")
        return []

    rows = []
    features = payload.get("features") or payload.get("Features") or []
    for feature in features:
        if len(rows) >= max_rows:
            break
        props = feature.get("properties", {})
        geom = feature.get("geometry", {})
        coords = geom.get("coordinates", [None, None])

        event_type = str(props.get("eventtype") or "").upper()
        alert_level = str(props.get("alertlevel") or "Green").capitalize()

        level_map = _GDACS_ALERT_MAP.get(event_type, {})
        mapping = level_map.get(alert_level) or level_map.get("Green")
        if not mapping:
            continue

        obs_class, mission, risk, conf = mapping

        try:
            lon = float(coords[0])
            lat = float(coords[1])
        except (TypeError, ValueError, IndexError):
            continue

        rows.append(
            {
                "class": obs_class,
                "confidence": round(conf, 3),
                "lat": round(lat, 5),
                "lon": round(lon, 5),
                "mission_type": mission,
                "risk_level": risk,
                "source": f"gdacs_{event_type.lower()}",
            }
        )

    print(f"  GDACS: {len(rows)} disaster alerts")
    return rows


def _parse_gdacs_rss(text: str, max_rows: int) -> List[Dict]:
    """Fallback: extract lat/lon/event type from GDACS RSS XML."""
    import re

    rows = []
    items = re.findall(r"<item>(.*?)</item>", text, re.DOTALL)
    for item in items[:max_rows]:
        try:
            lat_m = re.search(r"<geo:lat>([\d.\-]+)", item)
            lon_m = re.search(r"<geo:long>([\d.\-]+)", item)
            type_m = re.search(r"<gdacs:eventtype>(.*?)</gdacs:eventtype>", item)
            alert_m = re.search(r"<gdacs:alertlevel>(.*?)</gdacs:alertlevel>", item)
            if not (lat_m and lon_m):
                continue
            lat = float(lat_m.group(1))
            lon = float(lon_m.group(1))
            event_type = (type_m.group(1) if type_m else "EQ").upper()
            alert_level = (alert_m.group(1) if alert_m else "Green").capitalize()
            level_map = _GDACS_ALERT_MAP.get(event_type, {})
            mapping = level_map.get(alert_level) or level_map.get("Green")
            if not mapping:
                continue
            obs_class, mission, risk, conf = mapping
            rows.append(
                {
                    "class": obs_class,
                    "confidence": round(conf, 3),
                    "lat": round(lat, 5),
                    "lon": round(lon, 5),
                    "mission_type": mission,
                    "risk_level": risk,
                    "source": f"gdacs_{event_type.lower()}",
                }
            )
        except (ValueError, AttributeError):
            continue
    return rows


# ── ReliefWeb — Humanitarian Crisis Events ────────────────────────────────────

_RELIEFWEB_TYPE_MAP = {
    "Flood": ("flood", "SURVEY", "HIGH", 0.85),
    "Flash Flood": ("flash flood", "PERIMETER", "CRITICAL", 0.90),
    "Earthquake": ("earthquake", "SEARCH", "CRITICAL", 0.92),
    "Tsunami": ("tsunami", "PERIMETER", "CRITICAL", 0.95),
    "Tropical Cyclone": ("tropical cyclone", "PERIMETER", "CRITICAL", 0.92),
    "Drought": ("drought", "SURVEY", "MEDIUM", 0.70),
    "Fire": ("wildfire", "SURVEY", "CRITICAL", 0.90),
    "Volcano": ("volcanic eruption", "PERIMETER", "HIGH", 0.88),
    "Landslide": ("landslide", "SEARCH", "CRITICAL", 0.88),
    "Cold Wave": ("extreme cold", "SURVEY", "MEDIUM", 0.72),
    "Heat Wave": ("extreme heat", "SURVEY", "MEDIUM", 0.72),
    "Epidemic": ("disease outbreak", "DELIVER", "HIGH", 0.82),
    "Food Insecurity": ("aid required", "DELIVER", "HIGH", 0.78),
    "Industrial Accident": ("industrial incident", "PERIMETER", "HIGH", 0.85),
    "Transport Accident": ("vehicle incident", "SEARCH", "HIGH", 0.82),
    "Civil Unrest": ("security threat", "PERIMETER", "HIGH", 0.80),
    "Insect Infestation": ("crop threat", "SURVEY", "MEDIUM", 0.68),
    "Storm": ("severe storm", "PERIMETER", "HIGH", 0.82),
    "Mud Slide": ("mudslide", "SEARCH", "CRITICAL", 0.88),
    "Snow Avalanche": ("avalanche", "SEARCH", "CRITICAL", 0.90),
    "Sea Level Rise": ("coastal flood", "SURVEY", "HIGH", 0.75),
    "Oil Spill": ("hazmat spill", "PERIMETER", "HIGH", 0.88),
    "Chemical Incident": ("chemical release", "PERIMETER", "CRITICAL", 0.92),
    "Nuclear/Radiological": ("radiation hazard", "PERIMETER", "CRITICAL", 0.95),
}


def download_reliefweb(max_rows: int = 5000) -> List[Dict]:
    """
    Download humanitarian disaster events from ReliefWeb API.
    Covers 200+ countries, 1980-present, 20+ disaster types.
    No auth required.

    Coordinates come from country centroids when event-level geo not available —
    this adds geographic signal even without exact incident coords.
    """
    print("Downloading ReliefWeb humanitarian disasters...")
    import urllib.parse

    # Country centroids (lat, lon) for fallback geo
    _COUNTRY_CENTROIDS = {
        "Afghanistan": (33.9, 67.7),
        "Bangladesh": (23.7, 90.4),
        "Brazil": (-14.2, -51.9),
        "Cambodia": (12.6, 104.9),
        "China": (35.9, 104.2),
        "Colombia": (4.6, -74.1),
        "DR Congo": (-4.0, 21.8),
        "Ethiopia": (9.1, 40.5),
        "Haiti": (18.9, -72.3),
        "India": (20.6, 79.0),
        "Indonesia": (-0.8, 113.9),
        "Iraq": (33.2, 43.7),
        "Japan": (36.2, 138.3),
        "Kenya": (-0.0, 37.9),
        "Mexico": (23.6, -102.6),
        "Mozambique": (-18.7, 35.5),
        "Myanmar": (17.1, 96.0),
        "Nepal": (28.4, 84.1),
        "Nigeria": (9.1, 8.7),
        "Pakistan": (30.4, 69.3),
        "Peru": (-9.2, -75.0),
        "Philippines": (12.9, 121.8),
        "Somalia": (5.2, 46.2),
        "South Sudan": (6.9, 31.3),
        "Sudan": (15.6, 32.5),
        "Syria": (34.8, 38.9),
        "Turkey": (38.9, 35.2),
        "United States": (37.1, -95.7),
        "Venezuela": (6.4, -66.6),
        "Yemen": (15.6, 48.5),
        "Zimbabwe": (-20.0, 30.0),
        "Chile": (-30.0, -71.0),
        "Australia": (-25.3, 133.8),
        "Italy": (41.9, 12.6),
        "Greece": (39.1, 21.8),
        "Thailand": (15.9, 100.9),
        "Vietnam": (14.1, 108.3),
        "Malaysia": (4.2, 108.0),
    }

    params = {
        "appname": "summit-os-training",
        "fields[include][]": ["name", "type", "country", "date"],
        "limit": str(min(max_rows, 1000)),
        "sort[]": "date:desc",
        "filter[operator]": "AND",
    }
    url = "https://api.reliefweb.int/v1/disasters?" + urllib.parse.urlencode(
        params, doseq=True
    )
    data = _fetch(url, timeout=30)
    if not data:
        print("  ReliefWeb unavailable — skipping")
        return []

    try:
        payload = json.loads(data)
    except Exception as e:
        print(f"  ReliefWeb parse error: {e}")
        return []

    rows = []
    for item in payload.get("data", []):
        fields = item.get("fields", {})
        types = fields.get("type", [{}])
        disaster_type = types[0].get("name", "") if types else ""
        mapping = _RELIEFWEB_TYPE_MAP.get(disaster_type)
        if not mapping:
            continue

        obs_class, mission, risk, conf = mapping

        # Get country for centroid fallback
        countries = fields.get("country", [{}])
        country_name = countries[0].get("name", "") if countries else ""
        centroid = _COUNTRY_CENTROIDS.get(country_name)
        if not centroid:
            continue  # skip if no geo signal at all

        lat, lon = centroid
        # Small jitter so multiple events from same country don't stack exactly
        lat = round(lat + (hash(fields.get("name", "")) % 100) / 500.0 - 0.1, 5)
        lon = round(lon + (hash(disaster_type) % 100) / 500.0 - 0.1, 5)

        rows.append(
            {
                "class": obs_class,
                "confidence": round(conf, 3),
                "lat": lat,
                "lon": lon,
                "mission_type": mission,
                "risk_level": risk,
                "source": "reliefweb",
            }
        )

    print(f"  ReliefWeb: {len(rows)} humanitarian events")
    return rows


# ── Combine and deduplicate ───────────────────────────────────────────────────

ALL_SOURCES = [
    "firms",
    "noaa",
    "phmsa",
    "gbif",
    "usgs",
    "eonet",
    "inat",
    "openaq",
    "gdacs",
    "reliefweb",
]


def download_all(
    year: int = 2023, years: List[int] = None, sources: List[str] = None
) -> str:
    all_rows = []
    active = sources or ALL_SOURCES

    if "firms" in active:
        all_rows.extend(download_firms(max_rows=50000))

    if "noaa" in active:
        noaa_years = years or [year]
        for y in noaa_years:
            all_rows.extend(download_noaa_storm_events(year=y, max_rows=30000))

    if "phmsa" in active:
        all_rows.extend(download_phmsa(max_rows=10000))

    if "gbif" in active:
        all_rows.extend(download_gbif_wildlife(max_rows=3000))

    if "usgs" in active:
        all_rows.extend(download_usgs_earthquakes(min_magnitude=4.0, max_rows=10000))

    if "eonet" in active:
        all_rows.extend(download_nasa_eonet(max_rows=5000))

    if "inat" in active:
        all_rows.extend(download_inat_wildlife(max_rows=8000))

    if "openaq" in active:
        all_rows.extend(download_openaq(max_rows=5000))

    if "gdacs" in active:
        all_rows.extend(download_gdacs(max_rows=3000))

    if "reliefweb" in active:
        all_rows.extend(download_reliefweb(max_rows=5000))

    # Print breakdown by source
    from collections import Counter

    by_source = Counter(r["source"].split("_")[0] for r in all_rows)
    print("\nBreakdown by source:")
    for src, count in sorted(by_source.items(), key=lambda x: -x[1]):
        print(f"  {src:<20} {count:>6} rows")

    # Basic dedup on (class, lat, lon) rounded to 2 decimal places
    seen = set()
    deduped = []
    for r in all_rows:
        key = (r["class"], round(r["lat"], 2), round(r["lon"], 2))
        if key not in seen:
            seen.add(key)
            deduped.append(r)

    out_path = os.path.join(OUTPUT_DIR, "real_observations.csv")
    _write_csv(deduped, out_path)
    print(f"\nTotal: {len(deduped)} real observations → {out_path}")
    return out_path


if __name__ == "__main__":
    import urllib.parse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        choices=ALL_SOURCES,
        action="append",
        dest="sources",
        help="Specific source(s) to download (default: all)",
    )
    parser.add_argument("--year", type=int, default=2023)
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        help="Multiple NOAA years: --years 2020 2021 2022 2023",
    )
    args = parser.parse_args()
    download_all(year=args.year, years=args.years, sources=args.sources)
