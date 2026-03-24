"""
Download and preprocess public real-world datasets for Summit.OS model training.

Sources (all public, no auth required):
  NASA FIRMS    — active fire detections, global, updated every 10 min
  NOAA Storm    — severe weather events 1950-present (flood, tornado, hurricane, etc.)
  PHMSA         — pipeline and hazmat incidents 2010-present
  USCG SAR      — search and rescue incident summaries
  GBIF          — wildlife occurrence observations (API)

Output: data/real_observations.csv
  Columns: class, confidence, lat, lon, mission_type, risk_level, source

Usage:
  python download_real_data.py                  # download all sources
  python download_real_data.py --source firms   # single source
  python download_real_data.py --year 2023      # NOAA specific year
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
                time.sleep(2 ** attempt)
            else:
                print(f"  Failed: {e}")
                return None
    return None


def _write_csv(rows: List[Dict], path: str):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["class", "confidence", "lat", "lon",
                                                "mission_type", "risk_level", "source"])
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
        ("https://firms.modaps.eosdis.nasa.gov/data/active_fire/noaa-20-viirs-c2/csv/J1_VIIRS_C2_Global_7d.csv", "viirs_noaa20"),
        # VIIRS SUOMI NPP
        ("https://firms.modaps.eosdis.nasa.gov/data/active_fire/suomi-npp-viirs-c2/csv/SUOMI_VIIRS_C2_Global_7d.csv", "viirs_suomi"),
        # MODIS (older, broader coverage)
        ("https://firms.modaps.eosdis.nasa.gov/data/active_fire/modis-c6.1/csv/MODIS_C6_1_Global_7d.csv", "modis"),
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

                risk = "CRITICAL" if (conf >= 0.80 or frp > 100) else ("HIGH" if conf >= 0.60 else "MEDIUM")

                rows.append({
                    "class": obs_class,
                    "confidence": round(conf, 3),
                    "lat": round(lat, 5),
                    "lon": round(lon, 5),
                    "mission_type": "SURVEY",
                    "risk_level": risk,
                    "source": f"firms_{sensor}",
                })
                count += 1
            except (ValueError, KeyError):
                continue

        print(f"  {sensor}: {count} fire detections")

    return rows


# ── NOAA Storm Events ─────────────────────────────────────────────────────────

# Maps NOAA event types to (mission_type, risk_level)
_NOAA_EVENT_MAP = {
    # Flood events
    "flash flood":           ("PERIMETER", "CRITICAL"),
    "flood":                 ("SURVEY",    "HIGH"),
    "coastal flood":         ("SURVEY",    "HIGH"),
    "lakeshore flood":       ("SURVEY",    "MEDIUM"),
    "storm surge/tide":      ("PERIMETER", "CRITICAL"),

    # Wind/tornado
    "tornado":               ("PERIMETER", "CRITICAL"),
    "funnel cloud":          ("MONITOR",   "HIGH"),
    "waterspout":            ("MONITOR",   "HIGH"),
    "hurricane (typhoon)":   ("SURVEY",    "CRITICAL"),
    "tropical storm":        ("SURVEY",    "HIGH"),

    # Fire weather
    "wildfire":              ("SURVEY",    "CRITICAL"),
    "dense smoke":           ("SURVEY",    "HIGH"),

    # Winter/ice
    "avalanche":             ("SEARCH",    "CRITICAL"),
    "blizzard":              ("SURVEY",    "HIGH"),
    "ice storm":             ("INSPECT",   "MEDIUM"),

    # Hazmat-adjacent
    "dust storm":            ("SURVEY",    "MEDIUM"),
    "dense fog":             ("MONITOR",   "MEDIUM"),

    # Structural
    "landslide":             ("SEARCH",    "CRITICAL"),
    "debris flow":           ("PERIMETER", "CRITICAL"),

    # Marine
    "marine thunderstorm wind": ("MONITOR", "HIGH"),
    "high surf":             ("MONITOR",   "HIGH"),
    "rip current":           ("SEARCH",    "CRITICAL"),
    "marine dense fog":      ("MONITOR",   "MEDIUM"),
}

# NOAA magnitude scale → confidence proxy
def _noaa_magnitude_to_conf(mag: str, mag_type: str) -> float:
    try:
        v = float(mag)
        if mag_type in ("EF", "F"):        # tornado scale 0-5
            return min(0.95, 0.50 + v * 0.09)
        if mag_type in ("MPH", "KTS"):     # wind speed
            return min(0.95, 0.40 + v / 200.0)
        if mag_type in ("FT",):            # hail/surge
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
            matches = re.findall(rf'StormEvents_details-ftp_v1\.0_d{year}_c\d+\.csv\.gz', text)
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

        rows.append({
            "class": event_type,
            "confidence": round(conf, 3),
            "lat": round(lat, 5),
            "lon": round(lon, 5),
            "mission_type": mission_type,
            "risk_level": risk_level,
            "source": f"noaa_storm_{year}",
        })

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
                injuries   = int(rec.get("INJURE") or rec.get("INJURIES") or 0)
                ignite     = str(rec.get("IGNITE_IND") or rec.get("IGNITION") or "").upper()

                if fatalities > 0 or ignite == "YES":
                    conf, risk = 0.92, "CRITICAL"
                elif injuries > 0:
                    conf, risk = 0.82, "HIGH"
                else:
                    conf, risk = 0.68, "MEDIUM"

                rows.append({
                    "class": "pipeline leak",
                    "confidence": conf,
                    "lat": round(lat, 5),
                    "lon": round(lon, 5),
                    "mission_type": "PERIMETER",
                    "risk_level": risk,
                    "source": "phmsa_pipeline",
                })
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
                rows.append({
                    "class": f"{common} sighting",
                    "confidence": round(conf, 3),
                    "lat": round(float(lat), 5),
                    "lon": round(float(lon), 5),
                    "mission_type": mission_type,
                    "risk_level": risk_level,
                    "source": "gbif_wildlife",
                })
        except Exception as e:
            print(f"  GBIF {common}: {e}")
        time.sleep(0.5)  # be a polite API consumer

    print(f"  GBIF: {len(rows)} wildlife observations")
    return rows


# ── Combine and deduplicate ───────────────────────────────────────────────────

def download_all(year: int = 2023, years: List[int] = None, sources: List[str] = None) -> str:
    all_rows = []
    active = sources or ["firms", "noaa", "phmsa", "gbif"]

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
    try:
        import urllib.parse
    except ImportError:
        pass

    import urllib.parse

    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["firms", "noaa", "phmsa", "gbif"],
                        action="append", dest="sources")
    parser.add_argument("--year", type=int, default=2023)
    parser.add_argument("--years", type=int, nargs="+",
                        help="Multiple NOAA years, e.g. --years 2018 2019 2020 2021 2022 2023")
    args = parser.parse_args()
    download_all(year=args.year, years=args.years, sources=args.sources)
