"""
Conflict Event Loaders — ACLED + UCDP
==========================================
Two open armed-conflict event databases for federal SA, civilian crisis
mapping, and pattern-of-life modeling.

ACLED (Armed Conflict Location and Event Data Project)
  https://api.acleddata.com/acled/read
  Auth:    free academic / research key (ACLED_API_KEY + ACLED_EMAIL).
  Rows:    ~1.8M+ events globally, daily updates. Categorized
           (Battles, Explosions/Remote violence, Riots, Protests,
           Strategic developments, Violence against civilians).

UCDP (Uppsala Conflict Data Program)
  https://ucdpapi.pcr.uu.se/api
  Auth:    none.
  Rows:    GED (Georeferenced Event Dataset) covering 1989-present.

Why this matters:
  - Federal: global situational awareness, conflict context for COCOM
    briefings, deployment-environment risk overlays
  - Civilian: humanitarian crisis mapping, NGO operations support
  - Pattern of life: regional violence rate as feature for downstream
    risk models

Usage:
    from packages.training.datasets.conflict_events import (
        load_acled, load_ucdp_ged,
    )
    events = load_acled(start="2024-01-01", end="2024-12-31",
                        countries=["Ukraine", "Israel"])
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)

OUT_DIR = Path(__file__).parent.parent / "data" / "conflict_events"

ACLED_API = "https://api.acleddata.com/acled/read"
UCDP_GED  = "https://ucdpapi.pcr.uu.se/api/gedevents/24.1"

ACLED_EVENT_TYPES = [
    "Battles", "Explosions/Remote violence", "Violence against civilians",
    "Protests", "Riots", "Strategic developments",
]


def _env(name: str) -> str:
    val = os.environ.get(name, "")
    if val:
        return val
    env = Path(__file__).parent.parent.parent.parent / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith(f"{name}="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


# -- ACLED -----------------------------------------------------------------


def load_acled(start: str, end: str,
               countries: Optional[list[str]] = None,
               event_types: Optional[list[str]] = None,
               limit: int = 100_000) -> list[dict]:
    """ACLED REST query — returns event dicts.

    Args:
        start, end: ISO dates (YYYY-MM-DD)
        countries: optional list of country names (e.g. ["Ukraine"])
        event_types: optional list from ACLED_EVENT_TYPES
        limit: cap total rows pulled across paged calls
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cache_key = f"acled_{start}_{end}_" + "_".join(countries or ["all"])
    cache = OUT_DIR / f"{cache_key.replace('/', '_')}.json"
    if cache.exists():
        try:
            return json.loads(cache.read_text())
        except Exception:
            pass

    api_key = _env("ACLED_API_KEY")
    email   = _env("ACLED_EMAIL")
    if not api_key or not email:
        logger.warning(
            "[acled] ACLED_API_KEY / ACLED_EMAIL not set — register at "
            "developer.acleddata.com (free academic) and put creds in .env")
        return []

    out: list[dict] = []
    page = 1
    while len(out) < limit:
        params = {
            "key": api_key,
            "email": email,
            "limit": 5000,
            "page": page,
            "event_date": f"{start}|{end}",
            "event_date_where": "BETWEEN",
        }
        if countries:
            params["country"] = "|".join(countries)
            params["country_where"] = "IN"
        if event_types:
            params["event_type"] = "|".join(event_types)
            params["event_type_where"] = "IN"

        try:
            r = requests.get(ACLED_API, params=params, timeout=120,
                             headers={"User-Agent": "Heli.OS/1.0"})
            if r.status_code == 429:
                time.sleep(10); continue
            r.raise_for_status()
            data = r.json().get("data") or []
        except Exception as e:
            logger.warning("[acled] page %d failed: %s", page, e)
            break
        if not data:
            break
        out.extend(data)
        if len(data) < 5000:
            break
        page += 1
        time.sleep(0.5)

    out = out[:limit]
    cache.write_text(json.dumps(out))
    logger.info("[acled] %d events fetched (countries=%s)", len(out), countries)
    return out


# -- UCDP GED --------------------------------------------------------------


def load_ucdp_ged(year: int, country: Optional[str] = None,
                  pagesize: int = 1000, limit: int = 50_000) -> list[dict]:
    """UCDP Georeferenced Event Dataset for one year (no auth)."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    key = f"ucdp_ged_{year}" + (f"_{country}" if country else "")
    cache = OUT_DIR / f"{key.replace('/', '_')}.json"
    if cache.exists():
        try:
            return json.loads(cache.read_text())
        except Exception:
            pass

    out: list[dict] = []
    page = 0
    while len(out) < limit:
        params = {
            "pagesize": pagesize,
            "page": page,
            "StartDate": f"{year}-01-01",
            "EndDate":   f"{year}-12-31",
        }
        if country:
            params["Country"] = country
        try:
            r = requests.get(UCDP_GED, params=params, timeout=60,
                             headers={"User-Agent": "Heli.OS/1.0"})
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            logger.warning("[ucdp] page %d failed: %s", page, e)
            break
        results = data.get("Result") or []
        if not results:
            break
        out.extend(results)
        total_pages = data.get("TotalPages", 1)
        if page + 1 >= total_pages:
            break
        page += 1
        time.sleep(0.5)

    out = out[:limit]
    cache.write_text(json.dumps(out))
    logger.info("[ucdp] %d events fetched (year=%d country=%s)",
                len(out), year, country)
    return out


# -- Common normalized output ---------------------------------------------


def normalize_acled(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        try:
            out.append({
                "source":      "acled",
                "event_id":    r.get("event_id_cnty") or r.get("event_id_no_cnty"),
                "date":        r.get("event_date"),
                "country":     r.get("country"),
                "actor1":      r.get("actor1"),
                "actor2":      r.get("actor2"),
                "event_type":  r.get("event_type"),
                "sub_event":   r.get("sub_event_type"),
                "fatalities":  int(r.get("fatalities") or 0),
                "lat":         float(r.get("latitude") or 0),
                "lon":         float(r.get("longitude") or 0),
                "notes":       r.get("notes") or "",
            })
        except (TypeError, ValueError):
            continue
    return out


def normalize_ucdp(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        try:
            out.append({
                "source":      "ucdp_ged",
                "event_id":    str(r.get("id")),
                "date":        r.get("date_start"),
                "country":     r.get("country"),
                "side_a":      r.get("side_a"),
                "side_b":      r.get("side_b"),
                "event_type":  r.get("type_of_violence_name") or r.get("type_of_violence"),
                "fatalities":  int(r.get("best") or 0),
                "lat":         float(r.get("latitude") or 0),
                "lon":         float(r.get("longitude") or 0),
                "notes":       r.get("source_article") or "",
            })
        except (TypeError, ValueError):
            continue
    return out
