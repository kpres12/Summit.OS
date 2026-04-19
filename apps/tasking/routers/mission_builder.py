"""Mission builder endpoints — NLP parse and waypoint preview."""
import math
import os
import re
import logging
from typing import Optional, List, Dict, Any

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()
logger = logging.getLogger("tasking")

# ── Request / Response models ─────────────────────────────────────────────────

class NlpParseRequest(BaseModel):
    text: str

class NlpParseResponse(BaseModel):
    mission_type: str
    pattern: str
    altitude_m: int
    asset_hint: Optional[str] = None
    objectives: List[str]
    confidence: float
    interpretation: str

class PreviewRequest(BaseModel):
    pattern: str
    altitude_m: int
    area: List[Dict[str, float]]  # [{lat, lon}, ...]

class PreviewResponse(BaseModel):
    waypoints: List[Dict[str, float]]
    pattern: str
    count: int

# ── Keyword maps ──────────────────────────────────────────────────────────────

_MISSION_TYPES: List[tuple] = [
    (r'\bsearch\b|\bsar\b|\brescue\b|\blook for\b|\bfind\b|\bmissing\b', 'SEARCH'),
    (r'\bperimeter\b|\bpatrol\b|\bboundary\b|\bfence\b', 'PERIMETER'),
    (r'\borbit\b|\bcircle\b|\bloop\b|\boverwatch\b', 'ORBIT'),
    (r'\bmonitor\b|\bwatch\b|\bobserve\b|\bsurveillance\b|\bhold\b', 'MONITOR'),
    (r'\bdeliver\b|\bdrop\b|\bpackage\b|\bsupply\b', 'DELIVER'),
    (r'\binspect\b|\bcheck\b|\bscan\b|\baudit\b', 'INSPECT'),
    (r'\bsurvey\b|\bmap\b|\brecon\b', 'SURVEY'),
]

_PATTERNS: List[tuple] = [
    (r'\bspiral\b', 'spiral'),
    (r'\bexpand\b|\bexpanding\b|\bsweep out\b', 'expanding_square'),
    (r'\borbit\b|\bcircle\b|\bloop\b', 'orbit'),
    (r'\bperimeter\b|\boutline\b|\bfence line\b|\bborder\b', 'perimeter'),
    (r'\bgrid\b|\blawnmower\b|\brows\b|\bsweep\b', 'grid'),
]

_PATTERN_DEFAULTS = {
    'SURVEY': 'grid',
    'MONITOR': 'orbit',
    'SEARCH': 'expanding_square',
    'PERIMETER': 'perimeter',
    'ORBIT': 'orbit',
    'DELIVER': 'grid',
    'INSPECT': 'grid',
}


# ── Rule-based parser (always available) ─────────────────────────────────────

def _parse_rules(text: str) -> NlpParseResponse:
    t = text.lower()

    mission_type = 'SURVEY'
    for pat, mt in _MISSION_TYPES:
        if re.search(pat, t):
            mission_type = mt
            break

    pattern = _PATTERN_DEFAULTS.get(mission_type, 'grid')
    for pat, p in _PATTERNS:
        if re.search(pat, t):
            pattern = p
            break

    # Altitude — prefer metres, fall back to feet
    alt = 120
    m = re.search(r'(\d+)\s*(?:m\b|meters?\b|metres?\b)', t)
    if m:
        alt = min(max(int(m.group(1)), 20), 500)
    else:
        m = re.search(r'(\d+)\s*(?:ft\b|feet\b|foot\b)', t)
        if m:
            alt = min(max(int(round(int(m.group(1)) * 0.3048)), 20), 500)

    # Asset hint — "use Echo-1", "assign to UAV-3", or bare callsign-like token
    asset_hint = None
    m = re.search(r'(?:assign(?:ed)?\s+to|use|send|deploy|task)\s+([A-Za-z0-9][A-Za-z0-9\-]+)', t)
    if m:
        asset_hint = m.group(1).upper()
    else:
        m = re.search(r'\b([A-Z][a-z]+-\d+|[A-Z]+-\d+)\b', text)
        if m:
            asset_hint = m.group(1)

    confidence = 0.65
    explicit_pattern = any(re.search(p, t) for p, _ in _PATTERNS)
    if explicit_pattern:
        confidence += 0.1
    if asset_hint:
        confidence += 0.1
    if re.search(r'\d+\s*(?:m\b|meters?\b|ft\b|feet\b)', t):
        confidence += 0.05

    interpretation = (
        f"{mission_type.title()} via {pattern} pattern at {alt}m"
        + (f" · preferred asset: {asset_hint}" if asset_hint else "")
    )

    return NlpParseResponse(
        mission_type=mission_type,
        pattern=pattern,
        altitude_m=alt,
        asset_hint=asset_hint,
        objectives=[f"{mission_type.title()}: {text[:120]}"],
        confidence=round(min(confidence, 0.95), 2),
        interpretation=interpretation,
    )


async def _parse_claude(text: str, api_key: str) -> Optional[NlpParseResponse]:
    """Use Claude API for NLP mission parsing. Returns None on any failure."""
    try:
        import httpx, json as _json
        system_prompt = (
            "You are a mission planning assistant for Heli.OS, an autonomous systems coordination platform.\n"
            "Parse the operator's natural-language mission description and return structured JSON.\n\n"
            "Return ONLY a JSON object — no prose, no markdown:\n"
            '{\n'
            '  "mission_type": one of [SURVEY, MONITOR, SEARCH, PERIMETER, ORBIT, DELIVER, INSPECT],\n'
            '  "pattern": one of [grid, spiral, expanding_square, orbit, perimeter],\n'
            '  "altitude_m": integer 20-500,\n'
            '  "asset_hint": callsign string if mentioned or null,\n'
            '  "objectives": [short objective strings],\n'
            '  "confidence": float 0.0-1.0 (how clearly the command specified the mission),\n'
            '  "interpretation": one-line summary of what was understood\n'
            '}\n\n'
            "Pattern guidance: SEARCH→expanding_square, MONITOR→orbit, PERIMETER/PATROL→perimeter, "
            "SURVEY/MAP→grid. Default altitude 120m unless specified."
        )
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": "claude-haiku-4-5-20251001",
                    "max_tokens": 300,
                    "system": system_prompt,
                    "messages": [{"role": "user", "content": text}],
                },
            )
        if not resp.is_success:
            logger.debug("Claude NLP parse HTTP error: %s", resp.status_code)
            return None
        content = resp.json().get("content", [])
        raw = content[0].get("text", "") if content else ""
        m = re.search(r'\{[\s\S]+\}', raw)
        if not m:
            return None
        data = _json.loads(m.group())
        return NlpParseResponse(**data)
    except Exception as exc:
        logger.debug("Claude NLP parse failed: %s", exc)
        return None


async def _parse_ollama(text: str, base_url: str) -> Optional[NlpParseResponse]:
    """Attempt Ollama-backed NLP parsing. Returns None on any failure."""
    try:
        import httpx, json as _json
        prompt = (
            'Parse this mission command and return ONLY a JSON object — no extra text.\n\n'
            f'Command: "{text}"\n\n'
            'Return this exact structure:\n'
            '{\n'
            '  "mission_type": one of [SURVEY, MONITOR, SEARCH, PERIMETER, ORBIT, DELIVER, INSPECT],\n'
            '  "pattern": one of [grid, spiral, expanding_square, orbit, perimeter],\n'
            '  "altitude_m": integer 20-500,\n'
            '  "asset_hint": string or null,\n'
            '  "objectives": [string],\n'
            '  "confidence": float 0.0-1.0,\n'
            '  "interpretation": string\n'
            '}'
        )
        async with httpx.AsyncClient(timeout=6.0) as client:
            resp = await client.post(
                f"{base_url}/api/generate",
                json={"model": os.getenv("OLLAMA_MODEL", "mistral"), "prompt": prompt, "stream": False},
            )
        if not resp.is_success:
            return None
        raw = resp.json().get("response", "")
        m = re.search(r'\{[\s\S]+\}', raw)
        if not m:
            return None
        data = _json.loads(m.group())
        return NlpParseResponse(**data)
    except Exception as exc:
        logger.debug("Ollama NLP parse failed: %s", exc)
        return None


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/api/v1/missions/parse", response_model=NlpParseResponse)
async def parse_mission_nlp(req: NlpParseRequest):
    """Convert a natural-language mission description into structured parameters."""
    text = req.text.strip()
    if not text:
        return NlpParseResponse(
            mission_type="SURVEY", pattern="grid", altitude_m=120,
            asset_hint=None, objectives=["Survey mission"],
            confidence=0.5, interpretation="Default survey mission (no input)",
        )

    # Claude (primary LLM) — requires ANTHROPIC_API_KEY
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if anthropic_key:
        result = await _parse_claude(text, anthropic_key)
        if result:
            return result

    # Ollama (self-hosted fallback) — requires OLLAMA_URL
    ollama_url = os.getenv("OLLAMA_URL")
    if ollama_url:
        result = await _parse_ollama(text, ollama_url)
        if result:
            return result

    return _parse_rules(text)


@router.post("/api/v1/missions/preview", response_model=PreviewResponse)
async def preview_mission_waypoints(req: PreviewRequest):
    """Generate waypoints for a drawn area without persisting a mission."""
    if len(req.area) < 3:
        return PreviewResponse(waypoints=[], pattern=req.pattern, count=0)

    polygon = [(p["lon"], p["lat"]) for p in req.area]
    waypoints = _generate_waypoints(polygon, req.pattern, req.altitude_m)
    return PreviewResponse(waypoints=waypoints, pattern=req.pattern, count=len(waypoints))


# ── Waypoint generation ───────────────────────────────────────────────────────

def _generate_waypoints(polygon: list, pattern: str, altitude_m: int) -> list:
    lons = [p[0] for p in polygon]
    lats = [p[1] for p in polygon]
    min_lon, max_lon = min(lons), max(lons)
    min_lat, max_lat = min(lats), max(lats)
    c_lat = (min_lat + max_lat) / 2
    c_lon = (min_lon + max_lon) / 2

    lat_per_m = 1 / 111_320
    lon_per_m = 1 / (111_320 * math.cos(math.radians(c_lat)) + 1e-9)
    spacing_m = 80  # metres between passes

    wps: list = []

    if pattern == 'grid':
        lat = min_lat
        col = 0
        while lat <= max_lat and len(wps) < 120:
            lon_start, lon_end = (min_lon, max_lon) if col % 2 == 0 else (max_lon, min_lon)
            wps.append({"lat": lat, "lon": lon_start, "alt": altitude_m})
            wps.append({"lat": lat, "lon": lon_end,   "alt": altitude_m})
            lat += spacing_m * lat_per_m
            col += 1

    elif pattern == 'spiral':
        radius = min(max_lon - min_lon, max_lat - min_lat) / 2
        total = 3 * 36
        for i in range(total):
            angle = i * (2 * math.pi / 36)
            r = radius * (1 - i / total)
            if r < 0:
                break
            wps.append({"lat": c_lat + r * math.sin(angle), "lon": c_lon + r * math.cos(angle), "alt": altitude_m})

    elif pattern == 'expanding_square':
        dlat = spacing_m * lat_per_m
        dlon = spacing_m * lon_per_m
        dirs = [(dlon, 0), (0, dlat), (-dlon, 0), (0, -dlat)]
        x, y = 0.0, 0.0
        direction = 0
        steps_per_side = 1
        side_count = 0

        for _ in range(80):
            wps.append({"lat": c_lat + y, "lon": c_lon + x, "alt": altitude_m})
            dx, dy = dirs[direction % 4]
            x += dx * steps_per_side
            y += dy * steps_per_side
            direction += 1
            side_count += 1
            if side_count == 2:
                side_count = 0
                steps_per_side += 1
            if len(wps) >= 60:
                break

    elif pattern == 'orbit':
        radius = min(max_lon - min_lon, max_lat - min_lat) / 2
        for i in range(37):
            angle = i * (2 * math.pi / 36)
            wps.append({
                "lat": c_lat + radius * math.sin(angle),
                "lon": c_lon + radius * math.cos(angle),
                "alt": altitude_m,
            })

    elif pattern == 'perimeter':
        for p in polygon:
            wps.append({"lat": p[1], "lon": p[0], "alt": altitude_m})
        wps.append({"lat": polygon[0][1], "lon": polygon[0][0], "alt": altitude_m})

    return wps
