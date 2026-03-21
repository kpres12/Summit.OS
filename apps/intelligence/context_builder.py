"""
WorldStore → LLM Context Builder

Converts the live world state into a compact, structured context string
suitable for a local LLM (Llama 3.1 via Ollama). Prioritises CRITICAL
and WARNING entities. Enforces a token budget to fit within context windows.

Usage:
    builder = ContextBuilder(world_url="http://localhost:8001")
    ctx = await builder.build(mission_objective="Find and track the fire perimeter")
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# Shared sanitization — avoids duplicating injection pattern lists
from prompt_guard import _safe_str, sanitize_text

logger = logging.getLogger("summit.intelligence.context")

# Rough chars-per-token estimate for budget enforcement
_CHARS_PER_TOKEN = 4

# State priority — lower number = shown first / higher priority
_STATE_PRIORITY = {"CRITICAL": 0, "WARNING": 1, "ACTIVE": 2, "INACTIVE": 3}


def _age_str(ts_iso: Optional[str]) -> str:
    if not ts_iso:
        return "unknown age"
    try:
        from datetime import datetime
        dt = datetime.fromisoformat(ts_iso.replace("Z", "+00:00"))
        age = (datetime.now(timezone.utc) - dt).total_seconds()
        if age < 60:
            return f"{int(age)}s ago"
        if age < 3600:
            return f"{int(age/60)}m ago"
        return f"{int(age/3600)}h ago"
    except Exception:
        return "?"


def _summarise_entity(e: Dict) -> str:
    eid = e.get("entity_id", "?")
    etype = e.get("entity_type", "?")
    state = e.get("state", "?")
    domain = e.get("domain", "")
    label = e.get("class_label", e.get("metadata", {}).get("class_label", ""))
    name = e.get("name", e.get("metadata", {}).get("name", label or eid))

    kin = e.get("kinematics") or {}
    pos = kin.get("position") or {}
    lat = pos.get("latitude")
    lon = pos.get("longitude")
    alt = pos.get("altitude")

    vel = kin.get("velocity") or {}
    speed = vel.get("speed")
    heading = vel.get("heading")

    aerial = e.get("aerial") or {}
    battery = aerial.get("battery_pct") or e.get("battery_pct")
    flight_mode = aerial.get("flight_mode", "")

    meta = e.get("metadata") or {}
    ts = e.get("updated_at") or e.get("ts")

    # Sanitize all entity-sourced strings before embedding in LLM context
    etype = _safe_str(etype, 32)
    state = _safe_str(state, 32)
    name  = _safe_str(name, 80)
    domain = _safe_str(domain, 40)

    parts = [f"[{etype}/{state}] {name}"]
    if domain:
        parts[0] += f" ({domain})"
    if lat is not None and lon is not None:
        coord = f"@{lat:.4f},{lon:.4f}"
        if alt is not None:
            coord += f" alt={alt:.0f}m"
        parts.append(coord)
    if speed is not None:
        parts.append(f"spd={speed:.1f}m/s")
    if heading is not None:
        parts.append(f"hdg={heading:.0f}°")
    if flight_mode:
        parts.append(f"mode={flight_mode}")
    if battery is not None:
        parts.append(f"batt={battery}%")
    if ts:
        parts.append(_age_str(str(ts)))

    # Surface critical metadata (sanitize values — they come from external sensors)
    for key in ("value", "unit", "message", "description"):
        if meta.get(key):
            parts.append(f"{key}={_safe_str(meta[key], 120)}")

    return " | ".join(parts)


class ContextBuilder:
    """Builds structured LLM prompts from live WorldStore state."""

    def __init__(
        self,
        world_url: str = "http://localhost:8001",
        max_tokens: int = 3000,
        max_entities: int = 50,
    ):
        self.world_url = world_url.rstrip("/")
        self.max_tokens = max_tokens
        self.max_entities = max_entities
        self._char_budget = max_tokens * _CHARS_PER_TOKEN

    async def _fetch_entities(self) -> List[Dict]:
        try:
            import httpx
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.get(f"{self.world_url}/entities")
                if r.status_code == 200:
                    data = r.json()
                    if isinstance(data, list):
                        return data
                    return data.get("entities", [])
        except Exception as e:
            logger.warning(f"Could not fetch entities from WorldStore: {e}")
        return []

    async def _fetch_alerts(self) -> List[Dict]:
        try:
            import httpx
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.get(f"{self.world_url}/alerts?limit=20")
                if r.status_code == 200:
                    data = r.json()
                    if isinstance(data, list):
                        return data
                    return data.get("alerts", [])
        except Exception as e:
            logger.debug(f"Could not fetch alerts: {e}")
        return []

    def _prioritise(self, entities: List[Dict]) -> List[Dict]:
        return sorted(
            entities,
            key=lambda e: (
                _STATE_PRIORITY.get(e.get("state", ""), 99),
                e.get("entity_type", ""),
            )
        )

    def _format_alerts(self, alerts: List[Dict]) -> str:
        if not alerts:
            return ""
        lines = ["ACTIVE ALERTS:"]
        for a in alerts[:10]:
            # Sanitize all alert fields — alerts can come from external sources
            # or be created by operators and are a primary injection vector
            sev = _safe_str(a.get("severity", "?"), 32)
            raw_desc = a.get("description", a.get("message", "?"))
            desc = sanitize_text(str(raw_desc), max_len=300, label="alert description")
            ts = _age_str(str(a.get("ts_iso", a.get("ts", ""))))
            lines.append(f"  [{sev}] {desc} ({ts})")
        return "\n".join(lines)

    def _format_entities(self, entities: List[Dict]) -> str:
        if not entities:
            return "No entities in world model."
        lines = [f"WORLD STATE ({len(entities)} entities):"]
        for e in entities:
            lines.append("  " + _summarise_entity(e))
        return "\n".join(lines)

    def build_sync(
        self,
        entities: List[Dict],
        alerts: List[Dict],
        mission_objective: str,
        additional_context: str = "",
    ) -> str:
        """Build the full LLM context string from pre-fetched data."""
        prioritised = self._prioritise(entities)

        # Enforce entity count budget
        prioritised = prioritised[: self.max_entities]

        # Build sections
        ts_now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        header = f"SUMMIT.OS OPERATOR AI | {ts_now}\n"
        mission_section = f"CURRENT MISSION: {mission_objective}\n" if mission_objective else ""
        alert_section = self._format_alerts(alerts)
        entity_section = self._format_entities(prioritised)

        sections = [header, mission_section, alert_section, entity_section]
        if additional_context:
            safe_ctx = sanitize_text(additional_context, max_len=500, label="additional_context")
            sections.append(f"\nADDITIONAL CONTEXT:\n{safe_ctx}")

        full = "\n".join(s for s in sections if s)

        # Trim to char budget
        if len(full) > self._char_budget:
            full = full[: self._char_budget - 200] + "\n...[context truncated for token budget]"

        return full

    async def build(
        self,
        mission_objective: str = "",
        additional_context: str = "",
    ) -> str:
        """Fetch world state and build LLM context."""
        entities, alerts = await asyncio.gather(
            self._fetch_entities(),
            self._fetch_alerts(),
        )
        return self.build_sync(entities, alerts, mission_objective, additional_context)


import asyncio  # noqa: E402 — imported here to avoid top-level asyncio dependency in tests
