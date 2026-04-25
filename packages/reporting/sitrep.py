"""
SITREP (Situation Report) Generator

Produces structured SITREPs from Heli.OS world model snapshots.
Supports military, emergency management, and commercial operations formats.

Output is both human-readable plaintext (for ATAK/radio) and structured
JSON (for API consumers and handoff briefs).
"""

from __future__ import annotations

from datetime import datetime, timezone


def generate_sitrep(
    entities: list[dict],
    alerts: list[dict],
    missions: list[dict],
    observer_callsign: str = "HELI-ACTUAL",
    report_period_minutes: int = 60,
    operational_area: str = "AO BRAVO",
    domain: str = "general",
    commander_intent: str = "",
    additional_info: str = "",
) -> dict:
    """
    Generate a SITREP from current world model state.

    Args:
        entities:               All entities in world model
        alerts:                 Active alerts from domain assessment
        missions:               Active or completed missions
        observer_callsign:      Reporting element
        report_period_minutes:  Time window this SITREP covers
        operational_area:       Named AO / area of operations
        domain:                 Domain context (military/maritime/utilities/etc.)
        commander_intent:       Optional commander's intent statement
        additional_info:        Free-text amplifying remarks

    Returns:
        {
            "format":     "SITREP",
            "report_id":  str,
            "ts_iso":     str,
            "sections":   {situation, assets, threats, missions, outlook},
            "text":       str,
            "structured": dict,
        }
    """
    ts = datetime.now(timezone.utc)
    time_str = ts.strftime("%d%H%MZ %b %Y").upper()

    # Situation summary
    entity_types: dict[str, int] = {}
    for e in entities:
        et = e.get("entity_type", "UNKNOWN")
        entity_types[et] = entity_types.get(et, 0) + 1

    critical_alerts = [a for a in alerts if a.get("severity") == "critical"]
    high_alerts = [a for a in alerts if a.get("severity") == "high"]
    total_alerts = len(alerts)

    # Asset summary
    friendly = [e for e in entities if e.get("classification") in ("FRIENDLY", "ASSET")]
    unknown = [e for e in entities if e.get("classification") == "UNKNOWN"]
    offline = [e for e in entities if e.get("metadata", {}).get("status") == "OFFLINE"]

    # Mission summary
    active_missions = [m for m in missions if m.get("status") in ("ACTIVE", "IN_PROGRESS")]
    completed_missions = [m for m in missions if m.get("status") == "COMPLETED"]

    # Overall threat/risk
    if critical_alerts:
        overall_status = "CRITICAL — Immediate action required"
    elif high_alerts:
        overall_status = "ELEVATED — Close monitoring required"
    elif alerts:
        overall_status = "GUARDED — Routine monitoring"
    else:
        overall_status = "NOMINAL"

    sections = {
        "situation": (
            f"As of {time_str}, {operational_area} status is {overall_status}. "
            f"{len(entities)} entities tracked. "
            f"{total_alerts} active alert(s): {len(critical_alerts)} critical, {len(high_alerts)} high."
        ),
        "assets": (
            f"Friendly/own assets: {len(friendly)}. Unknown: {len(unknown)}. "
            f"Offline: {len(offline)}. "
            f"Entity types: {', '.join(f'{v} {k}' for k, v in entity_types.items())}."
        ),
        "threats": _format_threats(critical_alerts + high_alerts),
        "missions": (
            f"Active missions: {len(active_missions)}. "
            f"Completed this period: {len(completed_missions)}. "
            + (", ".join(m.get("mission_type", "UNK") for m in active_missions) or "None")
        ),
        "outlook": (
            commander_intent
            or f"Continue ISR coverage. Reassess in {report_period_minutes} min."
        ),
    }

    text_lines = [
        f"SITREP — {observer_callsign} — {time_str}",
        f"AO: {operational_area} | PERIOD: {report_period_minutes} MIN | DOMAIN: {domain.upper()}",
        "",
        f"1. SITUATION:  {sections['situation']}",
        f"2. ASSETS:     {sections['assets']}",
        f"3. THREATS:    {sections['threats']}",
        f"4. MISSIONS:   {sections['missions']}",
        f"5. OUTLOOK:    {sections['outlook']}",
    ]
    if additional_info:
        text_lines.append(f"6. AMPN: {additional_info}")

    return {
        "format":    "SITREP",
        "report_id": f"SITREP-{observer_callsign}-{int(ts.timestamp())}",
        "observer":  observer_callsign,
        "ts_iso":    ts.isoformat(),
        "sections":  sections,
        "text":      "\n".join(text_lines),
        "structured": {
            "entity_count":      len(entities),
            "friendly_count":    len(friendly),
            "unknown_count":     len(unknown),
            "alert_count":       total_alerts,
            "critical_count":    len(critical_alerts),
            "high_count":        len(high_alerts),
            "active_missions":   len(active_missions),
            "overall_status":    overall_status,
            "domain":            domain,
            "operational_area":  operational_area,
        },
    }


def _format_threats(alerts: list[dict]) -> str:
    if not alerts:
        return "No significant threats."
    lines = []
    for a in alerts[:5]:  # top 5
        lines.append(f"[{a.get('severity', '?').upper()}] {a.get('alert_type')}: {a.get('description')}")
    if len(alerts) > 5:
        lines.append(f"... and {len(alerts) - 5} additional alert(s).")
    return " | ".join(lines)
