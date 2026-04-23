"""
SALUTE / SPOT Contact Report Generator

SALUTE: Size, Activity, Location, Unit/Uniform, Time, Equipment
SPOT:   Size, Position, Other info, Time (simplified SALUTE for rapid reporting)

Used by: military, law enforcement, border patrol, anti-poaching operations
Output: plain text + structured dict (for API / ATAK CoT remarks field)
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional


def generate_salute(
    entity: dict,
    observer_callsign: str = "HELI-1",
    additional_info: str = "",
) -> dict:
    """
    Generate a SALUTE report from a world model entity.

    Args:
        entity:             Entity dict from the world model
        observer_callsign:  Reporting unit callsign
        additional_info:    Free-text amplifying information

    Returns:
        {
            "format":       "SALUTE",
            "report_id":    str,
            "observer":     str,
            "ts_iso":       str,
            "fields":       {S, A, L, U, T, E},
            "text":         str,   # formatted plaintext
            "structured":   dict,  # machine-readable fields
        }
    """
    meta = entity.get("metadata", {})
    pos = entity.get("position") or {}
    ts = datetime.now(timezone.utc)

    # Size
    size = _estimate_size(entity)

    # Activity
    velocity = entity.get("velocity") or {}
    speed = velocity.get("speed_mps")
    heading = velocity.get("heading_deg")
    if speed is not None and speed > 0.5:
        activity = f"Moving {_cardinal(heading)} at {speed * 1.944:.1f} kts"
    else:
        activity = meta.get("activity", "Stationary / unknown activity")

    # Location
    if pos.get("lat") and pos.get("lon"):
        location = f"{pos['lat']:.5f}°N {pos['lon']:.5f}°E"
        if pos.get("alt_m"):
            location += f" ALT {pos['alt_m']:.0f}m"
    else:
        location = "Location unknown"

    # Unit/Uniform
    callsign = entity.get("callsign", "Unknown")
    classification = entity.get("classification", "UNKNOWN")
    entity_type = entity.get("entity_type", "UNKNOWN")
    unit_info = f"{classification} {entity_type} — {callsign}"
    cot_type = meta.get("cot_type", "")
    if cot_type:
        unit_info += f" (CoT: {cot_type})"

    # Time
    time_str = ts.strftime("%d%H%MZ %b %Y").upper()

    # Equipment
    equipment = meta.get("equipment", meta.get("vessel_type", meta.get("aircraft_type", "Unknown")))

    fields = {
        "S": size,
        "A": activity,
        "L": location,
        "U": unit_info,
        "T": time_str,
        "E": str(equipment),
    }

    text_lines = [
        f"SALUTE REPORT — {observer_callsign} — {time_str}",
        f"S (Size):       {fields['S']}",
        f"A (Activity):   {fields['A']}",
        f"L (Location):   {fields['L']}",
        f"U (Unit):       {fields['U']}",
        f"T (Time):       {fields['T']}",
        f"E (Equipment):  {fields['E']}",
    ]
    if additional_info:
        text_lines.append(f"AMPN: {additional_info}")

    return {
        "format":    "SALUTE",
        "report_id": f"SALUTE-{entity.get('entity_id', 'unk')}-{int(ts.timestamp())}",
        "observer":  observer_callsign,
        "ts_iso":    ts.isoformat(),
        "fields":    fields,
        "text":      "\n".join(text_lines),
        "structured": {
            "entity_id":       entity.get("entity_id"),
            "lat":             pos.get("lat"),
            "lon":             pos.get("lon"),
            "alt_m":           pos.get("alt_m"),
            "classification":  classification,
            "entity_type":     entity_type,
            "speed_mps":       speed,
            "heading_deg":     heading,
        },
    }


def generate_spot(
    entity: dict,
    observer_callsign: str = "HELI-1",
) -> dict:
    """
    Generate a SPOT report (rapid abbreviated contact report).

    Returns the same structure as generate_salute but in SPOT format.
    """
    meta = entity.get("metadata", {})
    pos = entity.get("position") or {}
    ts = datetime.now(timezone.utc)

    size = _estimate_size(entity)
    if pos.get("lat") and pos.get("lon"):
        position = f"{pos['lat']:.5f}N {pos['lon']:.5f}E"
    else:
        position = "POS UNKNOWN"

    other = f"{entity.get('classification', 'UNK')} {entity.get('entity_type', 'UNK')}"
    equipment = meta.get("equipment", "UNK")
    time_str = ts.strftime("%d%H%MZ").upper()

    fields = {
        "S": size,
        "P": position,
        "O": f"{other} / EQUIP: {equipment}",
        "T": time_str,
    }

    text = (
        f"SPOT REPORT — {observer_callsign}\n"
        f"S: {fields['S']} | P: {fields['P']} | O: {fields['O']} | T: {fields['T']}"
    )

    return {
        "format":    "SPOT",
        "report_id": f"SPOT-{entity.get('entity_id', 'unk')}-{int(ts.timestamp())}",
        "observer":  observer_callsign,
        "ts_iso":    ts.isoformat(),
        "fields":    fields,
        "text":      text,
        "structured": {
            "entity_id": entity.get("entity_id"),
            "lat": pos.get("lat"),
            "lon": pos.get("lon"),
        },
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _estimate_size(entity: dict) -> str:
    count = entity.get("metadata", {}).get("group_count", 1)
    if count == 1:
        return "1 individual"
    if count <= 5:
        return f"{count} individuals (small group)"
    if count <= 20:
        return f"{count} individuals (patrol-size)"
    return f"{count}+ (large group)"


def _cardinal(heading_deg: Optional[float]) -> str:
    if heading_deg is None:
        return "unknown direction"
    dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    idx = round(heading_deg / 45) % 8
    return dirs[idx]
