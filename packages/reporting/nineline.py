"""
9-Line MEDEVAC / CASEVAC Request Generator

Standard NATO 9-line MEDEVAC format used by military, FEMA USAR teams,
wilderness SAR, and mass casualty events.

Lines:
  1. Location of pickup site (grid/coordinates)
  2. Radio frequency and callsign
  3. Number of patients by precedence (U/P/R/C/E)
  4. Special equipment required
  5. Number of patients (litter / ambulatory)
  6. Security at pickup site
  7. Method of marking pickup site
  8. Patient nationality and status
  9. NBC contamination

Reference: ATP 4-02.2 (Medical Evacuation)
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional


# Precedence codes
PRECEDENCE = {
    "U": "Urgent (within 2 hours)",
    "P": "Priority (within 4 hours)",
    "R": "Routine (within 24 hours)",
    "C": "Convenience",
    "E": "Urgent Surgical",
}

# Special equipment codes
SPECIAL_EQUIPMENT = {
    "N": "None",
    "A": "Hoist",
    "B": "Extraction equipment",
    "C": "Ventilator",
}

# Security codes (line 6)
SECURITY_CODES = {
    "N": "No enemy troops in area",
    "P": "Possible enemy troops (approach with caution)",
    "E": "Enemy troops in area",
    "X": "Enemy troops in area — armed escort required",
}

# Marking methods (line 7)
MARKING_METHODS = {
    "A": "Panels",
    "B": "Pyrotechnic signal",
    "C": "Smoke signal",
    "D": "None",
    "E": "Other",
}

# Patient status (line 8)
PATIENT_STATUS = {
    "M": "Military",
    "C": "Civilian",
    "E": "Enemy prisoner of war",
}


def generate_9line(
    location: dict,
    patients: list[dict],
    observer_callsign: str = "HELI-1",
    frequency_mhz: float = 121.5,
    security: str = "N",
    marking: str = "C",
    special_equipment: str = "N",
    nbc: str = "N",
    additional_info: str = "",
) -> dict:
    """
    Generate a 9-line MEDEVAC request.

    Args:
        location:           {"lat": float, "lon": float, "alt_m": float, "grid": str}
        patients:           List of {"precedence": str, "ambulatory": bool, "status": str}
        observer_callsign:  Requesting unit callsign
        frequency_mhz:      Guard/medevac frequency
        security:           Line 6 code (N/P/E/X)
        marking:            Line 7 code (A/B/C/D/E)
        special_equipment:  Line 4 code (N/A/B/C)
        nbc:                "N" or contamination type
        additional_info:    Free-text amplifying info

    Returns:
        {
            "format":    "9LINE",
            "report_id": str,
            "ts_iso":    str,
            "lines":     {1..9: str},
            "text":      str,
            "structured": dict,
        }
    """
    ts = datetime.now(timezone.utc)
    time_str = ts.strftime("%d%H%MZ %b %Y").upper()

    # Line 1: Location
    if location.get("grid"):
        line1 = location["grid"]
    elif location.get("lat") and location.get("lon"):
        line1 = f"{location['lat']:.5f}N {location['lon']:.5f}E"
        if location.get("alt_m"):
            line1 += f" ALT {location['alt_m']:.0f}m"
    else:
        line1 = "LOCATION UNKNOWN"

    # Line 2: Frequency / callsign
    line2 = f"{frequency_mhz:.3f} MHz — {observer_callsign}"

    # Line 3: Patient count by precedence
    precedence_counts: dict[str, int] = {}
    for p in patients:
        prec = p.get("precedence", "R").upper()
        precedence_counts[prec] = precedence_counts.get(prec, 0) + 1
    line3_parts = [f"{cnt}{code}({PRECEDENCE[code].split(' ')[0]})"
                   for code, cnt in precedence_counts.items()]
    line3 = ", ".join(line3_parts) if line3_parts else "1R (Routine)"

    # Line 4: Special equipment
    line4 = f"{special_equipment} — {SPECIAL_EQUIPMENT.get(special_equipment, 'Unknown')}"

    # Line 5: Litter / ambulatory
    litter = sum(1 for p in patients if not p.get("ambulatory", True))
    ambulatory = sum(1 for p in patients if p.get("ambulatory", True))
    line5 = f"L: {litter} / A: {ambulatory}"

    # Line 6: Security
    line6 = f"{security} — {SECURITY_CODES.get(security, 'Unknown')}"

    # Line 7: Marking
    line7 = f"{marking} — {MARKING_METHODS.get(marking, 'Unknown')}"

    # Line 8: Patient nationality/status
    status_counts: dict[str, int] = {}
    for p in patients:
        stat = p.get("status", "M").upper()
        status_counts[stat] = status_counts.get(stat, 0) + 1
    line8_parts = [f"{cnt} {PATIENT_STATUS.get(code, code)}"
                   for code, cnt in status_counts.items()]
    line8 = ", ".join(line8_parts)

    # Line 9: NBC
    line9 = "None" if nbc == "N" else f"CONTAMINATED — {nbc}"

    lines = {
        "1": line1,
        "2": line2,
        "3": line3,
        "4": line4,
        "5": line5,
        "6": line6,
        "7": line7,
        "8": line8,
        "9": line9,
    }

    text_lines = [
        f"9-LINE MEDEVAC — {observer_callsign} — {time_str}",
        f"LINE 1 (Location):     {line1}",
        f"LINE 2 (Freq/CS):      {line2}",
        f"LINE 3 (Precedence):   {line3}",
        f"LINE 4 (Equipment):    {line4}",
        f"LINE 5 (Patients):     {line5}",
        f"LINE 6 (Security):     {line6}",
        f"LINE 7 (Marking):      {line7}",
        f"LINE 8 (Status):       {line8}",
        f"LINE 9 (NBC):          {line9}",
    ]
    if additional_info:
        text_lines.append(f"AMPN: {additional_info}")

    return {
        "format":    "9LINE",
        "report_id": f"9LINE-{observer_callsign}-{int(ts.timestamp())}",
        "observer":  observer_callsign,
        "ts_iso":    ts.isoformat(),
        "lines":     lines,
        "text":      "\n".join(text_lines),
        "structured": {
            "location":    location,
            "patients":    patients,
            "total_count": len(patients),
            "litter":      litter,
            "ambulatory":  ambulatory,
            "security":    security,
            "nbc":         nbc,
        },
    }
