"""
Infrastructure Inspection Report Generator

Produces standardized inspection reports for:
  - Electrical transmission lines (thermal, sag, corrosion)
  - Pipeline ROW (leak, corrosion, encroachment)
  - Bridges and structures (visual, thermal)
  - Construction site progress
  - Oil & gas facilities

Output: structured dict + formatted plaintext ready for work order systems,
FAA Part 107 compliance documentation, or regulatory filings.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional


def generate_inspection_report(
    asset_id: str,
    asset_type: str,
    findings: list[dict],
    inspector_callsign: str = "HELI-INSPECT",
    operator: str = "Branca.ai / Heli.OS",
    flight_params: Optional[dict] = None,
    regulatory_ref: str = "",
    additional_notes: str = "",
) -> dict:
    """
    Generate a structured infrastructure inspection report.

    Args:
        asset_id:           Asset identifier (e.g., "SEGMENT-47A")
        asset_type:         POWER_LINE | PIPELINE | BRIDGE | CONSTRUCTION | OIL_GAS
        findings:           List of finding dicts from domain assessment alerts
        inspector_callsign: UAV/operator callsign
        operator:           Operating company name
        flight_params:      {"altitude_m", "speed_mps", "duration_min", "coverage_km2"}
        regulatory_ref:     Applicable standard (e.g., "NERC FAC-003", "49 CFR 195")
        additional_notes:   Free-text notes

    Returns:
        {
            "format":         "INSPECTION",
            "report_id":      str,
            "ts_iso":         str,
            "asset_id":       str,
            "asset_type":     str,
            "overall_rating": str,
            "findings":       list[dict],
            "text":           str,
            "structured":     dict,
        }
    """
    ts = datetime.now(timezone.utc)
    time_str = ts.strftime("%Y-%m-%d %H:%M UTC")
    fp = flight_params or {}

    # Overall rating from findings
    critical = [f for f in findings if f.get("severity") == "critical"]
    high = [f for f in findings if f.get("severity") == "high"]
    medium = [f for f in findings if f.get("severity") == "medium"]
    low = [f for f in findings if f.get("severity") == "low"]

    if critical:
        overall_rating = "FAIL — Immediate intervention required"
        rating_code = "FAIL"
    elif high:
        overall_rating = "ACTION REQUIRED — Schedule repair within 30 days"
        rating_code = "ACTION_REQUIRED"
    elif medium:
        overall_rating = "MONITOR — Schedule inspection follow-up"
        rating_code = "MONITOR"
    else:
        overall_rating = "PASS — No significant defects detected"
        rating_code = "PASS"

    # Flight params block
    fp_lines = []
    if fp.get("altitude_m"):
        fp_lines.append(f"  Altitude:    {fp['altitude_m']:.0f} m AGL")
    if fp.get("speed_mps"):
        fp_lines.append(f"  Speed:       {fp['speed_mps']:.1f} m/s")
    if fp.get("duration_min"):
        fp_lines.append(f"  Duration:    {fp['duration_min']:.0f} min")
    if fp.get("coverage_km2"):
        fp_lines.append(f"  Coverage:    {fp['coverage_km2']:.2f} km²")
    fp_block = "\n".join(fp_lines) if fp_lines else "  Not recorded"

    # Findings block
    finding_lines = []
    for i, f in enumerate(findings, 1):
        sev = f.get("severity", "low").upper()
        atype = f.get("alert_type", "FINDING")
        desc = f.get("description", "")
        eid = f.get("entity_id", "")
        coords = ""
        if f.get("lat") and f.get("lon"):
            coords = f" @ {f['lat']:.5f}N {f['lon']:.5f}E"
        finding_lines.append(f"  [{i}] [{sev}] {atype}: {desc}{coords}")
        if eid:
            finding_lines.append(f"       Asset ref: {eid}")

    findings_block = "\n".join(finding_lines) if finding_lines else "  No defects detected."

    text = f"""INSPECTION REPORT
=================
Asset ID:       {asset_id}
Asset Type:     {asset_type}
Inspector:      {inspector_callsign} ({operator})
Inspection Date: {time_str}
{f"Regulatory Ref: {regulatory_ref}" if regulatory_ref else ""}

FLIGHT PARAMETERS
-----------------
{fp_block}

OVERALL RATING: {overall_rating}
Summary: {len(critical)} critical, {len(high)} high, {len(medium)} medium, {len(low)} low

FINDINGS
--------
{findings_block}
{f"NOTES: {additional_notes}" if additional_notes else ""}
"""

    return {
        "format":         "INSPECTION",
        "report_id":      f"INSP-{asset_id}-{int(ts.timestamp())}",
        "inspector":      inspector_callsign,
        "ts_iso":         ts.isoformat(),
        "asset_id":       asset_id,
        "asset_type":     asset_type,
        "overall_rating": rating_code,
        "rating_label":   overall_rating,
        "findings":       findings,
        "finding_counts": {
            "critical": len(critical),
            "high":     len(high),
            "medium":   len(medium),
            "low":      len(low),
            "total":    len(findings),
        },
        "flight_params": fp,
        "text":          text,
        "structured": {
            "asset_id":       asset_id,
            "asset_type":     asset_type,
            "operator":       operator,
            "rating":         rating_code,
            "total_findings": len(findings),
            "regulatory_ref": regulatory_ref,
        },
    }
