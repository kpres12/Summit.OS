"""
Modality Adapters — Heli.OS Fusion

Normalizes raw observations from different sensor modalities into a common
TrackDescriptor for cross-modal association. Each adapter extracts the features
that are comparable across modalities.

Modalities: RADAR, AIS, CAMERA, ADSB, SONAR, MESHTASTIC
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class TrackDescriptor:
    """Common representation of an observation across sensor modalities."""

    source_id: str
    modality: str  # RADAR / AIS / CAMERA / ADSB / SONAR / MESH
    lat: Optional[float]
    lon: Optional[float]
    alt_m: Optional[float]
    speed_mps: Optional[float]
    heading_deg: Optional[float]
    size_m: Optional[float]       # estimated physical size
    mmsi: Optional[str]           # AIS vessel identifier
    callsign: Optional[str]       # ADS-B / mesh callsign
    visual_id: Optional[str]      # re-ID embedding hash for camera tracks
    ts: float                     # unix timestamp
    confidence: float = 1.0


# ── Internal helpers ──────────────────────────────────────────────────────────

def _pos(entity: dict) -> tuple[Optional[float], Optional[float]]:
    pos = entity.get("position") or {}
    lat = pos.get("lat") or entity.get("latitude") or entity.get("lat")
    lon = pos.get("lon") or entity.get("longitude") or entity.get("lon")
    return (float(lat) if lat is not None else None,
            float(lon) if lon is not None else None)


def _alt(entity: dict) -> Optional[float]:
    pos = entity.get("position") or {}
    alt = (pos.get("alt_m") or pos.get("altitude_m") or pos.get("altitude")
           or entity.get("altitude_m") or entity.get("altitude"))
    return float(alt) if alt is not None else None


def _speed(entity: dict) -> Optional[float]:
    v = (entity.get("speed_mps") or entity.get("speed")
         or (entity.get("kinematics") or {}).get("speed_mps"))
    return float(v) if v is not None else None


def _heading(entity: dict) -> Optional[float]:
    h = (entity.get("heading_deg") or entity.get("heading") or entity.get("course_deg")
         or (entity.get("kinematics") or {}).get("heading_deg"))
    return float(h) if h is not None else None


def _ts(entity: dict) -> float:
    t = entity.get("last_seen") or entity.get("ts") or entity.get("timestamp")
    return float(t) if t is not None else time.time()


def _meta(entity: dict) -> dict:
    return entity.get("metadata") or entity.get("meta") or {}


# ── Adapter functions ─────────────────────────────────────────────────────────

def adapt_adsb(entity: dict) -> TrackDescriptor:
    """Normalize an ADS-B entity into a TrackDescriptor."""
    lat, lon = _pos(entity)
    meta = _meta(entity)
    callsign = (entity.get("callsign") or meta.get("callsign")
                or entity.get("flight") or entity.get("flight_id"))
    return TrackDescriptor(
        source_id=str(entity.get("entity_id") or entity.get("id") or ""),
        modality="ADSB",
        lat=lat,
        lon=lon,
        alt_m=_alt(entity),
        speed_mps=_speed(entity),
        heading_deg=_heading(entity),
        size_m=None,
        mmsi=None,
        callsign=str(callsign) if callsign else None,
        visual_id=None,
        ts=_ts(entity),
        confidence=float(entity.get("confidence", 1.0)),
    )


def adapt_ais(entity: dict) -> TrackDescriptor:
    """Normalize an AIS maritime entity into a TrackDescriptor."""
    lat, lon = _pos(entity)
    meta = _meta(entity)
    mmsi = (entity.get("mmsi") or meta.get("mmsi"))
    callsign = (entity.get("callsign") or meta.get("callsign"))
    # Length overall (LOA) as physical size estimate
    size_m = (entity.get("length_m") or meta.get("length_m") or entity.get("size_m"))
    return TrackDescriptor(
        source_id=str(entity.get("entity_id") or entity.get("id") or ""),
        modality="AIS",
        lat=lat,
        lon=lon,
        alt_m=None,
        speed_mps=_speed(entity),
        heading_deg=_heading(entity),
        size_m=float(size_m) if size_m is not None else None,
        mmsi=str(mmsi) if mmsi else None,
        callsign=str(callsign) if callsign else None,
        visual_id=None,
        ts=_ts(entity),
        confidence=float(entity.get("confidence", 1.0)),
    )


def adapt_camera(entity: dict) -> TrackDescriptor:
    """Normalize a camera/vision track into a TrackDescriptor.

    Extracts visual_id from metadata.reid_embedding if present by hashing the
    embedding vector to a short hex string for fast comparison.
    """
    lat, lon = _pos(entity)
    meta = _meta(entity)

    # Re-ID embedding hash — use first 16 hex chars of SHA-256 of the raw value
    reid_raw = meta.get("reid_embedding")
    if reid_raw is not None:
        reid_bytes = (
            reid_raw.encode() if isinstance(reid_raw, str)
            else str(reid_raw).encode()
        )
        visual_id = hashlib.sha256(reid_bytes).hexdigest()[:16]
    else:
        visual_id = entity.get("visual_id") or meta.get("visual_id")

    size_m = (entity.get("size_m") or meta.get("bbox_size_m")
              or meta.get("apparent_size_m"))
    return TrackDescriptor(
        source_id=str(entity.get("entity_id") or entity.get("id") or ""),
        modality="CAMERA",
        lat=lat,
        lon=lon,
        alt_m=_alt(entity),
        speed_mps=_speed(entity),
        heading_deg=_heading(entity),
        size_m=float(size_m) if size_m is not None else None,
        mmsi=None,
        callsign=None,
        visual_id=str(visual_id) if visual_id else None,
        ts=_ts(entity),
        confidence=float(entity.get("confidence", meta.get("detection_confidence", 1.0))),
    )


def adapt_radar(entity: dict) -> TrackDescriptor:
    """Normalize a radar track into a TrackDescriptor."""
    lat, lon = _pos(entity)
    meta = _meta(entity)
    size_m = (entity.get("size_m") or meta.get("rcs_size_m")
              or meta.get("target_size_m") or entity.get("rcs_m"))
    return TrackDescriptor(
        source_id=str(entity.get("entity_id") or entity.get("id") or ""),
        modality="RADAR",
        lat=lat,
        lon=lon,
        alt_m=_alt(entity),
        speed_mps=_speed(entity),
        heading_deg=_heading(entity),
        size_m=float(size_m) if size_m is not None else None,
        mmsi=None,
        callsign=None,
        visual_id=None,
        ts=_ts(entity),
        confidence=float(entity.get("confidence", meta.get("snr_confidence", 1.0))),
    )


def adapt_sonar(entity: dict) -> TrackDescriptor:
    """Normalize a sonar (underwater/surface acoustic) track into a TrackDescriptor."""
    lat, lon = _pos(entity)
    meta = _meta(entity)
    size_m = (entity.get("size_m") or meta.get("acoustic_size_m"))
    # Sonar tracks typically have depth, stored in alt_m as negative value
    depth_m = (entity.get("depth_m") or meta.get("depth_m"))
    alt = _alt(entity)
    if alt is None and depth_m is not None:
        alt = -float(depth_m)
    return TrackDescriptor(
        source_id=str(entity.get("entity_id") or entity.get("id") or ""),
        modality="SONAR",
        lat=lat,
        lon=lon,
        alt_m=alt,
        speed_mps=_speed(entity),
        heading_deg=_heading(entity),
        size_m=float(size_m) if size_m is not None else None,
        mmsi=None,
        callsign=None,
        visual_id=None,
        ts=_ts(entity),
        confidence=float(entity.get("confidence", meta.get("acoustic_confidence", 1.0))),
    )


def adapt_mesh(entity: dict) -> TrackDescriptor:
    """Normalize a Meshtastic mesh node entity into a TrackDescriptor."""
    lat, lon = _pos(entity)
    meta = _meta(entity)
    callsign = (entity.get("callsign") or meta.get("node_id")
                or meta.get("long_name") or meta.get("short_name")
                or entity.get("node_id"))
    return TrackDescriptor(
        source_id=str(entity.get("entity_id") or entity.get("id") or ""),
        modality="MESH",
        lat=lat,
        lon=lon,
        alt_m=_alt(entity),
        speed_mps=_speed(entity),
        heading_deg=_heading(entity),
        size_m=None,
        mmsi=None,
        callsign=str(callsign) if callsign else None,
        visual_id=None,
        ts=_ts(entity),
        confidence=float(entity.get("confidence", meta.get("snr", 1.0))),
    )


# ── Dispatch ─────────────────────────────────────────────────────────────────

_MODALITY_ADAPTERS = {
    "ADSB":     adapt_adsb,
    "AIS":      adapt_ais,
    "CAMERA":   adapt_camera,
    "RADAR":    adapt_radar,
    "SONAR":    adapt_sonar,
    "MESH":     adapt_mesh,
    "MESHTASTIC": adapt_mesh,
}

_TYPE_TO_MODALITY = {
    "adsb":       "ADSB",
    "ais":        "AIS",
    "camera":     "CAMERA",
    "vision":     "CAMERA",
    "radar":      "RADAR",
    "sonar":      "SONAR",
    "mesh":       "MESH",
    "meshtastic": "MESH",
}


def adapt_entity(entity: dict) -> Optional[TrackDescriptor]:
    """Dispatch to the correct modality adapter based on entity type or source_adapter field."""
    # Explicit adapter field takes precedence
    adapter_field = (entity.get("source_adapter") or entity.get("adapter")
                     or entity.get("sensor_type") or "")
    modality = _MODALITY_ADAPTERS.get(adapter_field.upper())
    if modality:
        return modality(entity)

    # Fall back to entity type
    entity_type = str(entity.get("type") or entity.get("entity_type") or "").lower()
    mapped = _TYPE_TO_MODALITY.get(entity_type)
    if mapped and mapped in _MODALITY_ADAPTERS:
        return _MODALITY_ADAPTERS[mapped](entity)

    # Try partial match on source field
    source = str(entity.get("source") or "").lower()
    for key, mod in _TYPE_TO_MODALITY.items():
        if key in source:
            return _MODALITY_ADAPTERS[mod](entity)

    return None
