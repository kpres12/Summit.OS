"""
Summit.OS — CAP (Common Alerting Protocol) Adapter
====================================================

Polls CAP/Atom feeds (NWS, FEMA, international emergency alert systems) and
emits ALERT entities for each active alert. Deduplicates by CAP identifier so
the same alert is not re-emitted.

Dependencies
------------
    pip install aiohttp
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import AsyncIterator, Optional

try:
    import aiohttp
except ImportError:
    raise ImportError(
        "aiohttp is required for CAPAdapter. Install with: pip install aiohttp>=3.9.0"
    )

from .base import AdapterConfig, BaseAdapter

logger = logging.getLogger("summit.adapters.cap")

DEFAULT_CAP_FEEDS: list[str] = [
    "https://alerts.weather.gov/cap/us.php?x=1",  # NWS all US alerts
]

# CAP XML namespaces
_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "cap": "urn:oasis:names:tc:emergency:cap:1.2",
    "cap11": "urn:oasis:names:tc:emergency:cap:1.1",
}

_SEVERITY_MAP = {
    "extreme": "CRITICAL",
    "severe": "HIGH",
    "moderate": "MEDIUM",
    "minor": "LOW",
    "unknown": "INFO",
}


def _cap_ns(tag: str) -> list[str]:
    """Return list of fully-qualified tag names for both CAP 1.1 and 1.2."""
    return [f"{{{_NS['cap']}}}{tag}", f"{{{_NS['cap11']}}}{tag}"]


def _find(element: ET.Element, tag: str) -> Optional[ET.Element]:
    """Find a child element trying both CAP namespace versions."""
    for qualified in _cap_ns(tag):
        found = element.find(qualified)
        if found is not None:
            return found
    return element.find(tag)


def _findtext(element: ET.Element, tag: str) -> Optional[str]:
    el = _find(element, tag)
    return el.text.strip() if el is not None and el.text else None


def _parse_polygon_centroid(polygon_text: str) -> Optional[tuple[float, float]]:
    """Parse a CAP polygon (space-separated lat,lon pairs) and return centroid."""
    try:
        pairs = polygon_text.strip().split()
        coords = []
        for pair in pairs:
            parts = pair.split(",")
            if len(parts) >= 2:
                coords.append((float(parts[0]), float(parts[1])))
        if coords:
            lat = sum(c[0] for c in coords) / len(coords)
            lon = sum(c[1] for c in coords) / len(coords)
            return lat, lon
    except Exception:
        pass
    return None


def _parse_circle(circle_text: str) -> Optional[tuple[float, float]]:
    """Parse a CAP circle ('lat,lon radius') and return the center."""
    try:
        parts = circle_text.strip().split()
        if parts:
            coord = parts[0].split(",")
            return float(coord[0]), float(coord[1])
    except Exception:
        pass
    return None


class CAPAdapter(BaseAdapter):
    """
    Polls CAP/Atom feed URLs and emits ALERT entities.

    Config extras
    -------------
    feed_urls              : list[str]   (default NWS US feed)
    poll_interval_seconds  : float       (default 60.0)
    event_filter           : list[str]   (empty = all events)
    bbox                   : dict | None  with min_lat/max_lat/min_lon/max_lon
    """

    adapter_type = "cap"

    def __init__(self, config: AdapterConfig, mqtt_client=None) -> None:
        super().__init__(config, mqtt_client)
        ex = config.extra

        self._feed_urls: list[str] = ex.get("feed_urls", DEFAULT_CAP_FEEDS)
        if isinstance(self._feed_urls, str):
            self._feed_urls = [self._feed_urls]

        self._poll_interval: float = float(
            ex.get("poll_interval_seconds", config.poll_interval_seconds or 60.0)
        )
        self._event_filter: list[str] = [e.lower() for e in ex.get("event_filter", [])]
        self._bbox: Optional[dict] = ex.get("bbox", None)

        self._session: Optional[aiohttp.ClientSession] = None
        self._seen_ids: set[str] = set()  # Deduplicate by CAP identifier

    async def connect(self) -> None:
        self._session = aiohttp.ClientSession(
            headers={"User-Agent": "SummitOS-CAPAdapter/1.0"}
        )
        self._log.info("CAP adapter ready — polling %d feed(s)", len(self._feed_urls))

    async def disconnect(self) -> None:
        try:
            if self._session is not None:
                await self._session.close()
                self._session = None
        except Exception:
            pass

    async def stream_observations(self) -> AsyncIterator[dict]:
        while not self._stop_event.is_set():
            for url in self._feed_urls:
                try:
                    alerts = await self._fetch_feed(url)
                    for obs in alerts:
                        yield obs
                except Exception as exc:
                    self._log.warning("CAP feed error [%s]: %s", url, exc)
            await self._interruptible_sleep(self._poll_interval)

    async def _fetch_feed(self, url: str) -> list[dict]:
        async with self._session.get(
            url, timeout=aiohttp.ClientTimeout(total=30)
        ) as resp:
            resp.raise_for_status()
            content = await resp.text()

        return self._parse_cap_feed(content, url)

    def _parse_cap_feed(self, content: str, source_url: str) -> list[dict]:
        results = []
        try:
            root = ET.fromstring(content)
        except ET.ParseError as exc:
            self._log.warning("XML parse error from %s: %s", source_url, exc)
            return results

        # Support both Atom feeds (with <entry> elements containing <content>)
        # and direct CAP XML (<alert> at root)
        entries = []
        tag = root.tag.split("}")[-1] if "}" in root.tag else root.tag

        if tag == "alert":
            # Direct CAP document
            entries = [root]
        elif tag == "feed":
            # Atom feed — each <entry> may embed a <content> with CAP XML
            # or have the alert inline
            for entry in root.findall(f"{{{_NS['atom']}}}entry"):
                content_el = entry.find(f"{{{_NS['atom']}}}content")
                if content_el is not None and content_el.text:
                    try:
                        entries.append(ET.fromstring(content_el.text.strip()))
                    except ET.ParseError:
                        pass
                else:
                    # Try to find an alert element directly in the entry
                    for alert_tag in _cap_ns("alert"):
                        alert_el = entry.find(f".//{alert_tag}")
                        if alert_el is not None:
                            entries.append(alert_el)
                            break
        elif tag == "rss":
            # RSS channel
            for item in root.findall(".//item"):
                for alert_tag in _cap_ns("alert"):
                    alert_el = item.find(f".//{alert_tag}")
                    if alert_el is not None:
                        entries.append(alert_el)
                        break

        for alert_el in entries:
            obs = self._parse_alert(alert_el)
            if obs is not None:
                results.append(obs)

        return results

    def _parse_alert(self, alert_el: ET.Element) -> Optional[dict]:
        # Top-level CAP fields
        identifier = _findtext(alert_el, "identifier")
        if not identifier:
            return None

        # Deduplicate
        if identifier in self._seen_ids:
            return None

        sender = _findtext(alert_el, "sender")
        sent = _findtext(alert_el, "sent")
        status = _findtext(alert_el, "status")
        msg_type = _findtext(alert_el, "msgType")
        scope = _findtext(alert_el, "scope")

        # Find <info> block (first one)
        info_el = None
        for info_tag in _cap_ns("info"):
            info_el = alert_el.find(info_tag)
            if info_el is not None:
                break

        if info_el is None:
            return None

        event = _findtext(info_el, "event") or ""
        severity = _findtext(info_el, "severity") or "Unknown"
        urgency = _findtext(info_el, "urgency")
        certainty = _findtext(info_el, "certainty")
        headline = _findtext(info_el, "headline")
        description = _findtext(info_el, "description")

        # Apply event filter
        if self._event_filter and event.lower() not in self._event_filter:
            return None

        # Parse area — polygon or circle
        lat: Optional[float] = None
        lon: Optional[float] = None
        polygon_text: Optional[str] = None
        circle_text: Optional[str] = None

        area_el = None
        for area_tag in _cap_ns("area"):
            area_el = info_el.find(area_tag)
            if area_el is not None:
                break

        if area_el is not None:
            polygon_text = _findtext(area_el, "polygon")
            circle_text = _findtext(area_el, "circle")

            if polygon_text:
                result = _parse_polygon_centroid(polygon_text)
                if result:
                    lat, lon = result
            elif circle_text:
                result = _parse_circle(circle_text)
                if result:
                    lat, lon = result

        # Apply bbox filter
        if self._bbox and lat is not None and lon is not None:
            b = self._bbox
            if not (
                b["min_lat"] <= lat <= b["max_lat"]
                and b["min_lon"] <= lon <= b["max_lon"]
            ):
                return None

        # Map severity
        summit_severity = _SEVERITY_MAP.get(severity.lower(), "INFO")

        self._seen_ids.add(identifier)
        now = datetime.now(timezone.utc)

        return {
            "source_id": f"cap-{hashlib.sha1(identifier.encode()).hexdigest()[:12]}",
            "adapter_id": self.config.adapter_id,
            "adapter_type": self.adapter_type,
            "entity_id": f"cap-{hashlib.sha1(identifier.encode()).hexdigest()[:16]}",
            "callsign": headline or event or identifier,
            "position": (
                {"lat": lat, "lon": lon, "alt_m": None}
                if lat is not None and lon is not None
                else None
            ),
            "velocity": None,
            "entity_type": "ALERT",
            "classification": summit_severity,
            "metadata": {
                "cap_identifier": identifier,
                "sender": sender,
                "sent": sent,
                "status": status,
                "msg_type": msg_type,
                "scope": scope,
                "event": event,
                "severity": severity,
                "summit_severity": summit_severity,
                "urgency": urgency,
                "certainty": certainty,
                "headline": headline,
                "description": description,
                "polygon": polygon_text,
                "circle": circle_text,
            },
            "ts_iso": now.isoformat(),
        }
