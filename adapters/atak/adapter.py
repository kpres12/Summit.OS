"""
Heli.OS CoT/ATAK Adapter

Bidirectional Cursor on Target (CoT) integration for ATAK-equipped teams.
Receives SA (situational awareness) CoT events and publishes them as Heli.OS
entities. Also broadcasts Heli.OS entities back as CoT so ATAK clients see
the full operating picture.

CoT is the universal protocol for first-responder interoperability:
  - Fire departments
  - Search and rescue teams
  - Law enforcement
  - FEMA / state OES

Environment variables:
    ATAK_ENABLED          - "true" to enable
    ATAK_UDP_HOST         - bind address for incoming CoT (default: 0.0.0.0)
    ATAK_UDP_PORT         - incoming CoT port (default: 4242)
    ATAK_MULTICAST        - "true" to join ATAK multicast group (239.2.3.1:6969)
    ATAK_MULTICAST_GROUP  - multicast group (default: 239.2.3.1)
    ATAK_MULTICAST_PORT   - multicast port (default: 6969)
    ATAK_SEND_HOST        - target host for outbound CoT (default: broadcast)
    ATAK_SEND_PORT        - target port for outbound CoT (default: 4242)
    ATAK_CALLSIGN         - this node's callsign for outbound SA events
    ATAK_ORG_ID           - org_id tag on published entities
    MQTT_HOST / MQTT_PORT - broker connection
"""
from __future__ import annotations

import asyncio
import logging
import os
import socket
import struct
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "packages"))
from sdk import BaseAdapter, AdapterManifest, EntityBuilder, Protocol, Capability

logger = logging.getLogger("summit.adapter.atak")

# CoT type prefix → Heli.OS entity classification
_COT_TYPE_MAP = {
    "a-f": ("friendly", "active"),
    "a-h": ("hostile", "alert"),
    "a-u": ("unknown", "neutral"),
    "a-n": ("neutral", "neutral"),
    "b-":  ("alert",   "alert"),   # broadcast / emergency
}

# CoT how → domain hint
_COT_HOW_DOMAIN = {
    "h-g-i-g-o": "GROUND",
    "m-g":        "GROUND",
    "h-e":        "AERIAL",
    "m-a":        "AERIAL",
}


def _parse_cot(xml_bytes: bytes) -> Optional[Dict[str, Any]]:
    """Parse a CoT XML message into a Heli.OS observation dict. Returns None on failure."""
    try:
        root = ET.fromstring(xml_bytes.decode("utf-8", errors="replace"))
        if root.tag != "event":
            return None

        uid      = root.get("uid", "")
        cot_type = root.get("type", "a-u")
        how      = root.get("how", "m-g")
        stale    = root.get("stale", "")

        point = root.find("point")
        if point is None:
            return None
        lat = float(point.get("lat", 0))
        lon = float(point.get("lon", 0))
        alt = float(point.get("hae", 0))   # height above ellipsoid (meters)
        ce  = float(point.get("ce", 15))   # circular error (meters) — use as sigma

        # Detail block — callsign, group, etc.
        detail   = root.find("detail") or ET.Element("detail")
        contact  = detail.find("contact")
        callsign = (contact.get("callsign") if contact is not None else None) or uid.split("-")[0]

        group_el  = detail.find("__group")
        group     = group_el.get("name", "") if group_el is not None else ""
        role_el   = detail.find("__group")
        role      = role_el.get("role", "") if role_el is not None else ""

        # Map CoT type to Heli classification
        entity_type, classification = "neutral", "neutral"
        for prefix, (etype, cls) in _COT_TYPE_MAP.items():
            if cot_type.startswith(prefix):
                entity_type, classification = etype, cls
                break

        domain = "GROUND"
        for how_prefix, dom in _COT_HOW_DOMAIN.items():
            if how.startswith(how_prefix[:3]):
                domain = dom
                break
        # Sub-type override
        if "-A-" in cot_type:
            domain = "AERIAL"
        if "-S-" in cot_type or "-V-" in cot_type:
            domain = "MARITIME"

        return {
            "entity_id": f"cot-{uid}",
            "callsign":   callsign,
            "lat": lat, "lon": lon, "alt": alt,
            "sigma_m": ce,
            "entity_type": entity_type,
            "classification": classification,
            "domain": domain,
            "cot_type": cot_type,
            "group": group,
            "role": role,
            "stale": stale,
        }
    except Exception as e:
        logger.debug(f"CoT parse error: {e}")
        return None


def _build_cot(entity: Dict[str, Any], callsign_self: str) -> bytes:
    """Build a minimal SA CoT XML event from a Heli.OS entity dict."""
    now  = datetime.now(timezone.utc)
    stale = (now + timedelta(seconds=60)).strftime("%Y-%m-%dT%H:%M:%S.0Z")
    now_s = now.strftime("%Y-%m-%dT%H:%M:%S.0Z")

    uid      = entity.get("entity_id", "summit-unknown")
    callsign = entity.get("callsign") or uid
    lat      = entity.get("position", {}).get("lat", 0.0)
    lon      = entity.get("position", {}).get("lon", 0.0)
    alt      = entity.get("position", {}).get("alt", 0.0)
    etype    = entity.get("entity_type", "neutral")
    cot_type = "a-f-G-U-C" if etype == "friendly" else "a-h-G" if etype == "hostile" else "a-u-G"

    xml = (
        f'<?xml version="1.0" encoding="UTF-8"?>'
        f'<event version="2.0" uid="{uid}" type="{cot_type}" '
        f'time="{now_s}" start="{now_s}" stale="{stale}" how="m-g">'
        f'<point lat="{lat:.7f}" lon="{lon:.7f}" hae="{alt:.1f}" ce="9999999.0" le="9999999.0"/>'
        f'<detail>'
        f'<contact callsign="{callsign}"/>'
        f'<remarks>Heli.OS entity — {uid}</remarks>'
        f'</detail>'
        f'</event>'
    )
    return xml.encode("utf-8")


class ATAKAdapter(BaseAdapter):
    """
    CoT/ATAK adapter — bidirectional situational awareness bridge.

    Receives CoT XML over UDP (unicast or multicast), publishes as Heli.OS
    TRACK entities. Also sends Heli.OS entities out as CoT so ATAK clients
    maintain a complete common operating picture.
    """

    MANIFEST = AdapterManifest(
        name="atak",
        version="1.0.0",
        protocol=Protocol.UDP,
        capabilities=[Capability.READ, Capability.WRITE, Capability.STREAM],
        entity_types=["TRACK", "ASSET"],
        description="CoT/ATAK bidirectional situational awareness adapter",
        optional_env=["ATAK_UDP_HOST", "ATAK_UDP_PORT", "ATAK_MULTICAST", "ATAK_SEND_HOST"],
    )

    def __init__(self, **kwargs):
        super().__init__(device_id="atak", **kwargs)
        self.udp_host       = os.getenv("ATAK_UDP_HOST", "0.0.0.0")
        self.udp_port       = int(os.getenv("ATAK_UDP_PORT", "4242"))
        self.multicast      = os.getenv("ATAK_MULTICAST", "false").lower() == "true"
        self.mc_group       = os.getenv("ATAK_MULTICAST_GROUP", "239.2.3.1")
        self.mc_port        = int(os.getenv("ATAK_MULTICAST_PORT", "6969"))
        self.send_host      = os.getenv("ATAK_SEND_HOST", "255.255.255.255")
        self.send_port      = int(os.getenv("ATAK_SEND_PORT", "4242"))
        self.callsign_self  = os.getenv("ATAK_CALLSIGN", "SUMMIT-01")
        self.org_id         = os.getenv("ATAK_ORG_ID", "")
        self._send_sock: Optional[socket.socket] = None

    @property
    def enabled(self) -> bool:
        return os.getenv("ATAK_ENABLED", "false").lower() == "true"

    async def run(self):
        loop = asyncio.get_event_loop()

        # Receive socket
        recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        recv_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        recv_sock.setblocking(False)

        if self.multicast:
            recv_sock.bind(("", self.mc_port))
            mreq = struct.pack("4sL", socket.inet_aton(self.mc_group), socket.INADDR_ANY)
            recv_sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
            logger.info(f"ATAK adapter joined multicast {self.mc_group}:{self.mc_port}")
        else:
            recv_sock.bind((self.udp_host, self.udp_port))
            logger.info(f"ATAK adapter listening on {self.udp_host}:{self.udp_port}")

        # Send socket (broadcast / unicast)
        self._send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._send_sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

        try:
            while not self.stopped:
                try:
                    data, addr = await loop.run_in_executor(None, self._recv_with_timeout, recv_sock)
                    if data:
                        obs = _parse_cot(data)
                        if obs:
                            entity = self._obs_to_entity(obs)
                            self.publish(entity, qos=0)
                except Exception as e:
                    logger.debug(f"CoT receive error: {e}")
                    await asyncio.sleep(0.1)
        finally:
            recv_sock.close()
            if self._send_sock:
                self._send_sock.close()

    def _recv_with_timeout(self, sock: socket.socket, timeout: float = 1.0):
        """Blocking receive with timeout — runs in executor."""
        sock.settimeout(timeout)
        try:
            return sock.recvfrom(65535)
        except socket.timeout:
            return None, None

    def send_cot(self, entity: Dict[str, Any]) -> bool:
        """Send a Heli.OS entity out as a CoT SA event to ATAK clients."""
        if not self._send_sock:
            return False
        try:
            xml_bytes = _build_cot(entity, self.callsign_self)
            self._send_sock.sendto(xml_bytes, (self.send_host, self.send_port))
            return True
        except Exception as e:
            logger.warning(f"CoT send error: {e}")
            return False

    def _obs_to_entity(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        b = (
            EntityBuilder(obs["entity_id"], obs.get("callsign", obs["entity_id"]))
            .track()
            .at(obs["lat"], obs["lon"], obs.get("alt", 0.0))
            .label(obs.get("cot_type", ""))
            .source("atak", obs["entity_id"])
            .org(self.org_id)
            .ttl(120)
            .meta_dict({
                "cot_type":       obs.get("cot_type", ""),
                "group":          obs.get("group", ""),
                "role":           obs.get("role", ""),
                "domain":         obs.get("domain", "GROUND"),
                "classification": obs.get("classification", "neutral"),
                "protocol":       "cot",
            })
        )
        if obs.get("entity_type") == "hostile":
            b = b.critical()
        elif obs.get("entity_type") == "unknown":
            b = b.warning()
        else:
            b = b.active()
        return b.build()
