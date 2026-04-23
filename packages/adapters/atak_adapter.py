"""
Heli.OS — ATAK / Cursor-on-Target (CoT) Adapter
=================================================
Bridges Heli.OS entities to the Android Team Awareness Kit (ATAK) ecosystem
and any CoT-compatible system (WinTAK, iTAK, TAK Server, ATAK-CIV).

Publishes Heli.OS WorldStore entities as CoT XML events and optionally
receives CoT from the TAK network, ingesting them as Heli.OS entities.

Transports:
  UDP broadcast  — direct peer-to-peer ATAK (default, no server needed)
  UDP unicast    — point-to-point to a specific host
  TCP → TAK Server — enterprise deployment (port 8087/8088 with optional TLS)

CoT type mapping:
  mavlink/dji (drone)  → a-f-A-M-F-Q  (Friendly Air Military Quadrotor)
  spot (ground robot)  → a-f-G-R-U    (Friendly Ground Robot Unit)
  onvif/camera         → a-f-G-I-S    (Friendly Ground Installation Sensor)
  ais (vessel)         → a-u-S-X-M    (Unknown Surface craft)
  detected person      → a-u-G-U      (Unknown Ground Unit)
  detected vehicle     → a-u-G-E-V    (Unknown Ground Equipment Vehicle)
  fire/hotspot         → t-x-c-f      (Hazard chemical/fire)
  waypoint             → b-m-p-w      (Waypoint)

Dependencies:
  None beyond stdlib — CoT is plain XML over UDP/TCP.

Config extras:
  transport         : str   — "udp_broadcast" | "udp_unicast" | "tcp" (default: udp_broadcast)
  host              : str   — Target IP for unicast/tcp (default: 239.2.3.1 multicast)
  port              : int   — Port (default: 4242)
  callsign_prefix   : str   — Callsign prefix for entity IDs (default: "HELI")
  stale_seconds     : int   — CoT stale time in seconds (default: 60)
  recv_enabled      : bool  — Whether to ingest incoming CoT (default: True)
  recv_port         : int   — Port to listen for incoming CoT (default: 4242)
  tls_cert          : str   — Path to client cert for TAK Server TLS
  tls_key           : str   — Path to client key for TAK Server TLS
"""
from __future__ import annotations

import asyncio
import logging
import socket
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta
from typing import AsyncIterator, Optional

from .base import AdapterConfig, BaseAdapter

logger = logging.getLogger("heli.adapters.atak")

_UDP_MULTICAST = "239.2.3.1"
_DEFAULT_PORT  = 4242
_TAK_TCP_PORT  = 8087

# Entity type → CoT 2525B type string
_ENTITY_COT_TYPE: dict[str, str] = {
    "mavlink":   "a-f-A-M-F-Q",    # Friendly Air Military Quadrotor
    "dji":       "a-f-A-M-F-Q",
    "spot":      "a-f-G-R-U",      # Friendly Ground Robot Unit
    "onvif":     "a-f-G-I-S",      # Friendly Ground Installation Sensor
    "rtsp":      "a-f-G-I-S",
    "ais":       "a-u-S-X-M",      # Unknown Surface Marine
    "aisstream": "a-u-S-X-M",
    "opensky":   "a-f-A-C-F",      # Friendly Air Civil Fixed-wing
    "default":   "a-u-G-U",        # Unknown Ground Unit
}

_CLASSIFICATION_COT: dict[str, str] = {
    "person":         "a-u-G-U",
    "person_thermal": "a-u-G-U",
    "vehicle":        "a-u-G-E-V",
    "vessel":         "a-u-S-X",
    "fire":           "t-x-c-f",
    "smoke":          "t-x-c",
    "hotspot":        "t-x-c-f",
}


def _dtg(dt: datetime) -> str:
    """Format datetime as CoT DTG string (ISO 8601 UTC)."""
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _entity_to_cot(entity: dict, callsign_prefix: str, stale_secs: int) -> Optional[str]:
    """Convert a Heli.OS entity dict to CoT XML string."""
    entity_id = entity.get("entity_id") or entity.get("id", "")
    if not entity_id:
        return None

    position = entity.get("position") or {}
    lat = position.get("lat") or entity.get("lat")
    lon = position.get("lon") or entity.get("lon")
    if lat is None or lon is None:
        return None

    lat   = float(lat)
    lon   = float(lon)
    alt   = float(position.get("alt_m") or position.get("alt") or 0)
    speed = 0.0
    course = 0.0

    vel = entity.get("velocity") or {}
    if vel:
        speed  = float(vel.get("speed_mps", 0) or 0)
        course = float(vel.get("heading_deg", 0) or 0)

    now   = datetime.now(timezone.utc)
    stale = now + timedelta(seconds=stale_secs)

    adapter_type = entity.get("adapter_type", "default")
    classification = entity.get("classification", "")
    cot_type = (
        _CLASSIFICATION_COT.get(classification)
        or _ENTITY_COT_TYPE.get(adapter_type)
        or _ENTITY_COT_TYPE["default"]
    )

    callsign = entity.get("callsign") or f"{callsign_prefix}-{entity_id[-6:].upper()}"
    battery  = (entity.get("metadata") or {}).get("battery_pct")

    root = ET.Element("event", {
        "version": "2.0",
        "uid":     f"HELI-{entity_id}",
        "type":    cot_type,
        "time":    _dtg(now),
        "start":   _dtg(now),
        "stale":   _dtg(stale),
        "how":     "m-g",
        "access":  "Undefined",
    })

    ET.SubElement(root, "point", {
        "lat": f"{lat:.7f}",
        "lon": f"{lon:.7f}",
        "hae": f"{alt:.1f}",
        "ce":  "50",   # circular error (meters)
        "le":  "50",   # linear error (meters)
    })

    detail = ET.SubElement(root, "detail")
    ET.SubElement(detail, "contact", {"callsign": callsign})

    if speed > 0 or course > 0:
        ET.SubElement(detail, "track", {
            "speed":  f"{speed:.2f}",
            "course": f"{course:.1f}",
        })

    if battery is not None:
        ET.SubElement(detail, "status", {"battery": str(int(battery))})

    remarks_parts = []
    if adapter_type:
        remarks_parts.append(f"adapter:{adapter_type}")
    if classification:
        remarks_parts.append(f"class:{classification}")
    conf = entity.get("confidence")
    if conf:
        remarks_parts.append(f"conf:{conf:.2f}")
    if remarks_parts:
        ET.SubElement(detail, "remarks").text = " | ".join(remarks_parts)

    return '<?xml version="1.0" encoding="UTF-8"?>' + ET.tostring(root, encoding="unicode")


def _cot_to_entity(xml_str: str) -> Optional[dict]:
    """Parse incoming CoT XML into a Heli.OS entity dict."""
    try:
        root = ET.fromstring(xml_str.encode("utf-8", errors="replace"))
        if root.tag != "event":
            return None

        uid   = root.get("uid", "")
        ctype = root.get("type", "a-u-G-U")
        how   = root.get("how", "")

        point = root.find("point")
        if point is None:
            return None

        lat = float(point.get("lat", 0))
        lon = float(point.get("lon", 0))
        alt = float(point.get("hae", 0))

        detail   = root.find("detail") or ET.Element("detail")
        contact  = detail.find("contact")
        callsign = (contact.get("callsign") if contact is not None else None) or uid.split("-")[0]

        track = detail.find("track")
        speed  = float(track.get("speed",  0)) if track is not None else 0.0
        course = float(track.get("course", 0)) if track is not None else 0.0

        remarks_el = detail.find("remarks")
        remarks    = (remarks_el.text or "") if remarks_el is not None else ""

        # Map CoT type to Heli.OS entity_type
        entity_type = "UNKNOWN"
        if ctype.startswith("a-f-A"):
            entity_type = "UAV"
        elif ctype.startswith("a-f-G"):
            entity_type = "GROUND_UNIT"
        elif ctype.startswith("a-u-G"):
            entity_type = "GROUND_UNIT"
        elif ctype.startswith("a-u-S") or ctype.startswith("a-f-S"):
            entity_type = "VESSEL"
        elif ctype.startswith("t-x"):
            entity_type = "HAZARD"
        elif ctype.startswith("b-m"):
            entity_type = "WAYPOINT"

        return {
            "entity_id":    uid,
            "callsign":     callsign,
            "adapter_type": "atak",
            "entity_type":  entity_type,
            "source":       "cot",
            "cot_type":     ctype,
            "how":          how,
            "position":     {"lat": lat, "lon": lon, "alt_m": alt},
            "velocity":     {"speed_mps": speed, "heading_deg": course},
            "ts_iso":       root.get("time", ""),
            "metadata":     {"remarks": remarks, "cot_raw_type": ctype},
        }
    except Exception as e:
        logger.debug("CoT parse error: %s", e)
        return None


class ATAKAdapter(BaseAdapter):
    """
    Publishes Heli.OS entity updates as Cursor-on-Target (CoT) XML events
    to ATAK-compatible systems. Optionally receives CoT and ingests entities.

    This adapter is OUTPUT-focused: it reads the Heli.OS entity stream (via
    MQTT observations) and broadcasts CoT. It does not control any hardware.

    For military coordination: provides blue force tracking, entity sharing,
    and situational awareness integration with ATAK/WinTAK/TAK Server.
    """

    adapter_type = "atak"

    def __init__(self, config: AdapterConfig, mqtt_client=None) -> None:
        super().__init__(config, mqtt_client)
        ex = config.extra
        self._transport       = ex.get("transport", "udp_broadcast")
        self._host            = ex.get("host", _UDP_MULTICAST)
        self._port            = int(ex.get("port", _DEFAULT_PORT))
        self._callsign_prefix = ex.get("callsign_prefix", "HELI")
        self._stale_secs      = int(ex.get("stale_seconds", 60))
        self._recv_enabled    = bool(ex.get("recv_enabled", True))
        self._recv_port       = int(ex.get("recv_port", _DEFAULT_PORT))
        self._tls_cert        = ex.get("tls_cert")
        self._tls_key         = ex.get("tls_key")

        self._udp_sock: Optional[socket.socket] = None
        self._tcp_writer: Optional[asyncio.StreamWriter] = None
        self._recv_queue: asyncio.Queue = asyncio.Queue(maxsize=200)

    async def connect(self) -> None:
        if self._transport in ("udp_broadcast", "udp_unicast"):
            self._udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._udp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if self._transport == "udp_broadcast":
                self._udp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
                # Join multicast group if using multicast address
                if self._host.startswith("239."):
                    mreq = socket.inet_aton(self._host) + socket.inet_aton("0.0.0.0")
                    self._udp_sock.setsockopt(
                        socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq
                    )
            self._udp_sock.setblocking(False)
            logger.info("ATAK UDP %s connected → %s:%d",
                        self._transport, self._host, self._port)

        elif self._transport == "tcp":
            ssl_ctx = None
            if self._tls_cert and self._tls_key:
                import ssl
                ssl_ctx = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
                ssl_ctx.load_cert_chain(self._tls_cert, self._tls_key)
                ssl_ctx.check_hostname = False
                ssl_ctx.verify_mode = ssl.CERT_NONE
            reader, writer = await asyncio.open_connection(
                self._host, self._port, ssl=ssl_ctx
            )
            self._tcp_writer = writer
            logger.info("ATAK TCP connected → %s:%d%s",
                        self._host, self._port, " (TLS)" if ssl_ctx else "")

        # Start CoT receiver if enabled
        if self._recv_enabled:
            asyncio.create_task(self._recv_loop())

    async def disconnect(self) -> None:
        if self._udp_sock:
            try:
                self._udp_sock.close()
            except Exception:
                pass
            self._udp_sock = None
        if self._tcp_writer:
            try:
                self._tcp_writer.close()
                await self._tcp_writer.wait_closed()
            except Exception:
                pass
            self._tcp_writer = None

    async def publish_entity(self, entity: dict) -> None:
        """Publish a single entity as a CoT event."""
        cot_xml = _entity_to_cot(entity, self._callsign_prefix, self._stale_secs)
        if not cot_xml:
            return
        await self._send_cot(cot_xml)

    async def _send_cot(self, xml_str: str) -> None:
        data = xml_str.encode("utf-8")
        try:
            if self._udp_sock:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None,
                    lambda: self._udp_sock.sendto(data, (self._host, self._port))
                )
            elif self._tcp_writer:
                self._tcp_writer.write(data)
                await self._tcp_writer.drain()
        except Exception as e:
            logger.debug("ATAK send failed: %s", e)

    async def _recv_loop(self) -> None:
        """Listen for incoming CoT and queue as entities."""
        try:
            loop = asyncio.get_event_loop()
            recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            recv_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            recv_sock.bind(("", self._recv_port))
            recv_sock.setblocking(False)

            logger.info("ATAK CoT receiver listening on UDP :%d", self._recv_port)

            while not self._stop_event.is_set():
                try:
                    data = await asyncio.wait_for(
                        loop.run_in_executor(None, lambda: recv_sock.recv(65535)),
                        timeout=1.0,
                    )
                    xml_str = data.decode("utf-8", errors="replace")
                    entity = _cot_to_entity(xml_str)
                    if entity:
                        if self._recv_queue.full():
                            self._recv_queue.get_nowait()
                        await self._recv_queue.put(entity)
                except (asyncio.TimeoutError, BlockingIOError):
                    continue
                except Exception as e:
                    logger.debug("CoT recv error: %s", e)
        except Exception as e:
            logger.warning("ATAK recv loop failed: %s", e)
        finally:
            try:
                recv_sock.close()
            except Exception:
                pass

    async def stream_observations(self) -> AsyncIterator[dict]:
        """Yield incoming CoT entities from the ATAK network."""
        while not self._stop_event.is_set():
            try:
                entity = await asyncio.wait_for(self._recv_queue.get(), timeout=1.0)
                # Wrap as observation event
                yield {
                    "source_id":    f"atak:{entity['entity_id']}",
                    "adapter_id":   self.config.adapter_id,
                    "adapter_type": "atak",
                    "entity_id":    entity["entity_id"],
                    "callsign":     entity.get("callsign", ""),
                    "entity_type":  entity.get("entity_type", "UNKNOWN"),
                    "event_type":   "PEER_OBSERVATION",
                    "position":     entity.get("position"),
                    "velocity":     entity.get("velocity"),
                    "ts_iso":       entity.get("ts_iso", ""),
                    "metadata":     entity.get("metadata", {}),
                }
            except asyncio.TimeoutError:
                continue

    async def send_command(self, command: str, params: dict | None = None) -> dict:
        """Publish an entity or waypoint to the ATAK network."""
        params = params or {}
        cmd = command.upper()

        if cmd == "PUBLISH_ENTITY":
            entity = params.get("entity", {})
            await self.publish_entity(entity)
            return {"status": "published", "uid": f"HELI-{entity.get('entity_id', '')}"}

        if cmd == "PUBLISH_WAYPOINT":
            wp = {
                "entity_id":    params.get("id", "wp-001"),
                "callsign":     params.get("name", "WP"),
                "adapter_type": "waypoint",
                "position": {
                    "lat": params.get("lat", 0),
                    "lon": params.get("lon", 0),
                    "alt_m": params.get("alt_m", 0),
                },
            }
            await self.publish_entity(wp)
            return {"status": "published", "type": "waypoint"}

        return {"status": "unknown_command", "command": command}
