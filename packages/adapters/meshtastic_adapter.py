"""
packages/adapters/meshtastic_adapter.py — Meshtastic mesh radio adapter.

Connects Heli.OS to a Meshtastic LoRa mesh network. Each mesh node
becomes a tracked entity in the world model — field teams, sensors, vehicles,
or any device running Meshtastic firmware appear on the operator map in
real-time without cellular infrastructure.

Why this matters for Heli.OS:
  - Meshtastic operates on LoRa (915MHz/868MHz/433MHz) — works where cellular,
    Wi-Fi, and internet don't. Dense smoke, remote terrain, urban canyons.
  - Range: 10–50km per hop (terrain-dependent), multi-hop mesh routing
  - GPS position broadcasting: every node with GPS reports lat/lon/alt/speed
  - Bidirectional: Heli.OS can send mission waypoints, alerts, and commands
    back through the mesh to field operators
  - No infrastructure required — pure off-grid coordination

Packet types handled:
  POSITION_APP     → entity position update (lat/lon/alt/speed/heading)
  NODEINFO_APP     → entity identity (callsign, hardware type, MAC)
  TELEMETRY_APP    → battery level, environment sensors (temp/pressure/humidity)
  TEXT_MESSAGE_APP → freetext mesh messages → Heli.OS alert feed

Connection modes:
  serial  → direct USB connection to a Meshtastic node (default: auto-detect)
  tcp     → TCP connection to meshtastic-python HTTP bridge or direct device
             (e.g., Meshtastic T-Beam with WiFi firmware)

Config (extra fields):
  connection_type: "serial" | "tcp"           (default: "serial")
  hostname:        TCP host or serial port     (default: "localhost" / auto)
  port:            TCP port or serial baud     (default: 4403 / 115200)
  channel:         channel index to receive on (default: 0 = primary)
  send_channel:    channel index to send on    (default: 0)
  node_filter:     list of node IDs to include (default: all)
  position_ttl_s:  drop positions older than N seconds (default: 300)

Requirements:
  pip install meshtastic>=2.3.0

Environment:
  MESHTASTIC_HOST   — override hostname
  MESHTASTIC_PORT   — override port
  MESHTASTIC_CONN   — "serial" | "tcp"

Reference:
  https://meshtastic.org/docs/development/python/
  https://python.meshtastic.org/
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict, List, Optional

from .base import BaseAdapter, AdapterConfig

logger = logging.getLogger("heli.adapters.meshtastic")

# Meshtastic PortNums (packet type identifiers)
_PORT_TEXT_MESSAGE = 1
_PORT_REMOTE_HARDWARE = 2
_PORT_POSITION = 3
_PORT_NODEINFO = 4
_PORT_TELEMETRY = 67
_PORT_WAYPOINT = 71

# Entity type mapping for Heli.OS world model
_HARDWARE_TYPE_MAP = {
    # Meshtastic hardware IDs → Summit entity type
    "TBEAM": "GROUND",
    "HELTEC_V1": "GROUND",
    "HELTEC_V2_0": "GROUND",
    "HELTEC_V2_1": "GROUND",
    "TBEAM_V0P7": "GROUND",
    "T_ECHO": "GROUND",
    "TLORA_V1": "GROUND",
    "RAK4631": "GROUND",
    "STATION_G1": "SENSOR",
    "RASPBERRY_PI_PICO_W": "SENSOR",
}


class MeshtasticAdapter(BaseAdapter):
    """
    Meshtastic LoRa mesh radio adapter.

    Publishes each mesh node as a Heli.OS entity with position, battery,
    and telemetry. Supports sending text messages and waypoints back through
    the mesh to field operators.

    Observation output format (inherits BaseAdapter schema):
      entity_type: "GROUND" | "SENSOR" | "MESH_NODE"
      metadata includes:
        - battery_level (0-100)
        - snr (signal-to-noise ratio, dB)
        - rssi (received signal strength, dBm)
        - hop_count (mesh hops from this node)
        - hardware_model (device type string)
        - last_heard (ISO timestamp)
        - channel_index
        - mesh_messages (recent text messages from this node)
    """

    adapter_type = "meshtastic"

    @classmethod
    def required_extra_fields(cls) -> list[str]:
        return []  # All fields have defaults

    def __init__(self, config: AdapterConfig, mqtt_client=None) -> None:
        super().__init__(config, mqtt_client)
        extra = config.extra or {}

        self._conn_type = (
            os.getenv("MESHTASTIC_CONN")
            or extra.get("connection_type", "serial")
        ).lower()
        self._hostname = (
            os.getenv("MESHTASTIC_HOST")
            or extra.get("hostname", "localhost")
        )
        self._port = int(
            os.getenv("MESHTASTIC_PORT")
            or extra.get("port", 4403 if self._conn_type == "tcp" else 0)
        )
        self._channel = int(extra.get("channel", 0))
        self._send_channel = int(extra.get("send_channel", 0))
        self._node_filter: Optional[List[str]] = extra.get("node_filter")
        self._position_ttl = float(extra.get("position_ttl_s", 300.0))

        # Meshtastic interface (lazy-init in connect())
        self._iface = None
        self._obs_queue: asyncio.Queue = asyncio.Queue(maxsize=500)
        self._node_cache: Dict[str, Dict[str, Any]] = {}

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def connect(self) -> None:
        """Open connection to Meshtastic node."""
        try:
            import meshtastic
            import meshtastic.serial_interface
            import meshtastic.tcp_interface
            from pubsub import pub
        except ImportError:
            raise RuntimeError(
                "meshtastic package not installed. "
                "Run: pip install meshtastic>=2.3.0"
            )

        loop = asyncio.get_event_loop()

        def _open_iface():
            if self._conn_type == "tcp":
                logger.info(
                    "Meshtastic: connecting via TCP to %s:%d",
                    self._hostname, self._port,
                )
                return meshtastic.tcp_interface.TCPInterface(
                    hostname=self._hostname,
                    portNumber=self._port if self._port else None,
                )
            else:
                port = self._hostname if self._hostname != "localhost" else None
                logger.info(
                    "Meshtastic: connecting via serial (port=%s)", port or "auto"
                )
                return meshtastic.serial_interface.SerialInterface(
                    devPath=port,
                )

        self._iface = await loop.run_in_executor(None, _open_iface)

        # Subscribe to packet events using pubsub
        pub.subscribe(self._on_receive, "meshtastic.receive")
        pub.subscribe(self._on_connection, "meshtastic.connection.established")
        pub.subscribe(self._on_lost, "meshtastic.connection.lost")

        logger.info(
            "Meshtastic connected. Node count: %d",
            len(self._iface.nodes or {}),
        )

        # Seed node cache from current mesh state
        if self._iface.nodes:
            for node_id, node_info in self._iface.nodes.items():
                self._node_cache[node_id] = node_info

    async def disconnect(self) -> None:
        """Close Meshtastic connection."""
        if self._iface is not None:
            try:
                from pubsub import pub
                pub.unsubscribe(self._on_receive, "meshtastic.receive")
                pub.unsubscribe(self._on_connection, "meshtastic.connection.established")
                pub.unsubscribe(self._on_lost, "meshtastic.connection.lost")
            except Exception:
                pass
            try:
                await asyncio.get_event_loop().run_in_executor(
                    None, self._iface.close
                )
            except Exception:
                pass
            self._iface = None

    async def stream_observations(self) -> AsyncIterator[dict]:
        """Yield observations from the mesh packet queue."""
        # First, yield current node states from the seeded cache
        for node_id, node_info in self._node_cache.items():
            obs = self._node_to_observation(node_id, node_info)
            if obs:
                yield obs

        # Then stream live updates
        while True:
            try:
                obs = await asyncio.wait_for(
                    self._obs_queue.get(), timeout=30.0
                )
                yield obs
            except asyncio.TimeoutError:
                # Emit a keepalive heartbeat for connected nodes
                for node_id, node_info in list(self._node_cache.items()):
                    obs = self._node_to_observation(node_id, node_info)
                    if obs:
                        yield obs
            except asyncio.CancelledError:
                break

    # ── pubsub callbacks (called from meshtastic thread) ──────────────────────

    def _on_receive(self, packet: dict, interface=None) -> None:
        """Handle incoming Meshtastic packet (called in meshtastic thread)."""
        try:
            obs = self._packet_to_observation(packet)
            if obs:
                # Thread-safe enqueue
                try:
                    self._obs_queue.put_nowait(obs)
                except asyncio.QueueFull:
                    logger.debug("Meshtastic obs queue full — dropping packet")
        except Exception as exc:
            logger.debug("Meshtastic packet parse error: %s", exc)

    def _on_connection(self, interface=None, topic=None) -> None:
        logger.info("Meshtastic connection established")

    def _on_lost(self, interface=None, topic=None) -> None:
        logger.warning("Meshtastic connection lost")

    # ── Packet parsing ─────────────────────────────────────────────────────────

    def _packet_to_observation(self, packet: dict) -> Optional[dict]:
        """Convert a raw Meshtastic packet to a Heli.OS observation dict."""
        from_id = packet.get("fromId") or packet.get("from")
        if not from_id:
            return None

        # Apply node filter
        if self._node_filter and str(from_id) not in self._node_filter:
            return None

        decoded = packet.get("decoded", {})
        portnum = decoded.get("portnum", "")

        # Update node cache
        if from_id not in self._node_cache:
            self._node_cache[from_id] = {}

        if portnum == "POSITION_APP" or portnum == _PORT_POSITION:
            return self._parse_position(from_id, decoded, packet)
        elif portnum == "NODEINFO_APP" or portnum == _PORT_NODEINFO:
            return self._parse_nodeinfo(from_id, decoded, packet)
        elif portnum == "TELEMETRY_APP" or portnum == _PORT_TELEMETRY:
            return self._parse_telemetry(from_id, decoded, packet)
        elif portnum == "TEXT_MESSAGE_APP" or portnum == _PORT_TEXT_MESSAGE:
            return self._parse_text_message(from_id, decoded, packet)
        else:
            return None

    def _parse_position(self, node_id: str, decoded: dict, packet: dict) -> Optional[dict]:
        """Parse POSITION_APP packet."""
        pos = decoded.get("position", decoded)
        lat = pos.get("latitudeI") or pos.get("latitude")
        lon = pos.get("longitudeI") or pos.get("longitude")

        if lat is None or lon is None:
            return None

        # latitudeI is stored as integer * 1e7
        if isinstance(lat, int) and abs(lat) > 180:
            lat = lat / 1e7
        if isinstance(lon, int) and abs(lon) > 180:
            lon = lon / 1e7

        lat = float(lat)
        lon = float(lon)

        # Basic sanity check
        if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
            return None

        alt_m = pos.get("altitude")
        speed = pos.get("groundSpeed") or pos.get("speed")
        heading = pos.get("groundTrack") or pos.get("heading")

        # Update cache
        cached = self._node_cache.get(node_id, {})
        cached.update({"lat": lat, "lon": lon, "alt_m": alt_m, "last_position_ts": time.time()})
        self._node_cache[node_id] = cached

        node_info = self._iface.nodes.get(node_id, {}) if self._iface else {}
        user = node_info.get("user", {})
        callsign = user.get("shortName") or user.get("longName") or str(node_id)

        return {
            "source_id": f"mesh-{node_id}-pos-{int(time.time())}",
            "adapter_id": self.config.adapter_id,
            "adapter_type": self.adapter_type,
            "entity_id": f"mesh-{node_id}",
            "callsign": callsign,
            "position": {
                "lat": lat,
                "lon": lon,
                "alt_m": float(alt_m) if alt_m is not None else None,
            },
            "velocity": {
                "heading_deg": float(heading) if heading else None,
                "speed_mps": float(speed) / 3.6 if speed else None,  # km/h → m/s
                "vertical_mps": None,
            },
            "entity_type": _entity_type_for_node(node_info),
            "classification": "MESH_NODE",
            "metadata": {
                "node_id": str(node_id),
                "hardware_model": user.get("hwModel", "UNKNOWN"),
                "snr": packet.get("rxSnr"),
                "rssi": packet.get("rxRssi"),
                "hop_count": packet.get("hopStart", 0),
                "channel_index": self._channel,
                "battery_level": cached.get("battery_level"),
                "network": "meshtastic",
            },
            "ts_iso": datetime.now(timezone.utc).isoformat(),
        }

    def _parse_nodeinfo(self, node_id: str, decoded: dict, packet: dict) -> Optional[dict]:
        """Parse NODEINFO_APP packet — update entity identity."""
        user = decoded.get("user", decoded)
        callsign = user.get("shortName") or user.get("longName") or str(node_id)
        hw_model = user.get("hwModel", "UNKNOWN")

        # Update cache
        cached = self._node_cache.get(node_id, {})
        cached.update({"callsign": callsign, "hw_model": hw_model})
        self._node_cache[node_id] = cached

        # Only emit an observation if we have a position
        lat = cached.get("lat")
        lon = cached.get("lon")
        if lat is None or lon is None:
            return None

        return {
            "source_id": f"mesh-{node_id}-info-{int(time.time())}",
            "adapter_id": self.config.adapter_id,
            "adapter_type": self.adapter_type,
            "entity_id": f"mesh-{node_id}",
            "callsign": callsign,
            "position": {
                "lat": lat,
                "lon": lon,
                "alt_m": cached.get("alt_m"),
            },
            "velocity": None,
            "entity_type": _entity_type_from_hw(hw_model),
            "classification": "MESH_NODE",
            "metadata": {
                "node_id": str(node_id),
                "hardware_model": hw_model,
                "long_name": user.get("longName"),
                "macaddr": user.get("macaddr"),
                "snr": packet.get("rxSnr"),
                "rssi": packet.get("rxRssi"),
                "network": "meshtastic",
            },
            "ts_iso": datetime.now(timezone.utc).isoformat(),
        }

    def _parse_telemetry(self, node_id: str, decoded: dict, packet: dict) -> Optional[dict]:
        """Parse TELEMETRY_APP — battery + environment sensors."""
        telem = decoded.get("telemetry", decoded)

        # Device metrics
        device = telem.get("deviceMetrics", {})
        battery = device.get("batteryLevel")
        voltage = device.get("voltage")
        channel_util = device.get("channelUtilization")

        # Environment metrics
        env = telem.get("environmentMetrics", {})
        temperature = env.get("temperature")
        humidity = env.get("relativeHumidity")
        pressure = env.get("barometricPressure")

        # Update cache
        cached = self._node_cache.get(node_id, {})
        if battery is not None:
            cached["battery_level"] = battery
        self._node_cache[node_id] = cached

        lat = cached.get("lat")
        lon = cached.get("lon")
        if lat is None or lon is None:
            return None

        metadata: Dict[str, Any] = {
            "node_id": str(node_id),
            "network": "meshtastic",
            "snr": packet.get("rxSnr"),
            "rssi": packet.get("rxRssi"),
        }
        if battery is not None:
            metadata["battery_level"] = battery
        if voltage is not None:
            metadata["voltage_v"] = voltage
        if channel_util is not None:
            metadata["channel_utilization"] = channel_util
        if temperature is not None:
            metadata["temperature_c"] = temperature
        if humidity is not None:
            metadata["humidity_pct"] = humidity
        if pressure is not None:
            metadata["pressure_hpa"] = pressure

        return {
            "source_id": f"mesh-{node_id}-telem-{int(time.time())}",
            "adapter_id": self.config.adapter_id,
            "adapter_type": self.adapter_type,
            "entity_id": f"mesh-{node_id}",
            "callsign": cached.get("callsign", str(node_id)),
            "position": {"lat": lat, "lon": lon, "alt_m": cached.get("alt_m")},
            "velocity": None,
            "entity_type": "SENSOR" if not cached.get("lat") else "GROUND",
            "classification": "MESH_NODE",
            "metadata": metadata,
            "ts_iso": datetime.now(timezone.utc).isoformat(),
        }

    def _parse_text_message(self, node_id: str, decoded: dict, packet: dict) -> Optional[dict]:
        """Parse TEXT_MESSAGE_APP — mesh text → Heli.OS alert."""
        text = decoded.get("text", decoded.get("payload", b""))
        if isinstance(text, bytes):
            try:
                text = text.decode("utf-8", errors="replace")
            except Exception:
                text = ""

        if not text:
            return None

        cached = self._node_cache.get(node_id, {})
        callsign = cached.get("callsign", str(node_id))

        # Publish as MQTT alert to summit/alerts/mesh so ops console picks it up
        if self.mqtt is not None:
            import json
            alert_payload = json.dumps({
                "alert_id": f"mesh-msg-{node_id}-{int(time.time())}",
                "severity": "info",
                "description": f"[MESH] {callsign}: {text}",
                "source": f"mesh-{node_id}",
                "ts_iso": datetime.now(timezone.utc).isoformat(),
            })
            try:
                self.mqtt.publish("summit/alerts/mesh", alert_payload, qos=0)
            except Exception:
                pass

        # Only return entity observation if we have position
        lat = cached.get("lat")
        lon = cached.get("lon")
        if lat is None or lon is None:
            return None

        return {
            "source_id": f"mesh-{node_id}-msg-{int(time.time())}",
            "adapter_id": self.config.adapter_id,
            "adapter_type": self.adapter_type,
            "entity_id": f"mesh-{node_id}",
            "callsign": callsign,
            "position": {"lat": lat, "lon": lon, "alt_m": cached.get("alt_m")},
            "velocity": None,
            "entity_type": "GROUND",
            "classification": "MESH_NODE",
            "metadata": {
                "node_id": str(node_id),
                "last_message": text,
                "network": "meshtastic",
                "snr": packet.get("rxSnr"),
                "rssi": packet.get("rxRssi"),
            },
            "ts_iso": datetime.now(timezone.utc).isoformat(),
        }

    def _node_to_observation(self, node_id: str, node_info: dict) -> Optional[dict]:
        """Convert a cached node entry to an observation (for initial seed + heartbeat)."""
        user = node_info.get("user", {})
        position = node_info.get("position", {})

        lat = position.get("latitudeI") or position.get("latitude")
        lon = position.get("longitudeI") or position.get("longitude")

        if lat is None or lon is None:
            return None

        if isinstance(lat, int) and abs(lat) > 180:
            lat = lat / 1e7
        if isinstance(lon, int) and abs(lon) > 180:
            lon = lon / 1e7

        lat, lon = float(lat), float(lon)
        if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
            return None

        callsign = user.get("shortName") or user.get("longName") or str(node_id)
        hw_model = user.get("hwModel", "UNKNOWN")
        last_heard = node_info.get("lastHeard")

        # Skip stale positions
        if last_heard and (time.time() - last_heard) > self._position_ttl:
            return None

        device_metrics = node_info.get("deviceMetrics", {})
        battery = device_metrics.get("batteryLevel")

        return {
            "source_id": f"mesh-{node_id}-seed-{int(time.time())}",
            "adapter_id": self.config.adapter_id,
            "adapter_type": self.adapter_type,
            "entity_id": f"mesh-{node_id}",
            "callsign": callsign,
            "position": {
                "lat": lat,
                "lon": lon,
                "alt_m": float(position.get("altitude", 0)) or None,
            },
            "velocity": {
                "heading_deg": position.get("groundTrack"),
                "speed_mps": (
                    float(position["groundSpeed"]) / 3.6
                    if position.get("groundSpeed")
                    else None
                ),
                "vertical_mps": None,
            },
            "entity_type": _entity_type_from_hw(hw_model),
            "classification": "MESH_NODE",
            "metadata": {
                "node_id": str(node_id),
                "hardware_model": hw_model,
                "long_name": user.get("longName"),
                "battery_level": battery,
                "last_heard_ts": (
                    datetime.fromtimestamp(last_heard, tz=timezone.utc).isoformat()
                    if last_heard else None
                ),
                "network": "meshtastic",
            },
            "ts_iso": datetime.now(timezone.utc).isoformat(),
        }

    # ── Outbound: send commands through the mesh ───────────────────────────────

    async def send_text(self, message: str, destination_id: str = "^all") -> bool:
        """
        Send a text message through the Meshtastic mesh.

        destination_id: node ID (e.g. "!a1b2c3d4") or "^all" for broadcast.
        Returns True on success.
        """
        if self._iface is None:
            return False
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(
                None,
                lambda: self._iface.sendText(
                    message,
                    destinationId=destination_id,
                    channelIndex=self._send_channel,
                ),
            )
            logger.info("Meshtastic: sent text to %s: %s", destination_id, message[:50])
            return True
        except Exception as exc:
            logger.error("Meshtastic: send_text failed: %s", exc)
            return False

    async def send_waypoint(
        self,
        name: str,
        lat: float,
        lon: float,
        description: str = "",
        destination_id: str = "^all",
    ) -> bool:
        """
        Broadcast a named waypoint to all mesh nodes (or a specific node).
        Field operators' devices will display the waypoint on their Meshtastic map.
        """
        if self._iface is None:
            return False
        loop = asyncio.get_event_loop()
        try:
            # Build waypoint packet
            from meshtastic import mesh_pb2
            wp = mesh_pb2.Waypoint()
            wp.name = name[:30]  # Meshtastic name limit
            wp.description = description[:100]
            wp.latitudeI = int(lat * 1e7)
            wp.longitudeI = int(lon * 1e7)

            await loop.run_in_executor(
                None,
                lambda: self._iface.sendWaypoint(
                    wp,
                    destinationId=destination_id,
                    channelIndex=self._send_channel,
                ),
            )
            logger.info(
                "Meshtastic: sent waypoint '%s' (%.5f, %.5f) to %s",
                name, lat, lon, destination_id,
            )
            return True
        except ImportError:
            # Fallback: encode as text message
            msg = f"WPT {name}: {lat:.5f},{lon:.5f} — {description}"
            return await self.send_text(msg, destination_id)
        except Exception as exc:
            logger.error("Meshtastic: send_waypoint failed: %s", exc)
            return False

    async def broadcast_alert(self, alert_text: str) -> bool:
        """
        Broadcast a Heli.OS alert to all mesh nodes.
        Appears as an incoming message on all connected Meshtastic devices.
        """
        return await self.send_text(f"[SUMMIT] {alert_text}", "^all")


# ── Helpers ────────────────────────────────────────────────────────────────────


def _entity_type_for_node(node_info: dict) -> str:
    user = node_info.get("user", {})
    return _entity_type_from_hw(user.get("hwModel", ""))


def _entity_type_from_hw(hw_model: str) -> str:
    hw = str(hw_model).upper()
    for key, entity_type in _HARDWARE_TYPE_MAP.items():
        if key in hw:
            return entity_type
    return "GROUND"
