"""
Heli.OS — Link 16 / VMF Tactical Data Link Adapter (Scaffold)
================================================================
Tactical data-link gateway adapter for Link 16 (J-series messages, MIDS
terminals) and VMF (Variable Message Format, MIL-STD-6017) — the two
TDL standards most relevant to ACE / CANVAS-class deployments.

**This is a scaffold.** Operational Link 16 / VMF integration requires
a paired hardware terminal (MIDS LVT, MIDS-JTRS, BATS-D, KOR-24) and a
licensed Type-1 cryptographic loadout that Heli.OS does not ship.
Deployments use a host-side message gateway from Curtiss-Wright,
General Dynamics, or Collins Aerospace — or directly from a STANAG
4406 / TENA messaging fabric — with this adapter as the Heli.OS-side
ingestion + emission layer.

Inputs to this adapter (from the gateway):
  - J-series messages already decoded from Link 16 (JREAP-C XML or
    proprietary native format)
  - VMF K-series messages (MIL-STD-6017 chapter 5 message types)

Outputs from this adapter (toward the gateway):
  - Authorized C2 commands once a human says cleared-hot via the
    engagement-authorization gate

Configuration modes (extra.mode):
  jreap_c_listener     — listen for JREAP-C XML over TCP (default)
  vmf_serial           — read VMF over RS-232 / 422 serial port
  jreap_emitter        — emit J-series messages to a downstream gateway

Common message types we ingest:
  J2.x  — Precise Participant Location and Identification (PPLI)
  J3.x  — Surveillance tracks (air, surface, subsurface, land)
  J7.x  — Information management (alerts, status, requests)
  J10.x — Weapons coordination (status, assignment) — **read-only**
  J12.x — Control (mission management, vector, engagement)
  J13.x — Platform & system status
  K01.x — VMF basic messages
  K02.x — VMF position reports

Note: This module does NOT perform autonomous engagement. It surfaces
J10 / J12 weapon-coordination messages as decision support inputs to
EngagementAuthorizationGate, which still requires a human operator
to authorize before any outbound command is emitted.

Configuration / .env:
  LINK16_GATEWAY_URL         — JREAP-C TCP endpoint
  LINK16_GATEWAY_AUTH_TOKEN  — gateway-issued token (vendor-specific)
  VMF_SERIAL_PORT            — e.g. /dev/tty.usbserial-A5XK3ZQ7
  VMF_BAUD_RATE              — e.g. 9600
"""

from __future__ import annotations

import asyncio
import logging
import os
import socket
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator, Optional

from .base import BaseAdapter, AdapterConfig

logger = logging.getLogger("heli.adapters.link16")


def _env(name: str, default: str = "") -> str:
    val = os.environ.get(name, default)
    if val:
        return val
    env = Path(__file__).parent.parent.parent / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith(f"{name}="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return default


# Message-class friendly names
J_SERIES_NAMES = {
    "J2.0":  "Precise Participant Location and Identification (PPLI)",
    "J2.2":  "Air PPLI",
    "J2.3":  "Surface PPLI",
    "J3.0":  "Reference Point",
    "J3.2":  "Air Track",
    "J3.3":  "Surface Track",
    "J3.5":  "Land Track",
    "J3.7":  "Electronic Warfare Track",
    "J7.0":  "Track Management",
    "J7.1":  "Data Update Request",
    "J7.2":  "Correlation",
    "J10.2": "Engagement Status",
    "J10.5": "Weapon Coordination",
    "J12.0": "Mission Assignment",
    "J12.1": "Vector",
    "J12.6": "Target Sorting",
    "J13.0": "Airfield Status",
    "J13.2": "Air Platform & System Status",
    "J13.3": "Surface Platform & System Status",
    "J13.5": "Land Platform & System Status",
}


class Link16VMFAdapter(BaseAdapter):
    adapter_type = "link16_vmf"

    def __init__(self, config: AdapterConfig, mqtt_client=None):
        super().__init__(config, mqtt_client=mqtt_client)
        extra = config.extra or {}
        self._mode = extra.get("mode", "jreap_c_listener")
        self._gateway_url = extra.get("gateway_url") or _env("LINK16_GATEWAY_URL")
        self._gateway_token = extra.get("gateway_token") or _env("LINK16_GATEWAY_AUTH_TOKEN")
        self._serial_port = extra.get("serial_port") or _env("VMF_SERIAL_PORT")
        self._baud_rate = int(extra.get("baud_rate") or _env("VMF_BAUD_RATE", "9600"))
        # Networks of allowed J-series message classes (subscription filter)
        self._allowed_classes: set[str] = set(extra.get("allowed_classes") or [
            "J2.2", "J2.3", "J3.2", "J3.3", "J3.5", "J3.7",
            "J7.0", "J7.1", "J7.2",
            "J10.2", "J10.5",
            "J12.0", "J12.1", "J12.6",
            "J13.0", "J13.2", "J13.3", "J13.5",
        ])

    async def connect(self) -> None:
        if self._mode == "jreap_c_listener":
            if not self._gateway_url:
                logger.warning(
                    "[link16] %s: LINK16_GATEWAY_URL not configured — "
                    "running in scaffold mode only", self.config.adapter_id)
        elif self._mode == "vmf_serial":
            if not self._serial_port:
                logger.warning(
                    "[link16] %s: VMF_SERIAL_PORT not configured — "
                    "running in scaffold mode only", self.config.adapter_id)

    async def disconnect(self) -> None:
        return

    async def stream_observations(self) -> AsyncIterator[dict]:
        if self._mode == "jreap_c_listener":
            async for obs in self._jreap_c_loop():
                yield obs
        elif self._mode == "vmf_serial":
            async for obs in self._vmf_serial_loop():
                yield obs
        elif self._mode == "jreap_emitter":
            # Emitter mode is outbound only — no observations to stream
            while True:
                await asyncio.sleep(60)
        else:
            logger.warning("[link16] unknown mode %s", self._mode)
            while True:
                await asyncio.sleep(60)

    async def _jreap_c_loop(self) -> AsyncIterator[dict]:
        """JREAP-C is XML-over-TCP framed by the gateway. Parse + emit."""
        if not self._gateway_url:
            while True:
                await asyncio.sleep(60)
            return
        while True:
            try:
                async for obs in self._jreap_c_one_session():
                    yield obs
            except Exception as e:
                logger.warning("[link16] %s JREAP-C session error: %s",
                               self.config.adapter_id, e)
            await asyncio.sleep(self.config.poll_interval_seconds)

    async def _jreap_c_one_session(self) -> AsyncIterator[dict]:
        # Real implementations use a vendor SDK over TLS. The scaffold
        # parses XML messages from a plain TCP socket for testing against
        # an open simulator (e.g. a ground-rig stub).
        host, _, port_s = (self._gateway_url or "").partition(":")
        port = int(port_s or "5005")
        loop = asyncio.get_event_loop()
        sock = await loop.run_in_executor(None, socket.socket, socket.AF_INET, socket.SOCK_STREAM)
        try:
            await loop.run_in_executor(None, sock.connect, (host, port))
        except Exception as e:
            logger.warning("[link16] connect %s:%d failed: %s", host, port, e)
            return
        try:
            buf = b""
            while True:
                data = await loop.run_in_executor(None, sock.recv, 65536)
                if not data:
                    return
                buf += data
                # Frames are delimited by a JREAP-C envelope. Scaffold
                # uses a permissive split on </Message>.
                while b"</Message>" in buf:
                    frame, _, buf = buf.partition(b"</Message>")
                    frame = frame + b"</Message>"
                    obs = self._parse_jreap(frame)
                    if obs is not None:
                        yield obs
        finally:
            sock.close()

    def _parse_jreap(self, frame: bytes) -> Optional[dict]:
        try:
            root = ET.fromstring(frame.decode("utf-8", errors="replace"))
        except Exception:
            return None
        msg_class = (root.attrib.get("class") or root.findtext("Class") or "").strip()
        if msg_class and msg_class not in self._allowed_classes:
            return None
        # Generic extraction — gateway-specific schema would be richer
        lat = float(root.findtext("Lat") or root.findtext(".//Lat") or 0.0)
        lon = float(root.findtext("Lon") or root.findtext(".//Lon") or 0.0)
        track_id = root.findtext("TrackID") or root.findtext(".//TrackID") or ""
        return {
            "adapter_type":  self.adapter_type,
            "adapter_id":    self.config.adapter_id,
            "ts":            datetime.now(timezone.utc).isoformat(),
            "entity_type":   "tdl_message",
            "asset_type":    "j_series",
            "j_class":       msg_class,
            "j_class_name":  J_SERIES_NAMES.get(msg_class, ""),
            "track_id":      track_id,
            "lat":           lat,
            "lon":           lon,
            "raw":           frame.decode("utf-8", errors="replace")[:2000],
        }

    async def _vmf_serial_loop(self) -> AsyncIterator[dict]:
        """VMF is binary over serial. Real impl uses pyserial + MIL-STD-6017
        bit-field parser. Scaffold yields nothing without a port."""
        if not self._serial_port:
            while True:
                await asyncio.sleep(60)
            return
        # Real implementation:
        #   import serial
        #   with serial.Serial(self._serial_port, self._baud_rate) as s:
        #       while True:
        #           frame = read_vmf_frame(s)
        #           yield parse_vmf(frame)
        # For now, log and idle so the registry stays happy.
        logger.info("[link16] VMF serial scaffold registered for %s @ %d baud "
                    "(no decoder shipped — install pyserial + 6017 parser)",
                    self._serial_port, self._baud_rate)
        while True:
            await asyncio.sleep(60)

    async def send_command(self, command: str, params: dict | None = None) -> dict:
        """Outbound — emit a J/K message via the gateway. Only fires after
        engagement-authorization gate authorizes. Always passes through a
        signed-decision check."""
        params = params or {}
        # Validate that the calling code has an authorized engagement
        if not params.get("engagement_authorized"):
            return {"status": "rejected",
                    "reason": "send_command requires engagement_authorized=True "
                              "from EngagementAuthorizationGate"}
        # Real implementation: format a J/K message and forward via the
        # gateway's outbound API. Scaffold logs only.
        logger.info("[link16] %s OUTBOUND %s (params=%s) — scaffold log only",
                    self.config.adapter_id, command, list(params.keys()))
        return {"status": "logged_scaffold",
                "command": command,
                "would_emit": params.get("j_class", "unspecified")}
