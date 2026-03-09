"""
Summit.OS — NMEA 0183 GPS Adapter
===================================

Connects to a GPS receiver via serial port or TCP socket, parses NMEA 0183
sentences (GGA, RMC, VTG, GSA), and emits a single GPS_ASSET entity
representing the device the GPS is attached to.

Dependencies
------------
    pip install pynmea2 pyserial
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import AsyncIterator, Optional

try:
    import pynmea2
except ImportError:
    raise ImportError(
        "pynmea2 is required for NMEAAdapter. Install with: pip install pynmea2"
    )

from .base import AdapterConfig, BaseAdapter

logger = logging.getLogger("summit.adapters.nmea")


class NMEAAdapter(BaseAdapter):
    """
    Parses NMEA 0183 sentences from a serial port or TCP socket.

    Config extras
    -------------
    connection_type : "serial" | "tcp"  (default "tcp")
    serial_port     : str               (default "/dev/ttyUSB0")
    baud_rate       : int               (default 4800)
    tcp_host        : str               (default "")
    tcp_port        : int               (default 10110)
    entity_id_override : str            (default "" — uses adapter_id)
    """

    adapter_type = "nmea"

    def __init__(self, config: AdapterConfig, mqtt_client=None) -> None:
        super().__init__(config, mqtt_client)
        ex = config.extra

        self._connection_type: str = ex.get("connection_type", "tcp")
        self._serial_port: str = ex.get("serial_port", "/dev/ttyUSB0")
        self._baud_rate: int = int(ex.get("baud_rate", 4800))
        self._tcp_host: str = ex.get("tcp_host", "")
        self._tcp_port: int = int(ex.get("tcp_port", 10110))
        self._entity_id: str = ex.get("entity_id_override", "") or config.adapter_id

        # Connection handles
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._serial = None  # pyserial handle (sync, wrapped with executor)

        # Accumulated state across sentences
        self._lat: Optional[float] = None
        self._lon: Optional[float] = None
        self._alt_m: Optional[float] = None
        self._speed_mps: Optional[float] = None
        self._course_deg: Optional[float] = None
        self._gps_quality: Optional[int] = None
        self._satellites: Optional[int] = None
        self._hdop: Optional[float] = None
        self._pdop: Optional[float] = None
        self._fix_type: Optional[str] = None

    async def connect(self) -> None:
        if self._connection_type == "serial":
            await self._connect_serial()
        else:
            await self._connect_tcp()

    async def _connect_tcp(self) -> None:
        if not self._tcp_host:
            raise ValueError("tcp_host must be set for connection_type=tcp")
        self._reader, self._writer = await asyncio.wait_for(
            asyncio.open_connection(self._tcp_host, self._tcp_port),
            timeout=10.0,
        )
        self._log.info("Connected to NMEA TCP %s:%d", self._tcp_host, self._tcp_port)

    async def _connect_serial(self) -> None:
        try:
            import serial
        except ImportError:
            raise ImportError(
                "pyserial is required for serial connections. Install with: pip install pyserial"
            )
        loop = asyncio.get_event_loop()
        self._serial = await loop.run_in_executor(
            None,
            lambda: serial.Serial(self._serial_port, self._baud_rate, timeout=1),
        )
        self._log.info(
            "Opened serial port %s @ %d baud", self._serial_port, self._baud_rate
        )

    async def disconnect(self) -> None:
        try:
            if self._writer is not None:
                self._writer.close()
                await self._writer.wait_closed()
                self._writer = None
                self._reader = None
        except Exception:
            pass
        try:
            if self._serial is not None:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self._serial.close)
                self._serial = None
        except Exception:
            pass

    async def stream_observations(self) -> AsyncIterator[dict]:
        async for line in self._read_lines():
            obs = self._process_sentence(line.strip())
            if obs is not None:
                yield obs

    async def _read_lines(self):
        """Read NMEA lines from TCP or serial."""
        if self._connection_type == "tcp":
            while not self._stop_event.is_set():
                try:
                    raw = await asyncio.wait_for(
                        self._reader.readline(), timeout=15.0
                    )
                except asyncio.TimeoutError:
                    continue
                line = raw.decode("ascii", errors="ignore").strip()
                if line:
                    yield line
        else:
            loop = asyncio.get_event_loop()
            while not self._stop_event.is_set():
                try:
                    raw = await loop.run_in_executor(
                        None, lambda: self._serial.readline()
                    )
                    line = raw.decode("ascii", errors="ignore").strip()
                    if line:
                        yield line
                except Exception as exc:
                    raise RuntimeError(f"Serial read error: {exc}") from exc

    def _process_sentence(self, sentence: str) -> Optional[dict]:
        """Parse a single NMEA sentence; return observation if position available."""
        if not sentence.startswith("$"):
            return None
        try:
            msg = pynmea2.parse(sentence)
        except pynmea2.ParseError:
            return None

        sentence_type = msg.sentence_type

        if sentence_type == "GGA":
            try:
                self._lat = msg.latitude if msg.latitude else None
                self._lon = msg.longitude if msg.longitude else None
                self._alt_m = float(msg.altitude) if msg.altitude else None
                self._gps_quality = int(msg.gps_qual) if msg.gps_qual else None
                self._satellites = int(msg.num_sats) if msg.num_sats else None
                self._hdop = float(msg.horizontal_dil) if msg.horizontal_dil else None
            except (ValueError, AttributeError):
                pass

        elif sentence_type == "RMC":
            try:
                if msg.status == "A":  # Active fix
                    self._lat = msg.latitude if msg.latitude else self._lat
                    self._lon = msg.longitude if msg.longitude else self._lon
                    if msg.spd_over_grnd is not None:
                        self._speed_mps = float(msg.spd_over_grnd) * 0.514444
                    if msg.true_course is not None:
                        self._course_deg = float(msg.true_course)
            except (ValueError, AttributeError):
                pass

        elif sentence_type == "VTG":
            try:
                if msg.mag_course is not None:
                    self._course_deg = float(msg.mag_course)
                if msg.spd_over_grnd_kts is not None:
                    self._speed_mps = float(msg.spd_over_grnd_kts) * 0.514444
            except (ValueError, AttributeError):
                pass

        elif sentence_type == "GSA":
            try:
                fix_map = {"1": "No Fix", "2": "2D Fix", "3": "3D Fix"}
                self._fix_type = fix_map.get(str(msg.mode_fix_type), "Unknown")
                self._pdop = float(msg.pdop) if msg.pdop else None
            except (ValueError, AttributeError):
                pass

        else:
            return None  # Not a sentence type we care about

        if self._lat is None or self._lon is None:
            return None  # No position yet

        return self._build_observation()

    def _build_observation(self) -> dict:
        now = datetime.now(timezone.utc)
        return {
            "source_id": f"{self._entity_id}:{now.timestamp():.3f}",
            "adapter_id": self.config.adapter_id,
            "adapter_type": self.adapter_type,
            "entity_id": self._entity_id,
            "callsign": self._entity_id,
            "position": {
                "lat": self._lat,
                "lon": self._lon,
                "alt_m": self._alt_m,
            },
            "velocity": {
                "heading_deg": self._course_deg,
                "speed_mps": self._speed_mps,
                "vertical_mps": None,
            },
            "entity_type": "GPS_ASSET",
            "classification": None,
            "metadata": {
                "gps_quality": self._gps_quality,
                "satellites": self._satellites,
                "hdop": self._hdop,
                "pdop": self._pdop,
                "fix_type": self._fix_type,
                "connection_type": self._connection_type,
            },
            "ts_iso": now.isoformat(),
        }
