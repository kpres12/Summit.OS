"""
Minimal MAVLink adapter template.
Usage:
  python -m summit_os.bridges.mavlink_bridge --device-id drone-001 --udp 127.0.0.1:14550
"""
from __future__ import annotations
import argparse
import asyncio
from typing import Optional

from .adapter_base import BaseAdapter, AdapterConfig

try:
    from pymavlink import mavutil  # type: ignore
except Exception as e:  # pragma: no cover
    mavutil = None  # type: ignore


class MavlinkAdapter(BaseAdapter):
    def __init__(self, cfg: AdapterConfig, udp: str):
        super().__init__(cfg)
        self._udp = udp
        self._conn = None

    async def run(self):
        if mavutil is None:
            raise RuntimeError("pymavlink not installed; install summit-os-sdk[adapters]")
        # open MAVLink
        self._conn = mavutil.mavlink_connection(f"udp:{self._udp}")
        self._conn.wait_heartbeat(timeout=10)
        while not self._stop.is_set():
            msg = self._conn.recv_match(blocking=True, timeout=1)
            if msg is None:
                continue
            mtype = msg.get_type()
            if mtype == "GLOBAL_POSITION_INT":
                lat = getattr(msg, "lat", 0) / 1e7
                lon = getattr(msg, "lon", 0) / 1e7
                alt = getattr(msg, "alt", 0) / 1000.0
                payload = {
                    "device_id": self.cfg.device_id,
                    "ts_iso": self.now_iso(),
                    "location": {"lat": lat, "lon": lon, "alt": alt},
                    "status": "ACTIVE",
                    "sensors": {"hdg": getattr(msg, "hdg", None)},
                }
                await self.publish(f"telemetry/{self.cfg.device_id}", payload)
            elif mtype == "HEARTBEAT":
                hb = {
                    "device_id": self.cfg.device_id,
                    "ts_iso": self.now_iso(),
                    "status": "ALIVE",
                    "mode": getattr(msg, "custom_mode", None),
                }
                await self.publish(f"health/{self.cfg.device_id}/heartbeat", hb)


def main(argv: Optional[list[str]] = None):
    p = argparse.ArgumentParser()
    p.add_argument("--device-id", required=True)
    p.add_argument("--udp", required=True, help="host:port for MAVLink UDP")
    args = p.parse_args(argv)
    cfg = AdapterConfig(device_id=args.device_id)
    adapter = MavlinkAdapter(cfg, udp=args.udp)
    asyncio.run(adapter.start())


if __name__ == "__main__":
    main()
