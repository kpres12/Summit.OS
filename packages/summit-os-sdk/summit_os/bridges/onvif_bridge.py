"""
Minimal ONVIF adapter template (camera metadata + heartbeat).
Usage:
  python -m summit_os.bridges.onvif_bridge --device-id cam-001 --host 192.168.1.10 --user admin --password pass
"""
from __future__ import annotations
import argparse
import asyncio
from typing import Optional

from .adapter_base import BaseAdapter, AdapterConfig

try:
    from onvif import ONVIFCamera  # type: ignore
except Exception:  # pragma: no cover
    ONVIFCamera = None  # type: ignore


class OnvifAdapter(BaseAdapter):
    def __init__(self, cfg: AdapterConfig, host: str, user: str, password: str):
        super().__init__(cfg)
        self._host = host
        self._user = user
        self._password = password
        self._cam = None

    async def run(self):
        if ONVIFCamera is None:
            raise RuntimeError("onvif-zeep not installed; install summit-os-sdk[adapters]")
        self._cam = ONVIFCamera(self._host, 80, self._user, self._password)
        media = self._cam.create_media_service()
        device = self._cam.create_devicemgmt_service()
        # Publish basic device info as heartbeat
        info = device.GetDeviceInformation()
        hb = {
            "device_id": self.cfg.device_id,
            "ts_iso": self.now_iso(),
            "status": "ALIVE",
            "vendor": getattr(info, "Manufacturer", None),
            "model": getattr(info, "Model", None),
        }
        await self.publish(f"health/{self.cfg.device_id}/heartbeat", hb)
        # Periodic keepalive
        while not self._stop.is_set():
            await asyncio.sleep(10)
            await self.publish(f"health/{self.cfg.device_id}/heartbeat", {"device_id": self.cfg.device_id, "ts_iso": self.now_iso(), "status": "ALIVE"})


def main(argv: Optional[list[str]] = None):
    p = argparse.ArgumentParser()
    p.add_argument("--device-id", required=True)
    p.add_argument("--host", required=True)
    p.add_argument("--user", required=True)
    p.add_argument("--password", required=True)
    args = p.parse_args(argv)
    cfg = AdapterConfig(device_id=args.device_id)
    adapter = OnvifAdapter(cfg, host=args.host, user=args.user, password=args.password)
    asyncio.run(adapter.start())


if __name__ == "__main__":
    main()
