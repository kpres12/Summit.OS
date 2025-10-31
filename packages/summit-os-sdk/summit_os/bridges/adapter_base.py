"""
Base adapter interface and MQTT publisher helpers.
"""
from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional

try:
    import paho.mqtt.client as mqtt  # type: ignore
except Exception:  # pragma: no cover
    mqtt = None  # type: ignore


@dataclass
class AdapterConfig:
    device_id: str
    org_id: Optional[str] = None
    mqtt_host: str = os.getenv("MQTT_HOST", "localhost")
    mqtt_port: int = int(os.getenv("MQTT_PORT", "1883"))
    mqtt_username: Optional[str] = os.getenv("MQTT_USERNAME")
    mqtt_password: Optional[str] = os.getenv("MQTT_PASSWORD")


class BaseAdapter:
    def __init__(self, cfg: AdapterConfig):
        self.cfg = cfg
        self._stop = asyncio.Event()
        self._mqtt: Optional[mqtt.Client] = None
        self._token: Optional[str] = None

    async def start(self):
        await self._connect_mqtt()
        await self.run()

    async def stop(self):
        self._stop.set()
        if self._mqtt:
            try:
                self._mqtt.loop_stop()
                self._mqtt.disconnect()
            except Exception:
                pass

    async def run(self):  # override
        raise NotImplementedError

    async def _connect_mqtt(self):
        if mqtt is None:
            raise RuntimeError("paho-mqtt not installed; install summit-os-sdk[mqtt]")
        client = mqtt.Client()
        if self.cfg.mqtt_username and self.cfg.mqtt_password:
            client.username_pw_set(self.cfg.mqtt_username, self.cfg.mqtt_password)
        client.connect(self.cfg.mqtt_host, self.cfg.mqtt_port, 60)
        client.loop_start()
        self._mqtt = client

    # Helpers
    async def publish(self, topic: str, payload: Dict[str, Any]):
        if not self._mqtt:
            raise RuntimeError("MQTT not connected")
        self._mqtt.publish(topic, json.dumps(payload), qos=0)

    def register_with_gateway(self, api_base: str, node_type: str = "GENERIC", capabilities: list[str] | None = None, comm: list[str] | None = None) -> dict:
        """Register device with API Gateway → Fabric registry.
        Returns response dict and caches token (if provided).
        """
        import requests
        payload = {
            "id": self.cfg.device_id,
            "type": node_type,
            "pubkey": None,
            "fw_version": None,
            "location": None,
            "capabilities": capabilities or [],
            "comm": comm or [],
        }
        headers = {"X-Org-ID": self.cfg.org_id} if self.cfg.org_id else None
        r = requests.post(f"{api_base.rstrip('/')}/api/v1/nodes/register", json=payload, headers=headers, timeout=5)
        r.raise_for_status()
        data = r.json()
        self._token = data.get("token") if isinstance(data, dict) else None
        return data

    @staticmethod
    def now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()
