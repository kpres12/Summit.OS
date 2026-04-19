"""
Heli.OS OPC-UA Adapter

Subscribes to an OPC-UA server and publishes node values as Heli.OS
ASSET Entities into the data fabric.

OPC-UA is the modern industrial standard — GE, Honeywell, Siemens, ABB,
Rockwell, and Schneider Electric all ship OPC-UA servers.

Node Map Config (OPCUA_NODE_MAP env var, path to JSON file):
    [
      {
        "node_id": "ns=2;i=1001",
        "name": "TankLevel_01",
        "class_label": "level_sensor",
        "unit": "liters",
        "domain": "GROUND",
        "warn_above": 9000.0,
        "critical_above": 9800.0,
        "warn_below": 500.0,
        "critical_below": 100.0
      }
    ]

Environment variables:
    OPCUA_ENABLED               - "true" to enable (default: "false")
    OPCUA_URL                   - OPC-UA server URL (default: "opc.tcp://localhost:4840")
    OPCUA_DEVICE_ID             - logical device name (default: "opcua-device-01")
    OPCUA_USERNAME / OPCUA_PASSWORD - optional credentials
    OPCUA_SUBSCRIPTION_INTERVAL - ms between server publishes (default: 500)
    OPCUA_NODE_MAP              - path to JSON node map file
    OPCUA_ORG_ID                - org_id (default: "")
    MQTT_HOST / MQTT_PORT       - broker connection
"""
from __future__ import annotations

import json
import logging
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "packages"))

from sdk import BaseAdapter, AdapterManifest, EntityBuilder, Protocol, Capability

logger = logging.getLogger("summit.adapter.opcua")

DEFAULT_NODE_MAP: List[Dict[str, Any]] = [
    {"node_id": "ns=2;i=1001", "name": "Tank_Level", "class_label": "level_sensor",
     "unit": "liters", "domain": "GROUND", "warn_above": 9000.0, "critical_above": 9800.0,
     "warn_below": 500.0, "critical_below": 100.0},
    {"node_id": "ns=2;i=1002", "name": "Pump_Speed", "class_label": "pump",
     "unit": "RPM", "domain": "GROUND", "warn_above": 3200.0, "critical_above": 3500.0},
    {"node_id": "ns=2;i=1003", "name": "Ambient_Temperature", "class_label": "temperature_sensor",
     "unit": "degC", "domain": "GROUND", "warn_above": 45.0, "critical_above": 60.0,
     "warn_below": -10.0, "critical_below": -20.0},
    {"node_id": "ns=2;i=1004", "name": "Emergency_Stop", "class_label": "safety_switch",
     "unit": "bool", "domain": "GROUND"},
]


def _load_node_map(path: Optional[str]) -> List[Dict[str, Any]]:
    if not path:
        return DEFAULT_NODE_MAP
    try:
        with open(path) as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} node definitions from {path}")
        return data
    except Exception as e:
        logger.error(f"Failed to load node map from {path}: {e} — using demo map")
        return DEFAULT_NODE_MAP


class OPCUAAdapter(BaseAdapter):
    """Subscribes to OPC-UA server and publishes node values as ASSET entities."""

    MANIFEST = AdapterManifest(
        name="opcua",
        version="1.0.0",
        protocol=Protocol.OPCUA,
        capabilities=[Capability.READ, Capability.SUBSCRIBE],
        entity_types=["ASSET"],
        description="Modern industrial OPC-UA adapter — GE, Honeywell, Siemens, ABB",
        required_env=["OPCUA_URL"],
        optional_env=["OPCUA_USERNAME", "OPCUA_PASSWORD", "OPCUA_NODE_MAP",
                      "OPCUA_SUBSCRIPTION_INTERVAL"],
    )

    def __init__(
        self,
        url: str = os.getenv("OPCUA_URL", "opc.tcp://localhost:4840"),
        device_id: str = os.getenv("OPCUA_DEVICE_ID", "opcua-device-01"),
        username: Optional[str] = os.getenv("OPCUA_USERNAME"),
        password: Optional[str] = os.getenv("OPCUA_PASSWORD"),
        subscription_interval: int = int(os.getenv("OPCUA_SUBSCRIPTION_INTERVAL", "500")),
        node_map_path: Optional[str] = os.getenv("OPCUA_NODE_MAP"),
        org_id: str = os.getenv("OPCUA_ORG_ID", ""),
        **kwargs,
    ):
        super().__init__(device_id=device_id, org_id=org_id, **kwargs)
        self.url = url
        self.username = username
        self.password = password
        self.subscription_interval = subscription_interval
        self.nodes = _load_node_map(node_map_path)
        self._node_map: Dict[str, Dict[str, Any]] = {n["node_id"]: n for n in self.nodes}
        self._stats = {"updates": 0, "published": 0, "errors": 0}

    @property
    def enabled(self) -> bool:
        return os.getenv("OPCUA_ENABLED", "false").lower() == "true"

    async def run(self):
        logger.info(
            f"OPC-UA adapter running (url={self.url}, device={self.device_id}, "
            f"nodes={len(self.nodes)}, interval={self.subscription_interval}ms)"
        )
        try:
            from asyncua import Client
            await self._run_subscription(Client)
        except ImportError:
            logger.warning("asyncua not installed — running in simulated poll mode")
            await self._run_simulated()

    async def _run_subscription(self, Client: Any):
        while not self.stopped:
            try:
                async with Client(url=self.url) as client:
                    if self.username and self.password:
                        await client.set_user(self.username)
                        await client.set_password(self.password)

                    logger.info(f"OPC-UA connected to {self.url}")
                    handler = _OPCUAHandler(self)
                    subscription = await client.create_subscription(
                        self.subscription_interval, handler
                    )
                    nodes_to_sub = []
                    for node_def in self.nodes:
                        try:
                            nodes_to_sub.append(client.get_node(node_def["node_id"]))
                        except Exception as e:
                            logger.warning(f"Could not resolve node {node_def['node_id']}: {e}")

                    if nodes_to_sub:
                        await subscription.subscribe_data_change(nodes_to_sub)

                    await self._initial_read(client)
                    # Wait until stopped
                    while not self.stopped:
                        await self.sleep(1.0)
                    await subscription.delete()

            except Exception as e:
                self._stats["errors"] += 1
                logger.error(f"OPC-UA connection error: {e} — retrying in 10s")
                await self.sleep(10.0)

    async def _initial_read(self, client: Any):
        published = 0
        for node_def in self.nodes:
            try:
                node = client.get_node(node_def["node_id"])
                value = await node.read_value()
                entity = self._node_to_entity(node_def, value)
                self.publish(entity, qos=1)
                published += 1
            except Exception as e:
                logger.debug(f"Initial read failed for {node_def.get('node_id')}: {e}")
        logger.info(f"OPC-UA initial read: published {published}/{len(self.nodes)} nodes")

    def _publish_node_update(self, node_id_str: str, value: Any):
        node_def = self._node_map.get(node_id_str)
        if not node_def:
            return
        entity = self._node_to_entity(node_def, value)
        self.publish(entity, qos=1)
        self._stats["updates"] += 1
        self._stats["published"] += 1

    def _node_to_entity(self, node_def: Dict[str, Any], value: Any) -> Dict[str, Any]:
        name = node_def["name"]
        unit = node_def.get("unit", "")

        try:
            fval = float(value)
        except (TypeError, ValueError):
            fval = 0.0

        b = (
            EntityBuilder(
                f"opcua-{self.device_id}-{name.lower().replace(' ', '-').replace('_', '-')}",
                f"{self.device_id}/{name}",
            )
            .asset()
            .ground()
            .label(node_def.get("class_label", "sensor"))
            .value(fval, unit)
            .source("opcua", f"opcua-{self.device_id}")
            .org(self.org_id)
            .ttl(60)
            .meta_dict({
                "node_id": node_def["node_id"],
                "device_id": self.device_id,
                "protocol": "opcua",
            })
        )

        if node_def.get("warn_above") is not None:
            b = b.warn_above(float(node_def["warn_above"]))
        if node_def.get("critical_above") is not None:
            b = b.critical_above(float(node_def["critical_above"]))
        if node_def.get("warn_below") is not None:
            b = b.warn_below(float(node_def["warn_below"]))
        if node_def.get("critical_below") is not None:
            b = b.critical_below(float(node_def["critical_below"]))

        return b.build()

    async def _run_simulated(self):
        poll_interval = max(self.subscription_interval / 1000.0, 2.0)
        logger.info("OPC-UA running in simulation mode")
        while not self.stopped:
            t = time.time()
            sim = {
                "ns=2;i=1001": 5000 + 2000 * math.sin(t * 0.05),
                "ns=2;i=1002": 2800 + 200 * math.cos(t * 0.1),
                "ns=2;i=1003": 25.0 + 5 * math.sin(t * 0.02),
                "ns=2;i=1004": bool(int(t) % 20 > 1),
            }
            for node_def in self.nodes:
                value = sim.get(node_def["node_id"], 0.0)
                entity = self._node_to_entity(node_def, value)
                entity["metadata"]["simulated"] = "true"
                self.publish(entity, qos=1)
            self._stats["updates"] += 1
            self._stats["published"] += len(self.nodes)
            await self.sleep(poll_interval)


class _OPCUAHandler:
    def __init__(self, adapter: OPCUAAdapter):
        self._adapter = adapter

    def datachange_notification(self, node: Any, val: Any, data: Any):
        try:
            self._adapter._publish_node_update(node.nodeid.to_string(), val)
        except Exception as e:
            logger.debug(f"OPC-UA handler error: {e}")

    def event_notification(self, event: Any):
        pass

    def status_change_notification(self, status: Any):
        logger.info(f"OPC-UA subscription status: {status}")
