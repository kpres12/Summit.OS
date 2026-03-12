"""
Summit.OS OPC-UA Adapter

Subscribes to an OPC-UA server and publishes node values as Summit.OS
Entities into the data fabric via MQTT.

OPC-UA is the modern industrial standard. GE, Honeywell, Siemens, ABB,
Rockwell, and Schneider Electric all ship OPC-UA servers. If Modbus is
the "legacy" bridge, OPC-UA is the "modern" bridge.

Unlike Modbus polling, OPC-UA supports server-push subscriptions:
the adapter subscribes to node changes and receives updates only when
values change, reducing network load and latency.

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
    OPCUA_ENABLED           - "true" to enable (default: "false")
    OPCUA_URL               - OPC-UA server URL (default: "opc.tcp://localhost:4840")
    OPCUA_DEVICE_ID         - logical device name used in entity IDs (default: "opcua-device-01")
    OPCUA_USERNAME          - optional credentials
    OPCUA_PASSWORD          - optional credentials
    OPCUA_SUBSCRIPTION_INTERVAL - ms between server publishes (default: 500)
    OPCUA_NODE_MAP          - path to JSON node map file
    OPCUA_ORG_ID            - org_id for multi-tenant filtering (default: "")
    MQTT_HOST / MQTT_PORT   - broker connection
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger("summit.adapter.opcua")

DEFAULT_NODE_MAP: List[Dict[str, Any]] = [
    {
        "node_id": "ns=2;i=1001",
        "name": "Tank_Level",
        "class_label": "level_sensor",
        "unit": "liters",
        "domain": "GROUND",
        "warn_above": 9000.0,
        "critical_above": 9800.0,
        "warn_below": 500.0,
        "critical_below": 100.0,
    },
    {
        "node_id": "ns=2;i=1002",
        "name": "Pump_Speed",
        "class_label": "pump",
        "unit": "RPM",
        "domain": "GROUND",
        "warn_above": 3200.0,
        "critical_above": 3500.0,
        "warn_below": None,
        "critical_below": None,
    },
    {
        "node_id": "ns=2;i=1003",
        "name": "Ambient_Temperature",
        "class_label": "temperature_sensor",
        "unit": "degC",
        "domain": "GROUND",
        "warn_above": 45.0,
        "critical_above": 60.0,
        "warn_below": -10.0,
        "critical_below": -20.0,
    },
    {
        "node_id": "ns=2;i=1004",
        "name": "Emergency_Stop",
        "class_label": "safety_switch",
        "unit": "bool",
        "domain": "GROUND",
        "warn_above": None,
        "critical_above": None,
        "warn_below": None,
        "critical_below": None,
    },
]


def _load_node_map(path: Optional[str]) -> List[Dict[str, Any]]:
    if not path:
        logger.info("No node map file specified — using built-in demo node map")
        return DEFAULT_NODE_MAP
    try:
        with open(path, "r") as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} node definitions from {path}")
        return data
    except Exception as e:
        logger.error(f"Failed to load node map from {path}: {e} — using demo map")
        return DEFAULT_NODE_MAP


def _compute_state(value: float, node: Dict[str, Any]) -> str:
    critical_above = node.get("critical_above")
    critical_below = node.get("critical_below")
    warn_above = node.get("warn_above")
    warn_below = node.get("warn_below")

    try:
        fval = float(value)
    except (TypeError, ValueError):
        return "ACTIVE"

    if critical_above is not None and fval >= critical_above:
        return "CRITICAL"
    if critical_below is not None and fval <= critical_below:
        return "CRITICAL"
    if warn_above is not None and fval >= warn_above:
        return "WARNING"
    if warn_below is not None and fval <= warn_below:
        return "WARNING"
    return "ACTIVE"


def _node_to_entity(
    node: Dict[str, Any],
    value: Any,
    device_id: str,
    org_id: str,
    now_iso: str,
) -> Dict[str, Any]:
    name = node["name"]
    node_id = node["node_id"]
    unit = node.get("unit", "")
    domain = node.get("domain", "GROUND")

    state = _compute_state(value, node)
    entity_id = f"opcua-{device_id}-{name.lower().replace(' ', '-').replace('_', '-')}"

    return {
        "entity_id": entity_id,
        "id": entity_id,
        "entity_type": "ASSET",
        "domain": domain,
        "state": state,
        "name": f"{device_id}/{name}",
        "class_label": node.get("class_label", "sensor"),
        "confidence": 1.0,
        "kinematics": {
            "position": {
                "latitude": 0.0,
                "longitude": 0.0,
                "altitude_msl": 0.0,
                "altitude_agl": 0.0,
            },
            "heading_deg": 0.0,
            "speed_mps": 0.0,
            "climb_rate": 0.0,
        },
        "provenance": {
            "source_id": f"opcua-{device_id}",
            "source_type": "opcua",
            "org_id": org_id,
            "created_at": time.time(),
            "updated_at": time.time(),
            "version": 1,
        },
        "metadata": {
            "value": str(value),
            "unit": unit,
            "node_id": node_id,
            "device_id": device_id,
            "protocol": "opcua",
            "state_reason": state,
        },
        "ttl_seconds": 60,
        "ts": now_iso,
    }


class OPCUAAdapter:
    """
    Subscribes to an OPC-UA server and publishes node values as
    Summit.OS ASSET entities to MQTT.

    Uses asyncua subscriptions for push-based updates. Falls back to
    polling if asyncua is not installed.
    """

    def __init__(
        self,
        mqtt_client: Any,
        url: str = os.getenv("OPCUA_URL", "opc.tcp://localhost:4840"),
        device_id: str = os.getenv("OPCUA_DEVICE_ID", "opcua-device-01"),
        username: Optional[str] = os.getenv("OPCUA_USERNAME"),
        password: Optional[str] = os.getenv("OPCUA_PASSWORD"),
        subscription_interval: int = int(os.getenv("OPCUA_SUBSCRIPTION_INTERVAL", "500")),
        node_map_path: Optional[str] = os.getenv("OPCUA_NODE_MAP"),
        org_id: str = os.getenv("OPCUA_ORG_ID", ""),
    ):
        self.mqtt = mqtt_client
        self.url = url
        self.device_id = device_id
        self.username = username
        self.password = password
        self.subscription_interval = subscription_interval
        self.org_id = org_id
        self.nodes = _load_node_map(node_map_path)
        self._stop = asyncio.Event()
        self._stats = {"updates": 0, "published": 0, "errors": 0}
        # Map node_id -> node config for fast lookup in subscription handler
        self._node_map: Dict[str, Dict[str, Any]] = {
            n["node_id"]: n for n in self.nodes
        }

    @property
    def enabled(self) -> bool:
        return os.getenv("OPCUA_ENABLED", "false").lower() == "true"

    async def start(self):
        if not self.enabled:
            logger.info("OPC-UA adapter disabled")
            return

        logger.info(
            f"OPC-UA adapter starting (url={self.url}, device={self.device_id}, "
            f"nodes={len(self.nodes)}, interval={self.subscription_interval}ms)"
        )

        try:
            from asyncua import Client
            await self._run_subscription(Client)
        except ImportError:
            logger.warning("asyncua not installed — running OPC-UA adapter in simulated poll mode")
            await self._run_simulated()

    async def stop(self):
        self._stop.set()

    async def _run_subscription(self, Client: Any):
        """Connect to OPC-UA server and subscribe to node changes."""
        while not self._stop.is_set():
            try:
                async with Client(url=self.url) as client:
                    if self.username and self.password:
                        await client.set_user(self.username)
                        await client.set_password(self.password)

                    logger.info(f"OPC-UA connected to {self.url}")

                    # Create subscription
                    handler = _OPCUASubscriptionHandler(self)
                    subscription = await client.create_subscription(
                        self.subscription_interval, handler
                    )

                    # Subscribe to all configured nodes
                    nodes_to_subscribe = []
                    for node_def in self.nodes:
                        try:
                            node = client.get_node(node_def["node_id"])
                            nodes_to_subscribe.append(node)
                        except Exception as e:
                            logger.warning(f"Could not resolve node {node_def['node_id']}: {e}")

                    if nodes_to_subscribe:
                        await subscription.subscribe_data_change(nodes_to_subscribe)
                        logger.info(f"OPC-UA subscribed to {len(nodes_to_subscribe)} nodes")

                    # Also do an initial read so entities appear immediately
                    await self._initial_read(client)

                    # Wait until stopped or connection lost
                    await self._stop.wait()
                    await subscription.delete()

            except Exception as e:
                self._stats["errors"] += 1
                logger.error(f"OPC-UA connection error: {e} — retrying in 10s")
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=10.0)
                    break
                except asyncio.TimeoutError:
                    pass

        logger.info(f"OPC-UA adapter stopped (stats={self._stats})")

    async def _initial_read(self, client: Any):
        """Read all nodes once on connect to populate world model immediately."""
        now_iso = datetime.now(timezone.utc).isoformat()
        published = 0
        for node_def in self.nodes:
            try:
                node = client.get_node(node_def["node_id"])
                value = await node.read_value()
                entity = _node_to_entity(node_def, value, self.device_id, self.org_id, now_iso)
                topic = f"entities/{entity['entity_id']}/update"
                self.mqtt.publish(topic, json.dumps(entity), qos=1)
                published += 1
            except Exception as e:
                logger.debug(f"Initial read failed for {node_def.get('node_id')}: {e}")
        logger.info(f"OPC-UA initial read: published {published}/{len(self.nodes)} nodes")

    def _publish_node_update(self, node_id_str: str, value: Any):
        """Called by subscription handler when a node value changes."""
        node_def = self._node_map.get(node_id_str)
        if not node_def:
            logger.debug(f"Received update for unknown node: {node_id_str}")
            return

        now_iso = datetime.now(timezone.utc).isoformat()
        entity = _node_to_entity(node_def, value, self.device_id, self.org_id, now_iso)
        topic = f"entities/{entity['entity_id']}/update"
        self.mqtt.publish(topic, json.dumps(entity), qos=1)
        self._stats["updates"] += 1
        self._stats["published"] += 1

    async def _run_simulated(self):
        """Simulate OPC-UA data when asyncua is not available."""
        import math

        logger.info("OPC-UA running in simulation mode")
        poll_interval = self.subscription_interval / 1000.0

        while not self._stop.is_set():
            t = time.time()
            now_iso = datetime.now(timezone.utc).isoformat()

            sim_values = {
                "ns=2;i=1001": 5000 + 2000 * math.sin(t * 0.05),   # tank level
                "ns=2;i=1002": 2800 + 200 * math.cos(t * 0.1),      # pump RPM
                "ns=2;i=1003": 25.0 + 5 * math.sin(t * 0.02),       # temperature
                "ns=2;i=1004": bool(int(t) % 20 > 1),                # e-stop (mostly false)
            }

            published = 0
            for node_def in self.nodes:
                node_id = node_def["node_id"]
                value = sim_values.get(node_id, 0.0)
                entity = _node_to_entity(node_def, value, self.device_id, self.org_id, now_iso)
                entity["metadata"]["simulated"] = "true"
                topic = f"entities/{entity['entity_id']}/update"
                self.mqtt.publish(topic, json.dumps(entity), qos=1)
                published += 1

            self._stats["updates"] += 1
            self._stats["published"] += published
            logger.debug(f"OPC-UA (simulated): published {published} nodes")

            try:
                await asyncio.wait_for(self._stop.wait(), timeout=max(poll_interval, 2.0))
                break
            except asyncio.TimeoutError:
                pass

        logger.info(f"OPC-UA adapter stopped (stats={self._stats})")


class _OPCUASubscriptionHandler:
    """
    asyncua subscription handler.
    Receives datachange notifications from the OPC-UA server
    and forwards them to the adapter for publishing.
    """

    def __init__(self, adapter: OPCUAAdapter):
        self._adapter = adapter

    def datachange_notification(self, node: Any, val: Any, data: Any):
        """Called by asyncua when a subscribed node value changes."""
        try:
            node_id_str = node.nodeid.to_string()
            self._adapter._publish_node_update(node_id_str, val)
        except Exception as e:
            logger.debug(f"Subscription handler error: {e}")

    def event_notification(self, event: Any):
        pass

    def status_change_notification(self, status: Any):
        logger.info(f"OPC-UA subscription status: {status}")
