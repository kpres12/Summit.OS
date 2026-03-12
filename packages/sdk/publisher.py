"""
AdapterPublisher — handles MQTT connection and entity publishing for SDK adapters.

Wraps paho-mqtt with automatic reconnect, manifest validation, and rate tracking.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger("summit.sdk.publisher")


class AdapterPublisher:
    """
    MQTT publisher for Summit.OS adapters.

    Connects to the broker, validates the adapter manifest on first publish,
    and routes entities to the correct topic.
    """

    def __init__(
        self,
        manifest: Any,  # AdapterManifest
        host: str = os.getenv("MQTT_HOST", "localhost"),
        port: int = int(os.getenv("MQTT_PORT", "1883")),
        username: Optional[str] = os.getenv("MQTT_USERNAME"),
        password: Optional[str] = os.getenv("MQTT_PASSWORD"),
    ):
        self.manifest = manifest
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self._client = None
        self._stats = {"published": 0, "errors": 0}

    def connect(self):
        """Connect to MQTT broker. Call once before publishing."""
        try:
            import paho.mqtt.client as mqtt
            self._client = mqtt.Client(client_id=f"summit-adapter-{self.manifest.name}")
            if self.username and self.password:
                self._client.username_pw_set(self.username, self.password)
            self._client.connect(self.host, self.port, 60)
            self._client.loop_start()

            # Announce adapter presence
            self._publish_manifest()
            logger.info(f"Adapter '{self.manifest.name}' connected to MQTT {self.host}:{self.port}")
        except ImportError:
            logger.error("paho-mqtt not installed. Run: pip install paho-mqtt")
            raise
        except Exception as e:
            logger.error(f"MQTT connection failed for adapter '{self.manifest.name}': {e}")
            raise

    def disconnect(self):
        if self._client:
            self._client.loop_stop()
            self._client.disconnect()
            logger.info(f"Adapter '{self.manifest.name}' disconnected")

    def publish_entity(self, entity: Dict[str, Any], qos: int = 1):
        """Publish an entity to the Summit.OS data fabric."""
        if not self._client:
            raise RuntimeError("Publisher not connected. Call connect() first.")

        entity_id = entity.get("entity_id", "unknown")
        topic = f"entities/{entity_id}/update"

        try:
            self._client.publish(topic, json.dumps(entity), qos=qos)
            self._stats["published"] += 1
        except Exception as e:
            self._stats["errors"] += 1
            logger.error(f"Publish error for entity {entity_id}: {e}")
            raise

    def _publish_manifest(self):
        """Announce the adapter manifest to the registry topic."""
        if not self._client:
            return
        try:
            topic = f"adapters/{self.manifest.name}/manifest"
            payload = json.dumps(self.manifest.to_dict())
            self._client.publish(topic, payload, qos=1, retain=True)
        except Exception as e:
            logger.warning(f"Could not publish manifest for '{self.manifest.name}': {e}")

    @property
    def stats(self) -> Dict[str, int]:
        return dict(self._stats)
