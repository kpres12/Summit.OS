"""
BaseAdapter — the base class every Heli.OS adapter should extend.

Handles lifecycle (start/stop), MQTT publishing via AdapterPublisher,
manifest validation, and missing env var warnings.
"""

from __future__ import annotations

import asyncio
import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from .manifest import AdapterManifest
from .publisher import AdapterPublisher

logger = logging.getLogger("heli.sdk.adapter")


class BaseAdapter(ABC):
    """
    Base class for all Heli.OS adapters.

    Subclasses must define:
        MANIFEST: AdapterManifest  — static class attribute
        run()                      — async generator of adapter work

    Example:
        class MyAdapter(BaseAdapter):
            MANIFEST = AdapterManifest(
                name="my-sensor",
                version="1.0.0",
                protocol=Protocol.CUSTOM,
                capabilities=[Capability.READ],
                entity_types=["ASSET"],
            )

            async def run(self):
                while not self.stopped:
                    entity = EntityBuilder(...).build()
                    self.publish(entity)
                    await self.sleep(5)
    """

    # Subclasses MUST define this
    MANIFEST: AdapterManifest

    def __init__(
        self,
        device_id: Optional[str] = None,
        org_id: str = "",
        mqtt_host: Optional[str] = None,
        mqtt_port: Optional[int] = None,
        mqtt_username: Optional[str] = None,
        mqtt_password: Optional[str] = None,
    ):
        if not hasattr(self, "MANIFEST") or self.MANIFEST is None:
            raise TypeError(
                f"{self.__class__.__name__} must define a MANIFEST class attribute"
            )

        # Validate manifest
        errors = self.MANIFEST.validate()
        if errors:
            raise ValueError(f"Invalid manifest for '{self.MANIFEST.name}': {errors}")

        self.device_id = device_id or f"{self.MANIFEST.name}-01"
        self.org_id = org_id or os.getenv("ADAPTER_ORG_ID", "")
        self._stop_event = asyncio.Event()
        self._publisher: Optional[AdapterPublisher] = None

        # MQTT config
        self._mqtt_host = mqtt_host or os.getenv("MQTT_HOST", "localhost")
        self._mqtt_port = mqtt_port or int(os.getenv("MQTT_PORT", "1883"))
        self._mqtt_username = mqtt_username or os.getenv("MQTT_USERNAME")
        self._mqtt_password = mqtt_password or os.getenv("MQTT_PASSWORD")

        # Warn about missing required env vars
        self._check_required_env()

    def _check_required_env(self):
        missing = [v for v in self.MANIFEST.required_env if not os.getenv(v)]
        if missing:
            logger.warning(
                f"Adapter '{self.MANIFEST.name}' is missing required env vars: {missing}"
            )

    @property
    def stopped(self) -> bool:
        return self._stop_event.is_set()

    def publish(self, entity: Dict[str, Any], qos: int = 1):
        """Publish an entity to the Heli.OS data fabric."""
        if self._publisher:
            self._publisher.publish_entity(entity, qos=qos)

    async def sleep(self, seconds: float):
        """Sleep that respects the stop signal."""
        try:
            await asyncio.wait_for(self._stop_event.wait(), timeout=seconds)
        except asyncio.TimeoutError:
            pass

    @abstractmethod
    async def run(self):
        """
        Adapter main loop. Called after MQTT connection is established.
        Must check `self.stopped` and use `await self.sleep(n)` instead of asyncio.sleep.
        """
        ...

    async def start(self):
        """Connect to MQTT and start the adapter run loop."""
        logger.info(
            f"Starting adapter '{self.MANIFEST.name}' v{self.MANIFEST.version} "
            f"(protocol={self.MANIFEST.protocol.value}, device={self.device_id})"
        )

        self._publisher = AdapterPublisher(
            manifest=self.MANIFEST,
            host=self._mqtt_host,
            port=self._mqtt_port,
            username=self._mqtt_username,
            password=self._mqtt_password,
        )

        try:
            self._publisher.connect()
        except Exception as e:
            logger.error(f"Adapter '{self.MANIFEST.name}' failed to connect: {e}")
            return

        try:
            await self.run()
        except Exception as e:
            logger.error(
                f"Adapter '{self.MANIFEST.name}' run() raised: {e}", exc_info=True
            )
        finally:
            self._publisher.disconnect()
            logger.info(
                f"Adapter '{self.MANIFEST.name}' stopped "
                f"(stats={self._publisher.stats})"
            )

    async def stop(self):
        """Signal the adapter to stop gracefully."""
        self._stop_event.set()

    @property
    def enabled(self) -> bool:
        """
        Override to add an ENABLED env var check.
        Default: check {ADAPTER_NAME_UPPER}_ENABLED env var.
        """
        env_key = f"{self.MANIFEST.name.upper().replace('-', '_')}_ENABLED"
        return os.getenv(env_key, "false").lower() == "true"
