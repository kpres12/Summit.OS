"""
Heli.OS Adapter SDK

pip install summit-sdk

The canonical way to write a Heli.OS adapter. Handles MQTT connection,
entity publishing, manifest registration, and graceful shutdown.

Quick start:
    from summit_sdk import BaseAdapter, AdapterManifest, EntityBuilder, Protocol, Capability

    class MyAdapter(BaseAdapter):
        MANIFEST = AdapterManifest(
            name="my-sensor",
            version="1.0.0",
            protocol=Protocol.CUSTOM,
            capabilities=[Capability.READ],
            entity_types=["ASSET"],
            description="Reads from my custom sensor",
        )

        async def run(self):
            while not self.stopped:
                value = await read_my_sensor()
                entity = (
                    EntityBuilder(self.device_id, "MySensor")
                    .asset()
                    .ground()
                    .value(value, "PSI")
                    .build()
                )
                self.publish(entity)
                await self.sleep(5)
"""

from .manifest import AdapterManifest, Protocol, Capability
from .base import BaseAdapter
from .entity import EntityBuilder
from .publisher import AdapterPublisher

__version__ = "1.0.0"
__all__ = [
    "AdapterManifest",
    "Protocol",
    "Capability",
    "BaseAdapter",
    "EntityBuilder",
    "AdapterPublisher",
]
