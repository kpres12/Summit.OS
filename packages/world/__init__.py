"""
Heli.OS World Model

The shared world model is the single source of truth for every entity
in Heli.OS. Drones, sensors, tracks, alerts, missions, geofences —
everything is an Entity, and every Entity lives here.

Usage:
    from packages.world.store import WorldStore

    store = WorldStore()
    await store.initialize(engine)  # pass SQLAlchemy async engine

    # Write
    entity = store.upsert(Entity(entity_type=EntityType.ASSET, name="drone-01", ...))

    # Read
    entity = store.get("entity-id")
    assets = store.query(entity_type=EntityType.ASSET, domain=EntityDomain.AERIAL)

    # Watch
    queue = store.subscribe()
    async for event in queue:
        print(event)  # {"event": "update", "entity": {...}, "timestamp": ...}
"""

from packages.world.store import WorldStore

__all__ = ["WorldStore"]
