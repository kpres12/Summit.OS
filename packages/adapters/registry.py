"""
Summit.OS Adapter Registry
===========================

Central registry for all configured adapters.  Responsibilities:

  - Maintain a catalogue of known adapter *types* (class objects).
  - Instantiate adapter *instances* from ``AdapterConfig`` objects.
  - Start / stop all enabled adapters as a group.
  - Surface health status for every adapter (used by the API gateway's
    ``GET /adapters`` endpoint and the DEV console view).

Usage
-----
::

    from adapters.registry import AdapterRegistry
    from adapters.base import AdapterConfig

    registry = AdapterRegistry()
    registry.register_type(OpenSkyAdapter)   # optional — built-ins auto-registered

    cfg = AdapterConfig(
        adapter_id="opensky-conus",
        adapter_type="opensky",
        display_name="OpenSky CONUS",
        extra={"bbox": "24,-125,49,-66"},
    )
    registry.add(cfg, mqtt_client=my_mqtt)

    await registry.start_all()
    # … on shutdown …
    await registry.stop_all()
"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

from .base import AdapterConfig, AdapterHealth, BaseAdapter

logger = logging.getLogger("summit.adapters.registry")


# ---------------------------------------------------------------------------
# Built-in adapter type catalogue
# ---------------------------------------------------------------------------

BUILT_IN_ADAPTERS: list[dict] = [
    {
        "type": "opensky",
        "name": "OpenSky ADS-B",
        "description": "Live aircraft positions via OpenSky Network",
    },
    {
        "type": "celestrak",
        "name": "CelesTrak Satellites",
        "description": "Satellite orbital positions via CelesTrak",
    },
    {
        "type": "mavlink",
        "name": "MAVLink",
        "description": "ArduPilot/PX4 drone telemetry via MAVLink",
    },
    {
        "type": "rtsp",
        "name": "RTSP Camera",
        "description": "RTSP video stream (any IP camera)",
    },
    {
        "type": "onvif",
        "name": "ONVIF Camera",
        "description": "ONVIF-compliant IP camera",
    },
    {
        "type": "nmea",
        "name": "NMEA GPS",
        "description": "NMEA 0183 GPS feed (serial or TCP)",
    },
    {
        "type": "cap",
        "name": "CAP Alerts",
        "description": "Common Alerting Protocol (FEMA, weather)",
    },
    {
        "type": "webhook",
        "name": "Webhook",
        "description": "Generic inbound HTTP webhook",
    },
    {
        "type": "mqtt_relay",
        "name": "MQTT Relay",
        "description": "Relay observations from another MQTT broker",
    },
]

# Index by type for O(1) lookup
_BUILT_IN_BY_TYPE: dict[str, dict] = {a["type"]: a for a in BUILT_IN_ADAPTERS}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class AdapterRegistry:
    """
    Central registry for all configured adapters.

    Thread-safe for read operations (health, list).  Mutations (add, start,
    stop) are expected to occur at startup/shutdown only and do not require
    locking.
    """

    def __init__(self) -> None:
        # adapter_id → BaseAdapter instance
        self._adapters: dict[str, BaseAdapter] = {}

        # adapter_type string → BaseAdapter subclass
        self._adapter_classes: dict[str, type[BaseAdapter]] = {}

        # adapter_id → running asyncio.Task
        self._tasks: dict[str, asyncio.Task] = {}

    # -------------------------------------------------------------------------
    # Type registration
    # -------------------------------------------------------------------------

    def register_type(self, adapter_class: type[BaseAdapter]) -> None:
        """
        Register an adapter class by its ``adapter_type`` class attribute.

        This must be called before ``add()`` can instantiate adapters of that
        type.  Built-in adapter classes (opensky, celestrak) are registered
        automatically when this module loads if their packages are importable.
        """
        key = adapter_class.adapter_type
        if key == "base":
            raise ValueError(
                "Cannot register BaseAdapter itself — subclass must override adapter_type."
            )
        self._adapter_classes[key] = adapter_class
        logger.debug("Registered adapter type: %s → %s", key, adapter_class.__name__)

    # -------------------------------------------------------------------------
    # Instance management
    # -------------------------------------------------------------------------

    def add(
        self,
        config: AdapterConfig,
        mqtt_client=None,
    ) -> BaseAdapter:
        """
        Instantiate an adapter from ``config`` and add it to the registry.

        The adapter type must have been registered via ``register_type()``
        first.  Raises ``KeyError`` if the type is unknown and ``ValueError``
        if an adapter with the same ``adapter_id`` already exists.
        """
        if config.adapter_id in self._adapters:
            raise ValueError(
                f"Adapter with id '{config.adapter_id}' is already registered."
            )

        cls = self._adapter_classes.get(config.adapter_type)
        if cls is None:
            raise KeyError(
                f"Unknown adapter type '{config.adapter_type}'. "
                f"Register it first with registry.register_type(MyAdapter)."
            )

        adapter = cls(config=config, mqtt_client=mqtt_client)
        self._adapters[config.adapter_id] = adapter
        logger.info(
            "Added adapter %s (type=%s, enabled=%s)",
            config.adapter_id,
            config.adapter_type,
            config.enabled,
        )
        return adapter

    def get(self, adapter_id: str) -> Optional[BaseAdapter]:
        """Return the adapter with the given id, or ``None``."""
        return self._adapters.get(adapter_id)

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    async def start_all(self) -> None:
        """
        Start all enabled adapters concurrently as background asyncio tasks.

        Disabled adapters are skipped.  Already-running adapters are
        skipped silently.
        """
        for adapter_id, adapter in self._adapters.items():
            if not adapter.config.enabled:
                logger.info("Skipping disabled adapter: %s", adapter_id)
                continue
            if adapter_id in self._tasks and not self._tasks[adapter_id].done():
                logger.debug("Adapter already running: %s", adapter_id)
                continue

            task = asyncio.create_task(
                adapter.start(),
                name=f"adapter:{adapter_id}",
            )
            adapter._task = task
            self._tasks[adapter_id] = task
            logger.info("Started adapter: %s", adapter_id)

        if self._tasks:
            logger.info(
                "Adapter registry: %d adapter(s) running.",
                sum(1 for t in self._tasks.values() if not t.done()),
            )

    async def stop_all(self) -> None:
        """
        Gracefully stop all running adapters and await their completion.

        Calls each adapter's ``stop()`` method (which sets its stop event
        and cancels its task), then gathers all tasks.
        """
        logger.info("Stopping all adapters...")
        stop_coros = [adapter.stop() for adapter in self._adapters.values()]
        if stop_coros:
            await asyncio.gather(*stop_coros, return_exceptions=True)

        pending = [t for t in self._tasks.values() if not t.done()]
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

        self._tasks.clear()
        logger.info("All adapters stopped.")

    # -------------------------------------------------------------------------
    # Observability
    # -------------------------------------------------------------------------

    def get_health(self) -> dict[str, AdapterHealth]:
        """
        Return a health snapshot for every registered adapter, keyed by
        ``adapter_id``.
        """
        return {
            adapter_id: adapter.health()
            for adapter_id, adapter in self._adapters.items()
        }

    def list_adapters(self) -> list[dict]:
        """
        Return a list of adapter summary dicts suitable for the API response.

        Each entry includes config fields plus the current health snapshot.
        """
        result = []
        for adapter_id, adapter in self._adapters.items():
            h = adapter.health()
            result.append({
                "adapter_id":     adapter_id,
                "adapter_type":   adapter.config.adapter_type,
                "display_name":   adapter.config.display_name,
                "description":    adapter.config.description,
                "enabled":        adapter.config.enabled,
                "poll_interval":  adapter.config.poll_interval_seconds,
                "status":         h.status,
                "last_connected": h.last_connected.isoformat() if h.last_connected else None,
                "last_observation": (
                    h.last_observation.isoformat() if h.last_observation else None
                ),
                "observations_total":      h.observations_total,
                "observations_per_minute": round(h.observations_per_minute, 2),
                "error_message":           h.error_message,
                "uptime_seconds":          round(h.uptime_seconds, 1),
            })
        return result


# ---------------------------------------------------------------------------
# Auto-register built-in adapter implementations
# ---------------------------------------------------------------------------
# We attempt to import the concrete adapter classes that ship with the repo.
# Failures are non-fatal: the registry still works, just without that type.

def _try_register_builtins(registry: AdapterRegistry) -> None:
    """Attempt to register built-in adapter classes into a registry."""
    try:
        import sys, os
        # adapters/ directory is at the repo root — add it if needed
        repo_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )
        adapters_dir = os.path.join(repo_root, "adapters")
        if adapters_dir not in sys.path:
            sys.path.insert(0, adapters_dir)
    except Exception:
        pass

    _builtins = [
        ("opensky.adapter",          "OpenSkyAdapter"),
        ("celestrak.adapter",        "CelesTrakAdapter"),
        ("adapters.mavlink_adapter", "MAVLinkAdapter"),
    ]

    for module_path, class_name in _builtins:
        try:
            import importlib
            mod = importlib.import_module(module_path)
            cls = getattr(mod, class_name)
            # Ensure the class has an adapter_type attribute set
            if not hasattr(cls, "adapter_type") or cls.adapter_type == "base":
                logger.debug(
                    "Skipping %s.%s — adapter_type not set.", module_path, class_name
                )
                continue
            registry.register_type(cls)
            logger.debug("Auto-registered built-in adapter: %s", cls.adapter_type)
        except Exception as exc:
            logger.debug("Could not auto-register %s.%s: %s", module_path, class_name, exc)
