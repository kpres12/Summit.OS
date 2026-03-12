"""
Summit.OS Adapter Runner

Single entry point that starts all enabled Summit.OS adapters.
Each adapter is a BaseAdapter subclass that self-manages its MQTT connection,
declares its capabilities via a manifest, and runs as a concurrent asyncio task.

Usage:
    python runner.py

Enable adapters via environment:
    OPENSKY_ENABLED=true      Live ADS-B aircraft positions
    CELESTRAK_ENABLED=true    Satellite tracking via SGP4
    MODBUS_ENABLED=true       Industrial PLCs, sensors, actuators (Modbus/TCP)
    OPCUA_ENABLED=true        Modern industrial systems (Siemens, GE, Honeywell)
    MAVLINK_ENABLED=true      Drones (ArduPilot, PX4, DJI via MAVLink)
    CAMERA_ENABLED=true       RTSP/ONVIF cameras with YOLO detection + ByteTrack
    AIS_ENABLED=true          Maritime vessel tracking (AIS/NMEA)
"""
from __future__ import annotations

import asyncio
import logging
import signal
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("summit.adapters")

_ADAPTER_CLASSES = [
    ("opensky",   "opensky.adapter",   "OpenSkyAdapter"),
    ("celestrak", "celestrak.adapter", "CelesTrakAdapter"),
    ("modbus",    "modbus.adapter",    "ModbusAdapter"),
    ("opcua",     "opcua.adapter",     "OPCUAAdapter"),
    ("mavlink",   "mavlink.adapter",   "MAVLinkAdapter"),
    ("camera",    "camera.adapter",    "CameraAdapter"),
    ("ais",       "ais.adapter",       "AISAdapter"),
]


def _load_adapter(module_name: str, class_name: str):
    import importlib
    try:
        mod = importlib.import_module(module_name)
        return getattr(mod, class_name)
    except (ImportError, AttributeError) as e:
        logger.warning(f"Could not load {class_name} from {module_name}: {e}")
        return None


async def run():
    """Instantiate, validate, and start all enabled adapters."""
    adapters = []

    for name, module, cls_name in _ADAPTER_CLASSES:
        cls = _load_adapter(module, cls_name)
        if cls is None:
            continue
        try:
            adapter = cls()
        except Exception as e:
            logger.warning(f"Could not instantiate {cls_name}: {e}")
            continue

        if adapter.enabled:
            # Validate manifest before starting
            errors = adapter.MANIFEST.validate()
            if errors:
                logger.error(f"Adapter '{name}' manifest invalid: {errors} — skipping")
                continue
            adapters.append((name, adapter))
            logger.info(f"  {name:<12} ENABLED  ({adapter.MANIFEST.protocol.value})")
        else:
            logger.info(f"  {name:<12} disabled")

    if not adapters:
        logger.warning(
            "No adapters enabled. Set MODBUS_ENABLED=true, MAVLINK_ENABLED=true, etc."
        )
        return

    logger.info(f"Starting {len(adapters)} adapter(s)")

    # Start all adapters as concurrent tasks
    tasks = [
        (name, adapter, asyncio.create_task(adapter.start()))
        for name, adapter in adapters
    ]

    # Wait for shutdown signal
    stop = asyncio.Event()
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except NotImplementedError:
            pass  # Windows

    await stop.wait()

    logger.info("Shutting down adapters...")
    for name, adapter, task in tasks:
        try:
            await adapter.stop()
        except Exception as e:
            logger.debug(f"Stop error for {name}: {e}")
    await asyncio.gather(*[t for _, _, t in tasks], return_exceptions=True)
    logger.info("All adapters stopped")


def main():
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
