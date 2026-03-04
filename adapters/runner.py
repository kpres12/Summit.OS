"""
Summit.OS Adapter Runner

Single entry point that starts all enabled external data adapters.
Each adapter runs as a concurrent asyncio task within one process,
connecting to the MQTT broker to publish entities into the data fabric.

Usage:
    python runner.py

Environment:
    MQTT_HOST / MQTT_PORT - broker connection
    OPENSKY_ENABLED       - enable OpenSky aircraft feed
    CELESTRAK_ENABLED     - enable CelesTrak satellite feed
"""
from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
from typing import Any

import paho.mqtt.client as mqtt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("summit.adapters")


def _create_mqtt_client() -> mqtt.Client:
    """Create and connect a paho MQTT client."""
    host = os.getenv("MQTT_HOST", "localhost")
    port = int(os.getenv("MQTT_PORT", "1883"))
    username = os.getenv("MQTT_USERNAME")
    password = os.getenv("MQTT_PASSWORD")

    client = mqtt.Client(client_id="summit-adapters")
    if username and password:
        client.username_pw_set(username, password)

    logger.info(f"Connecting to MQTT broker at {host}:{port}")
    client.connect(host, port, 60)
    client.loop_start()
    return client


async def run():
    """Start all enabled adapters and wait for shutdown."""
    mqtt_client = _create_mqtt_client()
    tasks = []

    # OpenSky Network
    try:
        from opensky.adapter import OpenSkyAdapter
        adapter = OpenSkyAdapter(mqtt_client=mqtt_client)
        if adapter.enabled:
            tasks.append(("opensky", adapter, asyncio.create_task(adapter.start())))
            logger.info("OpenSky adapter: ENABLED")
        else:
            logger.info("OpenSky adapter: DISABLED")
    except ImportError as e:
        logger.warning(f"OpenSky adapter unavailable: {e}")

    # CelesTrak
    try:
        from celestrak.adapter import CelesTrakAdapter
        adapter = CelesTrakAdapter(mqtt_client=mqtt_client)
        if adapter.enabled:
            tasks.append(("celestrak", adapter, asyncio.create_task(adapter.start())))
            logger.info("CelesTrak adapter: ENABLED")
        else:
            logger.info("CelesTrak adapter: DISABLED")
    except ImportError as e:
        logger.warning(f"CelesTrak adapter unavailable: {e}")

    if not tasks:
        logger.warning("No adapters enabled — exiting")
        mqtt_client.loop_stop()
        mqtt_client.disconnect()
        return

    logger.info(f"Running {len(tasks)} adapter(s)")

    # Wait for shutdown signal
    stop = asyncio.Event()
    loop = asyncio.get_event_loop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except NotImplementedError:
            pass  # Windows

    await stop.wait()

    # Graceful shutdown
    logger.info("Shutting down adapters...")
    for name, adapter, task in tasks:
        await adapter.stop()
    await asyncio.gather(*[t for _, _, t in tasks], return_exceptions=True)

    mqtt_client.loop_stop()
    mqtt_client.disconnect()
    logger.info("All adapters stopped")


def main():
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
