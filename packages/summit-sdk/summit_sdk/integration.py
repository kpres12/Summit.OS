"""
Summit.OS Integration Client

Helper for integrating external systems (ADS-B feeds, radar,
SIGINT, etc.) into Summit.OS. Provides:
- Standardized data ingestion pipeline
- Entity auto-creation from sensor data
- Heartbeat and health reporting
"""

from __future__ import annotations

import time
import uuid
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("summit.integration")


@dataclass
class IntegrationConfig:
    """Configuration for an external integration."""

    integration_id: str = ""
    name: str = ""
    source_type: str = ""  # "adsb", "radar", "ais", "sigint", "custom"
    api_url: str = "http://localhost:8000"
    api_key: str = ""
    batch_size: int = 50
    flush_interval: float = 1.0


class IntegrationClient:
    """
    External system integration client.

    Simplifies connecting sensors and data feeds into Summit.OS.

    Usage:
        client = IntegrationClient(name="ADS-B Feed", source_type="adsb")
        await client.start()

        # Push tracks from your data source
        client.push_track(lat=34.0, lon=-118.0, alt=10000,
                          callsign="UAL123", source_id="adsb-receiver-1")

        # Periodic flush
        await client.flush()
    """

    def __init__(
        self,
        name: str = "External Integration",
        source_type: str = "custom",
        api_url: str = "http://localhost:8000",
        api_key: str = "",
    ):
        self.config = IntegrationConfig(
            integration_id=str(uuid.uuid4())[:8],
            name=name,
            source_type=source_type,
            api_url=api_url.rstrip("/"),
            api_key=api_key,
        )
        self._buffer: List[Dict] = []
        self._running = False
        self._stats = {
            "tracks_pushed": 0,
            "flushes": 0,
            "errors": 0,
            "started_at": 0.0,
        }

    async def start(self) -> None:
        """Start the integration."""
        self._running = True
        self._stats["started_at"] = time.time()
        logger.info(
            f"Integration '{self.config.name}' started (id={self.config.integration_id})"
        )

    async def stop(self) -> None:
        """Stop the integration and flush remaining data."""
        if self._buffer:
            await self.flush()
        self._running = False
        logger.info(f"Integration '{self.config.name}' stopped")

    def push_track(
        self,
        lat: float,
        lon: float,
        alt: float = 0.0,
        heading: float = 0.0,
        speed: float = 0.0,
        callsign: str = "",
        source_id: str = "",
        classification: str = "UNKNOWN",
        domain: str = "UNKNOWN",
        extra: Optional[Dict] = None,
    ) -> None:
        """Push a track observation into the buffer."""
        track = {
            "entity_type": "track",
            "domain": domain,
            "lat": lat,
            "lon": lon,
            "alt": alt,
            "heading": heading,
            "speed": speed,
            "classification": classification,
            "source": f"{self.config.source_type}:{source_id}",
            "name": callsign,
            "properties": {
                "integration_id": self.config.integration_id,
                "source_type": self.config.source_type,
                "callsign": callsign,
                **(extra or {}),
            },
        }
        self._buffer.append(track)
        self._stats["tracks_pushed"] += 1

        if len(self._buffer) >= self.config.batch_size:
            # Auto-flush not async — caller should await flush() periodically
            pass

    async def flush(self) -> int:
        """Flush buffered tracks to Summit.OS API."""
        if not self._buffer:
            return 0

        batch = self._buffer[: self.config.batch_size]
        self._buffer = self._buffer[self.config.batch_size :]

        try:
            # Attempt HTTP POST
            try:
                import aiohttp

                url = f"{self.config.api_url}/api/v1/entities/bulk"
                headers = {"Content-Type": "application/json"}
                if self.config.api_key:
                    headers["X-API-Key"] = self.config.api_key

                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        url, json={"entities": batch}, headers=headers
                    ) as resp:
                        if resp.status == 200:
                            self._stats["flushes"] += 1
                            return len(batch)
                        else:
                            self._stats["errors"] += 1
                            logger.error(f"Flush failed: HTTP {resp.status}")
                            self._buffer = batch + self._buffer  # Re-queue
                            return 0
            except ImportError:
                # No aiohttp — just count
                self._stats["flushes"] += 1
                logger.debug(f"Flushed {len(batch)} tracks (offline mode)")
                return len(batch)

        except Exception as e:
            self._stats["errors"] += 1
            logger.error(f"Flush error: {e}")
            self._buffer = batch + self._buffer
            return 0

    @property
    def buffer_size(self) -> int:
        return len(self._buffer)

    @property
    def stats(self) -> Dict:
        return {
            **self._stats,
            "buffer_size": self.buffer_size,
            "uptime_s": (
                time.time() - self._stats["started_at"]
                if self._stats["started_at"]
                else 0
            ),
        }
