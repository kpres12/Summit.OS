"""
Summit.OS SDK Client

Provides a high-level Python API for Summit.OS services:
- Entity CRUD and streaming
- Task management
- Sensor data ingestion
- Mesh status monitoring

Mirrors Lattice SDK patterns with REST + WebSocket transports.
"""
from __future__ import annotations

import json
import time
import logging
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Dict, List, Optional
from urllib.parse import urljoin

from .errors import NotConnectedError, raise_for_status
from .retry import RetryPolicy, CircuitBreaker, retry_with_circuit

logger = logging.getLogger("summit.sdk")


@dataclass
class SDKConfig:
    """SDK configuration."""
    api_url: str = "http://localhost:8000"
    ws_url: str = "ws://localhost:8000/ws"
    api_key: str = ""
    jwt_token: str = ""
    timeout: float = 30.0
    max_retries: int = 3
    verify_ssl: bool = True


class SummitClient:
    """
    Summit.OS SDK client.

    Usage:
        client = SummitClient(api_url="http://localhost:8000")
        await client.connect()
        entities = await client.entities.list(domain="AIR")
        await client.disconnect()

    With custom retry/circuit breaker:
        from summit_sdk import RetryPolicy, CircuitBreaker
        client = SummitClient(
            api_url="http://localhost:8000",
            retry_policy=RetryPolicy(max_retries=5, base_delay=1.0),
            circuit_breaker=CircuitBreaker(failure_threshold=10),
        )
    """

    def __init__(self, api_url: str = "http://localhost:8000",
                 ws_url: str = "",
                 api_key: str = "",
                 jwt_token: str = "",
                 retry_policy: Optional[RetryPolicy] = None,
                 circuit_breaker: Optional[CircuitBreaker] = None):
        self.config = SDKConfig(
            api_url=api_url.rstrip("/"),
            ws_url=ws_url or api_url.replace("http", "ws") + "/ws",
            api_key=api_key,
            jwt_token=jwt_token,
        )
        self._session = None
        self._ws = None
        self._connected = False
        self._aiohttp_available = self._check_aiohttp()
        self._retry = retry_policy or RetryPolicy()
        self._breaker = circuit_breaker or CircuitBreaker()

        # Sub-clients
        self.entities = EntityClient(self)
        self.tasks = TaskClient(self)
        self.mesh = MeshClient(self)
        self.sensors = SensorClient(self)

    @staticmethod
    def _check_aiohttp() -> bool:
        try:
            import aiohttp
            return True
        except ImportError:
            return False

    async def connect(self) -> bool:
        """Connect to Summit.OS API."""
        if self._aiohttp_available:
            import aiohttp
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            headers = self._auth_headers()
            self._session = aiohttp.ClientSession(
                headers=headers, timeout=timeout,
            )
            self._connected = True
            logger.info(f"SDK connected to {self.config.api_url}")
            return True
        else:
            logger.warning("aiohttp not installed — SDK in offline mode")
            self._connected = True
            return True

    async def disconnect(self) -> None:
        """Disconnect from Summit.OS API."""
        if self._session:
            await self._session.close()
        self._connected = False

    def _auth_headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.config.jwt_token:
            headers["Authorization"] = f"Bearer {self.config.jwt_token}"
        if self.config.api_key:
            headers["X-API-Key"] = self.config.api_key
        return headers

    async def _raw_get(self, path: str, params: Optional[Dict] = None) -> Dict:
        """Single HTTP GET attempt (no retry)."""
        if not self._session:
            raise NotConnectedError()
        url = f"{self.config.api_url}{path}"
        async with self._session.get(url, params=params) as resp:
            body = await resp.json()
            raise_for_status(resp.status, body)
            return body

    async def _raw_post(self, path: str, data: Dict) -> Dict:
        if not self._session:
            raise NotConnectedError()
        url = f"{self.config.api_url}{path}"
        async with self._session.post(url, json=data) as resp:
            body = await resp.json()
            raise_for_status(resp.status, body)
            return body

    async def _raw_put(self, path: str, data: Dict) -> Dict:
        if not self._session:
            raise NotConnectedError()
        url = f"{self.config.api_url}{path}"
        async with self._session.put(url, json=data) as resp:
            body = await resp.json()
            raise_for_status(resp.status, body)
            return body

    async def _raw_delete(self, path: str) -> Dict:
        if not self._session:
            raise NotConnectedError()
        url = f"{self.config.api_url}{path}"
        async with self._session.delete(url) as resp:
            body = await resp.json()
            raise_for_status(resp.status, body)
            return body

    # ── Public methods with retry + circuit breaker ──────────

    async def _get(self, path: str, params: Optional[Dict] = None) -> Dict:
        return await retry_with_circuit(
            self._raw_get, path, params,
            retry=self._retry, breaker=self._breaker,
        )

    async def _post(self, path: str, data: Dict) -> Dict:
        return await retry_with_circuit(
            self._raw_post, path, data,
            retry=self._retry, breaker=self._breaker,
        )

    async def _put(self, path: str, data: Dict) -> Dict:
        return await retry_with_circuit(
            self._raw_put, path, data,
            retry=self._retry, breaker=self._breaker,
        )

    async def _delete(self, path: str) -> Dict:
        return await retry_with_circuit(
            self._raw_delete, path,
            retry=self._retry, breaker=self._breaker,
        )

    async def health(self) -> Dict:
        """Check API health."""
        return await self._get("/health")

    @property
    def is_connected(self) -> bool:
        return self._connected


# ── Sub-clients ─────────────────────────────────────────────

class EntityClient:
    """Entity operations."""
    def __init__(self, client: SummitClient):
        self._c = client

    async def get(self, entity_id: str) -> Dict:
        return await self._c._get(f"/api/v1/entities/{entity_id}")

    async def list(self, domain: Optional[str] = None,
                   entity_type: Optional[str] = None,
                   limit: int = 100) -> List[Dict]:
        params = {"limit": limit}
        if domain:
            params["domain"] = domain
        if entity_type:
            params["type"] = entity_type
        resp = await self._c._get("/api/v1/entities", params)
        return resp.get("entities", [])

    async def create(self, entity_data: Dict) -> Dict:
        return await self._c._post("/api/v1/entities", entity_data)

    async def update(self, entity_id: str, updates: Dict) -> Dict:
        return await self._c._put(f"/api/v1/entities/{entity_id}", updates)

    async def delete(self, entity_id: str) -> Dict:
        return await self._c._delete(f"/api/v1/entities/{entity_id}")

    async def bulk_upsert(self, entities: List[Dict]) -> Dict:
        return await self._c._post("/api/v1/entities/bulk", {"entities": entities})


class TaskClient:
    """Task operations."""
    def __init__(self, client: SummitClient):
        self._c = client

    async def get(self, task_id: str) -> Dict:
        return await self._c._get(f"/api/v1/tasks/{task_id}")

    async def list(self, state: Optional[str] = None,
                   mission_id: Optional[str] = None) -> List[Dict]:
        params = {}
        if state:
            params["state"] = state
        if mission_id:
            params["mission_id"] = mission_id
        resp = await self._c._get("/api/v1/tasks", params)
        return resp.get("tasks", [])

    async def create(self, task_data: Dict) -> Dict:
        return await self._c._post("/api/v1/tasks", task_data)

    async def assign(self, task_id: str, assignee_id: str) -> Dict:
        return await self._c._post(f"/api/v1/tasks/{task_id}/assign",
                                    {"assignee_id": assignee_id})

    async def complete(self, task_id: str, result: Optional[Dict] = None) -> Dict:
        return await self._c._post(f"/api/v1/tasks/{task_id}/complete",
                                    {"result": result or {}})

    async def cancel(self, task_id: str) -> Dict:
        return await self._c._post(f"/api/v1/tasks/{task_id}/cancel", {})


class MeshClient:
    """Mesh status operations."""
    def __init__(self, client: SummitClient):
        self._c = client

    async def status(self) -> Dict:
        return await self._c._get("/api/v1/mesh/status")

    async def peers(self) -> List[Dict]:
        resp = await self._c._get("/api/v1/mesh/peers")
        return resp.get("peers", [])


class SensorClient:
    """Sensor data ingestion."""
    def __init__(self, client: SummitClient):
        self._c = client

    async def ingest(self, sensor_id: str, readings: List[Dict]) -> Dict:
        return await self._c._post(f"/api/v1/sensors/{sensor_id}/data",
                                    {"readings": readings})

    async def list(self) -> List[Dict]:
        resp = await self._c._get("/api/v1/sensors")
        return resp.get("sensors", [])
