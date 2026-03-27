"""
Summit.OS Adapter Status Router
================================

FastAPI router that surfaces adapter status to the operator console and
any external tooling.

Endpoints
---------
GET  /adapters              List all configured adapters + live health data.
GET  /adapters/types        List built-in supported adapter types.
GET  /adapters/{id}         Health detail for a single adapter.
POST /adapters/{id}/enable  Enable a registered adapter.
POST /adapters/{id}/disable Disable a running adapter.

Initialisation
--------------
Call ``init_adapter_router(registry)`` once during application startup
(inside the lifespan context manager in ``main.py``) before the first
request arrives::

    from routers.adapters import init_adapter_router
    init_adapter_router(registry)

The router can be imported and included in the FastAPI app without a
registry — requests will return 503 until ``init_adapter_router`` is called.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException

# ---------------------------------------------------------------------------
# Path setup: make packages/ importable
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[3]  # apps/api-gateway/routers → repo root
_PACKAGES = str(_REPO_ROOT / "packages")
if _PACKAGES not in sys.path:
    sys.path.insert(0, _PACKAGES)

try:
    from adapters.registry import AdapterRegistry, BUILT_IN_ADAPTERS
except ImportError as _import_err:
    AdapterRegistry = None  # type: ignore[misc,assignment]
    BUILT_IN_ADAPTERS = []  # type: ignore[assignment]
    logging.getLogger("api-gateway.adapters").warning(
        "adapters package not importable: %s", _import_err
    )

logger = logging.getLogger("api-gateway.adapters")

# ---------------------------------------------------------------------------
# Module-level registry reference (set by init_adapter_router)
# ---------------------------------------------------------------------------

_registry: Optional["AdapterRegistry"] = None

router = APIRouter(prefix="/adapters", tags=["adapters"])


def init_adapter_router(registry: "AdapterRegistry") -> None:
    """
    Bind an AdapterRegistry instance to this router.

    Must be called once at startup before the first request.
    """
    global _registry
    _registry = registry
    logger.info(
        "Adapter router initialised with registry (%d adapter(s)).",
        len(registry._adapters),
    )


def _require_registry() -> "AdapterRegistry":
    if _registry is None:
        raise HTTPException(
            status_code=503,
            detail="Adapter registry not initialised — call init_adapter_router() at startup.",
        )
    return _registry


def _require_adapter(registry: "AdapterRegistry", adapter_id: str):
    adapter = registry.get(adapter_id)
    if adapter is None:
        raise HTTPException(
            status_code=404,
            detail=f"Adapter '{adapter_id}' not found.",
        )
    return adapter


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get(
    "",
    summary="List all adapters with live health",
    response_description="Array of adapter summaries including current status.",
)
async def list_adapters():
    """
    Return every registered adapter together with its current health data.

    Suitable for populating the DEV console Adapter Registry panel.
    """
    registry = _require_registry()
    return {"adapters": registry.list_adapters()}


@router.get(
    "/types",
    summary="List supported adapter types",
    response_description="Catalogue of adapter types that Summit.OS supports.",
)
async def list_adapter_types():
    """
    Return the built-in adapter type catalogue.

    Each entry has ``type``, ``name``, and ``description`` fields.  Use this
    to populate the 'Add adapter' UI in the DEV console.
    """
    return {"types": BUILT_IN_ADAPTERS}


@router.get(
    "/{adapter_id}",
    summary="Single adapter health detail",
)
async def get_adapter(adapter_id: str):
    """Return health detail for a single adapter by its ``adapter_id``."""
    registry = _require_registry()
    adapter = _require_adapter(registry, adapter_id)

    h = adapter.health()
    return {
        "adapter_id": adapter_id,
        "adapter_type": adapter.config.adapter_type,
        "display_name": adapter.config.display_name,
        "description": adapter.config.description,
        "enabled": adapter.config.enabled,
        "status": h.status,
        "last_connected": h.last_connected.isoformat() if h.last_connected else None,
        "last_observation": (
            h.last_observation.isoformat() if h.last_observation else None
        ),
        "observations_total": h.observations_total,
        "observations_per_minute": round(h.observations_per_minute, 2),
        "error_message": h.error_message,
        "uptime_seconds": round(h.uptime_seconds, 1),
    }


@router.post(
    "/{adapter_id}/enable",
    summary="Enable a registered adapter",
)
async def enable_adapter(adapter_id: str):
    """
    Mark a registered adapter as enabled and start it if it is not already
    running.

    Note: this modifies the in-memory config only.  To persist the change
    across restarts, update the ``adapters.json`` config file.
    """
    registry = _require_registry()
    adapter = _require_adapter(registry, adapter_id)

    if adapter.config.enabled:
        return {"adapter_id": adapter_id, "status": "already_enabled"}

    adapter.config.enabled = True

    import asyncio

    if adapter_id not in registry._tasks or registry._tasks[adapter_id].done():
        task = asyncio.create_task(
            adapter.start(),
            name=f"adapter:{adapter_id}",
        )
        adapter._task = task
        registry._tasks[adapter_id] = task
        logger.info("Adapter %s enabled and started.", adapter_id)

    return {"adapter_id": adapter_id, "status": "enabled"}


@router.post(
    "/{adapter_id}/disable",
    summary="Disable and stop a running adapter",
)
async def disable_adapter(adapter_id: str):
    """
    Mark a registered adapter as disabled and stop it gracefully.

    Note: this modifies the in-memory config only.  To persist the change
    across restarts, update the ``adapters.json`` config file.
    """
    registry = _require_registry()
    adapter = _require_adapter(registry, adapter_id)

    if not adapter.config.enabled:
        return {"adapter_id": adapter_id, "status": "already_disabled"}

    adapter.config.enabled = False
    await adapter.stop()
    logger.info("Adapter %s disabled and stopped.", adapter_id)

    return {"adapter_id": adapter_id, "status": "disabled"}
