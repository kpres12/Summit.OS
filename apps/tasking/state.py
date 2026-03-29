"""Shared mutable state for the tasking service."""
from typing import Optional, Dict, Any
import paho.mqtt.client as mqtt
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession
from sqlalchemy.orm import sessionmaker

engine: Optional[AsyncEngine] = None
SessionLocal: Optional[sessionmaker] = None
mqtt_client: Optional[mqtt.Client] = None

STATE_MACHINE_AVAILABLE: bool = False
mission_registry = None

ASSIGNMENT_ENGINE_AVAILABLE: bool = False

PROM_AVAILABLE: bool = False
METRIC_MISSIONS_CREATED = None
METRIC_MISSIONS_ACTIVE = None
METRIC_ASSETS_REGISTERED = None

OIDC_AVAILABLE: bool = False
OIDC_ENFORCE: bool = False
OIDC_ISSUER: Optional[str] = None
OIDC_AUDIENCE: Optional[str] = None

_ENTERPRISE_MULTI_TENANT: bool = False
TASKING_TEST_MODE: bool = False
DIRECT_AUTOPILOT: bool = False

_direct_queue = None
_autopilots: Dict[str, Any] = {}
