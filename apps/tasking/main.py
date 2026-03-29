import asyncio
import logging
import os
import sys

sys.path.insert(0, "/packages")

logger = logging.getLogger("tasking")
logging.basicConfig(level=logging.INFO)

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse as _JSONResponse
from fastapi.requests import Request as _FRequest
import paho.mqtt.client as mqtt
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

import state
from models import ErrorResponse
from helpers import _to_asyncpg_url
from tables import metadata
from planning import _init_direct_autopilot

# ── Optional feature detection ────────────────────────────────────────────────
try:
    from apps.tasking.state_machine import (
        MissionStateMachine,
        MissionStateMachineRegistry,
        MissionState,
    )
    state.STATE_MACHINE_AVAILABLE = True
    state.mission_registry = MissionStateMachineRegistry()
except Exception:
    state.STATE_MACHINE_AVAILABLE = False

try:
    from apps.tasking.assignment_engine import AssignmentEngine, resolve_pattern
    state.ASSIGNMENT_ENGINE_AVAILABLE = True
except Exception:
    state.ASSIGNMENT_ENGINE_AVAILABLE = False

try:
    from apps.tasking.coverage_patterns import (
        grid_coverage_pattern,
        spiral_coverage_pattern,
        perimeter_patrol_pattern,
        orbit_pattern,
        expand_search_pattern,
    )
    COVERAGE_PATTERNS_AVAILABLE = True
except Exception:
    COVERAGE_PATTERNS_AVAILABLE = False

try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
    from packages.world.store import WorldStore
    WORLD_STORE_AVAILABLE = True
except Exception:
    WORLD_STORE_AVAILABLE = False

try:
    from prometheus_client import Counter, Gauge, CONTENT_TYPE_LATEST, generate_latest
    state.PROM_AVAILABLE = True
    state.METRIC_MISSIONS_CREATED = Counter(
        "missions_created_total", "Number of missions created"
    )
    state.METRIC_MISSIONS_ACTIVE = Gauge("missions_active", "Active missions")
    state.METRIC_ASSETS_REGISTERED = Counter("assets_registered_total", "Assets registered")
except Exception:
    state.PROM_AVAILABLE = False

try:
    from jose import jwt
    state.OIDC_AVAILABLE = True
except Exception:
    state.OIDC_AVAILABLE = False

# ── Configuration ─────────────────────────────────────────────────────────────
state.TASKING_TEST_MODE = os.getenv("TASKING_TEST_MODE", "false").lower() == "true"
state.OIDC_ENFORCE = os.getenv("OIDC_ENFORCE", "false").lower() == "true"
state.OIDC_ISSUER = os.getenv("OIDC_ISSUER")
state.OIDC_AUDIENCE = os.getenv("OIDC_AUDIENCE")
state._ENTERPRISE_MULTI_TENANT = os.getenv("ENTERPRISE_MULTI_TENANT", "false").lower() == "true"
state.DIRECT_AUTOPILOT = os.getenv("TASKING_DIRECT_AUTOPILOT", "false").lower() == "true"


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # DB setup
    if state.TASKING_TEST_MODE:
        pg_url = "sqlite+aiosqlite://"
    else:
        pg_url = _to_asyncpg_url(
            os.getenv(
                "POSTGRES_URL",
                "postgresql://summit:summit_password@localhost:5432/summit_os",
            )
        )
    state.engine = create_async_engine(pg_url, echo=False, future=True)
    state.SessionLocal = sessionmaker(state.engine, expire_on_commit=False, class_=AsyncSession)

    async with state.engine.begin() as conn:
        await conn.run_sync(metadata.create_all)

    if not state.TASKING_TEST_MODE:
        # MQTT setup
        broker = os.getenv("MQTT_BROKER", "localhost")
        port = int(os.getenv("MQTT_PORT", "1883"))
        state.mqtt_client = mqtt.Client()
        mqtt_user = os.getenv("MQTT_USERNAME")
        mqtt_pass = os.getenv("MQTT_PASSWORD")
        if mqtt_user and mqtt_pass:
            state.mqtt_client.username_pw_set(mqtt_user, mqtt_pass)
        state.mqtt_client.connect(broker, port, 60)
        state.mqtt_client.loop_start()

        # Closed-loop execution monitor
        try:
            from execution_monitor import ExecutionMonitor
            _exec_monitor = ExecutionMonitor(state.SessionLocal, state.mqtt_client)
            asyncio.create_task(_exec_monitor.run())
            logger.info("ExecutionMonitor started")
        except Exception as _em_err:
            logger.warning(f"ExecutionMonitor not started: {_em_err}")

        # Optional: direct autopilot worker subscribes to dispatches
        if state.DIRECT_AUTOPILOT:
            await _init_direct_autopilot()

    try:
        yield
    finally:
        if state.mqtt_client:
            state.mqtt_client.loop_stop()
            state.mqtt_client.disconnect()
        if state.engine:
            await state.engine.dispose()


# ── App init ──────────────────────────────────────────────────────────────────
app = FastAPI(title="Summit Tasking", version="0.2.1", lifespan=lifespan)

# Mission replay API
try:
    from replay_router import router as _replay_router
    app.include_router(_replay_router)
except Exception as _rr_err:
    logger.warning(f"Replay router not loaded: {_rr_err}")

# ── OpenTelemetry tracing middleware ──────────────────────────────────────────
try:
    _otel_root = os.path.join(os.path.dirname(__file__), "../..")
    if _otel_root not in sys.path:
        sys.path.insert(0, _otel_root)
    from packages.observability.tracing import get_tracer, create_tracing_middleware
    _tracer = get_tracer("summit-tasking")
    app.middleware("http")(create_tracing_middleware(_tracer))
except Exception as _e:
    logging.warning("OTel middleware not wired: %s", _e)

# ── Exception handler ─────────────────────────────────────────────────────────
@app.exception_handler(HTTPException)
async def http_exc_handler(_req: _FRequest, exc: HTTPException):
    return _JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(error="HTTPException", detail=exc.detail).model_dump(),
    )

# ── Routers ───────────────────────────────────────────────────────────────────
from routers import health, valves, tasks, assets, missions, tiered_missions

app.include_router(health.router)
app.include_router(valves.router)
app.include_router(tasks.router)
app.include_router(assets.router)
app.include_router(missions.router)
app.include_router(tiered_missions.router)
