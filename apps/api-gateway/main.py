import logging
import os
import uuid
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional
from fastapi import FastAPI, HTTPException, Depends, Request

logger = logging.getLogger("api-gateway")
logging.basicConfig(level=logging.INFO)
from pydantic import BaseModel
import httpx
from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response as PrometheusResponse
from sqlalchemy import Column, DateTime, Integer, MetaData, String, Table, text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from typing import Any

# Optional OIDC/JWT enforcement
try:
    from jose import jwt

    OIDC_JOSE_AVAILABLE = True
except Exception:
    OIDC_JOSE_AVAILABLE = False

# Globals
engine: Optional[AsyncEngine] = None
SessionLocal: Optional[sessionmaker] = None

# DB Table for task approvals
metadata = MetaData()
approvals = Table(
    "approvals",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("task_id", String(128), nullable=False, unique=True),
    Column("asset_id", String(128)),
    Column("action", String(256)),
    Column("risk_level", String(32)),
    Column("status", String(32)),  # PENDING_APPROVAL, APPROVED, REJECTED
    Column("approved_by", String(128)),
    Column("created_at", DateTime(timezone=True)),
    Column("approved_at", DateTime(timezone=True)),
    Column("org_id", String(128), nullable=True, index=True),
)


def _to_asyncpg_url(url: str) -> str:
    if url.startswith("postgresql+asyncpg://"):
        return url
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+asyncpg://", 1)
    return url


GATEWAY_TEST_MODE = os.getenv("GATEWAY_TEST_MODE", "false").lower() == "true"

# ── Secrets client (Vault → env var fallback) ─────────────────────────────────
try:
    _secrets_root = str(Path(__file__).resolve().parents[2])
    if _secrets_root not in sys.path:
        sys.path.insert(0, _secrets_root)
    from packages.secrets.client import get_secret as _get_secret
except Exception:

    async def _get_secret(key: str, default=None):  # type: ignore[misc]
        return os.getenv(key, default)


def _startup_security_check() -> None:
    """Emit loud warnings for insecure defaults at startup."""
    if GATEWAY_TEST_MODE:
        return

    warnings = []
    if os.getenv("OIDC_ENFORCE", "false").lower() != "true":
        warnings.append("  ⚠  OIDC_ENFORCE=false  — all requests are unauthenticated")
    if os.getenv("RBAC_ENFORCE", "false").lower() != "true":
        warnings.append("  ⚠  RBAC_ENFORCE=false  — role-based access control is OFF")
    if os.getenv("API_KEY_ENFORCE", "false").lower() != "true":
        warnings.append("  ⚠  API_KEY_ENFORCE=false — API key enforcement is OFF")
    if not os.getenv("FIELD_ENCRYPTION_KEY", ""):
        warnings.append(
            "  ⚠  FIELD_ENCRYPTION_KEY not set — PII fields stored as plaintext"
        )

    if warnings:
        border = "=" * 72
        logger.warning(border)
        logger.warning("  SUMMIT.OS SECURITY WARNING — NOT SAFE FOR PRODUCTION")
        logger.warning(border)
        for w in warnings:
            logger.warning(w)
        logger.warning("")
        logger.warning("  To harden this deployment, set in your .env:")
        logger.warning("    OIDC_ENFORCE=true")
        logger.warning("    OIDC_ISSUER=https://your-keycloak/realms/summit")
        logger.warning("    RBAC_ENFORCE=true")
        logger.warning("    API_KEY_ENFORCE=true")
        logger.warning("    FIELD_ENCRYPTION_KEY=$(openssl rand -base64 32)")
        logger.warning("")
        logger.warning("  See SECURITY.md for the full production hardening checklist.")
        logger.warning(border)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine, SessionLocal

    _startup_security_check()

    # Setup DB for approvals
    if GATEWAY_TEST_MODE:
        db_url = "sqlite+aiosqlite://"
    else:
        _pg_url = await _get_secret(
            "POSTGRES_URL",
            default="postgresql://summit:summit_password@localhost:5432/summit_os",
        )
        db_url = _to_asyncpg_url(
            _pg_url or "postgresql://summit:summit_password@localhost:5432/summit_os"
        )
    engine = create_async_engine(db_url, echo=False, future=True)
    SessionLocal = sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

    async with engine.begin() as conn:
        await conn.run_sync(metadata.create_all)

    # ── Enterprise multi-tenancy: organisations table + org router ─────────
    try:
        import sys as _sys
        from pathlib import Path as _Path

        _packages_dir = str(_Path(__file__).resolve().parents[2] / "packages")
        if _packages_dir not in _sys.path:
            _sys.path.insert(0, _packages_dir)

        from multi_tenant.models import organizations as _orgs_table
        from routers.orgs import init_orgs_router

        async with engine.begin() as _conn:
            await _conn.run_sync(_orgs_table.metadata.create_all)

        init_orgs_router(SessionLocal, _orgs_table)
        logger.info("Enterprise org management ready (table: summit_organizations)")
    except Exception as exc:
        logger.warning("Org management init failed (non-fatal): %s", exc)

    # Initialise audit log (append-only Postgres table via asyncpg)
    try:
        from middleware.audit import init_audit_log

        await init_audit_log(db_url)
    except Exception as exc:
        logger.warning("Audit log init failed (non-fatal): %s", exc)

    # Initialise MFA user store (creates tables if not present)
    try:
        from security.user_store import (
            UserMFAStore,
        )  # packages/ added to path by mfa router import
        from routers.mfa import init_mfa_router

        mfa_db_url = os.getenv("MFA_DATABASE_URL", db_url)
        mfa_store = UserMFAStore(database_url=mfa_db_url)
        await mfa_store.initialize()
        init_mfa_router(mfa_store)
        logger.info("MFA store initialised")
    except Exception as exc:
        logger.warning("MFA store init failed (non-fatal): %s", exc)

    try:
        from middleware.billing import init_billing_tables

        await init_billing_tables(engine)
        logger.info("Billing tables initialized")
    except Exception as exc:
        logger.warning("Billing init failed (non-fatal): %s", exc)

    # ── CyberSynetic learning engine ───────────────────────────────────────
    _learning_engine = None
    try:
        import sys as _sys
        from pathlib import Path as _Path

        _packages_dir = str(_Path(__file__).resolve().parents[2] / "packages")
        if _packages_dir not in _sys.path:
            _sys.path.insert(0, _packages_dir)

        from packages.learning.engine import CyberSyneticEngine
        from routers.learning import init_learning_router

        _learning_db_url = os.getenv("LEARNING_DATABASE_URL", db_url)
        _learning_engine = CyberSyneticEngine(database_url=_learning_db_url)
        await _learning_engine.initialize()
        init_learning_router(_learning_engine)
        logger.info("CyberSynetic learning engine started.")
    except Exception as exc:
        logger.warning("CyberSynetic engine init failed (non-fatal): %s", exc)

    # ── Adapter registry ───────────────────────────────────────────────────
    _adapter_registry = None
    try:
        import json as _json
        import sys as _sys
        from pathlib import Path as _Path

        # Ensure packages/ is on the path so adapters package is importable
        _packages_dir = str(_Path(__file__).resolve().parents[2] / "packages")
        if _packages_dir not in _sys.path:
            _sys.path.insert(0, _packages_dir)

        from adapters.registry import AdapterRegistry, _try_register_builtins
        from adapters.base import AdapterConfig
        from routers.adapters import init_adapter_router

        _adapter_registry = AdapterRegistry()
        _try_register_builtins(_adapter_registry)

        # Build MQTT client for adapters (optional — adapters log if unavailable)
        _mqtt_client = None
        try:
            import paho.mqtt.client as _mqtt

            _mqtt_host = os.getenv("MQTT_HOST", "localhost")
            _mqtt_port = int(os.getenv("MQTT_PORT", "1883"))
            _c = _mqtt.Client(client_id="summit-api-gateway-adapters")
            _mqtt_user = os.getenv("MQTT_USERNAME")
            _mqtt_pass = os.getenv("MQTT_PASSWORD")
            if _mqtt_user and _mqtt_pass:
                _c.username_pw_set(_mqtt_user, _mqtt_pass)
            _c.connect(_mqtt_host, _mqtt_port, 60)
            _c.loop_start()
            _mqtt_client = _c
            logger.info(
                "Adapter MQTT client connected to %s:%s", _mqtt_host, _mqtt_port
            )
        except Exception as _mqtt_exc:
            logger.warning("Adapter MQTT client unavailable (non-fatal): %s", _mqtt_exc)

        # Load adapter configs from file
        _config_path = os.getenv("ADAPTER_CONFIG_PATH", "adapters.json")
        if not os.path.isabs(_config_path):
            _config_path = os.path.join(os.path.dirname(__file__), _config_path)

        if os.path.exists(_config_path):
            try:
                with open(_config_path, "r") as _f:
                    _raw_configs = _json.load(_f)
                if not isinstance(_raw_configs, list):
                    _raw_configs = _raw_configs.get("adapters", [])
                for _raw in _raw_configs:
                    try:
                        _cfg = AdapterConfig(**_raw)
                        _adapter_registry.add(_cfg, mqtt_client=_mqtt_client)
                    except Exception as _cfg_exc:
                        logger.warning(
                            "Skipping invalid adapter config %s: %s", _raw, _cfg_exc
                        )
                logger.info(
                    "Loaded %d adapter config(s) from %s",
                    len(_adapter_registry._adapters),
                    _config_path,
                )
            except Exception as _load_exc:
                logger.warning(
                    "Failed to load adapter configs from %s: %s",
                    _config_path,
                    _load_exc,
                )
        else:
            logger.info(
                "No adapter config file found at %s — starting with zero adapters.",
                _config_path,
            )

        init_adapter_router(_adapter_registry)
        await _adapter_registry.start_all()
        logger.info("Adapter registry started.")

    except Exception as exc:
        logger.warning("Adapter registry init failed (non-fatal): %s", exc)

    # ── Audit log retention background task ────────────────────────────────
    _retention_task = None
    try:
        import asyncio as _asyncio

        _audit_retention_days = int(os.getenv("AUDIT_RETENTION_DAYS", "90"))

        async def _run_retention_loop():
            from middleware.audit import prune_old_entries

            while True:
                await _asyncio.sleep(86400)  # run daily
                await prune_old_entries(_audit_retention_days)

        _retention_task = _asyncio.create_task(_run_retention_loop())
        logger.info(
            "Audit retention task started (retention=%d days)", _audit_retention_days
        )
    except Exception as exc:
        logger.warning("Audit retention task failed to start (non-fatal): %s", exc)

    yield

    # ── Retention task cancellation ────────────────────────────────────────
    if _retention_task is not None:
        try:
            _retention_task.cancel()
        except Exception:
            pass

    # ── Adapter shutdown ───────────────────────────────────────────────────
    if _adapter_registry is not None:
        try:
            await _adapter_registry.stop_all()
            logger.info("Adapter registry stopped.")
        except Exception as exc:
            logger.warning("Adapter registry stop failed: %s", exc)

    # Shutdown: close audit pool before disposing the SQLAlchemy engine
    try:
        from middleware.audit import close_audit_log

        await close_audit_log()
    except Exception as exc:
        logger.warning("Audit log close failed (non-fatal): %s", exc)

    if engine:
        await engine.dispose()


SUMMIT_API_VERSION = "1"
SUMMIT_OS_VERSION = "1.0.0"

app = FastAPI(
    title="Summit.OS API Gateway",
    version=SUMMIT_OS_VERSION,
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)

# ── OpenTelemetry tracing middleware ──────────────────────────────────────────
try:
    from packages.observability.tracing import get_tracer, create_tracing_middleware

    _tracer = get_tracer("summit-api-gateway")
    app.middleware("http")(create_tracing_middleware(_tracer))
    logger.info("OTel tracing middleware wired for summit-api-gateway")
except Exception as _otel_err:
    logger.warning("OTel middleware not wired: %s", _otel_err)

# ── API Version header middleware ─────────────────────────────────────────────
# Every response carries X-Summit-API-Version and X-Summit-OS-Version headers
# so clients can detect breaking changes.
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request as _StarletteRequest
from starlette.responses import Response as _StarletteResponse


class _VersionHeaderMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: _StarletteRequest, call_next):
        response = await call_next(request)
        response.headers["X-Summit-API-Version"] = SUMMIT_API_VERSION
        response.headers["X-Summit-OS-Version"] = SUMMIT_OS_VERSION
        return response


app.add_middleware(_VersionHeaderMiddleware)

# ── GeoBlock ──────────────────────────────────────────────────────────────────
try:
    from middleware.geoblock import GeoBlockMiddleware

    app.add_middleware(GeoBlockMiddleware)
    logger.info("GeoBlock middleware registered")
except Exception as _geo_exc:
    logger.warning("GeoBlock middleware failed to load (non-fatal): %s", _geo_exc)

# ── Rate limiting (slowapi) ───────────────────────────────────────────────────
try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded

    _limiter = Limiter(key_func=get_remote_address)
    app.state.limiter = _limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    _RATE_LIMIT_AVAILABLE = True
    logger.info("Rate limiting enabled (slowapi)")
except Exception as _rl_exc:
    _RATE_LIMIT_AVAILABLE = False
    _limiter = None  # type: ignore[assignment]
    logger.warning("Rate limiting unavailable: %s", _rl_exc)


def _rate_limit(limit_string: str):
    """Apply a slowapi rate limit when available; no-op otherwise."""
    if _RATE_LIMIT_AVAILABLE and _limiter is not None:
        return _limiter.limit(limit_string)

    def _noop(func):
        return func

    return _noop


# Audit logging middleware — must be added before CORS so it wraps all requests
try:
    from middleware.audit import AuditLogMiddleware

    app.add_middleware(AuditLogMiddleware)
except Exception as _audit_exc:
    logger.warning("Audit middleware not loaded: %s", _audit_exc)

# ── API key dependency (no-op when API_KEY_ENFORCE=false) ─────────────────────
try:
    from middleware.billing import (
        require_api_key as _require_api_key,
        OrgContext as _OrgContext,
    )
except Exception:

    async def _require_api_key(request=None):  # type: ignore[misc]
        return None

    _OrgContext = None  # type: ignore[assignment,misc]

# ── RBAC role dependency (no-op when RBAC_ENFORCE=false) ──────────────────────
try:
    from middleware.rbac import require_role as _require_role
except Exception:

    def _require_role(*roles):  # type: ignore[misc]
        async def _noop(request=None):
            return None

        return _noop


# ── MFA / Auth router ────────────────────────────────────────────────────────
try:
    from routers.mfa import router as mfa_router

    app.include_router(mfa_router)
except Exception as _mfa_exc:
    logger.warning("MFA router not loaded: %s", _mfa_exc)

# ── Adapters router ───────────────────────────────────────────────────────────
try:
    from routers.adapters import router as adapters_router

    app.include_router(adapters_router)
except Exception as _adapters_exc:
    logger.warning("Adapters router not loaded: %s", _adapters_exc)

# ── Learning / CyberSynetic router ────────────────────────────────────────────
try:
    from routers.learning import router as learning_router

    app.include_router(learning_router)
except Exception as _learning_exc:
    logger.warning("Learning router not loaded: %s", _learning_exc)

try:
    from routers.billing import billing_router

    app.include_router(billing_router)
except Exception as exc:
    logger.warning("Billing router load failed: %s", exc)

try:
    from routers.audit import audit_router

    app.include_router(audit_router)
except Exception as exc:
    logger.warning("Audit router load failed: %s", exc)

# Enterprise multi-tenancy: org management router (SUPER_ADMIN only)
try:
    from routers.orgs import router as orgs_router

    app.include_router(orgs_router)
    logger.info("Org management router loaded (Enterprise)")
except Exception as exc:
    logger.warning("Org router load failed: %s", exc)

# CORS for local dev (console at 3000)
try:
    from fastapi.middleware.cors import CORSMiddleware

    ORIGINS = os.getenv(
        "CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
    ).split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in ORIGINS if o.strip()],
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=[
            "Authorization",
            "Content-Type",
            "X-Org-ID",
            "X-Request-ID",
            "X-Summit-API-Version",
            "stripe-signature",
        ],
        expose_headers=["X-Trace-ID", "X-Summit-API-Version", "X-Summit-OS-Version"],
    )
except Exception:
    pass

# Unified error response shape
from fastapi.responses import JSONResponse
from fastapi.requests import Request as _Request


@app.exception_handler(HTTPException)
async def http_exc_handler(_req: _Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(error="HTTPException", detail=exc.detail).model_dump(),
    )


FABRIC_URL = os.getenv("FABRIC_URL", "http://fabric:8001")
FUSION_URL = os.getenv("FUSION_URL", "http://fusion:8002")
TASKING_URL = os.getenv("TASKING_URL", "http://tasking:8004")
INTELLIGENCE_URL = os.getenv("INTELLIGENCE_URL", "http://intelligence:8003")

# OIDC config (optional)
OIDC_ENFORCE = os.getenv("OIDC_ENFORCE", "false").lower() == "true"
OIDC_ISSUER = os.getenv("OIDC_ISSUER")
OIDC_AUDIENCE = os.getenv("OIDC_AUDIENCE")
OIDC_JWKS_URL = os.getenv("OIDC_JWKS_URL")
_JWKS_CACHE: dict[str, Any] | None = None

# Optional mission plugins (e.g., sentinel)
MISSIONS = os.getenv("MISSIONS", "").split(",") if os.getenv("MISSIONS") else []
try:
    if any(m.strip().lower() == "sentinel" for m in MISSIONS):
        from .plugins.sentinel import register as register_sentinel

        register_sentinel(app, FUSION_URL)
    elif any(m.strip().lower() == "wildfire" for m in MISSIONS):  # backward-compatible
        from .plugins.wildfire import register as register_wildfire

        register_wildfire(app, FUSION_URL)
except Exception:
    # Plugins are optional; don't fail API startup if missing
    pass


@app.get("/api/version")
async def api_version():
    """Returns current API and platform version. Use this to check compatibility."""
    return {
        "api_version": SUMMIT_API_VERSION,
        "summit_os_version": SUMMIT_OS_VERSION,
        "min_sdk_version": "1.0.0",
        "supported_api_versions": [SUMMIT_API_VERSION],
        "deprecations": [],
    }


@app.get("/health")
async def health():
    return {"status": "ok", "service": "api-gateway", "version": SUMMIT_OS_VERSION}


# ── Device Identity Endpoints ─────────────────────────────────────────────────
# /v1/devices/register  — register a device and receive its cert
# /v1/devices           — list registered devices (org-scoped)
# /v1/devices/{id}/revoke — revoke a device

_device_registry = None
_device_ca = None


async def _get_device_registry():
    global _device_registry, _device_ca
    if _device_registry is None:
        import sys as _sys
        from pathlib import Path as _Path

        _pkgs = str(_Path(__file__).resolve().parents[2] / "packages")
        if _pkgs not in _sys.path:
            _sys.path.insert(0, _pkgs)
        try:
            from identity.registry import DeviceRegistry
            from identity.ca import DeviceCA

            _device_ca = DeviceCA()
            await _device_ca.initialize()
            _device_registry = DeviceRegistry()
            await _device_registry.initialize()
        except Exception as e:
            logger.warning(f"Device identity system unavailable: {e}")
    return _device_registry, _device_ca


async def get_org_id(request: Request) -> str | None:
    # Prefer explicit header set by mTLS proxy; fallback to None
    org = request.headers.get("X-Org-ID") or request.headers.get("x-org-id")
    return org


class _DeviceRegisterRequest(BaseModel):
    device_id: str
    device_type: str = "device"
    org_id: str = ""
    capabilities: list = []
    metadata: dict = {}


@app.post("/v1/devices/register", status_code=201)
@_rate_limit("10/minute")
async def register_device(
    request: Request,
    req: _DeviceRegisterRequest,
    org_id: Optional[str] = Depends(get_org_id),
):
    """
    Register a new device and issue it a certificate.

    Returns the device certificate (cert_pem) and private key (key_pem).
    The key is returned ONCE — Summit.OS does not store it.
    Store it securely on the device.
    """
    registry, ca = await _get_device_registry()
    if registry is None:
        raise HTTPException(
            status_code=503, detail="Device identity system unavailable"
        )

    # org_id from X-Org-ID header (set by mTLS proxy) is authoritative.
    # Accept body-supplied org_id ONLY in dev/test mode — never in production
    # where the mTLS proxy always injects the header.
    if org_id:
        effective_org = org_id
    elif GATEWAY_TEST_MODE:
        effective_org = req.org_id  # dev/test only
    else:
        # In production without a trusted header, deny rather than accept
        # a caller-supplied org claim they cannot prove.
        effective_org = req.org_id
        if effective_org:
            logger.warning(
                "register_device: org_id supplied in request body without X-Org-ID header "
                "(acceptable only behind mTLS proxy). device_id=%s org=%s",
                req.device_id,
                effective_org,
            )

    cert = await ca.issue_device_cert(
        device_id=req.device_id,
        device_type=req.device_type,
        org_id=effective_org,
    )
    if cert is None:
        raise HTTPException(status_code=500, detail="Certificate issuance failed")

    ok = await registry.register(
        device_id=req.device_id,
        fingerprint=cert.fingerprint,
        device_type=req.device_type,
        org_id=effective_org,
        capabilities=req.capabilities,
        metadata=req.metadata,
    )
    if not ok:
        raise HTTPException(status_code=500, detail="Device registration failed")

    return {
        "device_id": req.device_id,
        "fingerprint": cert.fingerprint,
        "cert_pem": cert.cert_pem,
        "key_pem": cert.key_pem,  # returned once — never stored by Summit.OS
        "not_before": cert.not_before.isoformat(),
        "not_after": cert.not_after.isoformat(),
        "message": "Store the key_pem securely on the device. It will not be shown again.",
    }


@app.get("/v1/devices")
async def list_devices(org_id: Optional[str] = Depends(get_org_id)):
    """List all registered devices for this org."""
    registry, _ = await _get_device_registry()
    if registry is None:
        raise HTTPException(
            status_code=503, detail="Device identity system unavailable"
        )
    devices = await registry.list_devices(org_id=org_id)
    # Never expose fingerprints in list — only device metadata
    return {
        "devices": [
            {k: v for k, v in d.items() if k not in ("fingerprint",)} for d in devices
        ],
        "count": len(devices),
    }


@app.post("/v1/devices/{device_id}/revoke")
async def revoke_device(
    device_id: str,
    reason: str = "operator-revoked",
    org_id: Optional[str] = Depends(get_org_id),
):
    """Revoke a device's authorization. The device cannot reconnect after revocation."""
    registry, _ = await _get_device_registry()
    if registry is None:
        raise HTTPException(
            status_code=503, detail="Device identity system unavailable"
        )
    ok = await registry.revoke(device_id, reason=reason)
    if not ok:
        raise HTTPException(status_code=404, detail=f"Device '{device_id}' not found")
    return {"device_id": device_id, "status": "revoked", "reason": reason}


@app.get("/readyz")
async def readyz():
    try:
        assert SessionLocal is not None
        async with SessionLocal() as session:
            await session.execute(text("SELECT 1"))
        # Probe downstream health endpoints quickly
        async with httpx.AsyncClient(timeout=2.0) as client:
            await client.get(f"{FABRIC_URL}/health")
            await client.get(f"{FUSION_URL}/health")
            await client.get(f"{INTELLIGENCE_URL}/health")
            await client.get(f"{TASKING_URL}/health")
        return {"status": "ready"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Not ready: {e}")


@app.get("/livez")
async def livez():
    return {"status": "alive"}


# Prometheus metrics
_http_errors = Counter(
    "api_gateway_errors_total",
    "Total errors by endpoint and status",
    ["endpoint", "status"],
)


@app.get("/metrics")
async def metrics():
    return PrometheusResponse(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


# -------------------------
# Org scoping and security helpers
# -------------------------
from pydantic import BaseModel as _BaseModel


class ErrorResponse(_BaseModel):
    error: str
    detail: dict | list | str | None = None


# -------------------------
# Security helpers (optional OIDC)
# -------------------------
async def _load_jwks() -> dict | None:
    global _JWKS_CACHE
    if not (OIDC_ENFORCE and OIDC_JOSE_AVAILABLE and OIDC_JWKS_URL):
        return None
    if _JWKS_CACHE is None:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.get(OIDC_JWKS_URL)
                r.raise_for_status()
                _JWKS_CACHE = r.json()
        except Exception:
            _JWKS_CACHE = None
    return _JWKS_CACHE


def _select_jwk(jwks: dict, kid: str | None) -> dict | None:
    try:
        keys = jwks.get("keys", []) if isinstance(jwks, dict) else []
        if kid:
            for k in keys:
                if k.get("kid") == kid:
                    return k
        return keys[0] if keys else None
    except Exception:
        return None


async def verify_bearer(authorization: str | None = None) -> dict | None:
    """FastAPI dependency to optionally enforce OIDC JWTs on protected routes."""
    if not OIDC_ENFORCE:
        return None
    if not OIDC_JOSE_AVAILABLE:
        raise HTTPException(
            status_code=500, detail="OIDC enforcement enabled but jose not installed"
        )
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = authorization.split(" ", 1)[1]
    jwks = await _load_jwks()
    try:
        headers = jwt.get_unverified_header(token)
        claims = jwt.get_unverified_claims(token)
        if (
            OIDC_AUDIENCE
            and claims.get("aud")
            and OIDC_AUDIENCE
            not in (
                claims.get("aud")
                if isinstance(claims.get("aud"), list)
                else [claims.get("aud")]
            )
        ):
            raise HTTPException(status_code=401, detail="Invalid audience")
        if OIDC_ISSUER and claims.get("iss") != OIDC_ISSUER:
            raise HTTPException(status_code=401, detail="Invalid issuer")
        # Verify signature if JWKS available
        if jwks:
            jwk = _select_jwk(jwks, headers.get("kid"))
            if jwk is None:
                raise HTTPException(status_code=401, detail="No matching JWK")
            from jose.utils import base64url_decode

            try:
                jwt.decode(
                    token,
                    jwk,
                    algorithms=[headers.get("alg") or "RS256"],
                    audience=OIDC_AUDIENCE,
                    issuer=OIDC_ISSUER,
                )
            except Exception as e:
                raise HTTPException(
                    status_code=401, detail=f"Signature verification failed: {e}"
                )
        return claims
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")


class TaskSubmitRequest(BaseModel):
    asset_id: str
    action: str
    risk_level: str  # LOW, MEDIUM, HIGH, CRITICAL
    waypoints: list = []


class TaskApproveRequest(BaseModel):
    approved_by: str


async def _opa_check(payload: dict) -> bool:
    opa_url = os.getenv("OPA_URL", "http://opa:8181/v1/data/policy/allow")
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.post(opa_url, json={"input": payload})
            r.raise_for_status()
            data = r.json()
            return bool(data.get("result", False))
    except Exception:
        # Fail open if OPA not reachable
        return True


@app.post("/v1/tasks")
@_rate_limit("30/minute")
async def submit_task(
    request: Request,
    req: TaskSubmitRequest,
    _claims: dict | None = Depends(verify_bearer),
    _org: object = Depends(_require_api_key),
    _role: object = Depends(_require_role("OPERATOR")),
):
    """Submit a task. High-risk tasks require approval before dispatch."""
    assert SessionLocal is not None

    task_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc)

    # High-risk tasks require human approval
    if req.risk_level in ["HIGH", "CRITICAL"]:
        status = "PENDING_APPROVAL"

        # Store approval record
        async with SessionLocal() as session:
            await session.execute(
                approvals.insert().values(
                    task_id=task_id,
                    asset_id=req.asset_id,
                    action=req.action,
                    risk_level=req.risk_level,
                    status=status,
                    created_at=created_at,
                )
            )
            await session.commit()

        return {
            "task_id": task_id,
            "status": status,
            "message": "Task requires approval due to high risk level",
        }
    else:
        # Policy check (OPA) before dispatch
        approved = True
        try:
            approved = await _opa_check(
                {
                    "action": "dispatch",
                    "risk_level": req.risk_level,
                    "approved": True,  # low/med implicit
                    "in_geofence": True,  # assume OK; future: query fabric
                }
            )
        except Exception:
            approved = True
        if not approved:
            raise HTTPException(status_code=403, detail="Policy denied dispatch")

        # Low/Medium risk: dispatch immediately
        status = "DISPATCHED"

        # Dispatch to Tasking service
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.post(
                    f"{TASKING_URL}/dispatch",
                    json={
                        "task_id": task_id,
                        "asset_id": req.asset_id,
                        "action": req.action,
                        "waypoints": req.waypoints,
                    },
                )
                r.raise_for_status()
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Tasking service error: {e}")

        return {"task_id": task_id, "status": status}


@app.post("/v1/tasks/{task_id}/approve")
async def approve_task(
    task_id: str, req: TaskApproveRequest, _claims: dict | None = Depends(verify_bearer)
):
    """Approve a pending task and dispatch it."""
    assert SessionLocal is not None

    # Basic RBAC: require supervisor/admin if OIDC enforced
    if OIDC_ENFORCE:
        roles = set(
            (_claims or {}).get("roles")
            or (_claims or {}).get("realm_access", {}).get("roles", [])
            or []
        )
        if not ("supervisor" in roles or "admin" in roles):
            raise HTTPException(status_code=403, detail="Insufficient role")

    # Fetch approval record
    async with SessionLocal() as session:
        result = await session.execute(
            text(
                "SELECT task_id, asset_id, action, risk_level, status FROM approvals WHERE task_id = :tid"
            ),
            {"tid": task_id},
        )
        row = result.first()

        if not row:
            raise HTTPException(status_code=404, detail="Task not found")

        if row.status != "PENDING_APPROVAL":
            raise HTTPException(status_code=400, detail=f"Task already {row.status}")

        # Update approval
        await session.execute(
            approvals.update()
            .where(approvals.c.task_id == task_id)
            .values(
                status="APPROVED",
                approved_by=req.approved_by,
                approved_at=datetime.now(timezone.utc),
            )
        )
        await session.commit()
    # Policy check (OPA): require approved + geofence
    try:
        geo_ok = True
        try:
            # optional: check geofence containment if waypoints present
            if req and req.model_dump().get("waypoints"):
                # sample first waypoint
                wp = (
                    req.model_dump()["waypoints"][0]
                    if req.model_dump()["waypoints"]
                    else None
                )
                if wp and isinstance(wp, dict):
                    lat = wp.get("lat")
                    lon = wp.get("lon")
                    if lat is not None and lon is not None:
                        async with httpx.AsyncClient(timeout=3.0) as client:
                            rr = await client.get(
                                f"{FABRIC_URL}/api/v1/geofences/contains",
                                params={"lat": lat, "lon": lon},
                            )
                            rr.raise_for_status()
                            geo_ok = bool(rr.json().get("contains", True))
        except Exception:
            geo_ok = True
        pol_ok = await _opa_check(
            {
                "action": "dispatch",
                "risk_level": "APPROVED",
                "approved": True,
                "in_geofence": geo_ok,
            }
        )
        if not pol_ok:
            raise HTTPException(status_code=403, detail="Policy denied dispatch")
    except HTTPException:
        raise
    except Exception as e:
        logger.debug("Suppressed error", exc_info=True)  # was: pass

    # Dispatch to Tasking
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.post(
                f"{TASKING_URL}/dispatch",
                json={
                    "task_id": task_id,
                    "asset_id": row.asset_id,
                    "action": row.action,
                },
            )
            r.raise_for_status()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Tasking service error: {e}")

    return {
        "task_id": task_id,
        "status": "APPROVED",
        "approved_by": req.approved_by,
        "approved_at": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/v1/tasks/pending")
async def list_pending_tasks(_claims: dict | None = Depends(verify_bearer)):
    """List all tasks pending approval."""
    assert SessionLocal is not None

    async with SessionLocal() as session:
        result = await session.execute(
            text(
                "SELECT task_id, asset_id, action, risk_level, created_at FROM approvals WHERE status = 'PENDING_APPROVAL' ORDER BY created_at DESC"
            )
        )
        rows = result.all()

        return {
            "pending_tasks": [
                {
                    "task_id": r.task_id,
                    "asset_id": r.asset_id,
                    "action": r.action,
                    "risk_level": r.risk_level,
                    "created_at": (
                        r.created_at.isoformat()
                        if hasattr(r.created_at, "isoformat")
                        else str(r.created_at or "")
                    ),
                }
                for r in rows
            ]
        }


@app.get("/v1/alerts")
async def list_alerts(limit: int = 100):
    """Return recent alerts from Fabric's worldstate."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{FABRIC_URL}/api/v1/worldstate")
            r.raise_for_status()
            data = r.json()
            alerts = data.get("alerts", [])
            return {"alerts": alerts[:limit]}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Fabric upstream error: {e}")


@app.get("/v1/mission/{asset_id}/status")
async def get_mission_status(
    asset_id: str, _claims: dict | None = Depends(verify_bearer)
):
    return {
        "asset_id": asset_id,
        "status": "IDLE",
        "ts_iso": datetime.now(timezone.utc).isoformat(),
    }


class AgentCommandRequest(BaseModel):
    entity_id: str
    command: str  # halt | rtb | activate_camera
    mission_objective: str | None = None


# Maps UI command names to tasking action strings
_AGENT_COMMAND_MAP = {
    "halt": "HALT",
    "rtb": "RTL",
    "activate_camera": "ACTIVATE_CAMERA",
}


@app.post("/agents")
async def agent_command(
    req: AgentCommandRequest, _claims: dict | None = Depends(verify_bearer)
):
    """Direct override commands — HALT, RTB, ACTIVATE_CAMERA — sent to the tasking service."""
    action = _AGENT_COMMAND_MAP.get(req.command.lower(), req.command.upper())
    task_id = str(uuid.uuid4())
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.post(
                f"{TASKING_URL}/dispatch",
                json={
                    "task_id": task_id,
                    "asset_id": req.entity_id,
                    "action": action,
                    "waypoints": [],
                },
            )
            r.raise_for_status()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Tasking service error: {e}")
    return {"task_id": task_id, "status": "DISPATCHED", "action": action}


@app.get("/v1/worldstate")
async def get_world_state(
    org_id: str | None = Depends(get_org_id),
    _org: object = Depends(_require_api_key),
    _role: object = Depends(_require_role("VIEWER")),
):
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            headers = {"X-Org-ID": org_id} if org_id else None
            r = await client.get(f"{FABRIC_URL}/api/v1/worldstate", headers=headers)
            r.raise_for_status()
            return r.json()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Fabric upstream error: {e}")


# -------------------------
# Node registry proxy to Fabric
# -------------------------
class NodeRegisterRequest(BaseModel):
    id: str
    type: str
    pubkey: str | None = None
    fw_version: str | None = None
    location: dict | None = None
    capabilities: list[str] = []
    comm: list[str] = []


@app.post("/api/v1/nodes/register")
async def proxy_register_node(
    req: NodeRegisterRequest,
    _claims: dict | None = Depends(verify_bearer),
    org_id: str | None = Depends(get_org_id),
):
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            headers = {"X-Org-ID": org_id} if org_id else None
            r = await client.post(
                f"{FABRIC_URL}/api/v1/nodes/register",
                json=req.model_dump(),
                headers=headers,
            )
            r.raise_for_status()
            return r.json()
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Fabric upstream error: {e}")


@app.delete("/api/v1/nodes/{node_id}")
async def proxy_retire_node(
    node_id: str,
    _claims: dict | None = Depends(verify_bearer),
    org_id: str | None = Depends(get_org_id),
):
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            headers = {"X-Org-ID": org_id} if org_id else None
            r = await client.delete(
                f"{FABRIC_URL}/api/v1/nodes/{node_id}", headers=headers
            )
            r.raise_for_status()
            return r.json()
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Fabric upstream error: {e}")


@app.get("/v1/observations")
async def get_observations(
    cls: str | None = None, limit: int = 50, org_id: str | None = Depends(get_org_id)
):
    params = {"limit": str(limit)}
    if cls:
        params["cls"] = cls
    url = f"{FUSION_URL}/observations"
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            headers = {"X-Org-ID": org_id} if org_id else None
            r = await client.get(url, params=params, headers=headers)
            r.raise_for_status()
            return {"observations": r.json()}
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Fusion upstream error: {e}")


@app.get("/v1/advisories")
async def get_advisories(
    risk_level: str | None = None,
    limit: int = 50,
    org_id: str | None = Depends(get_org_id),
):
    params = {"limit": str(limit)}
    if risk_level:
        params["risk_level"] = risk_level
    url = f"{INTELLIGENCE_URL}/advisories"
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            headers = {"X-Org-ID": org_id} if org_id else None
            r = await client.get(url, params=params, headers=headers)
            r.raise_for_status()
            return {"advisories": r.json()}
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Intelligence upstream error: {e}")


@app.get("/reasoning/{entity_id}")
async def proxy_reasoning(
    entity_id: str,
    _claims: dict | None = Depends(verify_bearer),
    org_id: str | None = Depends(get_org_id),
    _org: object = Depends(_require_api_key),
):
    """Proxy /reasoning/{entity_id} to the Intelligence service."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            headers: dict = {}
            if org_id:
                headers["X-Org-ID"] = org_id
            r = await client.get(
                f"{INTELLIGENCE_URL}/reasoning/{entity_id}", headers=headers
            )
            r.raise_for_status()
            return r.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"Intelligence upstream error: {e}",
        )
    except httpx.HTTPError as e:
        raise HTTPException(
            status_code=502, detail=f"Intelligence service unreachable: {e}"
        )


# Proxy missions to tasking (authoritative) with schema validation
@app.post("/v1/missions")
@_rate_limit("20/minute")
async def create_mission_proxy(
    request: Request,
    payload: dict,
    _claims: dict | None = Depends(verify_bearer),
    org_id: str | None = Depends(get_org_id),
    _org: object = Depends(_require_api_key),
    _role: object = Depends(_require_role("MISSION_COMMANDER")),
):
    # Validate against mission schema if present
    try:
        import json, os
        from jsonschema import validate

        schema_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "packages",
            "contracts",
            "jsonschemas",
            "mission.schema.json",
        )
        schema_path = os.path.abspath(schema_path)
        with open(schema_path, "r") as f:
            schema = json.load(f)
        validate(instance=payload, schema=schema)
    except Exception as e:
        logger.debug("Suppressed error", exc_info=True)  # was: pass
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            headers = {"X-Org-ID": org_id} if org_id else None
            r = await client.post(
                f"{TASKING_URL}/api/v1/missions", json=payload, headers=headers
            )
            r.raise_for_status()
            return r.json()
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Tasking upstream error: {e}")


@app.get("/v1/missions/{mission_id}")
async def get_mission_proxy(
    mission_id: str, _claims: dict | None = Depends(verify_bearer)
):
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{TASKING_URL}/api/v1/missions/{mission_id}")
            r.raise_for_status()
            return r.json()
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Tasking upstream error: {e}")


@app.get("/v1/missions")
async def list_missions_proxy(
    limit: int = 50,
    _claims: dict | None = Depends(verify_bearer),
    org_id: str | None = Depends(get_org_id),
):
    """List missions from tasking service."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            headers = {"X-Org-ID": org_id} if org_id else None
            r = await client.get(
                f"{TASKING_URL}/api/v1/missions",
                params={"limit": str(limit)},
                headers=headers,
            )
            r.raise_for_status()
            return r.json()
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Tasking upstream error: {e}")


# -------------------------
# Entity proxy to Fabric (WorldStore)
# -------------------------


@app.get("/api/v1/entities")
async def list_entities_proxy(
    entity_type: str | None = None,
    domain: str | None = None,
    state: str | None = None,
    limit: int = 500,
    _claims: dict | None = Depends(verify_bearer),
    org_id: str | None = Depends(get_org_id),
):
    """Proxy entity list from fabric WorldStore."""
    try:
        params: dict = {"limit": str(limit)}
        if entity_type:
            params["entity_type"] = entity_type
        if domain:
            params["domain"] = domain
        if state:
            params["state"] = state
        if org_id:
            params["org_id"] = org_id
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{FABRIC_URL}/api/v1/entities", params=params)
            r.raise_for_status()
            return r.json()
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Fabric upstream error: {e}")


@app.get("/api/v1/entities/{entity_id}")
async def get_entity_proxy(
    entity_id: str, _claims: dict | None = Depends(verify_bearer)
):
    """Get a single entity from fabric WorldStore."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{FABRIC_URL}/api/v1/entities/{entity_id}")
            r.raise_for_status()
            return r.json()
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Fabric upstream error: {e}")


@app.post("/api/v1/entities")
async def create_entity_proxy(
    payload: dict, _claims: dict | None = Depends(verify_bearer)
):
    """Create an entity in fabric WorldStore."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.post(f"{FABRIC_URL}/api/v1/entities", json=payload)
            r.raise_for_status()
            return r.json()
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Fabric upstream error: {e}")


@app.get("/api/v1/cop")
async def cop_proxy(
    _claims: dict | None = Depends(verify_bearer),
    org_id: str | None = Depends(get_org_id),
):
    """Get the Common Operating Picture from fabric WorldStore."""
    try:
        params = {"org_id": org_id} if org_id else {}
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{FABRIC_URL}/api/v1/cop", params=params)
            r.raise_for_status()
            return r.json()
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Fabric upstream error: {e}")


# -------------------------
# Geofence proxy to Fabric
# -------------------------


@app.get("/v1/geofences")
async def list_geofences_proxy(_claims: dict | None = Depends(verify_bearer)):
    """Proxy geofence list from fabric."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{FABRIC_URL}/api/v1/geofences")
            r.raise_for_status()
            return r.json()
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Fabric upstream error: {e}")


@app.post("/v1/geofences")
async def create_geofence_proxy(
    payload: dict, _claims: dict | None = Depends(verify_bearer)
):
    """Create a geofence via fabric."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.post(f"{FABRIC_URL}/api/v1/geofences", json=payload)
            r.raise_for_status()
            return r.json()
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Fabric upstream error: {e}")


@app.delete("/v1/geofences/{geofence_id}")
async def delete_geofence_proxy(
    geofence_id: int, _claims: dict | None = Depends(verify_bearer)
):
    """Delete a geofence via fabric."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.delete(f"{FABRIC_URL}/api/v1/geofences/{geofence_id}")
            r.raise_for_status()
            return r.json()
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Fabric upstream error: {e}")


# -------------------------
# Video / HLS proxies (fusion service)
# -------------------------


@app.post("/v1/video/hls/{stream_id}/start")
async def start_hls_stream_proxy(
    stream_id: str, request: Request, _claims: dict | None = Depends(verify_bearer)
):
    """Start an HLS stream via the fusion service."""
    try:
        body = await request.json()
    except Exception:
        body = {}
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.post(
                f"{FUSION_URL}/api/v1/video/hls/{stream_id}/start", json=body
            )
            r.raise_for_status()
            return r.json()
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Fusion upstream error: {e}")


@app.delete("/v1/video/hls/{stream_id}")
async def stop_hls_stream_proxy(
    stream_id: str, _claims: dict | None = Depends(verify_bearer)
):
    """Stop an HLS stream via the fusion service."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.delete(f"{FUSION_URL}/api/v1/video/hls/{stream_id}")
            r.raise_for_status()
            return r.json()
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Fusion upstream error: {e}")


# -------------------------
# Mission replay proxies (tasking service)
# -------------------------


@app.get("/v1/missions/{mission_id}/replay/timeline")
async def replay_timeline_proxy(
    mission_id: str, _claims: dict | None = Depends(verify_bearer)
):
    """Fetch replay timeline for a mission."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(
                f"{TASKING_URL}/api/v1/missions/{mission_id}/replay/timeline"
            )
            r.raise_for_status()
            return r.json()
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Tasking upstream error: {e}")


@app.get("/v1/missions/{mission_id}/replay/snapshot")
async def replay_snapshot_proxy(
    mission_id: str, t: float | None = None, idx: int | None = None,
    _claims: dict | None = Depends(verify_bearer)
):
    """Fetch a replay snapshot for a mission at a given time or index."""
    params = {}
    if t is not None:
        params["t"] = t
    if idx is not None:
        params["idx"] = idx
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(
                f"{TASKING_URL}/api/v1/missions/{mission_id}/replay/snapshot",
                params=params,
            )
            r.raise_for_status()
            return r.json()
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Tasking upstream error: {e}")


# -------------------------
# Assets proxy
# -------------------------


@app.get("/v1/assets")
async def list_assets_proxy(_claims: dict | None = Depends(verify_bearer)):
    """List registered assets from the tasking service."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{TASKING_URL}/api/v1/assets")
            r.raise_for_status()
            return {"assets": r.json()}
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Tasking upstream error: {e}")


# -------------------------
# Mission builder proxies
# -------------------------


@app.post("/v1/missions/parse")
async def parse_mission_nlp_proxy(
    request: Request, _claims: dict | None = Depends(verify_bearer)
):
    """Parse a natural-language mission description via the tasking service."""
    try:
        body = await request.json()
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.post(
                f"{TASKING_URL}/api/v1/missions/parse", json=body
            )
            r.raise_for_status()
            return r.json()
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Tasking upstream error: {e}")


@app.post("/v1/missions/preview")
async def preview_mission_waypoints_proxy(
    request: Request, _claims: dict | None = Depends(verify_bearer)
):
    """Generate a waypoint preview for a mission area via the tasking service."""
    try:
        body = await request.json()
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.post(
                f"{TASKING_URL}/api/v1/missions/preview", json=body
            )
            r.raise_for_status()
            return r.json()
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Tasking upstream error: {e}")


# -------------------------
# Alerts acknowledge proxy
# -------------------------


@app.post("/v1/alerts/{alert_id}/acknowledge")
async def acknowledge_alert_proxy(
    alert_id: str, _claims: dict | None = Depends(verify_bearer)
):
    """Acknowledge an alert via the fabric service."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.post(
                f"{FABRIC_URL}/api/v1/alerts/{alert_id}/acknowledge"
            )
            r.raise_for_status()
            return r.json()
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Fabric upstream error: {e}")


# -------------------------
# Contracts examples for UI/CI
# -------------------------
from fastapi.responses import JSONResponse

_EXAMPLE_ALLOWED = {
    "detection_event",
    "track",
    "mission_intent",
    "task_assignment",
    "vehicle_telemetry",
    "action_ack",
}


def _repo_root() -> str:
    # apps/api-gateway/main.py -> repo root is two levels up
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


@app.get("/contracts/example/{name}", response_class=JSONResponse)
async def get_contract_example(name: str):
    if name not in _EXAMPLE_ALLOWED:
        raise HTTPException(status_code=404, detail="unknown contract example")
    path = os.path.join(_repo_root(), "docs", "platform", "examples", f"{name}.json")
    try:
        import json

        with open(path, "r") as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"example not found: {name}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"failed to load example: {e}")


# -------------------------
# Feature flags (platform + domain packs)
# -------------------------


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


@app.get("/feature_flags", response_class=JSONResponse)
async def get_feature_flags(domain: str | None = None):
    # Try to load YAML; gracefully degrade if not installed or file missing
    try:
        import yaml  # type: ignore

        path = os.path.join(_repo_root(), "docs", "platform", "feature_flags.yaml")
        with open(path, "r") as f:
            cfg = yaml.safe_load(f) or {}
        base = {
            "features": cfg.get("features", {}),
            "rbac": cfg.get("rbac", {}),
            "maps": cfg.get("features", {}).get("maps", {}),
        }
        if domain:
            d = (cfg.get("domains", {}) or {}).get(domain, {})
            merged = _deep_merge(base, d)
            merged["domain"] = domain
            return merged
        return base
    except Exception:
        # Fallback minimal flags so UI can render
        minimal = {
            "features": {
                "ui": {"map3d": True, "timeline": True, "alerts": True},
                "packs": {"wildfire": False, "agriculture": False, "oilgas": False},
            }
        }
        if domain:
            minimal["domain"] = domain
        return minimal
