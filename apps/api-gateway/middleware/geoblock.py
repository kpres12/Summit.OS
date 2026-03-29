"""
GeoBlock middleware — hard-blocks requests from sanctioned/adversarial countries.

Requires MaxMind GeoLite2-Country database (free, sign up at maxmind.com):
  1. Create free account at https://www.maxmind.com/en/geolite2/signup
  2. Download GeoLite2-Country.mmdb
  3. Place at infra/geoip/GeoLite2-Country.mmdb  (or set GEOIP_DB_PATH env var)
  4. In docker-compose, mount: ./infra/geoip:/geoip:ro
     and set GEOIP_DB_PATH=/geoip/GeoLite2-Country.mmdb

Without the database file, geoblock is DISABLED and a warning is logged at startup.
"""

import ipaddress
import logging
import os
from typing import Optional

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("api-gateway.geoblock")

# ── Blocked country codes (ISO 3166-1 alpha-2) ──────────────────────────────
#
# KP  North Korea (DPRK)
# CN  China
# RU  Russia
# IR  Iran
# AF  Afghanistan
# IL  Israel  (PS / Palestine is explicitly NOT blocked)
# BY  Belarus  — effectively a Russian proxy, significant state-sponsored ops
#
BLOCKED_COUNTRIES: frozenset[str] = frozenset({
    "KP",  # North Korea / DPRK
    "CN",  # China
    "RU",  # Russia
    "IR",  # Iran
    "AF",  # Afghanistan
    "IL",  # Israel  (NOT Palestine — PS is allowed)
    "BY",  # Belarus
})

# Private/reserved IP ranges — always allow (LAN, loopback, Docker networks)
_PRIVATE_NETWORKS = [
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
]

_reader = None  # geoip2.database.Reader, lazily loaded


def _load_reader() -> Optional[object]:
    global _reader
    if _reader is not None:
        return _reader

    db_path = os.getenv("GEOIP_DB_PATH", "/geoip/GeoLite2-Country.mmdb")
    try:
        import geoip2.database  # type: ignore

        _reader = geoip2.database.Reader(db_path)
        logger.info("GeoBlock: loaded GeoLite2-Country from %s", db_path)
        logger.info(
            "GeoBlock: blocking %d countries: %s",
            len(BLOCKED_COUNTRIES),
            ", ".join(sorted(BLOCKED_COUNTRIES)),
        )
        return _reader
    except FileNotFoundError:
        logger.warning(
            "GeoBlock DISABLED — database not found at %s. "
            "Download GeoLite2-Country.mmdb from maxmind.com and set GEOIP_DB_PATH.",
            db_path,
        )
        return None
    except ImportError:
        logger.warning(
            "GeoBlock DISABLED — geoip2 library not installed. "
            "Add 'geoip2' to requirements.txt and rebuild."
        )
        return None
    except Exception as exc:
        logger.warning("GeoBlock DISABLED — failed to load database: %s", exc)
        return None


def _is_private(ip_str: str) -> bool:
    try:
        addr = ipaddress.ip_address(ip_str)
        return any(addr in net for net in _PRIVATE_NETWORKS)
    except ValueError:
        return False


def _get_client_ip(request: Request) -> str:
    """
    Resolve real client IP, respecting X-Forwarded-For from trusted proxies.
    Takes the leftmost non-private address in X-Forwarded-For, falling back
    to request.client.host.
    """
    forwarded_for = request.headers.get("X-Forwarded-For", "")
    if forwarded_for:
        for candidate in (ip.strip() for ip in forwarded_for.split(",")):
            if candidate and not _is_private(candidate):
                return candidate

    cf_ip = request.headers.get("CF-Connecting-IP", "")
    if cf_ip and not _is_private(cf_ip):
        return cf_ip

    return request.client.host if request.client else "127.0.0.1"


def _lookup_country(ip_str: str) -> Optional[str]:
    reader = _load_reader()
    if reader is None:
        return None
    if _is_private(ip_str):
        return None
    try:
        response = reader.country(ip_str)
        return response.country.iso_code  # e.g. "CN", "RU", "IL"
    except Exception:
        return None


class GeoBlockMiddleware(BaseHTTPMiddleware):
    """
    Drop requests from blocked countries with HTTP 451 (Unavailable For Legal Reasons).
    Health check and metrics endpoints are always allowed.
    """

    ALWAYS_ALLOW = frozenset({"/health", "/metrics", "/ready"})

    async def dispatch(self, request: Request, call_next):
        if request.url.path in self.ALWAYS_ALLOW:
            return await call_next(request)

        client_ip = _get_client_ip(request)
        country = _lookup_country(client_ip)

        if country and country in BLOCKED_COUNTRIES:
            logger.warning(
                "GeoBlock: rejected request from %s (%s) → %s",
                client_ip,
                country,
                request.url.path,
            )
            return JSONResponse(
                status_code=451,
                content={
                    "detail": "Access restricted in your region.",
                    "code": "GEO_BLOCKED",
                },
                headers={"X-Blocked-Country": country},
            )

        return await call_next(request)
