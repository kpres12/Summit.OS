"""
Alert Escalation Service for Summit.OS

Monitors active alerts. Any alert unacknowledged for longer than
ESCALATION_UNACK_TIMEOUT_S is escalated:
  1. Alert status updated to 'escalated' in the DB
  2. Webhook POST (if ESCALATION_WEBHOOK_URL is set)
  3. SMTP email (if ESCALATION_SMTP_HOST is set)

Runs as a background asyncio task inside the Fabric service.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import httpx

logger = logging.getLogger("fabric.escalation")


@dataclass
class EscalationConfig:
    check_interval_s: float = float(os.getenv("ESCALATION_CHECK_INTERVAL_S", "30"))
    unack_timeout_s: float = float(os.getenv("ESCALATION_UNACK_TIMEOUT_S", "120"))
    webhook_url: Optional[str] = os.getenv("ESCALATION_WEBHOOK_URL")
    smtp_host: Optional[str] = os.getenv("ESCALATION_SMTP_HOST")
    smtp_port: int = int(os.getenv("ESCALATION_SMTP_PORT", "587"))
    smtp_user: Optional[str] = os.getenv("ESCALATION_SMTP_USER")
    smtp_password: Optional[str] = os.getenv("ESCALATION_SMTP_PASSWORD")
    escalation_email: Optional[str] = os.getenv("ESCALATION_EMAIL_TO")
    escalation_from: str = os.getenv("ESCALATION_EMAIL_FROM", "alerts@summit.local")


class AlertEscalationService:
    """
    Background service that watches the in-memory alert store and escalates
    unacknowledged alerts past the configured timeout.
    """

    def __init__(
        self, alert_store: Dict[str, Any], config: Optional[EscalationConfig] = None
    ):
        self._alerts = alert_store  # shared reference to Fabric's in-memory alert dict
        self._config = config or EscalationConfig()
        self._escalated: set = set()  # alert_ids already escalated this session

    async def run(self):
        logger.info(
            f"AlertEscalationService started "
            f"(timeout={self._config.unack_timeout_s}s, "
            f"check={self._config.check_interval_s}s)"
        )
        while True:
            try:
                await self._check()
            except Exception as e:
                logger.error(f"Escalation check error: {e}", exc_info=True)
            await asyncio.sleep(self._config.check_interval_s)

    async def _check(self):
        now = time.time()
        for alert_id, alert in list(self._alerts.items()):
            if alert.get("acknowledged"):
                continue
            if alert_id in self._escalated:
                continue

            ts_iso = alert.get("ts_iso") or alert.get("ts")
            if not ts_iso:
                continue
            try:
                ts = datetime.fromisoformat(
                    str(ts_iso).replace("Z", "+00:00")
                ).timestamp()
            except Exception:
                continue

            age_s = now - ts
            if age_s >= self._config.unack_timeout_s:
                await self._escalate(alert_id, alert, age_s)

    async def _escalate(self, alert_id: str, alert: Dict[str, Any], age_s: float):
        self._escalated.add(alert_id)
        alert["status"] = "escalated"
        alert["escalated_at"] = datetime.now(timezone.utc).isoformat()
        logger.warning(
            f"ESCALATING alert {alert_id} "
            f"(severity={alert.get('severity')}, unacked {age_s:.0f}s)"
        )

        payload = {
            "event": "alert_escalated",
            "alert_id": alert_id,
            "severity": alert.get("severity", "UNKNOWN"),
            "description": alert.get("description", ""),
            "source": alert.get("source", ""),
            "unacknowledged_for_s": round(age_s),
            "ts_iso": datetime.now(timezone.utc).isoformat(),
        }

        await asyncio.gather(
            self._send_webhook(payload),
            self._send_email(payload),
            return_exceptions=True,
        )

    async def _send_webhook(self, payload: Dict[str, Any]):
        if not self._config.webhook_url:
            return
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                await client.post(self._config.webhook_url, json=payload)
            logger.info(f"Escalation webhook sent for {payload['alert_id']}")
        except Exception as e:
            logger.warning(f"Escalation webhook failed: {e}")

    async def _send_email(self, payload: Dict[str, Any]):
        if not (self._config.smtp_host and self._config.escalation_email):
            return
        try:
            import aiosmtplib

            subject = f"[SUMMIT ALERT ESCALATED] {payload['severity']} — {payload['description'][:60]}"
            body = (
                f"Alert ID:    {payload['alert_id']}\n"
                f"Severity:    {payload['severity']}\n"
                f"Description: {payload['description']}\n"
                f"Source:      {payload['source']}\n"
                f"Unacked for: {payload['unacknowledged_for_s']}s\n"
                f"Time:        {payload['ts_iso']}\n"
            )
            message = (
                f"From: Summit.OS <{self._config.escalation_from}>\n"
                f"To: {self._config.escalation_email}\n"
                f"Subject: {subject}\n\n{body}"
            )
            await aiosmtplib.send(
                message,
                hostname=self._config.smtp_host,
                port=self._config.smtp_port,
                username=self._config.smtp_user,
                password=self._config.smtp_password,
                start_tls=True,
            )
            logger.info(f"Escalation email sent for {payload['alert_id']}")
        except ImportError:
            logger.debug("aiosmtplib not installed — email escalation disabled")
        except Exception as e:
            logger.warning(f"Escalation email failed: {e}")
