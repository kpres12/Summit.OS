"""
Engagement MQTT Bridge
========================
Subscribes to sensor-fusion outputs on MQTT and auto-opens engagement
cases for confirmed kinetic-relevant tracks. This is the production
wiring that closes the gap between the fusion pipeline and the
EngagementAuthorizationGate.

Topics consumed:
  summit/engagement/track_confirmed   — fusion-published confirmed tracks
                                          ready to enter the engagement
                                          workflow (counter-UAS, force
                                          protection, etc.)

Topics published:
  summit/engagement/case_opened       — case_id + track_id on every
                                          successful gate.open_case()
  summit/engagement/case_rejected     — track_id + reason when the gate
                                          refuses to open a case (e.g.
                                          PID confidence below threshold,
                                          unknown classification)

Each MQTT-borne track must include AT LEAST:
  {
    "track_id":       "string",
    "entity_id":      "string",
    "classification": "string",   # e.g. "rotary_uas"
    "confidence":     0.0..1.0,
    "sensors":        ["radar-1", "rf-1"],
    "last_position":  {"lat": ..., "lon": ..., "alt_m": ...}
  }

Optional fields are passed through to TrackEvidence.

Auth:
  MQTT_USERNAME / MQTT_PASSWORD env vars when the broker requires it.
  TLS is configurable via MQTT_TLS_CA_PATH (recommended in production).

Usage:
  from packages.c2_intel.engagement_mqtt_bridge import EngagementMQTTBridge
  bridge = EngagementMQTTBridge(gate=production_gate)
  await bridge.start(broker="mqtt://localhost:1883")
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from .engagement_authorization import (
    EngagementAuthorizationError, EngagementAuthorizationGate, TrackEvidence,
)

logger = logging.getLogger("c2_intel.engagement_mqtt_bridge")


TOPIC_TRACK_CONFIRMED = "summit/engagement/track_confirmed"
TOPIC_CASE_OPENED     = "summit/engagement/case_opened"
TOPIC_CASE_REJECTED   = "summit/engagement/case_rejected"


class EngagementMQTTBridge:
    """Bridge between the sensor-fusion MQTT firehose and the
    EngagementAuthorizationGate. Auto-opens cases for confirmed tracks
    of kinetic-relevant entity classes; surfaces them on the operator
    UI's pending-cases queue via the standard gate event stream."""

    # Classifications that should automatically enter the engagement
    # workflow when fusion confirms them. Anything else is logged and
    # ignored (operator can manually open a case via the HTTP endpoint).
    DEFAULT_KINETIC_CLASSES: set[str] = {
        "rotary_uas",
        "fixed_wing_uas",
        "small_uas",
        "loitering_munition",
        "vessel_unknown",
        "ground_vehicle_hostile",
    }

    def __init__(
        self,
        gate: EngagementAuthorizationGate,
        kinetic_classes: Optional[set[str]] = None,
        min_track_confidence: float = 0.7,
    ):
        self._gate = gate
        self._kinetic = kinetic_classes or set(self.DEFAULT_KINETIC_CLASSES)
        self._min_confidence = float(min_track_confidence)
        self._client = None
        self._stop = asyncio.Event()

    async def start(
        self,
        broker_host: str = "localhost",
        broker_port: int = 1883,
        username: Optional[str] = None,
        password: Optional[str] = None,
        tls_ca_path: Optional[str] = None,
    ) -> None:
        """Connect, subscribe, and run until stop() is called."""
        try:
            import paho.mqtt.client as mqtt
        except ImportError as e:
            raise ImportError(
                "paho-mqtt is required for EngagementMQTTBridge. "
                "Install with: pip install paho-mqtt") from e

        username = username or os.environ.get("MQTT_USERNAME")
        password = password or os.environ.get("MQTT_PASSWORD")
        tls_ca_path = tls_ca_path or os.environ.get("MQTT_TLS_CA_PATH")

        client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2,
            client_id="heli-engagement-bridge",
        )
        if username and password:
            client.username_pw_set(username, password)
        if tls_ca_path:
            client.tls_set(ca_certs=tls_ca_path)

        client.on_connect = self._on_connect
        client.on_message = self._on_message
        client.connect_async(broker_host, broker_port, keepalive=60)
        client.loop_start()
        self._client = client

        logger.info("[engagement-mqtt-bridge] connected to %s:%d, listening on %s",
                    broker_host, broker_port, TOPIC_TRACK_CONFIRMED)
        await self._stop.wait()

        client.loop_stop()
        client.disconnect()
        logger.info("[engagement-mqtt-bridge] stopped")

    def stop(self) -> None:
        self._stop.set()

    # --- MQTT callbacks ---------------------------------------------------

    def _on_connect(self, client, userdata, flags, reason_code, properties=None):
        if reason_code != 0:
            logger.error("[engagement-mqtt-bridge] connect failed: %s", reason_code)
            return
        client.subscribe(TOPIC_TRACK_CONFIRMED, qos=1)

    def _on_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
        except Exception as e:
            logger.warning("[engagement-mqtt-bridge] bad payload on %s: %s",
                           msg.topic, e)
            return
        try:
            self._handle_track(client, payload)
        except Exception as e:
            logger.exception("[engagement-mqtt-bridge] track handler error: %s", e)

    # --- Core logic -------------------------------------------------------

    def _handle_track(self, client, payload: Dict[str, Any]) -> None:
        # Basic validation — refuse malformed input loudly
        for required in ("track_id", "entity_id", "classification", "confidence"):
            if required not in payload:
                self._reject(client, payload.get("track_id", ""),
                             f"missing required field: {required}")
                return

        confidence = float(payload.get("confidence", 0.0))
        classification = str(payload.get("classification", "")).strip().lower()

        # Threshold check
        if confidence < self._min_confidence:
            self._reject(client, payload["track_id"],
                         f"confidence {confidence:.2f} below threshold "
                         f"{self._min_confidence:.2f}")
            return

        # Class filter
        if classification not in self._kinetic:
            self._reject(client, payload["track_id"],
                         f"classification '{classification}' not in kinetic "
                         f"set; manual open required")
            return

        # Build TrackEvidence and open via the gate (single API surface)
        track = TrackEvidence(
            track_id=str(payload["track_id"]),
            entity_id=str(payload["entity_id"]),
            classification=classification,
            confidence=confidence,
            sensors=list(payload.get("sensors") or []),
            last_position=payload.get("last_position"),
            last_seen=datetime.now(timezone.utc),
        )
        try:
            case = self._gate.open_case(track)
        except EngagementAuthorizationError as e:
            self._reject(client, track.track_id, f"gate refused: {e}")
            return

        # Publish case_opened event for the operator UI to pick up
        client.publish(TOPIC_CASE_OPENED, json.dumps({
            "case_id":   case.case_id,
            "track_id":  track.track_id,
            "entity_id": track.entity_id,
            "classification": track.classification,
            "confidence": track.confidence,
            "ts":        datetime.now(timezone.utc).isoformat(),
        }), qos=1)
        logger.info("[engagement-mqtt-bridge] opened case %s for track %s",
                    case.case_id, track.track_id)

    def _reject(self, client, track_id: str, reason: str) -> None:
        logger.info("[engagement-mqtt-bridge] rejected track %s: %s",
                    track_id, reason)
        client.publish(TOPIC_CASE_REJECTED, json.dumps({
            "track_id": track_id,
            "reason":   reason,
            "ts":       datetime.now(timezone.utc).isoformat(),
        }), qos=1)
