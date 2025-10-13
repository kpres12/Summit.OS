#!/usr/bin/env python3
"""
Publish a sample smoke detection message to the local MQTT broker for testing.
"""
import os
import json
from datetime import datetime, timezone
import paho.mqtt.client as mqtt

broker = os.getenv("MQTT_BROKER", "localhost")
port = int(os.getenv("MQTT_PORT", "1883"))

data = {
    "asset_id": "asset-001",
    "ts_iso": datetime.now(timezone.utc).isoformat(),
    "bbox_xywh": [100, 120, 50, 60],
    "confidence": 0.82,
    "lat": 37.4219999,
    "lon": -122.0840575,
}

client = mqtt.Client()
client.connect(broker, port, 60)
client.loop_start()
# Publish as a generic observation (Kernel)
client.publish("observations/smoke", json.dumps({**data, "class": "smoke", "source": "demo"}), qos=1)
# Back-compat legacy topic (optional)
client.publish("detections/smoke", json.dumps(data), qos=1)
client.loop_stop()
client.disconnect()
print("Published sample smoke detection to detections/smoke")
