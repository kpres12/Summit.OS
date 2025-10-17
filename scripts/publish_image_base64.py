#!/usr/bin/env python3
"""
Publish an image as base64 JSON to MQTT for Fusion vision ingestion.

Usage:
  python scripts/publish_image_base64.py --image /path/to/img.jpg \
      --topic images/camera-001 --device-id camera-001 \
      --lat 37.422 --lon -122.084 --broker localhost --port 1883
"""
import argparse
import base64
import json
from datetime import datetime, timezone

import paho.mqtt.client as mqtt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Path to image file")
    ap.add_argument("--topic", default="images/camera-001")
    ap.add_argument("--device-id", default="camera-001")
    ap.add_argument("--lat", type=float, default=None)
    ap.add_argument("--lon", type=float, default=None)
    ap.add_argument("--broker", default="localhost")
    ap.add_argument("--port", type=int, default=1883)
    args = ap.parse_args()

    with open(args.image, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")

    payload = {
        "image_b64": img_b64,
        "device_id": args.device_id,
        "lat": args.lat,
        "lon": args.lon,
        "ts_iso": datetime.now(timezone.utc).isoformat(),
    }

    client = mqtt.Client()
    client.connect(args.broker, args.port, 60)
    client.publish(args.topic, json.dumps(payload), qos=0)
    client.disconnect()
    print(f"Published image to {args.topic}")


if __name__ == "__main__":
    main()
