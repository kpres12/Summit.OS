#!/usr/bin/env python3
"""
Heli.OS Detection Demo

Sends a synthetic test image to the inference service and displays results.
Optionally routes through the fusion → WorldStore pipeline.

Usage:
  python scripts/demo_detection.py                        # direct to inference
  python scripts/demo_detection.py --via-fusion            # fusion → inference → WorldStore
  INFERENCE_URL=http://localhost:8005 python scripts/demo_detection.py
"""

import argparse
import asyncio
import base64
import io
import os
import sys

import httpx

INFERENCE_URL = os.getenv("INFERENCE_URL", "http://localhost:8005")
FUSION_URL = os.getenv("FUSION_URL", "http://localhost:8002")


def create_test_image() -> str:
    """Create a simple synthetic test image (640x480 RGB) and return as base64."""
    try:
        import numpy as np
        # Create a simple image with colored rectangles (simulating objects)
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        img[50:200, 100:250] = [0, 0, 200]    # red rectangle (person-like)
        img[100:300, 350:500] = [200, 0, 0]    # blue rectangle (vehicle-like)
        img[300:420, 200:400] = [0, 200, 0]    # green rectangle

        try:
            import cv2
            _, buf = cv2.imencode(".jpg", img)
            return base64.b64encode(buf.tobytes()).decode()
        except ImportError:
            pass

        from PIL import Image
        pil_img = Image.fromarray(img)
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG")
        return base64.b64encode(buf.getvalue()).decode()
    except ImportError:
        # Minimal: 1x1 pixel JPEG
        pixel = b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00'
        return base64.b64encode(pixel).decode()


async def run(via_fusion: bool = False):
    image_b64 = create_test_image()
    print(f"\n{'='*60}")
    print("  Heli.OS Detection Demo")
    print(f"  Mode: {'fusion → inference → WorldStore' if via_fusion else 'direct inference'}")
    print(f"{'='*60}\n")

    async with httpx.AsyncClient(timeout=30.0) as client:
        if via_fusion:
            # Route through fusion pipeline
            print(f"Sending to fusion: {FUSION_URL}/api/v1/detect")
            try:
                r = await client.post(f"{FUSION_URL}/api/v1/detect", json={
                    "image_b64": image_b64,
                    "lat": 37.7749,
                    "lon": -122.4194,
                    "device_id": "demo-camera",
                    "confidence_threshold": 0.3,
                })
                r.raise_for_status()
                result = r.json()
                print(f"\nModel: {result.get('model_name', 'unknown')}")
                print(f"Inference: {result.get('inference_ms', 0):.1f} ms")
                print(f"Detections: {result.get('count', 0)}")
                for det in result.get("detections", []):
                    print(f"  - {det['class_name']}: {det['confidence']:.2%}")
                entities = result.get("entities_created", [])
                if entities:
                    print(f"\nTRACK entities created in WorldStore: {len(entities)}")
                    for eid in entities:
                        print(f"  - {eid}")
                print("\n✓ Fusion → Inference → WorldStore pipeline working")
            except Exception as e:
                print(f"\n✗ Failed: {e}")
                return 1
        else:
            # Direct to inference
            print(f"Sending to inference: {INFERENCE_URL}/detect")
            try:
                r = await client.post(f"{INFERENCE_URL}/detect", json={
                    "image_b64": image_b64,
                    "confidence_threshold": 0.3,
                })
                r.raise_for_status()
                result = r.json()
                print(f"\nModel: {result.get('model_name', 'unknown')}")
                print(f"Inference: {result.get('inference_ms', 0):.1f} ms")
                print(f"Detections: {result.get('count', 0)}")
                for det in result.get("detections", []):
                    bbox = det.get("bbox", {})
                    print(f"  - {det['class_name']}: {det['confidence']:.2%}  "
                          f"bbox=[{bbox.get('x1',0):.0f},{bbox.get('y1',0):.0f},"
                          f"{bbox.get('x2',0):.0f},{bbox.get('y2',0):.0f}]")
                print("\n✓ Inference service working")
            except Exception as e:
                print(f"\n✗ Failed: {e}")
                return 1

    print(f"{'='*60}\n")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Heli.OS Detection Demo")
    parser.add_argument("--via-fusion", action="store_true",
                        help="Route through fusion → inference → WorldStore pipeline")
    args = parser.parse_args()
    rc = asyncio.run(run(via_fusion=args.via_fusion))
    sys.exit(rc)
