#!/usr/bin/env python3
"""
Download YOLOv8n ONNX model for Summit.OS inference service.

Downloads from Ultralytics GitHub releases (~12.8 MB).
The model is licensed under AGPL-3.0 — see models/LICENSE.

Usage:
  python scripts/download_model.py
"""

import os
import sys
import urllib.request

MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.onnx"
DEST_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
DEST_PATH = os.path.join(DEST_DIR, "yolov8n.onnx")


def download():
    dest = os.path.abspath(DEST_PATH)
    if os.path.isfile(dest):
        size_mb = os.path.getsize(dest) / (1024 * 1024)
        print(f"Model already exists: {dest} ({size_mb:.1f} MB)")
        return

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"Downloading YOLOv8n ONNX model...")
    print(f"  From: {MODEL_URL}")
    print(f"  To:   {dest}")

    try:
        urllib.request.urlretrieve(MODEL_URL, dest, _progress)
        size_mb = os.path.getsize(dest) / (1024 * 1024)
        print(f"\nDownload complete: {size_mb:.1f} MB")
        print("License: AGPL-3.0 (Ultralytics) — see models/LICENSE")
    except Exception as e:
        print(f"\nDownload failed: {e}")
        print("You can manually download from:")
        print(f"  {MODEL_URL}")
        sys.exit(1)


def _progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100, downloaded * 100 // total_size)
        bar = "=" * (pct // 2) + " " * (50 - pct // 2)
        print(f"\r  [{bar}] {pct}%", end="", flush=True)


if __name__ == "__main__":
    download()
