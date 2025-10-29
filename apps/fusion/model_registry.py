"""
Simple model registry helpers for Fusion.
"""
import os
from typing import List

from .vision_inference import VisionInference


def list_models(root: str) -> List[str]:
    try:
        entries = []
        for dirpath, dirnames, filenames in os.walk(root):
            for f in filenames:
                if f.endswith((".onnx", ".pt", ".engine")):
                    entries.append(os.path.join(dirpath, f))
        return sorted(entries)
    except Exception:
        return []


def select_model(path: str) -> None:
    # Reinitialize global vision object if present
    from . import main as fusion_main  # circular import safe for attribute access
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if fusion_main.vision is None:
        fusion_main.vision = VisionInference(model_path=path, conf_threshold=float(os.getenv("FUSION_CONF_THRESHOLD", "0.6")))
    else:
        fusion_main.vision.load(path)
