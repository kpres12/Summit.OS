"""
Generic Vision Inference for Fusion Service

- Loads an ONNX or Torch model (if available) from env-provided paths.
- Provides a simple detect(image: np.ndarray) -> List[dict] API.
- Falls back to a lightweight OpenCV heuristic if no model/runtime is available.

Env:
- MODEL_REGISTRY (default /models)
- FUSION_MODEL_PATH (optional absolute path; if empty, uses MODEL_REGISTRY/default_vision.onnx)
- FUSION_CONF_THRESHOLD (default 0.6)
"""

from __future__ import annotations
import os
from typing import List, Dict, Any, Optional

import numpy as np

# Optional backends
try:
    import onnxruntime as ort  # type: ignore
    ORT_AVAILABLE = True
except Exception:
    ORT_AVAILABLE = False

try:
    import torch  # type: ignore
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

import cv2  # OpenCV is in fusion requirements


class VisionInference:
    def __init__(self,
                 model_path: Optional[str] = None,
                 conf_threshold: float = 0.6,
                 labels: Optional[List[str]] = None):
        self.conf_threshold = conf_threshold
        self.labels = labels or ["object"]
        self.model_path = model_path or os.path.join(
            os.getenv("MODEL_REGISTRY", "/models"), "default_vision.onnx"
        )
        self.backend = None
        self.session = None
        self.torch_model = None

        # Try ONNX first
        if ORT_AVAILABLE and os.path.exists(self.model_path) and self.model_path.endswith(".onnx"):
            try:
                self.session = ort.InferenceSession(self.model_path, providers=["CPUExecutionProvider"])  # type: ignore
                self.backend = "onnx"
            except Exception:
                self.session = None
                self.backend = None

        # Optionally try torch if model_path points to a torch model
        if self.backend is None and TORCH_AVAILABLE and os.path.exists(self.model_path) and self.model_path.endswith(('.pt', '.pth')):
            try:
                self.torch_model = torch.jit.load(self.model_path, map_location="cpu")  # type: ignore
                self.torch_model.eval()
                self.backend = "torch"
            except Exception:
                self.torch_model = None
                self.backend = None

    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        if self.backend == "onnx" and self.session is not None:
            return self._detect_onnx(image)
        if self.backend == "torch" and self.torch_model is not None:
            return self._detect_torch(image)
        # Fallback heuristic
        return self._detect_heuristic(image)

    def _preprocess(self, image: np.ndarray, size=(320, 320)) -> np.ndarray:
        img = cv2.resize(image, size)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # CHW
        img = np.expand_dims(img, 0)
        return img

    def _detect_onnx(self, image: np.ndarray) -> List[Dict[str, Any]]:
        try:
            inp = self._preprocess(image)
            inputs = {self.session.get_inputs()[0].name: inp}
            outputs = self.session.run(None, inputs)
            return self._postprocess(outputs)
        except Exception:
            return []

    def _detect_torch(self, image: np.ndarray) -> List[Dict[str, Any]]:
        try:
            inp = self._preprocess(image)
            with torch.no_grad():  # type: ignore
                out = self.torch_model(torch.from_numpy(inp))  # type: ignore
            return self._postprocess([out.cpu().numpy()])
        except Exception:
            return []

    def _postprocess(self, outputs: List[np.ndarray]) -> List[Dict[str, Any]]:
        # Expecting a generic YOLO-like output or adapt your own.
        # For safety, treat any output as empty without a specific format.
        # Users should replace with their postprocess matching their model.
        detections: List[Dict[str, Any]] = []
        # Placeholder: no-op; encourage users to implement per-model mapping.
        return detections

    def _detect_heuristic(self, image: np.ndarray) -> List[Dict[str, Any]]:
        # Very simple motion/contrast heuristic to produce low-confidence detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections: List[Dict[str, Any]] = []
        h, w = gray.shape[:2]
        for c in cnts[:3]:
            x, y, bw, bh = cv2.boundingRect(c)
            if bw * bh < 100:  # filter tiny
                continue
            cx = (x + x + bw) / 2.0 / w
            cy = (y + y + bh) / 2.0 / h
            detections.append({
                "class": self.labels[0],
                "confidence": 0.3,
                "bbox": [float(x), float(y), float(bw), float(bh)],
                "center_norm": [float(cx), float(cy)]
            })
        return detections
