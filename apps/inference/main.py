"""
Summit.OS Shared Inference Service

Centralized AI inference endpoint. All services (fusion, intelligence, console)
call this instead of running their own models. Supports hot-swapping models
and batched requests.

Port: 8005
Endpoints:
  POST /detect          — Run object detection on an image
  POST /classify        — Classify a single crop/image
  GET  /models          — List available models
  POST /models/select   — Hot-swap the active model
  GET  /health          — Health check
"""
from __future__ import annotations

import asyncio
import base64
import io
import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Import detection pipeline
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from packages.ai.detection import (
    ObjectDetector,
    DetectionResult,
    Detection,
    create_detector,
    non_max_suppression,
)

logger = logging.getLogger("inference")
logging.basicConfig(level=logging.INFO)

# ── Globals ────────────────────────────────────────────────

active_detector: Optional[ObjectDetector] = None
active_model_name: str = "auto"
available_models: Dict[str, Dict[str, Any]] = {}

# Stats
inference_stats = {
    "total_requests": 0,
    "total_detections": 0,
    "avg_latency_ms": 0.0,
    "last_inference": None,
}


# ── Models ─────────────────────────────────────────────────

class DetectRequest(BaseModel):
    image_b64: Optional[str] = None  # Base64 encoded image
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    nms_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    max_detections: int = Field(default=100, ge=1)


class DetectionOut(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    bbox: Dict[str, float]  # x1, y1, x2, y2
    lat: float = 0.0
    lon: float = 0.0


class DetectResponse(BaseModel):
    detections: List[DetectionOut]
    count: int
    inference_ms: float
    model_name: str
    timestamp: str


class ClassifyRequest(BaseModel):
    image_b64: str
    top_k: int = Field(default=5, ge=1)


class ClassifyResponse(BaseModel):
    predictions: List[Dict[str, Any]]
    model_name: str
    inference_ms: float


class ModelInfo(BaseModel):
    name: str
    backend: str
    classes: int
    active: bool


class ModelSelectRequest(BaseModel):
    backend: str = "auto"  # "yolo", "opencv", "mock", "auto"
    model_path: Optional[str] = None


# ── Lifespan ───────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global active_detector, active_model_name, available_models

    logger.info("Starting Summit.OS Inference Service")

    # Discover available backends
    available_models = _discover_models()

    # Initialize default detector
    backend = os.getenv("INFERENCE_BACKEND", "auto")
    model_path = os.getenv("INFERENCE_MODEL_PATH", "")
    kwargs = {"model_path": model_path} if model_path else {}
    active_detector = create_detector(backend=backend, **kwargs)
    active_model_name = backend

    logger.info(f"Inference ready: backend={backend}, models={list(available_models.keys())}")

    yield

    logger.info("Shutting down Inference Service")


app = FastAPI(
    title="Summit.OS Inference Service",
    description="Centralized AI detection and classification",
    version="1.0.0",
    lifespan=lifespan,
)

# ── OpenTelemetry tracing middleware ──────────────────────────────────────────
try:
    import sys as _sys_otel, os as _os_otel
    _otel_root = _os_otel.path.join(_os_otel.path.dirname(__file__), "../..")
    if _otel_root not in _sys_otel.path:
        _sys_otel.path.insert(0, _otel_root)
    from packages.observability.tracing import get_tracer, create_tracing_middleware
    _tracer = get_tracer("summit-inference")
    app.middleware("http")(create_tracing_middleware(_tracer))
except Exception as _e:
    pass  # inference service runs without tracing if OTel unavailable

CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in CORS_ORIGINS if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Endpoints ──────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "inference",
        "model": active_model_name,
        "stats": inference_stats,
    }


@app.post("/detect", response_model=DetectResponse)
async def detect(req: DetectRequest):
    """Run object detection on a base64-encoded image."""
    if active_detector is None:
        raise HTTPException(status_code=503, detail="No detector initialized")

    if not req.image_b64:
        raise HTTPException(status_code=400, detail="image_b64 required")

    # Decode image
    try:
        image = _decode_image(req.image_b64)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    # Run detection
    result = active_detector.detect(image, confidence_threshold=req.confidence_threshold)

    # Apply NMS
    filtered = non_max_suppression(result.detections, iou_threshold=req.nms_threshold)
    filtered = filtered[:req.max_detections]

    # Update stats
    inference_stats["total_requests"] += 1
    inference_stats["total_detections"] += len(filtered)
    inference_stats["last_inference"] = datetime.now(timezone.utc).isoformat()
    _update_avg_latency(result.inference_ms)

    detections_out = [
        DetectionOut(
            class_id=d.class_id,
            class_name=d.class_name,
            confidence=round(d.confidence, 4),
            bbox={"x1": d.bbox.x1, "y1": d.bbox.y1, "x2": d.bbox.x2, "y2": d.bbox.y2},
            lat=d.lat,
            lon=d.lon,
        )
        for d in filtered
    ]

    return DetectResponse(
        detections=detections_out,
        count=len(detections_out),
        inference_ms=round(result.inference_ms, 2),
        model_name=result.model_name,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.post("/detect/upload", response_model=DetectResponse)
async def detect_upload(
    file: UploadFile = File(...),
    confidence_threshold: float = 0.5,
    nms_threshold: float = 0.5,
):
    """Run detection on an uploaded image file."""
    if active_detector is None:
        raise HTTPException(status_code=503, detail="No detector initialized")

    content = await file.read()
    image = _bytes_to_image(content)

    result = active_detector.detect(image, confidence_threshold=confidence_threshold)
    filtered = non_max_suppression(result.detections, iou_threshold=nms_threshold)

    inference_stats["total_requests"] += 1
    inference_stats["total_detections"] += len(filtered)
    _update_avg_latency(result.inference_ms)

    detections_out = [
        DetectionOut(
            class_id=d.class_id,
            class_name=d.class_name,
            confidence=round(d.confidence, 4),
            bbox={"x1": d.bbox.x1, "y1": d.bbox.y1, "x2": d.bbox.x2, "y2": d.bbox.y2},
        )
        for d in filtered
    ]

    return DetectResponse(
        detections=detections_out,
        count=len(detections_out),
        inference_ms=round(result.inference_ms, 2),
        model_name=result.model_name,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.post("/classify", response_model=ClassifyResponse)
async def classify(req: ClassifyRequest):
    """Classify an image crop. Uses the detector and returns top-K class scores."""
    if active_detector is None:
        raise HTTPException(status_code=503, detail="No detector initialized")

    try:
        image = _decode_image(req.image_b64)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    start = time.time()
    result = active_detector.detect(image, confidence_threshold=0.1)
    elapsed_ms = (time.time() - start) * 1000

    # Aggregate by class name
    class_scores: Dict[str, float] = {}
    for d in result.detections:
        if d.class_name not in class_scores or d.confidence > class_scores[d.class_name]:
            class_scores[d.class_name] = d.confidence

    # Sort and take top-K
    sorted_classes = sorted(class_scores.items(), key=lambda x: x[1], reverse=True)
    predictions = [
        {"class_name": name, "confidence": round(conf, 4)}
        for name, conf in sorted_classes[:req.top_k]
    ]

    return ClassifyResponse(
        predictions=predictions,
        model_name=result.model_name,
        inference_ms=round(elapsed_ms, 2),
    )


@app.get("/models", response_model=List[ModelInfo])
async def list_models():
    """List available detection models."""
    return [
        ModelInfo(
            name=name,
            backend=info["backend"],
            classes=info.get("classes", 0),
            active=(name == active_model_name),
        )
        for name, info in available_models.items()
    ]


@app.post("/models/select")
async def select_model(req: ModelSelectRequest):
    """Hot-swap the active detection model."""
    global active_detector, active_model_name

    kwargs = {}
    if req.model_path:
        kwargs["model_path"] = req.model_path

    try:
        active_detector = create_detector(backend=req.backend, **kwargs)
        active_model_name = req.backend
        logger.info(f"Model switched to: {req.backend}")
        return {"status": "ok", "model": req.backend}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load model: {e}")


# ── Helpers ────────────────────────────────────────────────

def _decode_image(b64_str: str):
    """Decode a base64 image string to a numpy array."""
    try:
        import numpy as np
        image_bytes = base64.b64decode(b64_str)
        # Try with OpenCV first
        try:
            import cv2
            arr = np.frombuffer(image_bytes, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is not None:
                return img
        except ImportError:
            pass
        # Fallback: PIL
        try:
            from PIL import Image
            img = Image.open(io.BytesIO(image_bytes))
            return np.array(img)
        except ImportError:
            pass
    except Exception:
        pass
    # Last resort: return raw bytes for mock detector
    return base64.b64decode(b64_str)


def _bytes_to_image(data: bytes):
    """Convert raw bytes to image array."""
    try:
        import numpy as np
        try:
            import cv2
            arr = np.frombuffer(data, dtype=np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except ImportError:
            pass
        try:
            from PIL import Image
            return np.array(Image.open(io.BytesIO(data)))
        except ImportError:
            pass
    except Exception:
        pass
    return data


def _discover_models() -> Dict[str, Dict[str, Any]]:
    """Discover available model backends."""
    models = {}

    # Always available
    models["mock"] = {"backend": "mock", "classes": 6}

    # Check YOLO
    try:
        from ultralytics import YOLO
        models["yolo"] = {"backend": "yolo", "classes": 80}
    except ImportError:
        pass

    # Check OpenCV DNN
    try:
        import cv2
        if hasattr(cv2, "dnn"):
            models["opencv"] = {"backend": "opencv", "classes": 80}
    except ImportError:
        pass

    # Check ONNX Runtime (for edge inference)
    try:
        import onnxruntime
        models["onnx"] = {"backend": "onnx", "classes": 0}
    except ImportError:
        pass

    return models


def _update_avg_latency(new_ms: float):
    """Rolling average of inference latency."""
    n = inference_stats["total_requests"]
    if n <= 1:
        inference_stats["avg_latency_ms"] = new_ms
    else:
        inference_stats["avg_latency_ms"] = (
            inference_stats["avg_latency_ms"] * (n - 1) + new_ms
        ) / n


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8005,
        reload=True,
        log_level="info",
    )
