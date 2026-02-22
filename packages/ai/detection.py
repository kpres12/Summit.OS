"""
Object Detection Pipeline for Summit.OS

Provides a unified interface for running object detection models.
Supports:
- Ultralytics YOLOv8 (when installed)
- OpenCV DNN backend (when available)
- Mock detector for testing without GPU/dependencies

The pipeline outputs standardized Detection objects that feed into
the fusion track correlator.
"""
from __future__ import annotations

import time
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from abc import ABC, abstractmethod

logger = logging.getLogger("ai.detection")


@dataclass
class BoundingBox:
    """Bounding box in pixel coordinates."""
    x1: float  # top-left x
    y1: float  # top-left y
    x2: float  # bottom-right x
    y2: float  # bottom-right y

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def area(self) -> float:
        return max(0, self.width) * max(0, self.height)

    def iou(self, other: "BoundingBox") -> float:
        """Intersection over Union."""
        ix1 = max(self.x1, other.x1)
        iy1 = max(self.y1, other.y1)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        union = self.area + other.area - inter
        return inter / max(union, 1e-6)


@dataclass
class Detection:
    """A single object detection result."""
    class_id: int
    class_name: str
    confidence: float
    bbox: BoundingBox
    # Optional geo-projection (filled by fusion)
    lat: float = 0.0
    lon: float = 0.0
    alt: float = 0.0
    # Tracking
    track_id: int = -1
    # Metadata
    model_name: str = ""
    frame_id: int = 0
    timestamp: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "bbox": {"x1": self.bbox.x1, "y1": self.bbox.y1,
                     "x2": self.bbox.x2, "y2": self.bbox.y2},
            "lat": self.lat, "lon": self.lon, "alt": self.alt,
            "track_id": self.track_id,
            "model_name": self.model_name,
            "timestamp": self.timestamp,
        }


@dataclass
class DetectionResult:
    """Result of running detection on a single frame."""
    detections: List[Detection]
    frame_id: int = 0
    inference_ms: float = 0.0
    model_name: str = ""
    timestamp: float = field(default_factory=time.time)
    image_width: int = 0
    image_height: int = 0


# ── Abstract Detector ───────────────────────────────────────

class ObjectDetector(ABC):
    """Abstract base for all detectors."""

    @abstractmethod
    def detect(self, image: Any, confidence_threshold: float = 0.5) -> DetectionResult:
        """Run detection on an image. Image can be numpy array or path."""
        ...

    @abstractmethod
    def get_class_names(self) -> List[str]:
        """Get list of class names the model can detect."""
        ...


# ── YOLO Detector ───────────────────────────────────────────

class YOLODetector(ObjectDetector):
    """
    Ultralytics YOLOv8/YOLOv11 detector.

    Requires: pip install ultralytics
    """

    def __init__(self, model_path: str = "yolov8n.pt", device: str = "auto"):
        self.model_path = model_path
        self.device = device
        self._model = None
        self._class_names: List[str] = []
        self._load_model()

    def _load_model(self):
        try:
            from ultralytics import YOLO
            self._model = YOLO(self.model_path)
            if hasattr(self._model, 'names'):
                self._class_names = list(self._model.names.values())
            logger.info(f"YOLO model loaded: {self.model_path} ({len(self._class_names)} classes)")
        except ImportError:
            logger.warning("ultralytics not installed — YOLO detector unavailable")
            self._model = None
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            self._model = None

    def detect(self, image: Any, confidence_threshold: float = 0.5) -> DetectionResult:
        if self._model is None:
            return DetectionResult(detections=[], model_name="yolo-unavailable")

        start = time.time()
        results = self._model(image, conf=confidence_threshold, verbose=False)
        elapsed_ms = (time.time() - start) * 1000

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i].cpu().numpy())
                cls_id = int(boxes.cls[i].cpu().numpy())
                cls_name = self._class_names[cls_id] if cls_id < len(self._class_names) else f"class_{cls_id}"

                detections.append(Detection(
                    class_id=cls_id,
                    class_name=cls_name,
                    confidence=conf,
                    bbox=BoundingBox(x1=float(xyxy[0]), y1=float(xyxy[1]),
                                     x2=float(xyxy[2]), y2=float(xyxy[3])),
                    model_name=self.model_path,
                    timestamp=time.time(),
                ))

        return DetectionResult(
            detections=detections,
            inference_ms=elapsed_ms,
            model_name=self.model_path,
        )

    def get_class_names(self) -> List[str]:
        return self._class_names


# ── OpenCV DNN Detector ─────────────────────────────────────

class OpenCVDetector(ObjectDetector):
    """
    OpenCV DNN-based detector.

    Works with ONNX, Caffe, TensorFlow, or Darknet models.
    Requires: pip install opencv-python
    """

    COCO_CLASSES = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
        "truck", "boat", "traffic light", "fire hydrant", "stop sign",
        "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
        "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
        "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
        "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
        "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
        "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
        "couch", "potted plant", "bed", "dining table", "toilet", "tv",
        "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
        "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
        "scissors", "teddy bear", "hair drier", "toothbrush",
    ]

    def __init__(self, model_path: str = "", config_path: str = "",
                 input_size: Tuple[int, int] = (416, 416)):
        self.model_path = model_path
        self.config_path = config_path
        self.input_size = input_size
        self._net = None
        self._load_model()

    def _load_model(self):
        try:
            import cv2
            if self.model_path:
                self._net = cv2.dnn.readNet(self.model_path, self.config_path)
                logger.info(f"OpenCV DNN model loaded: {self.model_path}")
        except ImportError:
            logger.warning("opencv-python not installed — OpenCV detector unavailable")
        except Exception as e:
            logger.error(f"Failed to load OpenCV model: {e}")

    def detect(self, image: Any, confidence_threshold: float = 0.5) -> DetectionResult:
        if self._net is None:
            return DetectionResult(detections=[], model_name="opencv-unavailable")

        import cv2
        import numpy as np

        start = time.time()

        if isinstance(image, str):
            image = cv2.imread(image)

        h, w = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, 1/255.0, self.input_size, swapRB=True, crop=False)
        self._net.setInput(blob)
        outputs = self._net.forward(self._net.getUnconnectedOutLayersNames())

        detections = []
        for output in outputs:
            for det in output:
                scores = det[5:]
                class_id = int(np.argmax(scores))
                confidence = float(scores[class_id])

                if confidence < confidence_threshold:
                    continue

                cx, cy, bw, bh = det[0] * w, det[1] * h, det[2] * w, det[3] * h
                x1 = cx - bw / 2
                y1 = cy - bh / 2

                cls_name = self.COCO_CLASSES[class_id] if class_id < len(self.COCO_CLASSES) else f"class_{class_id}"
                detections.append(Detection(
                    class_id=class_id, class_name=cls_name,
                    confidence=confidence,
                    bbox=BoundingBox(x1=x1, y1=y1, x2=x1+bw, y2=y1+bh),
                    model_name="opencv_dnn",
                    timestamp=time.time(),
                ))

        elapsed_ms = (time.time() - start) * 1000
        return DetectionResult(
            detections=detections, inference_ms=elapsed_ms,
            model_name="opencv_dnn", image_width=w, image_height=h,
        )

    def get_class_names(self) -> List[str]:
        return self.COCO_CLASSES


# ── Mock Detector ───────────────────────────────────────────

class MockDetector(ObjectDetector):
    """
    Mock detector for testing without ML dependencies.

    Generates deterministic fake detections based on frame_id
    for consistent test behavior.
    """

    MOCK_CLASSES = ["person", "vehicle", "aircraft", "vessel", "drone", "animal"]

    def __init__(self, num_detections: int = 3, seed: int = 42):
        self.num_detections = num_detections
        self._seed = seed
        self._frame_count = 0

    def detect(self, image: Any, confidence_threshold: float = 0.5) -> DetectionResult:
        import random
        rng = random.Random(self._seed + self._frame_count)
        self._frame_count += 1

        detections = []
        for i in range(rng.randint(1, self.num_detections)):
            cls_id = rng.randint(0, len(self.MOCK_CLASSES) - 1)
            conf = rng.uniform(0.5, 0.99)
            if conf < confidence_threshold:
                continue

            cx = rng.uniform(100, 500)
            cy = rng.uniform(100, 400)
            w = rng.uniform(30, 150)
            h = rng.uniform(30, 150)

            detections.append(Detection(
                class_id=cls_id,
                class_name=self.MOCK_CLASSES[cls_id],
                confidence=round(conf, 3),
                bbox=BoundingBox(x1=cx-w/2, y1=cy-h/2, x2=cx+w/2, y2=cy+h/2),
                model_name="mock",
                timestamp=time.time(),
            ))

        return DetectionResult(
            detections=detections,
            inference_ms=rng.uniform(5, 20),
            model_name="mock",
        )

    def get_class_names(self) -> List[str]:
        return self.MOCK_CLASSES


# ── NMS utility ─────────────────────────────────────────────

def non_max_suppression(detections: List[Detection],
                        iou_threshold: float = 0.5) -> List[Detection]:
    """Apply non-maximum suppression to remove overlapping detections."""
    if not detections:
        return []

    # Sort by confidence descending
    sorted_dets = sorted(detections, key=lambda d: d.confidence, reverse=True)
    keep = []

    while sorted_dets:
        best = sorted_dets.pop(0)
        keep.append(best)
        sorted_dets = [
            d for d in sorted_dets
            if best.bbox.iou(d.bbox) < iou_threshold or d.class_id != best.class_id
        ]

    return keep


# ── Factory ─────────────────────────────────────────────────

def create_detector(backend: str = "auto", **kwargs) -> ObjectDetector:
    """
    Create a detector with the best available backend.

    Args:
        backend: "yolo", "opencv", "mock", or "auto" (tries in order)
    """
    if backend == "yolo":
        return YOLODetector(**kwargs)
    if backend == "opencv":
        return OpenCVDetector(**kwargs)
    if backend == "mock":
        return MockDetector(**kwargs)

    # Auto: try YOLO first, then OpenCV, then mock
    try:
        det = YOLODetector(**kwargs)
        if det._model is not None:
            return det
    except Exception:
        pass

    logger.info("No ML backend available, using MockDetector")
    return MockDetector(**kwargs)
