"""
Edge Inference Runtime for Heli.OS

Lightweight ONNX-based inference for edge devices (towers, drones, UGVs)
that may be offline or bandwidth-constrained. Falls back to mock
detection when ONNX Runtime is not available.

Usage:
    detector = EdgeDetector(model_path="model.onnx")
    result = detector.detect(image_array)

For fusion service integration:
    from packages.ai.edge_inference import InferenceClient
    client = InferenceClient()  # calls shared inference service
    result = await client.detect(image_b64)  # remote
    result = client.detect_local(image_array)  # local fallback
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from packages.ai.detection import (
    ObjectDetector,
    DetectionResult,
    Detection,
    BoundingBox,
    MockDetector,
    non_max_suppression,
)

logger = logging.getLogger("ai.edge")


# ── ONNX Edge Detector ────────────────────────────────────


class ONNXDetector(ObjectDetector):
    """
    ONNX Runtime-based detector for edge deployment.

    Runs YOLO-exported ONNX models on CPU or GPU without
    needing the full ultralytics/PyTorch stack.
    """

    COCO_CLASSES = [
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic light",
        "fire hydrant",
        "stop sign",
        "parking meter",
        "bench",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "backpack",
        "umbrella",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
        "bottle",
        "wine glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "chair",
        "couch",
        "potted plant",
        "bed",
        "dining table",
        "toilet",
        "tv",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
        "hair drier",
        "toothbrush",
    ]

    def __init__(
        self,
        model_path: str = "yolov8n.onnx",
        input_size: Tuple[int, int] = (640, 640),
        providers: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None,
    ):
        self.model_path = model_path
        self.input_size = input_size
        self._session = None
        self._class_names = class_names or self.COCO_CLASSES
        self._providers = providers or ["CPUExecutionProvider"]
        self._load_model()

    def _load_model(self):
        try:
            import onnxruntime as ort

            self._session = ort.InferenceSession(
                self.model_path, providers=self._providers
            )
            logger.info(
                f"ONNX model loaded: {self.model_path} "
                f"(providers={self._providers})"
            )
        except ImportError:
            logger.warning("onnxruntime not installed — ONNX detector unavailable")
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}")

    def detect(self, image: Any, confidence_threshold: float = 0.5) -> DetectionResult:
        if self._session is None:
            return DetectionResult(detections=[], model_name="onnx-unavailable")

        import numpy as np

        start = time.time()

        # Preprocess
        if isinstance(image, str):
            try:
                import cv2

                image = cv2.imread(image)
            except ImportError:
                return DetectionResult(detections=[], model_name="onnx-no-cv2")

        orig_h, orig_w = image.shape[:2]
        input_tensor = self._preprocess(image)

        # Run inference
        input_name = self._session.get_inputs()[0].name
        outputs = self._session.run(None, {input_name: input_tensor})
        elapsed_ms = (time.time() - start) * 1000

        # Post-process (YOLO output format: [batch, num_detections, 4+num_classes])
        detections = self._postprocess(outputs, orig_w, orig_h, confidence_threshold)

        return DetectionResult(
            detections=detections,
            inference_ms=elapsed_ms,
            model_name=f"onnx:{self.model_path}",
            image_width=orig_w,
            image_height=orig_h,
        )

    def _preprocess(self, image) -> Any:
        """Preprocess image for ONNX model input."""
        import numpy as np

        try:
            import cv2

            resized = cv2.resize(image, self.input_size)
        except ImportError:
            # Basic resize with numpy
            resized = image

        # HWC -> CHW, normalize to 0-1, add batch dimension
        blob = resized.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))
        blob = np.expand_dims(blob, 0)
        return blob

    def _postprocess(
        self,
        outputs: List,
        orig_w: int,
        orig_h: int,
        conf_thresh: float,
    ) -> List[Detection]:
        """Post-process ONNX model outputs to Detection objects."""
        import numpy as np

        detections = []

        if not outputs or len(outputs) == 0:
            return detections

        output = outputs[0]  # [batch, ...]

        # Handle YOLOv8 output format: [1, 84, 8400] (transposed)
        if len(output.shape) == 3 and output.shape[1] < output.shape[2]:
            output = np.transpose(output, (0, 2, 1))  # → [1, 8400, 84]

        if len(output.shape) == 3:
            output = output[0]  # Remove batch dimension → [8400, 84]

        num_classes = output.shape[1] - 4

        for i in range(output.shape[0]):
            row = output[i]
            cx, cy, w, h = row[0], row[1], row[2], row[3]
            class_scores = row[4:]

            max_score = float(np.max(class_scores))
            if max_score < conf_thresh:
                continue

            class_id = int(np.argmax(class_scores))
            class_name = (
                self._class_names[class_id]
                if class_id < len(self._class_names)
                else f"class_{class_id}"
            )

            # Scale to original image
            scale_x = orig_w / self.input_size[0]
            scale_y = orig_h / self.input_size[1]

            x1 = (cx - w / 2) * scale_x
            y1 = (cy - h / 2) * scale_y
            x2 = (cx + w / 2) * scale_x
            y2 = (cy + h / 2) * scale_y

            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=class_name,
                    confidence=max_score,
                    bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                    model_name=f"onnx:{self.model_path}",
                    timestamp=time.time(),
                )
            )

        # NMS
        return non_max_suppression(detections)

    def get_class_names(self) -> List[str]:
        return self._class_names


# ── Edge Detector Factory ─────────────────────────────────


class EdgeDetector:
    """
    Factory that creates the best available edge detector.

    Priority: ONNX → Mock fallback
    """

    def __init__(
        self,
        model_path: str = "yolov8n.onnx",
        **kwargs,
    ):
        try:
            self._detector = ONNXDetector(model_path=model_path, **kwargs)
            if self._detector._session is not None:
                self.backend = "onnx"
                return
        except Exception:
            pass

        logger.info("ONNX unavailable, using MockDetector for edge inference")
        self._detector = MockDetector()
        self.backend = "mock"

    def detect(self, image: Any, confidence_threshold: float = 0.5) -> DetectionResult:
        return self._detector.detect(image, confidence_threshold)

    def get_class_names(self) -> List[str]:
        return self._detector.get_class_names()


# ── Inference Client (remote + local fallback) ────────────


class InferenceClient:
    """
    Client that calls the shared inference service (port 8005).
    Falls back to local edge inference if the service is unreachable.
    """

    def __init__(
        self,
        service_url: str = "http://localhost:8005",
        timeout: float = 5.0,
        local_model_path: str = "yolov8n.onnx",
    ):
        self.service_url = service_url.rstrip("/")
        self.timeout = timeout
        self._local_detector: Optional[EdgeDetector] = None
        self._local_model_path = local_model_path

    async def detect(
        self,
        image_b64: str,
        confidence_threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Call the shared inference service. Returns the response dict.
        Falls back to local inference if the service is unreachable.
        """
        try:
            import httpx

            async with httpx.AsyncClient(timeout=self.timeout) as client:
                r = await client.post(
                    f"{self.service_url}/detect",
                    json={
                        "image_b64": image_b64,
                        "confidence_threshold": confidence_threshold,
                    },
                )
                r.raise_for_status()
                return r.json()
        except Exception as e:
            logger.warning(f"Inference service unreachable ({e}), using local fallback")
            return self._detect_local_b64(image_b64, confidence_threshold)

    def detect_local(
        self, image: Any, confidence_threshold: float = 0.5
    ) -> DetectionResult:
        """Run inference locally using edge detector."""
        if self._local_detector is None:
            self._local_detector = EdgeDetector(model_path=self._local_model_path)
        return self._local_detector.detect(image, confidence_threshold)

    def _detect_local_b64(
        self, image_b64: str, confidence_threshold: float
    ) -> Dict[str, Any]:
        """Decode base64 and run local detection, returning API-compatible dict."""
        import base64

        if self._local_detector is None:
            self._local_detector = EdgeDetector(model_path=self._local_model_path)

        # Decode image
        image_bytes = base64.b64decode(image_b64)
        try:
            import numpy as np

            try:
                import cv2

                arr = np.frombuffer(image_bytes, dtype=np.uint8)
                image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            except ImportError:
                image = image_bytes
        except ImportError:
            image = image_bytes

        result = self._local_detector.detect(image, confidence_threshold)

        return {
            "detections": [d.to_dict() for d in result.detections],
            "count": len(result.detections),
            "inference_ms": result.inference_ms,
            "model_name": result.model_name,
            "fallback": True,
        }

    async def health(self) -> Dict[str, Any]:
        """Check inference service health."""
        try:
            import httpx

            async with httpx.AsyncClient(timeout=2.0) as client:
                r = await client.get(f"{self.service_url}/health")
                r.raise_for_status()
                return r.json()
        except Exception:
            return {"status": "unreachable", "fallback": "local"}
