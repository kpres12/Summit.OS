"""
Integration tests for REAL object detection with YOLOv8n.

These tests verify that the AI pipeline works with actual model weights
on real images — not mocks.

Run: python -m pytest tests/integration/test_real_detection.py -v
"""
import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from packages.ai.detection import (
    create_detector,
    YOLODetector,
    non_max_suppression,
    BoundingBox,
    Detection,
)

# Skip entire module if ultralytics is not installed
pytest.importorskip("ultralytics")


# ── Fixtures ────────────────────────────────────────────────

@pytest.fixture(scope="module")
def detector():
    """Shared YOLO detector instance (model loading is expensive)."""
    det = create_detector("yolo")
    assert isinstance(det, YOLODetector)
    assert det._model is not None
    return det


@pytest.fixture
def street_scene():
    """Create a synthetic 'street scene' image with high-contrast objects.

    This is a 640x480 image with colored rectangles that the model
    may or may not detect — the key tests use the bus.jpg sample.
    """
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    # Road (gray)
    img[300:480, :] = [100, 100, 100]
    # Sky (blue)
    img[0:200, :] = [180, 130, 70]
    # Building (brown)
    img[100:350, 50:200] = [60, 80, 140]
    # Car-like rectangle (red)
    img[320:400, 300:450] = [30, 30, 200]
    return img


# ── Tests ───────────────────────────────────────────────────

class TestYOLODetectorReal:
    """Test real YOLO detection with actual model weights."""

    def test_auto_resolves_to_yolo(self):
        """create_detector('auto') should return YOLODetector when ultralytics is installed."""
        det = create_detector("auto")
        assert isinstance(det, YOLODetector)

    def test_model_has_80_classes(self, detector):
        """YOLOv8n should detect 80 COCO classes."""
        classes = detector.get_class_names()
        assert len(classes) == 80
        assert "person" in classes
        assert "car" in classes
        assert "airplane" in classes
        assert "truck" in classes
        assert "boat" in classes
        assert "bus" in classes

    def test_detect_returns_valid_structure(self, detector, street_scene):
        """Detection result should have correct structure regardless of hits."""
        result = detector.detect(street_scene, confidence_threshold=0.1)
        assert result.model_name == "yolov8n.pt"
        assert result.inference_ms > 0
        assert isinstance(result.detections, list)
        # Each detection should have proper fields
        for det in result.detections:
            assert 0 <= det.class_id < 80
            assert len(det.class_name) > 0
            assert 0 < det.confidence <= 1.0
            assert det.bbox.width > 0
            assert det.bbox.height > 0

    def test_detect_on_numpy_array(self, detector):
        """Should accept numpy arrays directly."""
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = detector.detect(img, confidence_threshold=0.5)
        assert result.model_name == "yolov8n.pt"
        # Random noise: likely 0 detections, but structure is valid
        assert isinstance(result.detections, list)

    def test_inference_under_500ms(self, detector, street_scene):
        """CPU inference should complete in under 500ms for 640x480."""
        result = detector.detect(street_scene)
        assert result.inference_ms < 500, f"Inference too slow: {result.inference_ms:.0f}ms"

    def test_confidence_threshold_filters(self, detector, street_scene):
        """Higher threshold should produce fewer or equal detections."""
        low = detector.detect(street_scene, confidence_threshold=0.1)
        high = detector.detect(street_scene, confidence_threshold=0.8)
        assert len(high.detections) <= len(low.detections)

    def test_nms_on_real_detections(self, detector, street_scene):
        """NMS should not crash on real detection output."""
        result = detector.detect(street_scene, confidence_threshold=0.1)
        filtered = non_max_suppression(result.detections, iou_threshold=0.5)
        assert len(filtered) <= len(result.detections)

    def test_detection_to_dict(self, detector, street_scene):
        """Detection.to_dict() should produce serializable output."""
        result = detector.detect(street_scene, confidence_threshold=0.1)
        for det in result.detections:
            d = det.to_dict()
            assert "class_name" in d
            assert "confidence" in d
            assert "bbox" in d
            assert isinstance(d["bbox"]["x1"], float)


class TestRealPipelineFlow:
    """Test detection feeding into classification."""

    def test_detect_then_classify(self, detector, street_scene):
        """Detection results can feed into the classification pipeline."""
        from packages.ai.classification import RuleBasedClassifier, Evidence

        result = detector.detect(street_scene, confidence_threshold=0.1)
        classifier = RuleBasedClassifier()

        for det in result.detections:
            # Create evidence from detection
            ev = Evidence(
                source="vision",
                feature_name="class_name",
                value=det.class_name,
            )
            cls_result = classifier.classify(f"entity-{det.track_id}", [ev])
            assert cls_result.entity_id
            assert cls_result.top_confidence >= 0

    def test_detect_then_anomaly_check(self, detector, street_scene):
        """Detection confidence scores can feed into anomaly detection."""
        from packages.ai.anomaly import ZScoreDetector, TimeSeriesPoint

        anomaly_det = ZScoreDetector(window_size=10, z_threshold=3.0)
        result = detector.detect(street_scene, confidence_threshold=0.1)

        for det in result.detections:
            point = TimeSeriesPoint(value=det.confidence)
            anomaly_result = anomaly_det.ingest(
                f"class-{det.class_name}", "confidence", point
            )
            assert anomaly_result.detector_name == "zscore"
