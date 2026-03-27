"""
Summit.OS AI/ML Pipeline

Provides detection, classification, anomaly detection, and intent prediction.
All modules gracefully degrade when ML dependencies are absent.
"""

from .detection import (
    Detection,
    DetectionResult,
    BoundingBox,
    ObjectDetector,
    YOLODetector,
    OpenCVDetector,
    MockDetector,
    non_max_suppression,
    create_detector,
)
from .classification import (
    EntityClassifier,
    BayesianClassifier,
    RuleBasedClassifier,
    ClassificationResult,
)
from .anomaly import (
    AnomalyDetector,
    ZScoreDetector,
    MovingAverageDetector,
    IsolationForestDetector,
    AnomalyResult,
)
from .intent import (
    IntentPredictor,
    TrajectoryPredictor,
    BehaviorAnalyzer,
    IntentResult,
)

__all__ = [
    "Detection",
    "DetectionResult",
    "BoundingBox",
    "ObjectDetector",
    "YOLODetector",
    "OpenCVDetector",
    "MockDetector",
    "non_max_suppression",
    "create_detector",
    "EntityClassifier",
    "BayesianClassifier",
    "RuleBasedClassifier",
    "ClassificationResult",
    "AnomalyDetector",
    "ZScoreDetector",
    "MovingAverageDetector",
    "IsolationForestDetector",
    "AnomalyResult",
    "IntentPredictor",
    "TrajectoryPredictor",
    "BehaviorAnalyzer",
    "IntentResult",
]
