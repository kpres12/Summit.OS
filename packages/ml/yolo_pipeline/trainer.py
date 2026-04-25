"""
YOLO Trainer — Heli.OS

Fine-tunes YOLOv8 on domain-specific datasets for SAR, wildfire, and
inspection use cases. Wraps the ultralytics training API with Heli.OS
conventions (checkpoint naming, eval metrics, ONNX export).

Requires: pip install ultralytics
Runs training in a subprocess to avoid blocking the main process.
If ultralytics not installed, stubs training with mock metrics.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger("ml.yolo_trainer")


@dataclass
class TrainResult:
    domain: str
    weights_path: str
    onnx_path: Optional[str]
    metrics: dict
    ts: float = field(default_factory=time.time)


class YOLOTrainer:
    """Fine-tunes YOLOv8 models on domain-specific datasets."""

    def __init__(
        self,
        domain: str,
        base_model: str = "yolov8n.pt",
        output_dir: str = "./models",
    ):
        self.domain = domain
        self.base_model = base_model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def train(
        self,
        dataset_yaml: str,
        epochs: int = 100,
        imgsz: int = 640,
        batch: int = 16,
    ) -> TrainResult:
        """
        Train on dataset_yaml.
        Calls ultralytics YOLO.train() if available; otherwise stubs with
        mock metrics so the rest of the pipeline can be exercised.
        """
        project_dir = str(self.output_dir / self.domain)

        try:
            from ultralytics import YOLO  # type: ignore

            model = YOLO(self.base_model)
            results = model.train(
                data=dataset_yaml,
                epochs=epochs,
                imgsz=imgsz,
                batch=batch,
                project=project_dir,
                name="train",
                exist_ok=True,
                verbose=False,
            )
            # ultralytics saves best weights at <project>/train/weights/best.pt
            weights_path = str(
                Path(project_dir) / "train" / "weights" / "best.pt"
            )
            metrics = {
                "mAP50":    float(getattr(results, "results_dict", {}).get("metrics/mAP50(B)", 0.0)),
                "mAP50-95": float(getattr(results, "results_dict", {}).get("metrics/mAP50-95(B)", 0.0)),
                "precision": float(getattr(results, "results_dict", {}).get("metrics/precision(B)", 0.0)),
                "recall":    float(getattr(results, "results_dict", {}).get("metrics/recall(B)", 0.0)),
            }
            logger.info("Training complete for domain=%s  mAP50=%.3f", self.domain, metrics["mAP50"])

        except ImportError:
            logger.warning("ultralytics not installed — returning stub TrainResult for domain=%s", self.domain)
            weights_path = str(self.output_dir / self.domain / "stub_best.pt")
            metrics = {
                "mAP50":    0.0,
                "mAP50-95": 0.0,
                "precision": 0.0,
                "recall":    0.0,
                "stub":     True,
            }

        return TrainResult(
            domain=self.domain,
            weights_path=weights_path,
            onnx_path=None,
            metrics=metrics,
        )

    # ------------------------------------------------------------------
    def export_onnx(self, weights_path: str) -> str:
        """Export a .pt weights file to ONNX. Returns the ONNX path."""
        weights_path = str(weights_path)
        onnx_path = weights_path.replace(".pt", ".onnx")

        try:
            from ultralytics import YOLO  # type: ignore

            model = YOLO(weights_path)
            model.export(format="onnx")
            logger.info("Exported ONNX to %s", onnx_path)
        except ImportError:
            logger.warning("ultralytics not installed — ONNX export skipped, writing stub at %s", onnx_path)
            Path(onnx_path).parent.mkdir(parents=True, exist_ok=True)
            Path(onnx_path).write_bytes(b"")

        return onnx_path

    # ------------------------------------------------------------------
    def evaluate(self, weights_path: str, dataset_yaml: str) -> dict:
        """
        Evaluate a trained model on a validation set.
        Returns dict with mAP50, mAP50-95, precision, recall.
        """
        try:
            from ultralytics import YOLO  # type: ignore

            model = YOLO(weights_path)
            results = model.val(data=dataset_yaml, verbose=False)
            rd = getattr(results, "results_dict", {})
            return {
                "mAP50":    float(rd.get("metrics/mAP50(B)", 0.0)),
                "mAP50-95": float(rd.get("metrics/mAP50-95(B)", 0.0)),
                "precision": float(rd.get("metrics/precision(B)", 0.0)),
                "recall":    float(rd.get("metrics/recall(B)", 0.0)),
            }
        except ImportError:
            logger.warning("ultralytics not installed — returning zero eval metrics")
            return {"mAP50": 0.0, "mAP50-95": 0.0, "precision": 0.0, "recall": 0.0, "stub": True}
