"""
Auto Annotator — Summit.OS

Uses an existing YOLO model to pre-label new footage for human review.
Generates YOLO-format label files with confidence scores embedded in comments.
High-confidence detections (>0.8) are auto-accepted; lower confidence ones
are flagged for human review.

Output format: standard YOLO .txt labels (class cx cy w h, normalized)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger("ml.auto_annotator")

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


class AutoAnnotator:
    """Semi-automatic annotation using an existing YOLO model."""

    def __init__(self, model_path: str, conf_threshold: float = 0.8):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self._model = None  # lazy-loaded

    # ------------------------------------------------------------------
    def _load_model(self):
        """Lazy-load the ultralytics YOLO model; stub if unavailable."""
        if self._model is not None:
            return self._model

        try:
            from ultralytics import YOLO  # type: ignore

            self._model = YOLO(self.model_path)
            logger.info("Loaded YOLO model from %s", self.model_path)
        except ImportError:
            logger.warning("ultralytics not installed — AutoAnnotator will produce empty labels (stub mode)")
            self._model = None

        return self._model

    # ------------------------------------------------------------------
    def annotate_image(self, image_path: str, output_dir: str) -> dict:
        """
        Run inference on a single image and write YOLO-format label file.
        Returns {auto_accepted: int, needs_review: int}.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        img_path = Path(image_path)
        label_file = output_path / img_path.with_suffix(".txt").name

        model = self._load_model()
        auto_accepted = 0
        needs_review = 0

        if model is None:
            # Stub: write empty label
            label_file.write_text("")
            return {"auto_accepted": 0, "needs_review": 0}

        results = model(str(img_path), verbose=False)
        lines: list[str] = []

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                cls_id  = int(box.cls[0])
                conf    = float(box.conf[0])
                xywhn   = box.xywhn[0].tolist()  # normalized cx cy w h
                cx, cy, bw, bh = xywhn

                line = f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"

                if conf >= self.conf_threshold:
                    lines.append(f"{line}  # conf={conf:.3f} AUTO")
                    auto_accepted += 1
                else:
                    lines.append(f"{line}  # conf={conf:.3f} REVIEW")
                    needs_review += 1

        label_file.write_text("\n".join(lines) + ("\n" if lines else ""))
        return {"auto_accepted": auto_accepted, "needs_review": needs_review}

    # ------------------------------------------------------------------
    def annotate_directory(self, image_dir: str, output_dir: str) -> dict:
        """
        Batch-annotate all images in a directory.
        Returns aggregated {auto_accepted, needs_review, total_images}.
        """
        src = Path(image_dir)
        total_auto = 0
        total_review = 0
        total_images = 0

        for img_path in src.iterdir():
            if img_path.suffix.lower() not in _IMG_EXTS:
                continue
            try:
                result = self.annotate_image(str(img_path), output_dir)
                total_auto   += result["auto_accepted"]
                total_review += result["needs_review"]
                total_images += 1
            except Exception as exc:
                logger.error("Failed to annotate %s: %s", img_path, exc)

        logger.info(
            "annotate_directory complete: %d images, %d auto-accepted, %d need review",
            total_images, total_auto, total_review,
        )
        return {
            "auto_accepted":  total_auto,
            "needs_review":   total_review,
            "total_images":   total_images,
        }
