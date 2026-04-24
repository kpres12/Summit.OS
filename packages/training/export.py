"""
Model Export Script
====================
Copies trained models from the training output directory to their
integration paths in the Heli.OS codebase.

  Detection model → packages/models/heli-detect-v1.pt
  Anomaly models  → packages/c2_intel/models/anomaly_detector_*.joblib
  Battery models  → packages/c2_intel/models/battery_predictor_*.joblib
  Timing models   → packages/c2_intel/models/c2_timing_predictor*.joblib
  Intent model    → packages/c2_intel/models/intent_classifier.joblib

Usage:
    python export.py \\
        --models-dir    /tmp/heli-training-data/models \\
        --c2-models-dir ../../packages/c2_intel/models
"""
from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]

_COPY_RULES: list[tuple[str, str]] = [
    # (source glob relative to models_dir, dest relative to c2_models_dir)
    ("anomaly_detector_*.joblib",         "."),
    ("anomaly_detector_meta.json",        "."),
    ("battery_predictor_*.joblib",        "."),
    ("battery_predictor_meta.json",       "."),
    ("c2_timing_predictor*.joblib",       "."),
    ("c2_timing_predictor_meta.json",     "."),
    ("c2_relevance_classifier*.joblib",   "."),
    ("c2_relevance_classifier_meta.json", "."),
    ("intent_classifier.joblib",          "."),
    ("intent_vectorizer.joblib",          "."),
    ("intent_classifier_meta.json",       "."),
    # Tabular domain models (train_damage/corrosion/crowd output directly to MODELS_DIR)
    ("damage_classifier.joblib",          "."),
    ("damage_classifier_meta.json",       "."),
    ("corrosion_classifier.joblib",       "."),
    ("corrosion_classifier_meta.json",    "."),
    ("crowd_estimator.joblib",            "."),
    ("crowd_estimator_meta.json",         "."),
]

_DETECT_GLOB = "heli-detect-v1.pt"
_DETECT_DEST = _REPO_ROOT / "packages" / "models" / "heli-detect-v1.pt"


def export(models_dir: str, c2_models_dir: str) -> None:
    src_root = Path(models_dir)
    c2_root  = Path(c2_models_dir)
    c2_root.mkdir(parents=True, exist_ok=True)

    copied = 0
    skipped = 0

    # C2 intel models
    for glob_pat, dest_subdir in _COPY_RULES:
        dest_dir = c2_root / dest_subdir
        dest_dir.mkdir(parents=True, exist_ok=True)

        matches = list(src_root.glob(glob_pat))
        if not matches:
            logger.debug("No files matching %s in %s", glob_pat, src_root)
            skipped += 1
            continue

        for src in matches:
            dest = dest_dir / src.name
            shutil.copy(src, dest)
            logger.info("Copied %s → %s", src.name, dest)
            copied += 1

    # Detection model (goes to packages/models/)
    detect_src = src_root / _DETECT_GLOB
    if detect_src.exists():
        _DETECT_DEST.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(detect_src, _DETECT_DEST)
        logger.info("Detection model → %s", _DETECT_DEST)
        copied += 1
    else:
        logger.warning("Detection model not found: %s", detect_src)
        skipped += 1

    logger.info("Export complete: %d copied, %d skipped/not found", copied, skipped)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    p = argparse.ArgumentParser(description="Export trained models to integration paths")
    p.add_argument("--models-dir",    required=True,
                   help="Directory containing trained model files")
    p.add_argument("--c2-models-dir", default=str(_REPO_ROOT / "packages" / "c2_intel" / "models"),
                   help="Destination c2_intel/models directory")
    args = p.parse_args()
    export(args.models_dir, args.c2_models_dir)
