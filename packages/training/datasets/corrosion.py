"""
Corrosion / Structural Defect Detection Dataset

Sources:
  - CODEBRIM (COncrete DEfect BRIdge iMage): 1,590 images, 6 defect classes
    https://zenodo.org/record/2620293 (CC BY 4.0)
  - Corrosion CS (steel corrosion classification, IEEE DataPort, public)
  - Synthetic augmentation for pipeline/industrial contexts

Classes: crack(0), spalling(1), efflorescence(2), exposed_rebar(3),
         corrosion(4), delamination(5)

Used to train: structural defect classifier for:
  - packages/domains/utilities.py — CORROSION_DETECTED alert
  - packages/domains/oilgas.py — CORROSION_DETECTED alert
  - infrastructure inspection reports
"""

from __future__ import annotations

import json
import logging
import random
import zipfile
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

OUT_DIR = Path(__file__).parent.parent / "data" / "corrosion"

# CODEBRIM on Zenodo (CC BY 4.0) — URL often 404, Kaggle is more reliable
CODEBRIM_URL = "https://zenodo.org/record/2620293/files/CODEBRIM_benchmark_datasets.zip"

# Kaggle alternatives (real crack/defect imagery)
CRACK_DIR   = Path(__file__).parent.parent / "data" / "crack"    # arunrk7/surface-crack-detection
SDNET_DIR   = Path(__file__).parent.parent / "data" / "sdnet"    # aniruddhsharma/structural-defects-network

DEFECT_CLASSES = {
    "crack": 0,
    "spalling": 1,
    "efflorescence": 2,
    "exposed_rebar": 3,
    "corrosion": 4,
    "delamination": 5,
}


def download(out_dir: Path = OUT_DIR) -> Path:
    """Download CODEBRIM defect dataset."""
    out_dir.mkdir(parents=True, exist_ok=True)
    annot_file = out_dir / "codebrim_annotations.json"

    if annot_file.exists():
        logger.info("[Corrosion] Annotations already present")
        return out_dir

    logger.info("[Corrosion] Downloading CODEBRIM from Zenodo (CC BY 4.0)...")
    try:
        zip_path = out_dir / "codebrim.zip"
        resp = requests.get(CODEBRIM_URL, timeout=300, stream=True)
        resp.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in resp.iter_content(65536):
                f.write(chunk)

        with zipfile.ZipFile(zip_path) as zf:
            csv_members = [m for m in zf.namelist() if m.endswith(".csv")]
            zf.extractall(out_dir, members=csv_members)
            logger.info("[CODEBRIM] Extracted %d CSV files", len(csv_members))
        zip_path.unlink(missing_ok=True)
        _convert_csv_to_json(out_dir, annot_file)
    except Exception as e:
        logger.warning("[Corrosion] Download failed: %s — generating synthetic", e)
        _generate_synthetic(annot_file)

    return out_dir


def _load_from_kaggle_crack() -> list[dict]:
    """
    Load from arunrk7/surface-crack-detection (40k images: Positive=crack, Negative=no crack)
    and SDNET2018 (56k images: Decks/Walls/Pavements with D/U prefix for cracked/uncracked).
    Maps crack presence → crack defect class; no-crack → empty defect list.
    """
    samples = []

    # Surface crack detection (binary: Positive/Negative folders)
    for label_dir, has_crack in [(CRACK_DIR / "Positive", True), (CRACK_DIR / "Negative", False)]:
        if not label_dir.exists():
            continue
        for img in list(label_dir.glob("*.jpg"))[:5000]:  # cap at 5k per class
            defects = ["crack"] if has_crack else []
            samples.append({
                "image_path": str(img),
                "defect_classes": defects,
                "defect_indices": [DEFECT_CLASSES["crack"]] if has_crack else [],
                "severity": len(defects) / 6.0,
                "context": "concrete_surface",
                "split": "train",
                "source": "kaggle_surface_crack",
            })

    # SDNET2018 (D=cracked, U=uncracked prefix on filenames)
    for context_dir in (SDNET_DIR.iterdir() if SDNET_DIR.exists() else []):
        for img in list(context_dir.rglob("*.jpg"))[:2000]:
            has_crack = img.name.startswith("D")
            defects = ["crack"] if has_crack else []
            samples.append({
                "image_path": str(img),
                "defect_classes": defects,
                "defect_indices": [DEFECT_CLASSES["crack"]] if has_crack else [],
                "severity": 1/6.0 if has_crack else 0.0,
                "context": context_dir.name.lower(),
                "split": "train",
                "source": "kaggle_sdnet2018",
            })

    if samples:
        logger.info("[Corrosion] Loaded %d real crack samples from Kaggle datasets", len(samples))
    return samples


def load_as_training_samples(data_dir: Path) -> list[dict]:
    """
    Load corrosion/defect samples as classification training data.

    Returns:
        List of {image_path, defect_classes (multi-label), severity, split}

    Note: real Kaggle crack images (image-path-only records) are used by the
    CNN trainer (train_corrosion_vision.py), NOT here. This tabular pipeline
    needs feature vectors (rgb_rust_ratio, thermal_delta_c, etc.).
    """
    annot_file = data_dir / "codebrim_annotations.json"
    if not annot_file.exists():
        annot_file = data_dir / "synthetic_defects.json"
    if not annot_file.exists():
        return _synthetic_samples()

    try:
        data = json.loads(annot_file.read_text())
        logger.info("[Corrosion] Loaded %d defect samples", len(data))
        return data
    except Exception as e:
        logger.warning("[Corrosion] Failed to load annotations: %s", e)
        return _synthetic_samples()


def _convert_csv_to_json(data_dir: Path, out_file: Path) -> None:
    import csv
    samples = []
    for csv_file in data_dir.rglob("*.csv"):
        try:
            with open(csv_file, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    defects = []
                    for cls in DEFECT_CLASSES:
                        if str(row.get(cls, "0")).strip() in ("1", "True", "true"):
                            defects.append(cls)
                    if defects:
                        samples.append({
                            "image_path": row.get("image_name", ""),
                            "defect_classes": defects,
                            "defect_indices": [DEFECT_CLASSES[d] for d in defects],
                            "severity": len(defects) / 6.0,
                            "split": "train",
                        })
        except Exception:
            continue
    out_file.write_text(json.dumps(samples))


def _synthetic_samples() -> list[dict]:
    random.seed(42)
    samples = []
    contexts = ["bridge_pier", "pipeline_weld", "tank_exterior", "steel_beam", "concrete_deck"]
    for i in range(500):
        n_defects = random.choices([1, 2, 3], weights=[0.6, 0.3, 0.1])[0]
        defects = random.sample(list(DEFECT_CLASSES.keys()), n_defects)
        # Make corrosion more common in pipeline/tank contexts
        context = random.choice(contexts)
        if context in ("pipeline_weld", "tank_exterior") and "corrosion" not in defects:
            defects[0] = "corrosion"
        samples.append({
            "image_path": f"synth_{context}_{i:04d}.jpg",
            "defect_classes": defects,
            "defect_indices": [DEFECT_CLASSES[d] for d in defects],
            "severity": len(defects) / 6.0,
            "context": context,
            "split": "train" if i < 400 else "val",
            "source": "synthetic",
        })
    return samples


def _generate_synthetic(out_file: Path) -> None:
    samples = _synthetic_samples()
    out_file.write_text(json.dumps(samples))
    logger.info("[Corrosion] Generated %d synthetic defect samples", len(samples))
