"""
Crowd Counting / Density Estimation — Public Datasets

Sources:
  - UCF-QNRF: 1,535 aerial/ground images, 1.25M annotated people
    https://www.crcv.ucf.edu/data/ucf-qnrf/ (free for non-commercial use)
  - ShanghaiTech Part B: street-level crowd, 400 training images
    https://github.com/desenzhou/ShanghaiTechDataset
  - Drone-based crowd count: VisDrone DET track (already in visdrone.py)

Used to train: crowd density estimator for:
  - Mass casualty events (HADR) — estimate victim count from aerial view
  - Event security (FORCE_PROTECT) — track crowd size and movement
  - SAR assessment — estimate survivors in collapse zone
"""

from __future__ import annotations

import json
import logging
import os
import random
import math
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

OUT_DIR = Path(__file__).parent.parent / "data" / "crowd"

# ShanghaiTech Part B (smaller, reliable mirror)
SHANGHAITECH_B_URL = (
    "https://github.com/desenzhou/ShanghaiTechDataset/releases/download/v1.0/"
    "ShanghaiTech.zip"
)


def download(out_dir: Path = OUT_DIR) -> Path:
    """Download crowd counting dataset annotations."""
    out_dir.mkdir(parents=True, exist_ok=True)
    mat_dir = out_dir / "ground_truth"

    if mat_dir.exists() and any(mat_dir.rglob("*.mat")):
        logger.info("[Crowd] Ground truth already present at %s", mat_dir)
        return mat_dir

    logger.info("[Crowd] Downloading ShanghaiTech crowd dataset...")
    try:
        import zipfile
        zip_path = out_dir / "shanghaitech.zip"
        resp = requests.get(SHANGHAITECH_B_URL, timeout=300, stream=True)
        resp.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in resp.iter_content(65536):
                f.write(chunk)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(out_dir)
        zip_path.unlink(missing_ok=True)
        logger.info("[Crowd] ShanghaiTech downloaded → %s", out_dir)
    except Exception as e:
        logger.warning("[Crowd] Download failed: %s — generating synthetic", e)
        _generate_synthetic(out_dir)

    return out_dir


def load_as_training_samples(data_dir: Path) -> list[dict]:
    """
    Convert crowd ground truth to density map training samples.

    Each sample: {image_id, count, density_map_path or None, split}
    The count is used for regression; density maps for spatial models.
    """
    samples = []

    # Try scipy mat files (ShanghaiTech format)
    mat_files = list(data_dir.rglob("*.mat"))
    if mat_files:
        try:
            import scipy.io as sio
            for mf in mat_files:
                try:
                    mat = sio.loadmat(str(mf))
                    # ShanghaiTech: mat['image_info'][0,0][0,0][0] = annotations
                    ann = mat.get("image_info", mat.get("annPoints"))
                    if ann is not None:
                        if hasattr(ann, "shape"):
                            count = int(ann.shape[0]) if ann.ndim == 2 else 0
                        else:
                            count = 0
                        samples.append({
                            "image_id": mf.stem,
                            "count": count,
                            "split": "train" if "train" in str(mf) else "test",
                            "source": "shanghaitech",
                        })
                except Exception:
                    continue
        except ImportError:
            pass

    # JSON fallback
    json_files = list(data_dir.rglob("crowd_samples.json"))
    for jf in json_files:
        try:
            data = json.loads(jf.read_text())
            samples.extend(data)
        except Exception:
            continue

    if not samples:
        samples = _synthetic_samples()

    logger.info("[Crowd] Loaded %d crowd counting samples", len(samples))
    return samples


def _synthetic_samples() -> list[dict]:
    """Synthetic crowd counts across realistic scenarios."""
    random.seed(42)
    scenarios = [
        ("disaster_rubble", 5, 50),
        ("stadium_event", 500, 50000),
        ("street_protest", 100, 5000),
        ("evacuation_zone", 50, 2000),
        ("rural_sar", 1, 20),
    ]
    samples = []
    for scenario, lo, hi in scenarios:
        for i in range(100):
            count = random.randint(lo, hi)
            samples.append({
                "image_id": f"synth_{scenario}_{i:03d}",
                "count": count,
                "split": "train" if i < 80 else "val",
                "source": "synthetic",
                "scenario": scenario,
            })
    return samples


def _generate_synthetic(out_dir: Path) -> None:
    samples = _synthetic_samples()
    (out_dir / "crowd_samples.json").write_text(json.dumps(samples))
    logger.info("[Crowd] Generated %d synthetic crowd samples", len(samples))
