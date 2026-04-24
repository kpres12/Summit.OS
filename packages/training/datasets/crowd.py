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

# Kaggle mirror (use: kaggle datasets download -d tthien/shanghaitech)
KAGGLE_REF = "tthien/shanghaitech"
KAGGLE_OUT = Path(__file__).parent.parent / "data" / "crowd_shanghaitech"

# ShanghaiTech Part B (original mirror — often unavailable)
SHANGHAITECH_B_URL = (
    "https://github.com/desenzhou/ShanghaiTechDataset/releases/download/v1.0/"
    "ShanghaiTech.zip"
)


def download(out_dir: Path = OUT_DIR) -> Path:
    """Download crowd counting dataset annotations."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Check if Kaggle download already present (preferred real data path)
    if any(KAGGLE_OUT.rglob("*.mat")):
        logger.info("[Crowd] ShanghaiTech (Kaggle) already present at %s", KAGGLE_OUT)
        return KAGGLE_OUT

    mat_dir = out_dir / "ground_truth"
    if mat_dir.exists() and any(mat_dir.rglob("*.mat")):
        logger.info("[Crowd] Ground truth already present at %s", mat_dir)
        return mat_dir

    # Try Kaggle API if token available
    import os
    if os.getenv("KAGGLE_API_TOKEN") or os.getenv("KAGGLE_KEY"):
        try:
            import subprocess
            KAGGLE_OUT.mkdir(parents=True, exist_ok=True)
            result = subprocess.run(
                ["kaggle", "datasets", "download", "-d", KAGGLE_REF,
                 "--path", str(KAGGLE_OUT), "--unzip"],
                capture_output=True, text=True, timeout=600,
                env={**os.environ},
            )
            if result.returncode == 0 and any(KAGGLE_OUT.rglob("*.mat")):
                logger.info("[Crowd] ShanghaiTech downloaded via Kaggle → %s", KAGGLE_OUT)
                return KAGGLE_OUT
        except Exception as e:
            logger.warning("[Crowd] Kaggle download failed: %s", e)

    # Fallback: direct URL
    logger.info("[Crowd] Downloading ShanghaiTech crowd dataset (direct)...")
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
        return out_dir
    except Exception as e:
        logger.warning("[Crowd] Download failed: %s — generating synthetic", e)
        _generate_synthetic(out_dir)

    return out_dir


def load_as_training_samples(data_dir: Path = None) -> list[dict]:
    # Prefer Kaggle path if present
    if data_dir is None or not Path(data_dir).exists():
        if any(KAGGLE_OUT.rglob("*.mat")):
            data_dir = KAGGLE_OUT
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
                    count = 0

                    # ShanghaiTech Part A/B: image_info[0,0][0,0][0] = Nx2 annotation points
                    if "image_info" in mat:
                        try:
                            ann_pts = mat["image_info"][0, 0][0, 0][0]
                            count = int(ann_pts.shape[0])
                        except Exception:
                            pass

                    # ShanghaiTech older format: annPoints directly
                    elif "annPoints" in mat:
                        ann_pts = mat["annPoints"]
                        count = int(ann_pts.shape[0]) if ann_pts.ndim == 2 else 0

                    if count > 0:
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
