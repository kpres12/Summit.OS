"""
Heli.OS Aerial Detection Model Training
=========================================
Fine-tunes YOLOv8n on a combined aerial dataset:
  - VisDrone2019: person + vehicle detection from UAV altitude
  - D-Fire: smoke and fire detection
  - HERIDAL: human detection in SAR aerial imagery

Output: heli-detect-v1.pt (4 classes: person, vehicle, fire, smoke)
Place the output model in packages/models/ for adapter use.

Usage:
    python train_detection.py \\
        --visdrone /tmp/heli-training-data/visdrone \\
        --dfire    /tmp/heli-training-data/dfire \\
        --heridal  /tmp/heli-training-data/heridal \\
        --output   /tmp/heli-training-data/models \\
        --epochs   50
"""
from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

_CLASSES = {0: "person", 1: "vehicle", 2: "fire", 3: "smoke"}


def _merge_datasets(
    visdrone_dir: Path,
    dfire_dir: Path,
    heridal_dir: Path,
    merged_dir: Path,
) -> Path:
    """
    Merge VisDrone + D-Fire + HERIDAL into a single YOLO dataset directory.
    Returns path to the merged dataset.
    """
    merged_dir.mkdir(parents=True, exist_ok=True)
    counts = {"train": 0, "val": 0, "test": 0}

    for src_root, src_name in [
        (visdrone_dir, "visdrone"),
        (dfire_dir,    "dfire"),
        (heridal_dir,  "heridal"),
    ]:
        src_root = Path(src_root)
        if not src_root.exists():
            logger.warning("%s directory not found: %s — skipping", src_name, src_root)
            continue

        for split in ("train", "val", "test"):
            img_dir   = src_root / "images" / split
            label_dir = src_root / "labels" / split

            if not img_dir.exists():
                continue

            out_img   = merged_dir / "images" / split
            out_label = merged_dir / "labels" / split
            out_img.mkdir(parents=True, exist_ok=True)
            out_label.mkdir(parents=True, exist_ok=True)

            for img_path in img_dir.iterdir():
                if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                    continue
                label_path = label_dir / img_path.with_suffix(".txt").name
                if not label_path.exists():
                    continue

                # Prefix filename with source to avoid collisions
                new_name  = f"{src_name}_{img_path.name}"
                new_label = f"{src_name}_{label_path.name}"
                shutil.copy(img_path,   out_img / new_name)
                shutil.copy(label_path, out_label / new_label)
                counts[split] += 1

    logger.info("Merged dataset: train=%d val=%d test=%d images",
                counts["train"], counts["val"], counts["test"])
    return merged_dir


def _write_dataset_yaml(merged_dir: Path) -> Path:
    """Write YOLO dataset YAML pointing to the merged directory."""
    yaml_path = merged_dir / "dataset.yaml"
    yaml_path.write_text(f"""\
path: {merged_dir.resolve()}
train: images/train
val: images/val
test: images/test

nc: {len(_CLASSES)}
names:
""" + "\n".join(f"  {i}: {name}" for i, name in _CLASSES.items()) + "\n")
    return yaml_path


def train(
    visdrone_dir: str,
    dfire_dir: str,
    heridal_dir: str,
    output_dir: str,
    epochs: int = 50,
    imgsz: int = 640,
    batch: int = 16,
    device: str = "auto",
) -> Path:
    try:
        from ultralytics import YOLO
    except ImportError:
        raise RuntimeError("ultralytics not installed. Run: pip install ultralytics")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    merged = out / "merged_dataset"
    _merge_datasets(
        Path(visdrone_dir),
        Path(dfire_dir),
        Path(heridal_dir),
        merged,
    )

    yaml_path = _write_dataset_yaml(merged)
    logger.info("Dataset YAML: %s", yaml_path)

    # Load pretrained YOLOv8n
    model = YOLO("yolov8n.pt")

    logger.info("Starting fine-tuning: epochs=%d imgsz=%d batch=%d device=%s",
                epochs, imgsz, batch, device)

    model.train(
        data=str(yaml_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=str(out / "runs"),
        name="heli-detect",
        exist_ok=True,
        patience=15,          # early stopping
        save=True,
        plots=True,
        # Augmentation suited for aerial imagery
        degrees=15.0,         # rotation
        flipud=0.3,           # vertical flip (aerial = valid)
        fliplr=0.5,
        mosaic=0.8,
        mixup=0.1,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        translate=0.1,
        scale=0.5,
    )

    # Copy best weights to output
    best_weights = out / "runs" / "heli-detect" / "weights" / "best.pt"
    if best_weights.exists():
        dest = out / "heli-detect-v1.pt"
        shutil.copy(best_weights, dest)
        logger.info("Model saved: %s", dest)
        return dest
    else:
        logger.error("Training completed but best.pt not found at %s", best_weights)
        raise FileNotFoundError(f"best.pt not found at {best_weights}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="Train Heli.OS aerial detection model")
    p.add_argument("--visdrone", required=True)
    p.add_argument("--dfire",    required=True)
    p.add_argument("--heridal",  required=True)
    p.add_argument("--output",   required=True)
    p.add_argument("--epochs",   type=int, default=50)
    p.add_argument("--imgsz",    type=int, default=640)
    p.add_argument("--batch",    type=int, default=16)
    p.add_argument("--device",   default="auto",
                   help="Training device: auto, cpu, 0 (GPU index)")
    args = p.parse_args()

    model_path = train(
        visdrone_dir=args.visdrone,
        dfire_dir=args.dfire,
        heridal_dir=args.heridal,
        output_dir=args.output,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
    )
    print(f"Detection model: {model_path}")
