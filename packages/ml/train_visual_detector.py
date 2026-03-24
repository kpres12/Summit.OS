"""
Fine-tune YOLOv8n on the Summit.OS visual dataset for multi-domain object detection.

Covers all operational domains out of the box:
  Wildfire       — smoke, fire, fire_front
  Search & Rescue — person, person_water, life_raft
  Oil / Hazmat   — oil_spill, pipeline_damage, chemical_plume
  Agriculture    — crop_disease, pest_damage, dry_field
  Maritime       — vessel, vessel_distress
  Infrastructure — power_line_damage, structural_crack, solar_defect
  Wildlife       — dangerous_animal

Prerequisites:
  pip install ultralytics onnx onnxruntime

Usage:
  # Build dataset first (if not already done)
  python download_visual_datasets.py

  # Train (default: YOLOv8n, 100 epochs, all domains)
  python train_visual_detector.py

  # Quick smoke-test (5 epochs, CPU-safe)
  python train_visual_detector.py --epochs 5 --device cpu

  # Full training with GPU
  python train_visual_detector.py --epochs 200 --device 0

  # Export only (skip training, export existing checkpoint)
  python train_visual_detector.py --export-only --checkpoint runs/detect/summit_detector/weights/best.pt

Output:
  packages/ml/models/summit_detector.onnx   (dropped in place — inference service hot-swaps)
  packages/ml/models/summit_detector_classes.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

# ── paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).parent
DATA_DIR    = SCRIPT_DIR / "data"
MODELS_DIR  = SCRIPT_DIR / "models"
DATASET_DIR = DATA_DIR / "summit_detector"
YAML_PATH   = DATASET_DIR / "summit_detector.yaml"

MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ── class taxonomy ─────────────────────────────────────────────────────────────
# Must stay in sync with download_visual_datasets.py VISUAL_CLASSES
VISUAL_CLASSES = {
    0:  "smoke",
    1:  "fire",
    2:  "fire_front",
    3:  "person",
    4:  "person_water",
    5:  "life_raft",
    6:  "oil_spill",
    7:  "pipeline_damage",
    8:  "chemical_plume",
    9:  "crop_disease",
    10: "pest_damage",
    11: "dry_field",
    12: "vessel",
    13: "vessel_distress",
    14: "power_line_damage",
    15: "structural_crack",
    16: "solar_defect",
    17: "dangerous_animal",
}

# Domain groupings — used for per-domain augmentation strategies
DOMAIN_CLASSES = {
    "wildfire":        [0, 1, 2],           # smoke, fire, fire_front
    "search_rescue":   [3, 4, 5],           # person, person_water, life_raft
    "hazmat_pipeline": [6, 7, 8],           # oil_spill, pipeline_damage, chemical_plume
    "agriculture":     [9, 10, 11],         # crop_disease, pest_damage, dry_field
    "maritime":        [12, 13],            # vessel, vessel_distress
    "infrastructure":  [14, 15, 16],        # power_line_damage, structural_crack, solar_defect
    "wildlife":        [17],                # dangerous_animal
}


# ── augmentation profiles ──────────────────────────────────────────────────────
def _augmentation_kwargs() -> dict:
    """
    Augmentation settings tuned for aerial/UAV imagery across all operational domains.

    Key choices:
      - High HSV-Hue shift: fire and hazmat plumes have inconsistent colour rendering
        across different sensor types and times of day.
      - Mosaic=1.0: forces the model to learn partial objects (critical for smoke at edge
        of frame, partial persons in water, partially visible vessels).
      - Flipud=0.5: UAV downward-looking imagery has no canonical 'up'.
      - Degrees=30: rotational invariance for top-down infrastructure inspection.
      - Scale=0.6: handles extreme altitude variation (drone height 30m–300m).
      - Perspective=0.001: simulates lens tilt on oblique aerial angles.
    """
    return dict(
        hsv_h=0.02,        # hue jitter — fire/hazmat colour variance
        hsv_s=0.7,         # saturation — smoke opacity varies
        hsv_v=0.4,         # value — altitude/lighting variation
        degrees=30,        # rotation — top-down has no canonical orientation
        translate=0.1,
        scale=0.6,         # extreme zoom range for altitude variance
        shear=0.0,
        perspective=0.001, # oblique UAV angles
        flipud=0.5,        # UAV: vertical flip is valid
        fliplr=0.5,
        mosaic=1.0,        # always-on: critical for partial objects
        mixup=0.1,         # gentle mixup for domain blending
        copy_paste=0.1,    # copy-paste augmentation for rare classes
    )


# ── dataset yaml guard ─────────────────────────────────────────────────────────
def _ensure_dataset() -> None:
    if not YAML_PATH.exists():
        print(
            f"\n[ERROR] Dataset YAML not found at {YAML_PATH}\n"
            "Run first:\n"
            "  python download_visual_datasets.py\n"
        )
        sys.exit(1)

    # Check train/val image dirs are non-empty
    for split in ("train", "val"):
        img_dir = DATASET_DIR / "images" / split
        if not img_dir.exists() or not any(img_dir.iterdir()):
            print(
                f"\n[ERROR] {img_dir} is empty.\n"
                "Run first:\n"
                "  python download_visual_datasets.py\n"
            )
            sys.exit(1)

    # Count images
    train_count = len(list((DATASET_DIR / "images" / "train").iterdir()))
    val_count   = len(list((DATASET_DIR / "images" / "val").iterdir()))
    print(f"[dataset] train={train_count} images  val={val_count} images")


# ── training ───────────────────────────────────────────────────────────────────
def train(
    epochs: int,
    device: str,
    batch: int,
    imgsz: int,
    model_size: str,
    project: str,
    name: str,
    resume: bool,
    patience: int,
) -> Path:
    """
    Fine-tune YOLOv8 on summit_detector dataset.
    Returns path to the best checkpoint (.pt).
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] ultralytics not installed.\n  pip install ultralytics")
        sys.exit(1)

    base_model = f"yolov8{model_size}.pt"
    print(f"\n[train] base={base_model}  epochs={epochs}  device={device}  imgsz={imgsz}")
    print(f"[train] dataset={YAML_PATH}")

    model = YOLO(base_model)

    aug = _augmentation_kwargs()

    results = model.train(
        data=str(YAML_PATH),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=name,
        resume=resume,
        patience=patience,
        save=True,
        save_period=10,         # checkpoint every 10 epochs
        val=True,
        plots=True,
        # Augmentation
        **aug,
        # Loss weights — up-weight localization for small aerial targets
        box=7.5,
        cls=0.5,
        dfl=1.5,
        # Optimizer
        optimizer="AdamW",
        lr0=1e-3,
        lrf=0.01,               # final LR = lr0 * lrf
        warmup_epochs=3,
        weight_decay=5e-4,
        # Misc
        workers=min(8, os.cpu_count() or 4),
        seed=42,
        deterministic=True,
        verbose=True,
    )

    best_pt = Path(project) / name / "weights" / "best.pt"
    if not best_pt.exists():
        # Fallback: find in ultralytics default run dir
        import glob
        candidates = glob.glob(f"{project}/{name}*/weights/best.pt")
        if candidates:
            best_pt = Path(sorted(candidates)[-1])

    print(f"\n[train] best checkpoint: {best_pt}")
    return best_pt


# ── export to ONNX ─────────────────────────────────────────────────────────────
def export_onnx(checkpoint: Path, imgsz: int) -> Path:
    """
    Export YOLOv8 checkpoint to ONNX and copy to models/ for hot-swap.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] ultralytics not installed.\n  pip install ultralytics")
        sys.exit(1)

    print(f"\n[export] {checkpoint} → ONNX (imgsz={imgsz})")
    model = YOLO(str(checkpoint))

    # Export — ultralytics writes .onnx alongside the .pt file
    model.export(
        format="onnx",
        imgsz=imgsz,
        dynamic=False,          # static shapes for ONNX Runtime compatibility
        simplify=True,          # onnx-simplifier reduces node count
        opset=17,               # opset 17 = broad ONNX Runtime support
        half=False,             # FP32 — maximises CPU compatibility for edge devices
        nms=True,               # bake NMS into the graph for single-pass inference
    )

    exported_onnx = checkpoint.with_suffix(".onnx")
    dest_onnx     = MODELS_DIR / "summit_detector.onnx"

    if not exported_onnx.exists():
        print(f"[WARN] ONNX export not found at {exported_onnx} — check ultralytics output")
    else:
        import shutil
        shutil.copy2(exported_onnx, dest_onnx)
        size_mb = dest_onnx.stat().st_size / 1e6
        print(f"[export] {dest_onnx}  ({size_mb:.1f} MB)")

    # Write class map for the inference service
    classes_path = MODELS_DIR / "summit_detector_classes.json"
    with open(classes_path, "w") as f:
        json.dump(VISUAL_CLASSES, f, indent=2)
    print(f"[export] {classes_path}")

    return dest_onnx


# ── quick validation ───────────────────────────────────────────────────────────
def validate(checkpoint: Path, imgsz: int) -> None:
    """Run val loop and print per-class mAP50."""
    try:
        from ultralytics import YOLO
    except ImportError:
        return

    print(f"\n[validate] {checkpoint}")
    model   = YOLO(str(checkpoint))
    metrics = model.val(data=str(YAML_PATH), imgsz=imgsz, verbose=True)

    print("\n[validate] Per-class mAP50:")
    if hasattr(metrics, "ap_class_index") and hasattr(metrics, "box"):
        for i, cls_idx in enumerate(metrics.ap_class_index):
            cls_name = VISUAL_CLASSES.get(int(cls_idx), str(cls_idx))
            ap50     = float(metrics.box.ap50[i]) if i < len(metrics.box.ap50) else float("nan")
            print(f"  {cls_name:25s}  mAP50={ap50:.3f}")

    map50 = getattr(getattr(metrics, "box", None), "map50", None)
    if map50 is not None:
        print(f"\n[validate] Overall mAP50={float(map50):.3f}")


# ── ONNX smoke-test ────────────────────────────────────────────────────────────
def smoke_test_onnx(onnx_path: Path, imgsz: int) -> None:
    """
    Verify the exported ONNX model loads and runs a single forward pass.
    No real image needed — random tensor is sufficient for shape/type validation.
    """
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError:
        print("[smoke-test] onnxruntime/numpy not available — skipping")
        return

    if not onnx_path.exists():
        print(f"[smoke-test] {onnx_path} not found — skipping")
        return

    print(f"\n[smoke-test] {onnx_path}")
    sess    = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inp     = sess.get_inputs()[0]
    dummy   = np.random.rand(1, 3, imgsz, imgsz).astype(np.float32)
    outputs = sess.run(None, {inp.name: dummy})
    print(f"[smoke-test] output shapes: {[o.shape for o in outputs]}")
    print("[smoke-test] PASS")


# ── CLI ────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Summit.OS visual detector (YOLOv8) across all operational domains"
    )
    parser.add_argument("--epochs",       type=int,   default=100,
                        help="Training epochs (default 100; use 5 for quick smoke-test)")
    parser.add_argument("--device",       type=str,   default="",
                        help="'cpu', '0', '0,1', 'mps' — empty = auto-detect")
    parser.add_argument("--batch",        type=int,   default=16,
                        help="Batch size (reduce if OOM; -1 = auto)")
    parser.add_argument("--imgsz",        type=int,   default=640,
                        help="Input image size (default 640)")
    parser.add_argument("--model",        type=str,   default="n",
                        choices=["n", "s", "m", "l", "x"],
                        help="YOLOv8 model size: n=nano, s=small, m=medium (default: n)")
    parser.add_argument("--project",      type=str,   default="runs/detect",
                        help="Output directory for training runs")
    parser.add_argument("--name",         type=str,   default="summit_detector",
                        help="Run name within --project")
    parser.add_argument("--resume",       action="store_true",
                        help="Resume interrupted training from last checkpoint")
    parser.add_argument("--patience",     type=int,   default=50,
                        help="Early-stopping patience in epochs (0 = disabled)")
    parser.add_argument("--export-only",  action="store_true",
                        help="Skip training — export existing checkpoint to ONNX")
    parser.add_argument("--checkpoint",   type=str,   default=None,
                        help="Path to .pt checkpoint for --export-only or post-train export")
    parser.add_argument("--no-validate",  action="store_true",
                        help="Skip validation after training")
    parser.add_argument("--no-smoke-test", action="store_true",
                        help="Skip ONNX smoke-test after export")
    parser.add_argument("--list-classes", action="store_true",
                        help="Print class taxonomy and exit")
    args = parser.parse_args()

    if args.list_classes:
        print("Summit.OS visual detector — 18 classes:\n")
        for idx, name in VISUAL_CLASSES.items():
            domain = next(
                (d for d, ids in DOMAIN_CLASSES.items() if idx in ids), "—"
            )
            print(f"  {idx:2d}  {name:25s}  [{domain}]")
        return

    # ── export-only mode ────────────────────────────────────────────────────
    if args.export_only:
        if not args.checkpoint:
            print("[ERROR] --export-only requires --checkpoint <path/to/best.pt>")
            sys.exit(1)
        ckpt = Path(args.checkpoint)
        if not ckpt.exists():
            print(f"[ERROR] checkpoint not found: {ckpt}")
            sys.exit(1)
        onnx_path = export_onnx(ckpt, args.imgsz)
        if not args.no_smoke_test:
            smoke_test_onnx(onnx_path, args.imgsz)
        return

    # ── full training ───────────────────────────────────────────────────────
    _ensure_dataset()

    best_pt = train(
        epochs   = args.epochs,
        device   = args.device,
        batch    = args.batch,
        imgsz    = args.imgsz,
        model_size = args.model,
        project  = args.project,
        name     = args.name,
        resume   = args.resume,
        patience = args.patience,
    )

    # Use user-supplied checkpoint if provided (e.g. after manual export decision)
    if args.checkpoint:
        best_pt = Path(args.checkpoint)

    if not args.no_validate and best_pt.exists():
        validate(best_pt, args.imgsz)

    onnx_path = export_onnx(best_pt, args.imgsz)

    if not args.no_smoke_test:
        smoke_test_onnx(onnx_path, args.imgsz)

    print(f"""
╔══════════════════════════════════════════════════════════╗
║  Summit.OS visual detector training complete             ║
║                                                          ║
║  Model   : packages/ml/models/summit_detector.onnx       ║
║  Classes : packages/ml/models/summit_detector_classes.json║
║                                                          ║
║  Inference service will hot-swap automatically.          ║
║  Restart apps/inference to pick up new model.            ║
╚══════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    main()
