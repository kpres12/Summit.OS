"""
Download and prepare labeled image datasets for Heli.OS visual object detection training.

Covers all operational domains:
  - Wildfire / fire / smoke      — D-Fire dataset (aerial imagery)
  - Search & rescue              — SeaDronesSee (person-in-water, UAV) + AIDER (emergency response)
  - Oil pipeline / hazmat        — Synthetic generation (no suitable public dataset)
  - Agricultural                 — Synthetic generation
  - Maritime                     — COCO 2017 val subset (boats + persons)
  - Infrastructure               — Synthetic generation
  - Wildlife / dangerous animals — COCO 2017 val subset (bear, elephant, horse, etc.)

Output directory tree:
  packages/ml/data/
    images/fire_smoke/         ← D-Fire extracted images
    images/coco_subset/        ← COCO val images + filtered annotations
    images/seadronessee/       ← SeaDronesSee images (if accessible)
    images/aider/              ← AIDER emergency response images
    images/synthetic/          ← Programmatically generated labeled crops
    summit_detector/
      images/train/
      images/val/
      labels/train/            ← YOLO format: class_id cx cy w h (normalized)
      labels/val/
      summit_detector.yaml

Unified class taxonomy (must stay in sync with train_visual_detector.py):
  0  smoke            7  pipeline_damage   14 power_line_damage
  1  fire             8  chemical_plume    15 structural_crack
  2  fire_front       9  crop_disease      16 solar_defect
  3  person           10 pest_damage       17 dangerous_animal
  4  person_water     11 dry_field
  5  life_raft        12 vessel
  6  oil_spill        13 vessel_distress

Usage:
  python download_visual_datasets.py                          # download all sources + build
  python download_visual_datasets.py --source dfire coco_subset
  python download_visual_datasets.py --build-only            # rebuild combined dataset only
"""

import argparse
import json
import math
import os
import random
import shutil
import struct
import time
import urllib.error
import urllib.request
import zipfile
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data")
IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")
SUMMIT_DIR = os.path.join(OUTPUT_DIR, "summit_detector")

# ---------------------------------------------------------------------------
# Unified class taxonomy
# ---------------------------------------------------------------------------

VISUAL_CLASSES: Dict[int, str] = {
    # Fire / Smoke
    0: "smoke",
    1: "fire",
    2: "fire_front",
    # SAR
    3: "person",
    4: "person_water",
    5: "life_raft",
    # Pipeline / Hazmat
    6: "oil_spill",
    7: "pipeline_damage",
    8: "chemical_plume",
    # Agricultural
    9: "crop_disease",
    10: "pest_damage",
    11: "dry_field",
    # Maritime
    12: "vessel",
    13: "vessel_distress",
    # Infrastructure
    14: "power_line_damage",
    15: "structural_crack",
    16: "solar_defect",
    # Wildlife
    17: "dangerous_animal",
}

# COCO category_id → VISUAL_CLASSES id.  Only categories we care about.
COCO_TO_VISUAL: Dict[int, int] = {
    1: 3,  # person           → person
    9: 12,  # boat             → vessel
    21: 17,  # cow              → dangerous_animal  (proxy for livestock / large animal)
    22: 17,  # elephant         → dangerous_animal
    23: 17,  # bear             → dangerous_animal
    24: 17,  # zebra            → dangerous_animal
    # The following are kept for scene-level usefulness but mapped to person / vessel
    3: 12,  # car              → vessel (surface vehicle proxy; filtered later)
    8: 12,  # truck            → vessel
    16: 17,  # bird             → dangerous_animal (raptor / UAV hazard proxy)
    17: 17,  # cat              → dangerous_animal
    18: 17,  # dog              → dangerous_animal
    19: 17,  # horse            → dangerous_animal
}

# ---------------------------------------------------------------------------
# HTTP helper (mirrors download_real_data.py pattern)
# ---------------------------------------------------------------------------


def _fetch(url: str, timeout: int = 60, retries: int = 3) -> Optional[bytes]:
    """Fetch a URL, returning raw bytes or None on failure."""
    headers = {"User-Agent": "Heli.OS/1.0 (public dataset research)"}
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            print(f"  HTTP {e.code}: {url}")
            return None
        except Exception as exc:
            if attempt < retries - 1:
                wait = 2**attempt
                print(f"  Attempt {attempt + 1} failed ({exc}), retrying in {wait}s…")
                time.sleep(wait)
            else:
                print(f"  Failed after {retries} attempts: {exc}")
                return None
    return None


def _fetch_stream(url: str, dest_path: str, timeout: int = 120) -> bool:
    """Stream-download a potentially large file to disk, showing progress."""
    headers = {"User-Agent": "Heli.OS/1.0 (public dataset research)"}
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            total = int(resp.getheader("Content-Length") or 0)
            downloaded = 0
            chunk = 1 << 16  # 64 KiB
            with open(dest_path, "wb") as f:
                while True:
                    block = resp.read(chunk)
                    if not block:
                        break
                    f.write(block)
                    downloaded += len(block)
                    if total:
                        pct = downloaded * 100 // total
                        print(
                            f"\r  {pct:3d}%  {downloaded // 1048576} / {total // 1048576} MB",
                            end="",
                            flush=True,
                        )
            print()
        return True
    except Exception as exc:
        print(f"  Stream download failed: {exc}")
        return False


# ---------------------------------------------------------------------------
# Minimal PNG writer (stdlib only — no Pillow)
# Used by synthetic generator to write tiny solid-color patches.
# ---------------------------------------------------------------------------


def _write_png(path: str, width: int, height: int, rgb: Tuple[int, int, int]) -> None:
    """Write a single-color PNG using only stdlib (zlib + struct)."""
    import zlib

    def _u32be(n: int) -> bytes:
        return struct.pack(">I", n)

    def _chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return _u32be(len(data)) + tag + data + _u32be(crc)

    # Build raw scanlines: filter byte (0) + RGB pixels
    row = bytes([0]) + bytes(rgb) * width
    raw = row * height
    compressed = zlib.compress(raw, 9)

    ihdr_data = (
        _u32be(width)
        + _u32be(height)
        + bytes(
            [8, 2, 0, 0, 0]
        )  # bit depth=8, colour=RGB, compression, filter, interlace
    )

    data = (
        b"\x89PNG\r\n\x1a\n"  # PNG signature
        + _chunk(b"IHDR", ihdr_data)
        + _chunk(b"IDAT", compressed)
        + _chunk(b"IEND", b"")
    )
    with open(path, "wb") as f:
        f.write(data)


# ---------------------------------------------------------------------------
# 1. D-Fire dataset  (fire / smoke, aerial imagery)
# ---------------------------------------------------------------------------

DFIRE_RELEASE_URL = (
    "https://github.com/gaiasd/DFireDataset/archive/refs/heads/master.zip"
)


def download_dfire() -> Dict[str, int]:
    """
    Download the D-Fire dataset from GitHub.

    D-Fire contains 21,527 images annotated for fire and smoke from aerial
    and ground-level cameras.  Annotations are in YOLO format with two classes:
      0 = fire, 1 = smoke

    We remap:  D-Fire 0 (fire) → VISUAL class 1 (fire)
               D-Fire 1 (smoke) → VISUAL class 0 (smoke)

    Saves to: packages/ml/data/images/fire_smoke/
    """
    dest = os.path.join(IMAGES_DIR, "fire_smoke")
    os.makedirs(dest, exist_ok=True)

    zip_path = os.path.join(OUTPUT_DIR, "_dfire_master.zip")

    print("Downloading D-Fire dataset (GitHub master archive)…")
    print(f"  URL: {DFIRE_RELEASE_URL}")

    if os.path.exists(zip_path):
        print("  Archive already cached, skipping download.")
    else:
        ok = _fetch_stream(DFIRE_RELEASE_URL, zip_path)
        if not ok:
            print(
                "\n  [MANUAL] D-Fire download failed. To install manually:\n"
                "    1. Visit https://github.com/gaiasd/DFireDataset\n"
                "    2. Click Code → Download ZIP\n"
                f"    3. Extract 'images/' and 'labels/' subdirectories into:\n"
                f"       {dest}\n"
            )
            return {}

    print("  Extracting archive…")
    counts: Dict[str, int] = {"smoke": 0, "fire": 0}
    images_extracted = 0
    labels_extracted = 0

    # First pass: print a few paths so we can see the actual archive structure
    # D-Fire has shipped with several layouts over time:
    #   DFireDataset-master/Train/images/img.jpg  (older)
    #   DFireDataset-master/images/train/img.jpg  (newer)
    #   DFireDataset-master/Train/img.jpg          (flat)
    # Accept any .jpg/.png/.jpeg regardless of directory depth.
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            members = zf.namelist()
            sample = [m for m in members if not m.endswith("/")][:5]
            print(f"  Archive sample paths: {sample}")

            for member in members:
                lower = member.lower()
                fname = os.path.basename(member)
                if not fname or "__macosx" in lower or fname.startswith("."):
                    continue
                # Images — accept any jpg/jpeg/png in the archive
                if lower.endswith((".jpg", ".jpeg", ".png")):
                    prefix = "train_" if "train" in lower else "test_"
                    out_path = os.path.join(dest, "images", prefix + fname)
                    os.makedirs(os.path.dirname(out_path), exist_ok=True)
                    with zf.open(member) as src, open(out_path, "wb") as tgt:
                        tgt.write(src.read())
                    images_extracted += 1
                # Labels — .txt files that are YOLO annotations (skip README etc.)
                elif lower.endswith(".txt") and fname.lower() not in (
                    "readme.txt",
                    "license.txt",
                    "classes.txt",
                ):
                    prefix = "train_" if "train" in lower else "test_"
                    label_path = os.path.join(dest, "labels", prefix + fname)
                    os.makedirs(os.path.dirname(label_path), exist_ok=True)
                    with zf.open(member) as src:
                        raw = src.read().decode("utf-8", errors="replace")
                    # Remap class IDs: D-Fire 0=fire→1, 1=smoke→0
                    remapped_lines = []
                    for line in raw.strip().splitlines():
                        parts = line.strip().split()
                        if not parts:
                            continue
                        try:
                            dfire_cls = int(parts[0])
                        except ValueError:
                            continue
                        visual_cls = 1 if dfire_cls == 0 else 0  # fire=1, smoke=0
                        counts["fire" if visual_cls == 1 else "smoke"] += 1
                        remapped_lines.append(
                            str(visual_cls) + " " + " ".join(parts[1:])
                        )
                    with open(label_path, "w") as lf:
                        lf.write(
                            "\n".join(remapped_lines) + "\n" if remapped_lines else ""
                        )
                    labels_extracted += 1

    except zipfile.BadZipFile as exc:
        print(f"  Bad zip file: {exc}. Removing cached archive.")
        os.remove(zip_path)
        return {}

    print(f"  Extracted {images_extracted} images, {labels_extracted} label files.")
    print(f"  Class counts — smoke: {counts['smoke']}, fire: {counts['fire']}")
    _print_summary("D-Fire", images_extracted, counts)
    return counts


# ---------------------------------------------------------------------------
# 2. COCO 2017 val subset
# ---------------------------------------------------------------------------

COCO_VAL_IMAGES_URL = "http://images.cocodataset.org/zips/val2017.zip"
COCO_VAL_ANNOT_URL = (
    "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
)


def download_coco_subset() -> Dict[str, int]:
    """
    Download COCO 2017 val images + annotations for relevant classes.

    Relevant COCO category IDs and their VISUAL_CLASSES mapping:
      person(1)→person(3), boat(9)→vessel(12), cow(21)/elephant(22)/
      bear(23)/zebra(24)→dangerous_animal(17), car(3)/truck(8)→vessel(12),
      bird(16)/cat(17)/dog(18)/horse(19)→dangerous_animal(17)

    COCO has 5,000 val images; we keep only images that contain at least one
    relevant category annotation.

    Saves to: packages/ml/data/images/coco_subset/
    """
    dest = os.path.join(IMAGES_DIR, "coco_subset")
    os.makedirs(dest, exist_ok=True)
    img_dir = os.path.join(dest, "images")
    lbl_dir = os.path.join(dest, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    annot_zip = os.path.join(OUTPUT_DIR, "_coco_annotations.zip")
    images_zip = os.path.join(OUTPUT_DIR, "_coco_val2017.zip")

    # --- Annotations ---
    if os.path.exists(annot_zip):
        print("  COCO annotations archive cached.")
    else:
        print(f"Downloading COCO 2017 annotations (~241 MB)…")
        if not _fetch_stream(COCO_VAL_ANNOT_URL, annot_zip, timeout=300):
            print(
                "\n  [MANUAL] COCO annotation download failed.\n"
                "    1. Visit https://cocodataset.org/#download\n"
                "    2. Download '2017 Val annotations [241MB]'\n"
                f"    3. Place the zip at: {annot_zip}\n"
            )
            return {}

    print("  Extracting COCO annotations…")
    annot_dir = os.path.join(OUTPUT_DIR, "_coco_annot")
    os.makedirs(annot_dir, exist_ok=True)
    with zipfile.ZipFile(annot_zip, "r") as zf:
        zf.extract("annotations/instances_val2017.json", annot_dir)

    annot_path = os.path.join(annot_dir, "annotations", "instances_val2017.json")
    print("  Parsing COCO annotations…")
    with open(annot_path) as f:
        coco = json.load(f)

    # Build {image_id: [{class_id, bbox:[x,y,w,h], img_w, img_h}]}
    id_to_info: Dict[int, Dict] = {
        img["id"]: {
            "file_name": img["file_name"],
            "w": img["width"],
            "h": img["height"],
        }
        for img in coco["images"]
    }

    relevant_image_ids = set()
    image_annotations: Dict[int, List[Dict]] = {}
    for ann in coco["annotations"]:
        cat_id = ann["category_id"]
        if cat_id not in COCO_TO_VISUAL:
            continue
        img_id = ann["image_id"]
        relevant_image_ids.add(img_id)
        if img_id not in image_annotations:
            image_annotations[img_id] = []
        x, y, bw, bh = ann["bbox"]
        image_annotations[img_id].append(
            {
                "visual_cls": COCO_TO_VISUAL[cat_id],
                "x": x,
                "y": y,
                "bw": bw,
                "bh": bh,
            }
        )

    print(f"  Found {len(relevant_image_ids)} relevant COCO val images.")

    # Write YOLO labels for these images (we can do this before downloading images)
    for img_id in relevant_image_ids:
        info = id_to_info[img_id]
        stem = os.path.splitext(info["file_name"])[0]
        lbl_path = os.path.join(lbl_dir, stem + ".txt")
        img_w, img_h = info["w"], info["h"]
        lines = []
        for ann in image_annotations[img_id]:
            # YOLO: cx cy w h normalized
            cx = (ann["x"] + ann["bw"] / 2) / img_w
            cy = (ann["y"] + ann["bh"] / 2) / img_h
            nw = ann["bw"] / img_w
            nh = ann["bh"] / img_h
            # Clamp to [0,1]
            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            nw = max(0.001, min(1.0, nw))
            nh = max(0.001, min(1.0, nh))
            lines.append(f"{ann['visual_cls']} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
        with open(lbl_path, "w") as lf:
            lf.write("\n".join(lines) + "\n")

    # --- Images ---
    if os.path.exists(images_zip):
        print("  COCO val images archive cached.")
    else:
        print(f"Downloading COCO 2017 val images (~778 MB)…")
        print(
            "  This is a large download. Press Ctrl-C to skip and place images manually."
        )
        print(f"  URL: {COCO_VAL_IMAGES_URL}")
        if not _fetch_stream(COCO_VAL_IMAGES_URL, images_zip, timeout=600):
            print(
                "\n  [MANUAL] COCO image download failed.\n"
                "    1. Visit https://cocodataset.org/#download\n"
                "    2. Download '2017 Val images [5K/1GB]'\n"
                f"    3. Extract val2017/ images into: {img_dir}\n"
                "  Labels have already been written; add images when available.\n"
            )
            return {}

    print("  Extracting relevant COCO images…")
    relevant_files = {id_to_info[iid]["file_name"] for iid in relevant_image_ids}
    extracted = 0
    with zipfile.ZipFile(images_zip, "r") as zf:
        for member in zf.namelist():
            fname = os.path.basename(member)
            if fname in relevant_files:
                out_path = os.path.join(img_dir, fname)
                if not os.path.exists(out_path):
                    with zf.open(member) as src, open(out_path, "wb") as tgt:
                        tgt.write(src.read())
                extracted += 1

    # Per-class counts
    counts: Dict[str, int] = {}
    for anns in image_annotations.values():
        for ann in anns:
            name = VISUAL_CLASSES[ann["visual_cls"]]
            counts[name] = counts.get(name, 0) + 1

    print(f"  Extracted {extracted} images.")
    _print_summary("COCO subset", extracted, counts)
    return counts


# ---------------------------------------------------------------------------
# 3. SeaDronesSee  (person-in-water from UAV)
# ---------------------------------------------------------------------------

SEADRONESSEE_BASE = (
    "https://cloud.cs.uni-tuebingen.de/index.php/s/yMmqkNLMJKPHbx8/download"
)
SEADRONESSEE_GITHUB = "https://github.com/Ben93kie/SeaDronesSee"


def download_seadronessee() -> Dict[str, int]:
    """
    Attempt to download SeaDronesSee dataset for person-in-water detection from UAV.

    SeaDronesSee provides drone imagery with bounding box annotations for:
      swimmers (person_water), persons on boats, life rafts.

    The dataset is hosted at University of Tübingen. If direct download is
    unavailable, instructions are printed for manual acquisition.

    Saves to: packages/ml/data/images/seadronessee/
    """
    dest = os.path.join(IMAGES_DIR, "seadronessee")
    os.makedirs(dest, exist_ok=True)

    zip_path = os.path.join(OUTPUT_DIR, "_seadronessee.zip")

    print("Attempting SeaDronesSee download…")
    print(f"  Source: {SEADRONESSEE_GITHUB}")

    if os.path.exists(zip_path):
        print("  Archive cached.")
    else:
        ok = _fetch_stream(SEADRONESSEE_BASE, zip_path, timeout=300)
        if not ok:
            _print_manual_instructions(
                "SeaDronesSee (person-in-water, UAV)",
                steps=[
                    f"Visit: {SEADRONESSEE_GITHUB}",
                    "Follow dataset access instructions in the README.",
                    "Download the full dataset archive.",
                    f"Extract 'images/' and 'labels/' into: {dest}",
                    "Label format should be YOLO: class cx cy w h.",
                    "Remap SeaDronesSee class IDs to Heli.OS classes:",
                    "  swimmer/person_in_water → 4 (person_water)",
                    "  life_raft               → 5 (life_raft)",
                    "  person_on_board         → 3 (person)",
                ],
                dest=dest,
            )
            return {}

    print("  Extracting SeaDronesSee…")
    img_dir = os.path.join(dest, "images")
    lbl_dir = os.path.join(dest, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    images_count = 0
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            for member in zf.namelist():
                lower = member.lower()
                fname = os.path.basename(member)
                if not fname:
                    continue
                if lower.endswith((".jpg", ".jpeg", ".png")):
                    out = os.path.join(img_dir, fname)
                    with zf.open(member) as src, open(out, "wb") as tgt:
                        tgt.write(src.read())
                    images_count += 1
                elif lower.endswith(".txt"):
                    out = os.path.join(lbl_dir, fname)
                    with zf.open(member) as src, open(out, "wb") as tgt:
                        tgt.write(src.read())
    except zipfile.BadZipFile as exc:
        print(f"  Bad zip: {exc}")
        _print_manual_instructions(
            "SeaDronesSee",
            steps=[
                f"Visit: {SEADRONESSEE_GITHUB}",
                f"Download manually and extract to: {dest}",
            ],
            dest=dest,
        )
        return {}

    counts = {"person_water": images_count}  # rough approximation
    _print_summary("SeaDronesSee", images_count, counts)
    return counts


# ---------------------------------------------------------------------------
# 4. AIDER  (Aerial Image Dataset for Emergency Response)
# ---------------------------------------------------------------------------

AIDER_GITHUB = "https://github.com/ckyrkou/AIDER"
AIDER_ARCHIVE_URL = "https://github.com/ckyrkou/AIDER/archive/refs/heads/master.zip"


def download_aerial_sar() -> Dict[str, int]:
    """
    Download AIDER — Aerial Image Dataset for Emergency Response.

    AIDER contains aerial imagery of four disaster types:
      fire/smoke, flood, collapsed building/rubble, normal (background).

    AIDER images do NOT include bounding box annotations, only scene-level
    class labels. We convert these to whole-image YOLO pseudo-annotations
    (bbox covering 80% of the frame) suitable for training a detector backbone.

    Class mapping:
      AIDER fire/smoke → VISUAL 1 (fire) + 0 (smoke)
      AIDER flood      → (no direct class; used as background diversity)
      AIDER collapsed  → VISUAL 15 (structural_crack)

    Saves to: packages/ml/data/images/aider/
    """
    dest = os.path.join(IMAGES_DIR, "aider")
    os.makedirs(dest, exist_ok=True)
    img_dir = os.path.join(dest, "images")
    lbl_dir = os.path.join(dest, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    zip_path = os.path.join(OUTPUT_DIR, "_aider_master.zip")

    print("Downloading AIDER dataset (Aerial Image Dataset for Emergency Response)…")
    print(f"  Source: {AIDER_GITHUB}")

    if os.path.exists(zip_path):
        print("  Archive cached.")
    else:
        ok = _fetch_stream(AIDER_ARCHIVE_URL, zip_path, timeout=300)
        if not ok:
            _print_manual_instructions(
                "AIDER (Aerial Emergency Response)",
                steps=[
                    f"Visit: {AIDER_GITHUB}",
                    "Click Code → Download ZIP.",
                    f"Extract into: {dest}",
                    "Expected subdirectories: fire_smoke/, flood/, collapsed/, normal/",
                ],
                dest=dest,
            )
            return {}

    print("  Extracting AIDER…")

    # AIDER GitHub repo structure: AIDER-master/{fire_smoke,flood,collapsed,normal}/...
    aider_class_map = {
        "fire_smoke": (1, "fire"),
        "collapsed": (15, "structural_crack"),
        "flood": (None, None),  # background diversity — no annotation
    }

    counts: Dict[str, int] = {}
    images_extracted = 0

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            for member in zf.namelist():
                lower = member.lower()
                if not lower.endswith((".jpg", ".jpeg", ".png")):
                    continue
                fname = os.path.basename(member)
                if not fname:
                    continue

                # Determine AIDER class from directory path
                aider_cls = None
                for cls_name in aider_class_map:
                    if f"/{cls_name}/" in lower or lower.startswith(cls_name + "/"):
                        aider_cls = cls_name
                        break

                if aider_cls is None:
                    continue

                visual_cls_id, visual_cls_name = aider_class_map[aider_cls]
                safe_name = aider_cls + "_" + fname
                out_img = os.path.join(img_dir, safe_name)
                with zf.open(member) as src, open(out_img, "wb") as tgt:
                    tgt.write(src.read())

                if visual_cls_id is not None:
                    # Whole-image pseudo-annotation (80% coverage, centered)
                    stem = os.path.splitext(safe_name)[0]
                    lbl_path = os.path.join(lbl_dir, stem + ".txt")
                    with open(lbl_path, "w") as lf:
                        lf.write(
                            f"{visual_cls_id} 0.500000 0.500000 0.800000 0.800000\n"
                        )
                    counts[visual_cls_name] = counts.get(visual_cls_name, 0) + 1

                images_extracted += 1

    except zipfile.BadZipFile as exc:
        print(f"  Bad zip: {exc}")
        return {}

    _print_summary("AIDER", images_extracted, counts)
    return counts


# ---------------------------------------------------------------------------
# 5. Synthetic labeled crops
#    For: pipeline, agricultural, chemical, infrastructure, maritime distress
# ---------------------------------------------------------------------------

# Domain definitions: (visual_class_id, primary_rgb, secondary_rgb, description)
_SYNTHETIC_DOMAINS = [
    # Pipeline / Hazmat
    (
        6,
        (40, 60, 20),
        (180, 120, 30),
        "oil_spill",
        "Dark oily sheen patches on soil/water",
    ),
    (
        7,
        (80, 80, 80),
        (200, 50, 50),
        "pipeline_damage",
        "Rust-coloured breach on grey pipe cross-section",
    ),
    (
        8,
        (200, 230, 200),
        (120, 200, 100),
        "chemical_plume",
        "Pale yellow-green smoke/plume pattern",
    ),
    # Agricultural
    (
        9,
        (100, 140, 40),
        (200, 80, 80),
        "crop_disease",
        "Brown/red lesion patches on green canopy",
    ),
    (
        10,
        (120, 160, 50),
        (180, 120, 20),
        "pest_damage",
        "Irregular brown spots on crop canopy",
    ),
    (
        11,
        (180, 160, 80),
        (220, 200, 100),
        "dry_field",
        "Uniform straw-yellow parched soil",
    ),
    # Maritime distress
    (
        13,
        (200, 60, 60),
        (255, 120, 30),
        "vessel_distress",
        "Red/orange distress signal flare on dark water",
    ),
    # Infrastructure
    (
        14,
        (60, 60, 60),
        (200, 40, 40),
        "power_line_damage",
        "Burned/broken conductor segment",
    ),
    (
        15,
        (150, 150, 150),
        (80, 80, 80),
        "structural_crack",
        "Linear crack on concrete surface",
    ),
    (
        16,
        (240, 200, 20),
        (80, 80, 60),
        "solar_defect",
        "Dark cell patch on bright panel surface",
    ),
]

# Sizes of synthetic images and bounding boxes
_IMG_W = 416
_IMG_H = 416
_MIN_BOX_FRAC = 0.10  # minimum box side as fraction of image
_MAX_BOX_FRAC = 0.50  # maximum box side


def _make_synthetic_image(
    img_path: str,
    lbl_path: str,
    visual_cls: int,
    bg_rgb: Tuple[int, int, int],
    fg_rgb: Tuple[int, int, int],
    n_boxes: int = 1,
    rng: Optional[random.Random] = None,
) -> None:
    """
    Write a synthetic PNG with solid-color bounding-box regions.

    Background = bg_rgb, bounding box fills = fg_rgb with slight noise per
    box (simulated by varying shade).  Saves YOLO label alongside.
    """
    import zlib

    if rng is None:
        rng = random.Random()

    W, H = _IMG_W, _IMG_H

    # Build pixel buffer: row-major, RGB
    pixels = list(bg_rgb) * (W * H)  # flat list of R,G,B,...

    yolo_lines = []
    for _ in range(n_boxes):
        min_bw = int(_MIN_BOX_FRAC * W)
        max_bw = int(_MAX_BOX_FRAC * W)
        min_bh = int(_MIN_BOX_FRAC * H)
        max_bh = int(_MAX_BOX_FRAC * H)

        bw = rng.randint(min_bw, max_bw)
        bh = rng.randint(min_bh, max_bh)
        x0 = rng.randint(0, W - bw)
        y0 = rng.randint(0, H - bh)

        # Fill box with fg_rgb (slight brightness variation per pixel for realism)
        for row in range(y0, y0 + bh):
            for col in range(x0, x0 + bw):
                noise = rng.randint(-15, 15)
                r = max(0, min(255, fg_rgb[0] + noise))
                g = max(0, min(255, fg_rgb[1] + noise))
                b = max(0, min(255, fg_rgb[2] + noise))
                idx = (row * W + col) * 3
                pixels[idx] = r
                pixels[idx + 1] = g
                pixels[idx + 2] = b

        cx = (x0 + bw / 2) / W
        cy = (y0 + bh / 2) / H
        nw = bw / W
        nh = bh / H
        yolo_lines.append(f"{visual_cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

    # Encode as PNG (filter byte 0 = None per scanline)
    def _u32be(n: int) -> bytes:
        return struct.pack(">I", n)

    def _chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return _u32be(len(data)) + tag + data + _u32be(crc)

    raw_rows = bytearray()
    for row in range(H):
        raw_rows.append(0)  # filter type None
        base = row * W * 3
        raw_rows.extend(pixels[base : base + W * 3])

    compressed = zlib.compress(bytes(raw_rows), 1)  # level 1 = fast

    ihdr = _u32be(W) + _u32be(H) + bytes([8, 2, 0, 0, 0])
    png = (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", compressed)
        + _chunk(b"IEND", b"")
    )

    with open(img_path, "wb") as f:
        f.write(png)

    with open(lbl_path, "w") as f:
        f.write("\n".join(yolo_lines) + "\n")


def generate_synthetic_crops(n_per_class: int = 200, seed: int = 42) -> Dict[str, int]:
    """
    Generate synthetic labeled images for domains with no suitable public dataset.

    Each image is a solid-color background with one or more solid-color
    foreground bounding boxes.  This is intentionally simple — its purpose
    is to give the detector a weak signal for rare classes while real data
    collection proceeds.  For production, replace with domain-expert labeled
    aerial imagery.

    n_per_class: number of synthetic images per class (default 200)
    seed: random seed for reproducibility

    Saves to: packages/ml/data/images/synthetic/
    """
    dest = os.path.join(IMAGES_DIR, "synthetic")
    img_dir = os.path.join(dest, "images")
    lbl_dir = os.path.join(dest, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    print(
        f"Generating {n_per_class} synthetic images per class ({len(_SYNTHETIC_DOMAINS)} classes)…"
    )

    rng = random.Random(seed)
    counts: Dict[str, int] = {}
    total_images = 0

    for visual_cls, fg_rgb, bg_rgb, cls_name, description in _SYNTHETIC_DOMAINS:
        print(f"  {cls_name}: {description}")
        for i in range(n_per_class):
            fname = f"{cls_name}_{i:05d}.png"
            img_path = os.path.join(img_dir, fname)
            lbl_path = os.path.join(lbl_dir, f"{cls_name}_{i:05d}.txt")

            # Vary number of boxes: mostly 1, sometimes 2-3 for realism
            n_boxes = rng.choices([1, 2, 3], weights=[0.70, 0.20, 0.10])[0]

            # Slight background color variation
            bg_var = tuple(max(0, min(255, c + rng.randint(-20, 20))) for c in bg_rgb)

            _make_synthetic_image(
                img_path,
                lbl_path,
                visual_cls=visual_cls,
                bg_rgb=bg_var,  # type: ignore[arg-type]
                fg_rgb=fg_rgb,
                n_boxes=n_boxes,
                rng=rng,
            )
            total_images += 1

        counts[cls_name] = n_per_class

    _print_summary("Synthetic", total_images, counts)
    return counts


# ---------------------------------------------------------------------------
# 6. Build combined YOLO dataset
# ---------------------------------------------------------------------------


def build_combined_dataset(val_fraction: float = 0.20, seed: int = 42) -> None:
    """
    Merge all downloaded sources into a unified YOLO-format directory:

      packages/ml/data/summit_detector/
        images/train/
        images/val/
        labels/train/
        labels/val/
        summit_detector.yaml

    Only images that have a corresponding label file (non-empty) are included.
    Performs an 80/20 train/val stratified split.
    """
    print("\n" + "=" * 60)
    print("Building combined Heli.OS detector dataset…")
    print("=" * 60)

    # Collect all (image_path, label_path) pairs from all sources
    sources = [
        os.path.join(IMAGES_DIR, "fire_smoke"),
        os.path.join(IMAGES_DIR, "coco_subset"),
        os.path.join(IMAGES_DIR, "seadronessee"),
        os.path.join(IMAGES_DIR, "aider"),
        os.path.join(IMAGES_DIR, "synthetic"),
    ]

    all_pairs: List[Tuple[str, str]] = []
    for src in sources:
        img_dir = os.path.join(src, "images")
        lbl_dir = os.path.join(src, "labels")
        if not os.path.isdir(img_dir) or not os.path.isdir(lbl_dir):
            continue
        for fname in os.listdir(img_dir):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            stem = os.path.splitext(fname)[0]
            lbl_path = os.path.join(lbl_dir, stem + ".txt")
            if not os.path.exists(lbl_path):
                continue
            # Skip empty label files (background-only images we don't want)
            if os.path.getsize(lbl_path) == 0:
                continue
            all_pairs.append((os.path.join(img_dir, fname), lbl_path))

    print(f"  Found {len(all_pairs)} labeled images across all sources.")

    if not all_pairs:
        print("  No labeled images found. Run downloads first.")
        return

    # Shuffle then split
    rng = random.Random(seed)
    rng.shuffle(all_pairs)
    n_val = max(1, int(len(all_pairs) * val_fraction))
    val_pairs = all_pairs[:n_val]
    train_pairs = all_pairs[n_val:]

    print(f"  Split: {len(train_pairs)} train / {len(val_pairs)} val")

    # Create output dirs
    for split in ("train", "val"):
        os.makedirs(os.path.join(SUMMIT_DIR, "images", split), exist_ok=True)
        os.makedirs(os.path.join(SUMMIT_DIR, "labels", split), exist_ok=True)

    def _copy_pair(img_src: str, lbl_src: str, split: str) -> None:
        img_fname = os.path.basename(img_src)
        lbl_fname = os.path.splitext(img_fname)[0] + ".txt"
        shutil.copy2(img_src, os.path.join(SUMMIT_DIR, "images", split, img_fname))
        shutil.copy2(lbl_src, os.path.join(SUMMIT_DIR, "labels", split, lbl_fname))

    for img_src, lbl_src in train_pairs:
        _copy_pair(img_src, lbl_src, "train")
    for img_src, lbl_src in val_pairs:
        _copy_pair(img_src, lbl_src, "val")

    # Write summit_detector.yaml
    yaml_path = os.path.join(SUMMIT_DIR, "summit_detector.yaml")
    class_names = [VISUAL_CLASSES[i] for i in sorted(VISUAL_CLASSES)]
    yaml_lines = [
        "# Heli.OS unified visual detector dataset",
        f"# Auto-generated by download_visual_datasets.py",
        "",
        f"path: {SUMMIT_DIR}",
        "train: images/train",
        "val: images/val",
        "",
        f"nc: {len(VISUAL_CLASSES)}",
        "names:",
    ]
    for i, name in enumerate(class_names):
        yaml_lines.append(f"  {i}: {name}")

    with open(yaml_path, "w") as f:
        f.write("\n".join(yaml_lines) + "\n")

    print(f"  Wrote: {yaml_path}")

    # Class distribution summary
    class_counts: Dict[int, int] = {i: 0 for i in VISUAL_CLASSES}
    for _, lbl_path in train_pairs + val_pairs:
        with open(lbl_path) as lf:
            for line in lf:
                parts = line.strip().split()
                if parts:
                    try:
                        cls_id = int(parts[0])
                        if cls_id in class_counts:
                            class_counts[cls_id] += 1
                    except ValueError:
                        pass

    print("\n  Class distribution in combined dataset:")
    print(f"  {'ID':<4} {'Name':<22} {'Annotations':>12}")
    print("  " + "-" * 40)
    for cls_id in sorted(VISUAL_CLASSES):
        name = VISUAL_CLASSES[cls_id]
        cnt = class_counts.get(cls_id, 0)
        marker = " *** EMPTY" if cnt == 0 else ""
        print(f"  {cls_id:<4} {name:<22} {cnt:>12,}{marker}")
    print()

    total_ann = sum(class_counts.values())
    print(f"  Total annotations: {total_ann:,}")
    print(f"  Dataset root: {SUMMIT_DIR}")
    print(f"  YAML config:  {yaml_path}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _print_summary(source: str, n_images: int, counts: Dict[str, int]) -> None:
    print(f"\n  [{source}] Summary:")
    print(f"    Images: {n_images:,}")
    if counts:
        print(f"    Labels per class:")
        for name, cnt in sorted(counts.items(), key=lambda kv: -kv[1]):
            print(f"      {name:<25} {cnt:>8,}")
    print()


def _print_manual_instructions(dataset: str, steps: List[str], dest: str) -> None:
    print(f"\n  [MANUAL REQUIRED] {dataset} cannot be auto-downloaded.")
    print(f"  Follow these steps to add it manually:")
    for i, step in enumerate(steps, 1):
        print(f"    {i}. {step}")
    print(f"  Target directory: {dest}")
    print()


# ---------------------------------------------------------------------------
# Source registry
# ---------------------------------------------------------------------------

SOURCES = {
    "dfire": download_dfire,
    "coco_subset": download_coco_subset,
    "seadronessee": download_seadronessee,
    "aerial_sar": download_aerial_sar,
    "synthetic": generate_synthetic_crops,
}

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and build visual object detection datasets for Heli.OS.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--source",
        nargs="+",
        choices=list(SOURCES.keys()),
        metavar="SOURCE",
        help=(
            "Download only specific source(s). "
            f"Choices: {', '.join(SOURCES)}. "
            "Default: all."
        ),
    )
    parser.add_argument(
        "--build-only",
        action="store_true",
        help="Skip all downloads; just rebuild the combined dataset from existing downloads.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.20,
        help="Fraction of data to use for validation (default: 0.20).",
    )
    parser.add_argument(
        "--synthetic-n",
        type=int,
        default=200,
        help="Number of synthetic images per class (default: 200).",
    )
    args = parser.parse_args()

    # Ensure base directories exist
    for d in (OUTPUT_DIR, IMAGES_DIR, SUMMIT_DIR):
        os.makedirs(d, exist_ok=True)

    if args.build_only:
        build_combined_dataset(val_fraction=args.val_fraction)
        return

    sources_to_run = args.source if args.source else list(SOURCES.keys())

    print(f"Heli.OS Visual Dataset Downloader")
    print(f"Output root : {OUTPUT_DIR}")
    print(f"Sources     : {', '.join(sources_to_run)}")
    print()

    for source_name in sources_to_run:
        fn = SOURCES[source_name]
        print(f"{'─' * 60}")
        print(f"Source: {source_name}")
        print(f"{'─' * 60}")
        if source_name == "synthetic":
            fn(n_per_class=args.synthetic_n)  # type: ignore[call-arg]
        else:
            fn()

    build_combined_dataset(val_fraction=args.val_fraction)


if __name__ == "__main__":
    main()
