"""
Train a small CNN appearance embedding model for Summit.OS cross-camera entity re-identification.

Produces the ONNX model consumed by apps/fusion/reid.py (_ONNXEmbedder).

Input:  image crop (B, 3, 128, 64) — float32, normalized
Output: (B, 128) L2-normalized appearance embedding

Cross-camera matching in reid.py uses cosine similarity between embeddings.
Match threshold (REID_MATCH_THRESHOLD_ONNX env var) defaults to 0.75.

NOTE on reid.py INPUT_H/INPUT_W:
  _ONNXEmbedder in reid.py resizes crops to (W=128, H=256) before passing to the model,
  producing NCHW tensor (1, 3, 256, 128).  This training script uses (3, 128, 64) as
  specified — the production model should be retrained with (3, 256, 128) once real data
  is available. For the synthetic stub the stored config reflects the actual trained shape.

Outputs:
  packages/ml/models/reid_embedder.onnx
  packages/ml/models/reid_embedder_config.json

Usage:
  python train_reid_embedder.py
  python train_reid_embedder.py --epochs 100 --samples 600 --output-dir ./models
"""

import onnx_compat  # noqa: F401 — Python 3.14 compat patch
import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "models"

# ── Torch availability check ──────────────────────────────────────────────────

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# ── Architecture ──────────────────────────────────────────────────────────────

if TORCH_AVAILABLE:

    class _DepthwiseSeparable(nn.Module):
        """Depthwise separable convolution: depthwise + pointwise."""

        def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
            super().__init__()
            self.dw = nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1,
                                groups=in_ch, bias=False)
            self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
            self.bn_dw = nn.BatchNorm2d(in_ch)
            self.bn_pw = nn.BatchNorm2d(out_ch)

        def forward(self, x):
            x = F.relu6(self.bn_dw(self.dw(x)))
            x = F.relu6(self.bn_pw(self.pw(x)))
            return x

    class _InvertedResidual(nn.Module):
        """MobileNetV2-style inverted residual block."""

        def __init__(self, in_ch: int, out_ch: int, stride: int = 1,
                     expansion: int = 6):
            super().__init__()
            mid_ch = in_ch * expansion
            self.use_residual = (stride == 1 and in_ch == out_ch)

            layers = []
            if expansion != 1:
                layers += [
                    nn.Conv2d(in_ch, mid_ch, 1, bias=False),
                    nn.BatchNorm2d(mid_ch),
                    nn.ReLU6(inplace=True),
                ]
            layers += [
                nn.Conv2d(mid_ch, mid_ch, 3, stride=stride, padding=1,
                          groups=mid_ch, bias=False),
                nn.BatchNorm2d(mid_ch),
                nn.ReLU6(inplace=True),
                nn.Conv2d(mid_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
            ]
            self.conv = nn.Sequential(*layers)

        def forward(self, x):
            out = self.conv(x)
            if self.use_residual:
                out = out + x
            return out

    class ReIDEmbedder(nn.Module):
        """
        Lightweight re-ID embedding network.

        Input:  (B, 3, 128, 64) — person/vehicle crop, standard re-ID aspect ratio
        Output: (B, 128) L2-normalized embedding

        Architecture: MobileNetV2-inspired with depthwise separable convolutions
        for edge deployment (<2 MB parameter budget).
        """

        def __init__(self, embedding_dim: int = 128):
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU6(inplace=True),
            )
            # InvertedResidual blocks: 32 → 64 → 128 → 128
            self.blocks = nn.Sequential(
                _InvertedResidual(32, 64,  stride=2, expansion=6),
                _InvertedResidual(64, 64,  stride=1, expansion=6),
                _InvertedResidual(64, 128, stride=2, expansion=6),
                _InvertedResidual(128, 128, stride=1, expansion=6),
            )
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.head = nn.Linear(128, embedding_dim)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            x = self.stem(x)
            x = self.blocks(x)
            x = self.pool(x)
            x = x.flatten(1)
            x = self.head(x)
            # L2 normalize
            x = F.normalize(x, p=2, dim=1)
            return x

# ── Synthetic data generation ─────────────────────────────────────────────────

def _make_identity_base_colors(n_ids: int, rng: np.random.Generator
                               ) -> np.ndarray:
    """Each identity gets a unique base RGB color (float32, 0-1)."""
    return rng.uniform(0.0, 1.0, size=(n_ids, 3)).astype(np.float32)


def _generate_crop(base_color: np.ndarray, rng: np.random.Generator,
                   h: int = 128, w: int = 64) -> np.ndarray:
    """
    Synthesize a (3, H, W) float32 image patch for one identity.

    The patch is a color-textured rectangle with the identity's base color
    as the dominant hue, plus random noise and texture.
    """
    # Start with base color broadcast to full patch
    patch = np.tile(base_color[:, None, None], (1, h, w)).astype(np.float32)

    # Add spatial texture: random low-frequency pattern
    freq = rng.integers(2, 6)
    grid_x = np.linspace(0, freq * np.pi, w, dtype=np.float32)
    grid_y = np.linspace(0, freq * np.pi, h, dtype=np.float32)
    texture = (np.sin(grid_y)[:, None] * np.cos(grid_x)[None, :]) * 0.15
    patch += texture[None, :, :]  # broadcast to C

    # Gaussian noise
    patch += rng.normal(0, 0.05, patch.shape).astype(np.float32)

    return np.clip(patch, 0.0, 1.0)


def _augment_crop(patch: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply random augmentations to a synthetic crop."""
    h, w = patch.shape[1], patch.shape[2]

    # Brightness jitter ±30%
    factor = 1.0 + rng.uniform(-0.3, 0.3)
    patch = patch * factor

    # Horizontal flip
    if rng.random() > 0.5:
        patch = patch[:, :, ::-1].copy()

    # Random crop: take 90-100% of the patch
    scale = rng.uniform(0.9, 1.0)
    ch = int(h * scale)
    cw = int(w * scale)
    y0 = rng.integers(0, h - ch + 1)
    x0 = rng.integers(0, w - cw + 1)
    cropped = patch[:, y0:y0 + ch, x0:x0 + cw]

    # Resize back to original size via simple repeat (no cv2 dependency)
    # Use numpy resize (wraps) then crop to correct size
    ry = np.round(np.linspace(0, ch - 1, h)).astype(int)
    rx = np.round(np.linspace(0, cw - 1, w)).astype(int)
    patch = cropped[:, ry][:, :, rx]

    # Channel-wise hue shift (approximate: shift per-channel)
    shift = rng.uniform(-0.15, 0.15, (3, 1, 1)).astype(np.float32)
    patch = patch + shift

    return np.clip(patch, 0.0, 1.0).astype(np.float32)


# ── Triplet dataset ───────────────────────────────────────────────────────────

if TORCH_AVAILABLE:

    class TripletDataset(Dataset):
        """
        Online triplet sampling dataset.

        Each __getitem__ call returns (anchor, positive, negative) where
        anchor and positive are augmented crops of the same identity and
        negative is an augmented crop of a different identity.
        """

        def __init__(self, n_ids: int, triplets_per_epoch: int,
                     rng_seed: int = 42):
            self.n_ids = n_ids
            self.triplets_per_epoch = triplets_per_epoch
            self._rng = np.random.default_rng(rng_seed)

            # Pre-generate base colors for each identity
            self._bases = _make_identity_base_colors(n_ids, self._rng)

        def __len__(self):
            return self.triplets_per_epoch

        def __getitem__(self, idx):
            rng = np.random.default_rng(idx * 9973 + 1)  # deterministic per idx

            # Anchor identity
            anchor_id = rng.integers(0, self.n_ids)
            neg_id = rng.integers(0, self.n_ids - 1)
            if neg_id >= anchor_id:
                neg_id += 1  # ensure different identity

            base = self._bases[anchor_id]
            anchor = _augment_crop(_generate_crop(base, rng), rng)
            positive = _augment_crop(_generate_crop(base, rng), rng)
            negative = _augment_crop(
                _generate_crop(self._bases[neg_id], rng), rng
            )

            return (
                torch.from_numpy(anchor),
                torch.from_numpy(positive),
                torch.from_numpy(negative),
            )


# ── Training ──────────────────────────────────────────────────────────────────

def _recall_at_1(model: "nn.Module", n_ids: int = 50, crops_per_id: int = 10,
                 device: "torch.device" = None) -> float:
    """
    Estimate Recall@1 on a small held-out identity set.

    For each identity, one crop is the query; the remainder are gallery.
    Returns fraction of queries where the nearest gallery entry matches.
    """
    if device is None:
        device = torch.device("cpu")
    model.eval()
    rng = np.random.default_rng(999)
    bases = _make_identity_base_colors(n_ids, rng)

    # Build gallery: (crops_per_id - 1) crops per identity
    gallery_embs, gallery_ids = [], []
    query_embs, query_ids = [], []

    with torch.no_grad():
        for iid in range(n_ids):
            crops = [
                _augment_crop(_generate_crop(bases[iid], rng), rng)
                for _ in range(crops_per_id)
            ]
            batch = torch.from_numpy(
                np.stack(crops, axis=0)
            ).to(device)
            embs = model(batch).cpu().numpy()
            query_embs.append(embs[0])
            query_ids.append(iid)
            for e in embs[1:]:
                gallery_embs.append(e)
                gallery_ids.append(iid)

    gallery_embs = np.stack(gallery_embs)  # (G, 128)
    query_embs   = np.stack(query_embs)    # (Q, 128)

    # Cosine similarity: dot product (embeddings are L2-normalized)
    sims = query_embs @ gallery_embs.T     # (Q, G)
    preds = np.argmax(sims, axis=1)
    predicted_ids = [gallery_ids[p] for p in preds]
    correct = sum(p == q for p, q in zip(predicted_ids, query_ids))
    return correct / len(query_ids)


def _train_torch(n_ids: int, epochs: int, batch_size: int,
                 output_dir: Path) -> Path:
    """Train with PyTorch and export to ONNX."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    triplets_per_epoch = n_ids * 200  # 200 triplets per identity per epoch
    # Cap to avoid very long runs during development
    triplets_per_epoch = min(triplets_per_epoch, 20000)

    dataset = TripletDataset(n_ids=n_ids, triplets_per_epoch=triplets_per_epoch)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                         num_workers=0, pin_memory=(device.type == "cuda"))

    model = ReIDEmbedder(embedding_dim=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.TripletMarginLoss(margin=0.3, p=2, reduction="mean")

    print(f"\n  Training ReIDEmbedder | {epochs} epochs | "
          f"{triplets_per_epoch} triplets/epoch | batch={batch_size}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()

    loss_history = []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses = []

        for anchor, positive, negative in loader:
            anchor   = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            emb_a = model(anchor)
            emb_p = model(positive)
            emb_n = model(negative)

            loss = criterion(emb_a, emb_p, emb_n)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        scheduler.step()
        mean_loss = float(np.mean(epoch_losses))
        loss_history.append(mean_loss)

        if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
            lr = scheduler.get_last_lr()[0]
            print(f"  Epoch {epoch:>4}/{epochs}  "
                  f"triplet_loss={mean_loss:.4f}  lr={lr:.6f}")

    # Recall@1 on held-out set
    recall = _recall_at_1(model, n_ids=100, crops_per_id=10, device=device)
    print(f"\n  Recall@1 (held-out 100 identities): {recall:.3f}")

    # Export to ONNX
    model.eval()
    dummy = torch.randn(1, 3, 128, 64, device=device)
    onnx_path = output_dir / "reid_embedder.onnx"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(
            model,
            dummy,
            str(onnx_path),
            input_names=["image"],
            output_names=["embedding"],
            dynamic_axes={
                "image":     {0: "batch"},
                "embedding": {0: "batch"},
            },
            opset_version=17,
        )

    size_kb = onnx_path.stat().st_size / 1024
    print(f"\n  ONNX model saved: {onnx_path}  ({size_kb:.1f} KB)")
    return onnx_path


# ── Fallback: PCA stub via skl2onnx ──────────────────────────────────────────

def _train_pca_stub(output_dir: Path) -> Path:
    """
    Create a stub ONNX embedder using sklearn PCA on flattened random crops.

    This produces the correct input/output interface for reid.py even though
    embedding quality is low.  Use the PyTorch path for any real deployment.
    """
    print()
    print("  *** WARNING: torch not available — creating PCA stub embedder ***")
    print("  *** Install PyTorch for a real re-ID model.                   ***")
    print()

    from sklearn.decomposition import PCA
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    H, W, C = 128, 64, 3
    flat_dim = H * W * C  # 24576
    embedding_dim = 128
    n_samples = 2000

    rng = np.random.default_rng(42)
    n_ids = 200
    bases = _make_identity_base_colors(n_ids, rng)

    X = []
    for _ in range(n_samples):
        iid = rng.integers(0, n_ids)
        crop = _augment_crop(_generate_crop(bases[iid], rng), rng)
        X.append(crop.flatten())
    X = np.array(X, dtype=np.float32)

    print(f"  Fitting PCA ({embedding_dim} components) on {n_samples} synthetic crops ...")
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("pca",    PCA(n_components=embedding_dim, whiten=True, random_state=42)),
    ])
    pipe.fit(X)

    onnx_path = output_dir / "reid_embedder.onnx"
    initial_type = [("image_flat", FloatTensorType([None, flat_dim]))]
    onnx_model = convert_sklearn(pipe, initial_types=initial_type, target_opset=12)
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    size_kb = onnx_path.stat().st_size / 1024
    print(f"  PCA stub ONNX saved: {onnx_path}  ({size_kb:.1f} KB)")
    print("  NOTE: This stub takes a flat (B, 24576) input, not (B, 3, 128, 64).")
    print("        Replace with the PyTorch-trained model for production.")
    return onnx_path


# ── Config JSON ───────────────────────────────────────────────────────────────

def _save_config(output_dir: Path, *, input_size: list, embedding_dim: int,
                 backend: str) -> Path:
    cfg = {
        "input_size":    input_size,   # [H, W]
        "embedding_dim": embedding_dim,
        "channels":      3,
        "normalize":     True,
        "backend":       backend,
        "note": (
            "reid.py _ONNXEmbedder resizes crops to (W=128, H=256). "
            "Retrain with input_size=[256,128] once real labeled crops are available."
        ),
    }
    cfg_path = output_dir / "reid_embedder_config.json"
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"  Config saved: {cfg_path}")
    return cfg_path


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train Summit.OS re-ID appearance embedder"
    )
    parser.add_argument(
        "--samples", type=int, default=500,
        help="Number of synthetic identities to generate (default: 500)"
    )
    parser.add_argument(
        "--epochs", type=int, default=50,
        help="Training epochs (PyTorch path only, default: 50)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Triplet batch size (default: 64)"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for .onnx + .json (default: {DEFAULT_OUTPUT_DIR})"
    )
    args = parser.parse_args()

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Summit.OS — ReID Embedder Training")
    print("=" * 60)
    print(f"  Identities:   {args.samples}")
    print(f"  Output dir:   {output_dir}")

    if TORCH_AVAILABLE:
        print(f"  Backend:      PyTorch {torch.__version__}")
        print(f"  Epochs:       {args.epochs}")
        print(f"  Batch size:   {args.batch_size}")
        onnx_path = _train_torch(
            n_ids=args.samples,
            epochs=args.epochs,
            batch_size=args.batch_size,
            output_dir=output_dir,
        )
        _save_config(
            output_dir,
            input_size=[128, 64],
            embedding_dim=128,
            backend="pytorch",
        )
    else:
        print("  Backend:      PCA stub (torch not available)")
        onnx_path = _train_pca_stub(output_dir)
        _save_config(
            output_dir,
            input_size=[128, 64],
            embedding_dim=128,
            backend="pca_stub",
        )

    print()
    print("Done.")
    print(f"  Model:  {onnx_path}")
    print(f"  Config: {output_dir / 'reid_embedder_config.json'}")
    print()
    print("To use: set REID_MODEL_PATH to the .onnx path before starting the fusion service.")


if __name__ == "__main__":
    main()
