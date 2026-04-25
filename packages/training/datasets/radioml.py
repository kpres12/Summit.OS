"""
DeepSig RadioML 2016 / 2018 — RF Modulation Classification Dataset
======================================================================
DeepSig's open RadioML datasets are the standard benchmark for RF
signal modulation classification — a core capability for counter-UAS,
EW awareness, drone detection, and signal intelligence.

Source:    https://www.deepsig.ai/datasets
License:   CC BY-NC-SA 4.0 (research / non-commercial). Commercial use
           requires DeepSig license.
Auth:      Free registration at deepsig.ai gates the download links.

Editions supported by this loader:
    RML2016.10a   600 MB  HDF5  11 modulations  20 SNR levels  220k samples
    RML2018.01a   21 GB   HDF5  24 modulations  26 SNR levels  2.6M samples

Both contain raw I/Q samples in (2, 1024) per example (I and Q channels)
along with the modulation label and SNR.

Expected on-disk layout:
  packages/training/data/radioml/
    RML2016.10a_dict.pkl          (legacy 2016 release, pickle)
    GOLD_XYZ_OSC.0001_1024.hdf5   (2018 release, HDF5)

Usage:
    from packages.training.datasets.radioml import (
        load_radioml_2016, load_radioml_2018, MODULATION_CLASSES_2018,
    )
    X, y_mod, y_snr = load_radioml_2018()
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

OUT_DIR = Path(__file__).parent.parent / "data" / "radioml"

PATH_2016 = OUT_DIR / "RML2016.10a_dict.pkl"
PATH_2018 = OUT_DIR / "GOLD_XYZ_OSC.0001_1024.hdf5"

# Mod class lists (canonical order published by DeepSig)
MODULATION_CLASSES_2016 = [
    "8PSK", "AM-DSB", "AM-SSB", "BPSK", "CPFSK", "GFSK",
    "PAM4", "QAM16", "QAM64", "QPSK", "WBFM",
]

MODULATION_CLASSES_2018 = [
    "OOK", "4ASK", "8ASK", "BPSK", "QPSK", "8PSK", "16PSK", "32PSK",
    "16APSK", "32APSK", "64APSK", "128APSK",
    "16QAM", "32QAM", "64QAM", "128QAM", "256QAM",
    "AM-SSB-WC", "AM-SSB-SC", "AM-DSB-WC", "AM-DSB-SC", "FM",
    "GMSK", "OQPSK",
]


# ---------------------------------------------------------------------------
# 2016 loader (pickle)
# ---------------------------------------------------------------------------


def load_radioml_2016(path: Optional[Path] = None,
                      max_samples: Optional[int] = None
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load RML2016.10a from the DeepSig pickle.

    Returns:
        X:     (N, 2, 128) float32 — I/Q samples
        y_mod: (N,) int64 — modulation class index
        y_snr: (N,) int64 — SNR in dB
    """
    p = path or PATH_2016
    if not p.exists():
        logger.warning(
            "[radioml] %s not found. Register at deepsig.ai/datasets, "
            "download RML2016.10a, and place the pickle at %s",
            p, OUT_DIR,
        )
        return np.zeros((0, 2, 128), dtype=np.float32), np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)

    import pickle
    with p.open("rb") as f:
        data = pickle.load(f, encoding="latin1")

    cls_to_idx = {m: i for i, m in enumerate(MODULATION_CLASSES_2016)}
    X_all, y_mod_all, y_snr_all = [], [], []
    for (mod, snr), arr in data.items():
        idx = cls_to_idx.get(mod)
        if idx is None:
            continue
        X_all.append(arr.astype(np.float32))
        y_mod_all.append(np.full(arr.shape[0], idx, dtype=np.int64))
        y_snr_all.append(np.full(arr.shape[0], int(snr), dtype=np.int64))

    X = np.concatenate(X_all)
    y_mod = np.concatenate(y_mod_all)
    y_snr = np.concatenate(y_snr_all)
    if max_samples is not None and len(X) > max_samples:
        rng = np.random.default_rng(7)
        idx = rng.choice(len(X), max_samples, replace=False)
        X, y_mod, y_snr = X[idx], y_mod[idx], y_snr[idx]
    logger.info("[radioml-2016] loaded %d samples × %d classes × %d SNRs",
                len(X), len(set(y_mod.tolist())), len(set(y_snr.tolist())))
    return X, y_mod, y_snr


# ---------------------------------------------------------------------------
# 2018 loader (HDF5)
# ---------------------------------------------------------------------------


def load_radioml_2018(path: Optional[Path] = None,
                      max_samples: Optional[int] = None,
                      min_snr: Optional[int] = None
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load RML2018.01a from HDF5.

    Args:
        max_samples: subsample to this many examples (uniform random)
        min_snr:     drop examples below this SNR (dB)

    Returns:
        X:     (N, 2, 1024) float32
        y_mod: (N,) int64
        y_snr: (N,) int64
    """
    p = path or PATH_2018
    if not p.exists():
        logger.warning(
            "[radioml] %s not found. Register at deepsig.ai/datasets, "
            "download RML2018.01a (21 GB HDF5), and place at %s",
            p, OUT_DIR,
        )
        return np.zeros((0, 2, 1024), dtype=np.float32), np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)

    try:
        import h5py
    except ImportError:
        logger.error("[radioml] h5py required for RML2018 — pip install h5py")
        return np.zeros((0, 2, 1024), dtype=np.float32), np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)

    with h5py.File(p, "r") as h:
        # The DeepSig 2018 file conventionally has X (I/Q), Y (one-hot mod), Z (SNR)
        X_raw = np.asarray(h["X"])              # (N, 1024, 2)
        Y_raw = np.asarray(h["Y"])              # (N, 24) one-hot
        Z_raw = np.asarray(h["Z"]).reshape(-1)  # (N,)

    X = np.transpose(X_raw, (0, 2, 1)).astype(np.float32)  # → (N, 2, 1024)
    y_mod = np.argmax(Y_raw, axis=1).astype(np.int64)
    y_snr = Z_raw.astype(np.int64)

    if min_snr is not None:
        mask = y_snr >= int(min_snr)
        X, y_mod, y_snr = X[mask], y_mod[mask], y_snr[mask]
    if max_samples is not None and len(X) > max_samples:
        rng = np.random.default_rng(7)
        idx = rng.choice(len(X), max_samples, replace=False)
        X, y_mod, y_snr = X[idx], y_mod[idx], y_snr[idx]

    logger.info("[radioml-2018] loaded %d samples × %d classes × %d SNRs",
                len(X), len(set(y_mod.tolist())), len(set(y_snr.tolist())))
    return X, y_mod, y_snr


# ---------------------------------------------------------------------------
# Synthetic fallback (pipeline test only)
# ---------------------------------------------------------------------------


def synthetic_radioml(n_per_class: int = 200, seq_len: int = 128,
                      classes: Optional[list[str]] = None
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate I/Q samples with crude per-modulation signatures so the
    pipeline can be validated without the real DeepSig dataset."""
    rng = np.random.default_rng(7)
    classes = classes or MODULATION_CLASSES_2016
    n_classes = len(classes)

    X = rng.normal(0, 0.3, (n_classes * n_per_class, 2, seq_len)).astype(np.float32)
    y_mod = np.repeat(np.arange(n_classes, dtype=np.int64), n_per_class)
    y_snr = rng.choice([-10, -5, 0, 5, 10, 15, 20], n_classes * n_per_class).astype(np.int64)

    t = np.arange(seq_len) / seq_len
    for i in range(len(X)):
        cls = classes[y_mod[i]]
        # Per-class crude carrier
        if "BPSK" in cls or "PSK" in cls or "QAM" in cls:
            phase = rng.choice([0, np.pi]) if "BPSK" in cls else rng.uniform(0, 2*np.pi)
            f = 0.05 + 0.005 * y_mod[i]
            X[i, 0, :] += np.cos(2*np.pi*f*np.arange(seq_len) + phase)
            X[i, 1, :] += np.sin(2*np.pi*f*np.arange(seq_len) + phase)
        elif "FSK" in cls or cls in ("GFSK", "CPFSK"):
            f = 0.05 if rng.random() < 0.5 else 0.10
            X[i, 0, :] += np.cos(2*np.pi*f*np.arange(seq_len))
            X[i, 1, :] += np.sin(2*np.pi*f*np.arange(seq_len))
        elif "AM" in cls:
            f = 0.05
            env = 1.0 + 0.3*np.sin(2*np.pi*0.01*np.arange(seq_len))
            X[i, 0, :] += env * np.cos(2*np.pi*f*np.arange(seq_len))
            X[i, 1, :] += env * np.sin(2*np.pi*f*np.arange(seq_len))
        elif "FM" in cls:
            f = 0.04 + 0.02*np.sin(2*np.pi*0.01*np.arange(seq_len))
            X[i, 0, :] += np.cos(2*np.pi*f*np.arange(seq_len))
            X[i, 1, :] += np.sin(2*np.pi*f*np.arange(seq_len))
        # SNR scaling
        snr_lin = 10 ** (y_snr[i] / 10.0)
        X[i] = X[i] / np.sqrt(snr_lin) + rng.normal(0, 0.1, X[i].shape).astype(np.float32)

    logger.info("[radioml-synth] %d samples × %d classes (pipeline test only)",
                len(X), n_classes)
    return X, y_mod, y_snr
