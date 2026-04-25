"""
NASA Li-Ion Battery Aging Dataset Downloader
=============================================
Downloads the NASA Prognostics Center battery aging datasets (B0005–B0018).
These contain charge/discharge cycle data for Li-ion batteries at room temperature.

Used to train the Heli.OS battery degradation predictor:
  current SOC + discharge rate → minutes to critical threshold

Source: https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/
Data mirror: https://data.nasa.gov/Aerospace/BATTERY-DATA-SET/
License: NASA Open Data (public domain)
"""
from __future__ import annotations

import argparse
import logging
import zipfile
from pathlib import Path

import numpy as np
import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

# NASA Prognostics Center — publicly accessible zip files
_BATTERY_URLS = {
    "B0005": "https://ti.arc.nasa.gov/c/6/",    # redirect, use mirror
    "B0006": "https://ti.arc.nasa.gov/c/7/",
    "B0007": "https://ti.arc.nasa.gov/c/8/",
    "B0018": "https://ti.arc.nasa.gov/c/9/",
}

# Public Zenodo mirror (more reliable than NASA direct)
_ZENODO_URL = "https://zenodo.org/record/3678771/files/NASA_Battery_Dataset.zip"
# Fallback: CALCE dataset from UMD (open access)
_CALCE_BASE = "https://calce.umd.edu/files/batteries"
_CALCE_CELLS = ["CS2_33", "CS2_34", "CS2_35", "CS2_36"]


def _download_file(url: str, dest: Path, desc: str = "file") -> bool:
    if dest.exists():
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        r = requests.get(url, stream=True, timeout=120)
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=desc) as bar:
            for chunk in r.iter_content(65536):
                f.write(chunk)
                bar.update(len(chunk))
        return True
    except Exception as e:
        logger.debug("Download failed %s: %s", url, e)
        return False


def _generate_synthetic_battery_data(out: Path) -> None:
    """
    Generate realistic synthetic Li-ion discharge curves when NASA data
    is unavailable. Uses empirical Li-ion discharge characteristics.
    """
    logger.info("Generating synthetic battery training data (NASA mirror unavailable) ...")
    rng = np.random.default_rng(42)
    records = []

    for _ in range(5000):
        # Initial state
        soc_start = rng.uniform(0.40, 1.0)           # starting SOC
        capacity  = rng.uniform(0.85, 1.15)           # relative capacity (degradation)
        temp      = rng.uniform(15, 35)               # °C
        c_rate    = rng.uniform(0.5, 2.0)             # C rate (discharge speed)

        # Empirical discharge model: time = capacity / (c_rate * degradation_factor)
        # At 1C: full discharge takes ~60 min. At 2C: ~30 min.
        base_time_full = 60.0 / c_rate * capacity
        time_to_empty  = base_time_full * soc_start

        # Time to critical (15% SOC)
        time_to_critical = base_time_full * max(0.0, soc_start - 0.15)

        # Add temperature effect (cold = faster discharge)
        temp_factor = 1.0 - max(0, (20 - temp)) * 0.008
        time_to_critical *= temp_factor

        # Add noise
        time_to_critical += rng.normal(0, 2.0)
        time_to_critical = max(0.0, time_to_critical)

        records.append({
            "soc_pct":             soc_start * 100,
            "discharge_rate_c":    c_rate,
            "temp_celsius":        temp,
            "capacity_ratio":      capacity,
            "minutes_to_critical": time_to_critical,
            "minutes_to_empty":    max(0.0, time_to_empty * temp_factor + rng.normal(0, 3)),
        })

    import pandas as pd
    df = pd.DataFrame(records)
    out_file = out / "battery_training.parquet"
    df.to_parquet(out_file, index=False)
    logger.info("Generated %d synthetic discharge records → %s", len(df), out_file)


def download(output_dir: str = "/tmp/heli-training-data/battery") -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Try Zenodo mirror first
    zip_path = out / "nasa_battery.zip"
    success = _download_file(_ZENODO_URL, zip_path, "NASA Battery (Zenodo)")

    if success:
        extract_dir = out / "raw"
        if not extract_dir.exists():
            logger.info("Extracting NASA battery data ...")
            try:
                with zipfile.ZipFile(zip_path) as zf:
                    zf.extractall(extract_dir)

                # Convert MATLAB .mat files to parquet
                _convert_mat_to_parquet(extract_dir, out)
            except Exception as e:
                logger.warning("Extraction failed: %s — using synthetic data", e)
                _generate_synthetic_battery_data(out)
    else:
        logger.warning("NASA battery download failed — using synthetic discharge model")
        _generate_synthetic_battery_data(out)

    logger.info("Battery dataset ready at %s", out)
    return out


def _convert_mat_to_parquet(src: Path, out: Path) -> None:
    """Convert NASA .mat files to pandas parquet for training."""
    try:
        import scipy.io as sio
        import pandas as pd
    except ImportError:
        logger.warning("scipy not installed — cannot convert .mat files. Using synthetic data.")
        _generate_synthetic_battery_data(out)
        return

    records = []
    for mat_file in src.rglob("*.mat"):
        try:
            data = sio.loadmat(str(mat_file), simplify_cells=True)
            cycles = data.get("B", {}).get("cycle", [])
            if not hasattr(cycles, "__iter__"):
                continue

            for cycle in cycles:
                if not isinstance(cycle, dict):
                    continue
                ctype = cycle.get("type", "")
                if ctype != "discharge":
                    continue

                measurements = cycle.get("data", {})
                voltage = np.asarray(measurements.get("Voltage_measured", []))
                current = np.asarray(measurements.get("Current_measured", []))
                temp    = np.asarray(measurements.get("Temperature_measured", []))
                t       = np.asarray(measurements.get("Time", []))

                if len(voltage) < 10:
                    continue

                # Compute discharge rate (C-rate proxy)
                capacity_ah = 2.0  # nominal 2 Ah
                avg_current = float(np.abs(np.mean(current)))
                c_rate      = avg_current / capacity_ah

                # SOC at each sample (integrate current)
                dt  = np.diff(t, prepend=t[0])
                ah_discharged = np.cumsum(np.abs(current) * dt / 3600)
                soc = np.maximum(0, 1.0 - ah_discharged / capacity_ah)

                # Find time to 15% SOC
                critical_mask = soc <= 0.15
                if critical_mask.any():
                    t_critical = float(t[critical_mask][0]) / 60  # minutes
                else:
                    t_critical = float(t[-1]) / 60 * (soc[-1] / 0.15)

                records.append({
                    "soc_pct":             float(soc[0]) * 100,
                    "discharge_rate_c":    c_rate,
                    "temp_celsius":        float(np.mean(temp)) if len(temp) > 0 else 25.0,
                    "capacity_ratio":      1.0,
                    "minutes_to_critical": t_critical,
                    "minutes_to_empty":    float(t[-1]) / 60,
                })
        except Exception as e:
            logger.debug("Skipping %s: %s", mat_file.name, e)

    if not records:
        logger.warning("No discharge cycles extracted from .mat files — using synthetic")
        _generate_synthetic_battery_data(out)
        return

    import pandas as pd
    df = pd.DataFrame(records)
    out_file = out / "battery_training.parquet"
    df.to_parquet(out_file, index=False)
    logger.info("NASA battery: %d discharge cycles extracted → %s", len(df), out_file)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="/tmp/heli-training-data/battery")
    args = p.parse_args()
    download(args.output)
