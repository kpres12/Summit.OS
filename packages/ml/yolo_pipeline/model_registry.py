"""
ML Model Registry — Heli.OS

Tracks trained models by domain and version. When a new model is promoted
(after passing eval metrics threshold), it's copied to the hot-reload
directory watched by the fusion service.

Hot-reload dir: SUMMIT_MODEL_HOT_DIR env (default ./models/hot)
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger("ml.model_registry")

_DEFAULT_DB = os.getenv("SUMMIT_ML_REGISTRY_DB", "./ml_registry.db")
_DEFAULT_HOT_DIR = os.getenv("SUMMIT_MODEL_HOT_DIR", "./models/hot")

_MAP50_THRESHOLD = 0.6


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class ModelRecord:
    model_id: str
    domain: str
    version: int
    weights_path: str
    onnx_path: str
    metrics: dict
    ts_trained: float
    promoted: bool = False


# ── Registry ──────────────────────────────────────────────────────────────────

class MLModelRegistry:
    """SQLite-backed registry for trained YOLO domain models."""

    def __init__(self, db_path: str = None, hot_dir: str = None):
        self.db_path = db_path or _DEFAULT_DB
        self.hot_dir = Path(hot_dir or _DEFAULT_HOT_DIR)
        self.hot_dir.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ------------------------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ml_models (
                    model_id    TEXT PRIMARY KEY,
                    domain      TEXT NOT NULL,
                    version     INTEGER NOT NULL,
                    weights_path TEXT NOT NULL,
                    onnx_path   TEXT NOT NULL,
                    metrics     TEXT NOT NULL,
                    ts_trained  REAL NOT NULL,
                    promoted    INTEGER NOT NULL DEFAULT 0
                )
            """)
            conn.commit()

    # ------------------------------------------------------------------
    def register(self, result) -> ModelRecord:
        """
        Store a TrainResult, auto-increment version for the domain.
        Returns the created ModelRecord.
        """
        # Resolve import lazily to avoid circular imports
        from packages.ml.yolo_pipeline.trainer import TrainResult  # noqa: F401  type: ignore

        domain = result.domain
        current_max = self._max_version(domain)
        version = current_max + 1
        model_id = f"{domain}-v{version}"
        onnx_path = result.onnx_path or ""

        record = ModelRecord(
            model_id=model_id,
            domain=domain,
            version=version,
            weights_path=result.weights_path,
            onnx_path=onnx_path,
            metrics=result.metrics,
            ts_trained=result.ts,
            promoted=False,
        )

        with self._connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO ml_models
                   (model_id, domain, version, weights_path, onnx_path, metrics, ts_trained, promoted)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    record.model_id,
                    record.domain,
                    record.version,
                    record.weights_path,
                    record.onnx_path,
                    json.dumps(record.metrics),
                    record.ts_trained,
                    int(record.promoted),
                ),
            )
            conn.commit()

        logger.info("Registered model %s  mAP50=%.3f", model_id, result.metrics.get("mAP50", 0.0))
        return record

    # ------------------------------------------------------------------
    def promote(self, model_id: str) -> bool:
        """
        Promote a model to the hot-reload directory if mAP50 >= threshold.
        Returns True on success.
        """
        record = self._get_record(model_id)
        if record is None:
            logger.warning("promote: model_id=%s not found", model_id)
            return False

        map50 = record.metrics.get("mAP50", 0.0)
        if map50 < _MAP50_THRESHOLD:
            logger.warning(
                "promote: model_id=%s mAP50=%.3f < threshold %.2f — not promoted",
                model_id, map50, _MAP50_THRESHOLD,
            )
            return False

        onnx_src = Path(record.onnx_path)
        if not onnx_src.exists():
            logger.warning("promote: ONNX file not found at %s", onnx_src)
            return False

        dst = self.hot_dir / onnx_src.name
        shutil.copy2(onnx_src, dst)

        with self._connect() as conn:
            conn.execute("UPDATE ml_models SET promoted=1 WHERE model_id=?", (model_id,))
            conn.commit()

        logger.info("Promoted model %s → %s", model_id, dst)
        return True

    # ------------------------------------------------------------------
    def get_best(self, domain: str) -> Optional[ModelRecord]:
        """Return highest-version promoted model for a domain."""
        with self._connect() as conn:
            row = conn.execute(
                """SELECT * FROM ml_models
                   WHERE domain=? AND promoted=1
                   ORDER BY version DESC LIMIT 1""",
                (domain,),
            ).fetchone()
        return self._row_to_record(row) if row else None

    # ------------------------------------------------------------------
    def list_models(self, domain: str = None) -> List[ModelRecord]:
        """List all models, optionally filtered by domain."""
        with self._connect() as conn:
            if domain:
                rows = conn.execute(
                    "SELECT * FROM ml_models WHERE domain=? ORDER BY version DESC",
                    (domain,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM ml_models ORDER BY domain, version DESC"
                ).fetchall()
        return [self._row_to_record(r) for r in rows]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _max_version(self, domain: str) -> int:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT MAX(version) FROM ml_models WHERE domain=?", (domain,)
            ).fetchone()
        return row[0] or 0

    def _get_record(self, model_id: str) -> Optional[ModelRecord]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM ml_models WHERE model_id=?", (model_id,)
            ).fetchone()
        return self._row_to_record(row) if row else None

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> ModelRecord:
        return ModelRecord(
            model_id=row["model_id"],
            domain=row["domain"],
            version=row["version"],
            weights_path=row["weights_path"],
            onnx_path=row["onnx_path"],
            metrics=json.loads(row["metrics"]),
            ts_trained=row["ts_trained"],
            promoted=bool(row["promoted"]),
        )
