"""
Simple model registry helpers for Fusion.
"""

import logging
import os
import threading
from typing import Callable, List

from vision_inference import VisionInference

logger = logging.getLogger("fusion.model_registry")


def list_models(root: str) -> List[str]:
    try:
        entries = []
        for dirpath, dirnames, filenames in os.walk(root):
            for f in filenames:
                if f.endswith((".onnx", ".pt", ".engine")):
                    entries.append(os.path.join(dirpath, f))
        return sorted(entries)
    except Exception:
        return []


def select_model(path: str) -> None:
    # Reinitialize global vision object if present
    import main as fusion_main  # circular import safe for attribute access

    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if fusion_main.vision is None:
        fusion_main.vision = VisionInference(
            model_path=path,
            conf_threshold=float(os.getenv("FUSION_CONF_THRESHOLD", "0.6")),
        )
    else:
        fusion_main.vision.load(path)


# ── Hot-reload watcher ────────────────────────────────────────────────────────

def watch_hot_dir(hot_dir: str, reload_fn: Callable[[str], None]) -> None:
    """
    Watch hot_dir for new .onnx files and call reload_fn(path) when found.

    Uses the watchdog library if available; falls back to polling every 30 s
    via a background threading.Timer chain.

    The watcher runs as a daemon thread — it stops automatically when the
    main process exits.
    """
    try:
        from watchdog.observers import Observer  # type: ignore
        from watchdog.events import FileSystemEventHandler, FileCreatedEvent  # type: ignore

        class _OnnxHandler(FileSystemEventHandler):
            def on_created(self, event: FileCreatedEvent) -> None:
                if not event.is_directory and event.src_path.endswith(".onnx"):
                    logger.info("Hot-reload: new model detected at %s", event.src_path)
                    try:
                        reload_fn(event.src_path)
                    except Exception as exc:
                        logger.error("Hot-reload failed for %s: %s", event.src_path, exc)

        observer = Observer()
        observer.schedule(_OnnxHandler(), path=hot_dir, recursive=False)
        observer.daemon = True
        observer.start()
        logger.info("Hot-reload watchdog started on %s (watchdog library)", hot_dir)

    except ImportError:
        logger.warning(
            "watchdog not installed — falling back to 30-second polling for hot-reload on %s",
            hot_dir,
        )

        _seen: set = set()

        def _poll() -> None:
            try:
                if os.path.isdir(hot_dir):
                    for fname in os.listdir(hot_dir):
                        if fname.endswith(".onnx"):
                            fpath = os.path.join(hot_dir, fname)
                            if fpath not in _seen:
                                _seen.add(fpath)
                                logger.info("Hot-reload (poll): new model at %s", fpath)
                                try:
                                    reload_fn(fpath)
                                except Exception as exc:
                                    logger.error("Hot-reload failed for %s: %s", fpath, exc)
            finally:
                # Reschedule; daemon=True so this doesn't block shutdown
                t = threading.Timer(30.0, _poll)
                t.daemon = True
                t.start()

        # Seed _seen with whatever already exists so we don't reload on startup
        if os.path.isdir(hot_dir):
            for fname in os.listdir(hot_dir):
                if fname.endswith(".onnx"):
                    _seen.add(os.path.join(hot_dir, fname))

        t = threading.Timer(30.0, _poll)
        t.daemon = True
        t.start()
