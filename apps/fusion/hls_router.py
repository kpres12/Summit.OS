"""
HLS Video Streaming Router for Heli.OS Fusion Service (Gap 4)

Wraps RTSP streams in HLS using ffmpeg so the operator UI can play them
with hls.js in a standard <video> element.

For each stream_id:
  POST /api/v1/video/hls/{stream_id}/start  — start transcoding
  DELETE /api/v1/video/hls/{stream_id}      — stop transcoding
  GET  /api/v1/video/hls/{stream_id}/index.m3u8  — HLS playlist
  GET  /api/v1/video/hls/{stream_id}/{segment}.ts — TS segments

Env vars:
  HLS_OUTPUT_DIR   — directory to write .m3u8 + .ts files (default /tmp/hls)
  FFMPEG_BIN       — path to ffmpeg binary (default: ffmpeg)
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
from pathlib import Path
from typing import Dict

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

logger = logging.getLogger("fusion.hls")

HLS_OUTPUT_DIR = Path(os.getenv("HLS_OUTPUT_DIR", "/tmp/hls"))
FFMPEG_BIN = os.getenv("FFMPEG_BIN", "ffmpeg")

# Active transcoder processes: stream_id → asyncio.subprocess.Process
_procs: Dict[str, asyncio.subprocess.Process] = {}

router = APIRouter(tags=["video"])


def _stream_dir(stream_id: str) -> Path:
    d = HLS_OUTPUT_DIR / stream_id
    d.mkdir(parents=True, exist_ok=True)
    return d


# ── Control endpoints ─────────────────────────────────────────────────────────


@router.post("/hls/{stream_id}/start")
async def start_hls(stream_id: str, payload: dict):
    """
    Start HLS transcoding for an RTSP stream.

    Body: { "rtsp_url": "rtsp://..." }
    """
    rtsp_url = payload.get("rtsp_url")
    if not rtsp_url:
        raise HTTPException(status_code=400, detail="rtsp_url required")

    if stream_id in _procs:
        proc = _procs[stream_id]
        if proc.returncode is None:
            return {"status": "already_running", "stream_id": stream_id}
        del _procs[stream_id]

    if not shutil.which(FFMPEG_BIN):
        raise HTTPException(
            status_code=501, detail="ffmpeg not found — install ffmpeg to enable HLS"
        )

    out_dir = _stream_dir(stream_id)
    playlist = str(out_dir / "index.m3u8")

    cmd = [
        FFMPEG_BIN,
        "-rtsp_transport",
        "tcp",
        "-i",
        rtsp_url,
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-tune",
        "zerolatency",
        "-f",
        "hls",
        "-hls_time",
        "2",
        "-hls_list_size",
        "10",
        "-hls_flags",
        "delete_segments",
        "-hls_segment_filename",
        str(out_dir / "%05d.ts"),
        playlist,
    ]

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        _procs[stream_id] = proc
        logger.info(f"HLS transcoder started for {stream_id} → {rtsp_url}")
        return {
            "status": "started",
            "stream_id": stream_id,
            "playlist_url": f"/api/v1/video/hls/{stream_id}/index.m3u8",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start ffmpeg: {e}")


@router.delete("/hls/{stream_id}")
async def stop_hls(stream_id: str):
    """Stop HLS transcoding for a stream."""
    proc = _procs.pop(stream_id, None)
    if proc and proc.returncode is None:
        proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), timeout=3.0)
        except asyncio.TimeoutError:
            proc.kill()
    # Clean up segment files
    out_dir = HLS_OUTPUT_DIR / stream_id
    if out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)
    return {"status": "stopped", "stream_id": stream_id}


@router.get("/hls/{stream_id}/index.m3u8")
async def get_playlist(stream_id: str):
    """Serve the HLS playlist file."""
    path = HLS_OUTPUT_DIR / stream_id / "index.m3u8"
    if not path.exists():
        raise HTTPException(
            status_code=404, detail="Playlist not ready — stream may not be started"
        )
    return FileResponse(str(path), media_type="application/vnd.apple.mpegurl")


@router.get("/hls/{stream_id}/{segment}")
async def get_segment(stream_id: str, segment: str):
    """Serve an HLS transport stream segment."""
    if not segment.endswith(".ts"):
        raise HTTPException(status_code=400, detail="Only .ts segments served here")
    path = HLS_OUTPUT_DIR / stream_id / segment
    if not path.exists():
        raise HTTPException(status_code=404, detail="Segment not found")
    return FileResponse(str(path), media_type="video/mp2t")


@router.get("/hls")
async def list_streams():
    """List all active HLS streams."""
    active = {sid: proc.returncode is None for sid, proc in _procs.items()}
    return {"streams": active}
