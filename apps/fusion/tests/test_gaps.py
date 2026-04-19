"""
Tests for the 8 LatticeOS parity gaps implemented in Heli.OS fusion service.

Gap 1: Multi-sensor track fusion (TrackManager + /api/v1/tracks endpoint)
Gap 4: HLS video streaming (hls_router endpoints)
Gap 8: Cross-camera re-identification (reid.py)
"""

import os

os.environ["FUSION_DISABLE_STARTUP"] = "1"

import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from fastapi.testclient import TestClient
from main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


# ── Gap 1: /api/v1/tracks ─────────────────────────────────────────────────────


def test_tracks_endpoint_exists(client):
    """GET /api/v1/tracks returns 200 with tracks list."""
    r = client.get("/api/v1/tracks")
    assert r.status_code == 200
    body = r.json()
    assert "tracks" in body
    assert isinstance(body["tracks"], list)


def test_tracks_endpoint_count_field(client):
    r = client.get("/api/v1/tracks")
    body = r.json()
    assert "count" in body
    assert body["count"] == len(body["tracks"])


def test_tracks_status_filter(client):
    """Status filter param is accepted without error."""
    r = client.get("/api/v1/tracks?status=CONFIRMED")
    assert r.status_code == 200


def test_tracks_limit_param(client):
    r = client.get("/api/v1/tracks?limit=5")
    assert r.status_code == 200


# ── Gap 4: HLS router ─────────────────────────────────────────────────────────


def test_hls_list_streams(client):
    """GET /api/v1/video/hls lists active streams."""
    r = client.get("/api/v1/video/hls")
    assert r.status_code == 200
    body = r.json()
    assert "streams" in body


def test_hls_start_missing_rtsp_url(client):
    """POST without rtsp_url returns 400."""
    r = client.post("/api/v1/video/hls/test-stream/start", json={})
    assert r.status_code == 400


def test_hls_playlist_not_found(client):
    """Playlist 404 when stream not started."""
    r = client.get("/api/v1/video/hls/nonexistent-stream/index.m3u8")
    assert r.status_code == 404


def test_hls_segment_bad_extension(client):
    r = client.get("/api/v1/video/hls/test-stream/segment.mp4")
    assert r.status_code == 400


def test_hls_stop_nonexistent_stream(client):
    """DELETE on nonexistent stream returns 200 (idempotent)."""
    r = client.delete("/api/v1/video/hls/nonexistent-stream")
    assert r.status_code == 200


# ── Gap 8: AppearanceReID ─────────────────────────────────────────────────────


def test_reid_import():
    """AppearanceReID can be imported and instantiated."""
    from reid import AppearanceReID

    reid = AppearanceReID()
    assert reid is not None
    assert reid.gallery_size() == 0


def test_reid_histogram_embedding():
    """_color_histogram returns a non-zero vector for a valid image."""
    import numpy as np
    from reid import _color_histogram

    # Create a small BGR test image (red square)
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    img[:, :, 2] = 200  # red channel
    hist = _color_histogram(img)
    assert hist.shape == (512,)  # 8^3
    assert float(hist.sum()) > 0.0


def test_reid_update_and_query_same_camera():
    """Query from same camera as gallery entry returns no match (cross-camera only)."""
    import numpy as np
    from reid import AppearanceReID

    reid = AppearanceReID()
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[:, :, 1] = 150
    reid.update("track-001", img, camera_id="cam-a")
    # Query from the same camera — should not match (cross-camera only)
    match_id, score = reid.query(img, camera_id="cam-a")
    assert match_id is None


def test_reid_cross_camera_match():
    """Track registered on cam-a should match query from cam-b."""
    import numpy as np
    from reid import AppearanceReID

    reid = AppearanceReID()
    # Distinctive green image
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[:, :, 1] = 200
    reid.update("track-green", img, camera_id="cam-a")
    # Query from different camera with same appearance
    match_id, score = reid.query(img, camera_id="cam-b")
    assert match_id == "track-green"
    assert score > 0.5


def test_reid_no_match_different_appearance():
    """Different appearance should not match."""
    import numpy as np
    from reid import AppearanceReID

    reid = AppearanceReID()
    green_img = np.zeros((64, 64, 3), dtype=np.uint8)
    green_img[:, :, 1] = 200
    reid.update("track-green", green_img, camera_id="cam-a")
    # Red image — very different colour histogram
    red_img = np.zeros((64, 64, 3), dtype=np.uint8)
    red_img[:, :, 2] = 200
    match_id, score = reid.query(red_img, camera_id="cam-b")
    # May or may not match depending on threshold, but score should be lower
    assert score < 0.9  # shouldn't be a high-confidence match


def test_reid_get_track_cameras():
    import numpy as np
    from reid import AppearanceReID

    reid = AppearanceReID()
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    reid.update("track-001", img, camera_id="cam-a")
    reid.update("track-001", img, camera_id="cam-b")
    cams = reid.get_track_cameras("track-001")
    assert set(cams) == {"cam-a", "cam-b"}


def test_reid_gallery_eviction():
    """Stale tracks are evicted after MAX_GALLERY_AGE_S."""
    import time
    import numpy as np
    from reid import AppearanceReID, _GalleryEntry

    reid = AppearanceReID()
    reid.MAX_GALLERY_AGE_S = 0.01  # 10ms
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    reid.update("stale-track", img, camera_id="cam-a")
    assert reid.gallery_size() == 1
    time.sleep(0.02)
    # Trigger eviction via another update
    reid.update("fresh-track", img, camera_id="cam-b")
    assert "stale-track" not in reid._gallery
