"""
Triangulation helpers: compute intersection point of two bearings from two observer points.
Approximates on small distances using rhumb line to convert bearing to a ray in lat/lon.
"""
from math import radians, degrees, sin, cos, tan, atan2
from typing import Optional

# Earth radius in meters
R = 6371000.0

def _to_vector(lat: float, lon: float, bearing_deg: float):
    b = radians(bearing_deg)
    # Unit vector in ENU plane (approx)
    return cos(b), sin(b)


def bearing_intersection(lat1: float, lon1: float, b1_deg: float, lat2: float, lon2: float, b2_deg: float) -> Optional[tuple[float, float]]:
    # Convert to local tangent plane (meters) with origin at point1
    # Equirectangular approximation
    import math
    lat1r = radians(lat1)
    x1, y1 = 0.0, 0.0
    x2 = (radians(lon2 - lon1)) * cos(lat1r) * R
    y2 = (radians(lat2 - lat1)) * R

    v1x, v1y = _to_vector(lat1, lon1, b1_deg)
    v2x, v2y = _to_vector(lat2, lon2, b2_deg)

    # Solve x1 + t*v1 = x2 + s*v2
    # -> t*v1 - s*v2 = (x2 - x1)
    # 2x2 system
    det = v1x * (-v2y) - v1y * (-v2x)
    if abs(det) < 1e-6:
        return None
    dx = x2 - x1
    dy = y2 - y1
    t = (dx * (-v2y) - dy * (-v2x)) / det
    ix = x1 + t * v1x
    iy = y1 + t * v1y

    # Convert back to lat/lon
    lat = lat1 + degrees(iy / R)
    lon = lon1 + degrees(ix / (R * cos(lat1r)))
    return (lat, lon)
