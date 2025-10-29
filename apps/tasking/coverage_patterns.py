"""
Coverage Pattern Algorithms for Mission Planning

Provides algorithms for generating systematic coverage patterns for
surveillance, search, and area monitoring missions.
"""

import math
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass


@dataclass
class Waypoint:
    """Geographic waypoint with altitude and optional metadata"""
    lat: float
    lon: float
    alt: float  # meters AGL
    speed: float = 5.0  # m/s
    action: str = "waypoint"  # waypoint, loiter, land, etc.
    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lat": self.lat,
            "lon": self.lon,
            "alt": self.alt,
            "speed": self.speed,
            "action": self.action,
            "metadata": self.metadata or {},
        }


@dataclass
class BoundingBox:
    """Geographic bounding box"""
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float

    @property
    def center(self) -> Tuple[float, float]:
        return (
            (self.lat_min + self.lat_max) / 2,
            (self.lon_min + self.lon_max) / 2,
        )

    @property
    def width_km(self) -> float:
        """Approximate width in km"""
        lat_avg = (self.lat_min + self.lat_max) / 2
        return haversine_distance(
            self.lat_min, self.lon_min,
            self.lat_min, self.lon_max
        ) * math.cos(math.radians(lat_avg))

    @property
    def height_km(self) -> float:
        """Approximate height in km"""
        return haversine_distance(
            self.lat_min, self.lon_min,
            self.lat_max, self.lon_min
        )


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two points in kilometers"""
    R = 6371.0  # Earth radius in km
    
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)
    
    a = (math.sin(delta_lat / 2) ** 2 +
         math.cos(lat1_rad) * math.cos(lat2_rad) *
         math.sin(delta_lon / 2) ** 2)
    c = 2 * math.asin(math.sqrt(a))
    
    return R * c


def grid_coverage_pattern(
    bbox: BoundingBox,
    altitude: float,
    spacing_m: float = 50,
    orientation: float = 0,
    speed: float = 5.0
) -> List[Waypoint]:
    """
    Generate a lawnmower/grid pattern for systematic area coverage.
    
    Args:
        bbox: Bounding box to cover
        altitude: Flight altitude in meters AGL
        spacing_m: Distance between parallel legs in meters
        orientation: Pattern orientation in degrees (0 = North-South)
        speed: Flight speed in m/s
    
    Returns:
        List of waypoints forming a grid pattern
    """
    waypoints = []
    
    # Calculate approximate spacing in degrees
    lat_center, lon_center = bbox.center
    meters_per_deg_lat = 111000  # approximately
    meters_per_deg_lon = 111000 * math.cos(math.radians(lat_center))
    
    spacing_lat = spacing_m / meters_per_deg_lat
    spacing_lon = spacing_m / meters_per_deg_lon
    
    # Start at southwest corner
    lat = bbox.lat_min
    direction = 1  # 1 for east, -1 for west
    leg_num = 0
    
    while lat <= bbox.lat_max:
        if direction == 1:
            # Going east
            waypoints.append(Waypoint(
                lat=lat,
                lon=bbox.lon_min,
                alt=altitude,
                speed=speed,
                metadata={"leg": leg_num, "direction": "east"}
            ))
            waypoints.append(Waypoint(
                lat=lat,
                lon=bbox.lon_max,
                alt=altitude,
                speed=speed,
                metadata={"leg": leg_num, "direction": "east"}
            ))
        else:
            # Going west
            waypoints.append(Waypoint(
                lat=lat,
                lon=bbox.lon_max,
                alt=altitude,
                speed=speed,
                metadata={"leg": leg_num, "direction": "west"}
            ))
            waypoints.append(Waypoint(
                lat=lat,
                lon=bbox.lon_min,
                alt=altitude,
                speed=speed,
                metadata={"leg": leg_num, "direction": "west"}
            ))
        
        # Move to next leg
        lat += spacing_lat
        direction *= -1
        leg_num += 1
    
    return waypoints


def spiral_coverage_pattern(
    center_lat: float,
    center_lon: float,
    radius_m: float,
    altitude: float,
    spacing_m: float = 30,
    direction: str = "outward",  # "outward" or "inward"
    speed: float = 5.0
) -> List[Waypoint]:
    """
    Generate an Archimedean spiral pattern for radial coverage.
    
    Args:
        center_lat: Center latitude
        center_lon: Center longitude
        radius_m: Maximum radius in meters
        altitude: Flight altitude in meters AGL
        spacing_m: Distance between spiral loops in meters
        direction: "outward" (center to edge) or "inward" (edge to center)
        speed: Flight speed in m/s
    
    Returns:
        List of waypoints forming a spiral pattern
    """
    waypoints = []
    
    # Convert to approximate degree offsets
    meters_per_deg_lat = 111000
    meters_per_deg_lon = 111000 * math.cos(math.radians(center_lat))
    
    # Number of complete rotations needed
    num_rotations = radius_m / spacing_m
    
    # Angular resolution (radians per waypoint)
    angular_resolution = 0.1  # radians, ~6 degrees
    total_angle = num_rotations * 2 * math.pi
    num_points = int(total_angle / angular_resolution)
    
    for i in range(num_points):
        angle = i * angular_resolution
        
        if direction == "outward":
            # Spiral outward from center
            r = (angle / (2 * math.pi)) * spacing_m
        else:
            # Spiral inward from edge
            r = radius_m - (angle / (2 * math.pi)) * spacing_m
        
        if r < 0 or r > radius_m:
            continue
        
        # Convert polar to Cartesian in meters
        x_m = r * math.cos(angle)
        y_m = r * math.sin(angle)
        
        # Convert to lat/lon offsets
        lat = center_lat + (y_m / meters_per_deg_lat)
        lon = center_lon + (x_m / meters_per_deg_lon)
        
        waypoints.append(Waypoint(
            lat=lat,
            lon=lon,
            alt=altitude,
            speed=speed,
            metadata={"angle_rad": angle, "radius_m": r}
        ))
    
    return waypoints


def perimeter_patrol_pattern(
    points: List[Tuple[float, float]],
    altitude: float,
    offset_m: float = 0,
    num_loops: int = 1,
    speed: float = 5.0
) -> List[Waypoint]:
    """
    Generate a perimeter patrol pattern following a polygon boundary.
    
    Args:
        points: List of (lat, lon) tuples defining the perimeter
        altitude: Flight altitude in meters AGL
        offset_m: Offset distance from perimeter in meters (positive = outward)
        num_loops: Number of times to complete the perimeter
        speed: Flight speed in m/s
    
    Returns:
        List of waypoints forming perimeter patrol
    """
    if len(points) < 3:
        raise ValueError("Need at least 3 points to define a perimeter")
    
    waypoints = []
    
    # For simplicity, we'll just offset each point radially from the centroid
    # In production, use proper polygon offsetting algorithms
    centroid_lat = sum(p[0] for p in points) / len(points)
    centroid_lon = sum(p[1] for p in points) / len(points)
    
    meters_per_deg_lat = 111000
    meters_per_deg_lon = 111000 * math.cos(math.radians(centroid_lat))
    
    offset_points = []
    for lat, lon in points:
        # Vector from centroid to point
        dlat = lat - centroid_lat
        dlon = lon - centroid_lon
        
        # Normalize and scale by offset
        distance = math.sqrt((dlat * meters_per_deg_lat) ** 2 +
                            (dlon * meters_per_deg_lon) ** 2)
        if distance > 0:
            scale = (distance + offset_m) / distance
            offset_lat = centroid_lat + dlat * scale
            offset_lon = centroid_lon + dlon * scale
        else:
            offset_lat = lat
            offset_lon = lon
        
        offset_points.append((offset_lat, offset_lon))
    
    # Create waypoints for each loop
    for loop in range(num_loops):
        for i, (lat, lon) in enumerate(offset_points):
            waypoints.append(Waypoint(
                lat=lat,
                lon=lon,
                alt=altitude,
                speed=speed,
                metadata={"loop": loop, "vertex": i}
            ))
        
        # Close the loop by returning to first point
        if loop < num_loops - 1:
            first_lat, first_lon = offset_points[0]
            waypoints.append(Waypoint(
                lat=first_lat,
                lon=first_lon,
                alt=altitude,
                speed=speed,
                metadata={"loop": loop, "vertex": "close"}
            ))
    
    return waypoints


def orbit_pattern(
    center_lat: float,
    center_lon: float,
    radius_m: float,
    altitude: float,
    num_points: int = 16,
    direction: str = "cw",  # "cw" or "ccw"
    speed: float = 5.0
) -> List[Waypoint]:
    """
    Generate a circular orbit pattern for observation/loiter.
    
    Args:
        center_lat: Center latitude
        center_lon: Center longitude
        radius_m: Orbit radius in meters
        altitude: Flight altitude in meters AGL
        num_points: Number of waypoints in the circle
        direction: "cw" (clockwise) or "ccw" (counter-clockwise)
        speed: Flight speed in m/s
    
    Returns:
        List of waypoints forming circular orbit
    """
    waypoints = []
    
    meters_per_deg_lat = 111000
    meters_per_deg_lon = 111000 * math.cos(math.radians(center_lat))
    
    angle_step = (2 * math.pi) / num_points
    if direction == "cw":
        angle_step = -angle_step
    
    for i in range(num_points):
        angle = i * angle_step
        
        x_m = radius_m * math.cos(angle)
        y_m = radius_m * math.sin(angle)
        
        lat = center_lat + (y_m / meters_per_deg_lat)
        lon = center_lon + (x_m / meters_per_deg_lon)
        
        waypoints.append(Waypoint(
            lat=lat,
            lon=lon,
            alt=altitude,
            speed=speed,
            action="waypoint" if i < num_points - 1 else "loiter",
            metadata={"angle_deg": math.degrees(angle)}
        ))
    
    # Close the orbit
    if waypoints:
        first = waypoints[0]
        waypoints.append(Waypoint(
            lat=first.lat,
            lon=first.lon,
            alt=altitude,
            speed=speed,
            action="loiter",
            metadata={"angle_deg": 0}
        ))
    
    return waypoints


def expand_search_pattern(
    start_lat: float,
    start_lon: float,
    altitude: float,
    leg_length_m: float = 100,
    num_legs: int = 8,
    speed: float = 5.0
) -> List[Waypoint]:
    """
    Generate an expanding square search pattern (common in SAR operations).
    
    Pattern: fly north, turn right, fly east (same distance), turn right,
    fly south (double distance), turn right, fly west (double distance), repeat.
    
    Args:
        start_lat: Starting latitude
        start_lon: Starting longitude
        altitude: Flight altitude in meters AGL
        leg_length_m: Initial leg length in meters
        num_legs: Number of legs to generate
        speed: Flight speed in m/s
    
    Returns:
        List of waypoints forming expanding search pattern
    """
    waypoints = [Waypoint(start_lat, start_lon, altitude, speed)]
    
    meters_per_deg_lat = 111000
    meters_per_deg_lon = 111000 * math.cos(math.radians(start_lat))
    
    current_lat = start_lat
    current_lon = start_lon
    current_heading = 0  # 0=North, 90=East, 180=South, 270=West
    current_leg_length = leg_length_m
    
    for leg in range(num_legs):
        # Calculate new position based on heading
        if current_heading == 0:  # North
            new_lat = current_lat + (current_leg_length / meters_per_deg_lat)
            new_lon = current_lon
        elif current_heading == 90:  # East
            new_lat = current_lat
            new_lon = current_lon + (current_leg_length / meters_per_deg_lon)
        elif current_heading == 180:  # South
            new_lat = current_lat - (current_leg_length / meters_per_deg_lat)
            new_lon = current_lon
        else:  # 270, West
            new_lat = current_lat
            new_lon = current_lon - (current_leg_length / meters_per_deg_lon)
        
        waypoints.append(Waypoint(
            lat=new_lat,
            lon=new_lon,
            alt=altitude,
            speed=speed,
            metadata={"leg": leg, "heading_deg": current_heading}
        ))
        
        current_lat = new_lat
        current_lon = new_lon
        
        # Turn right 90 degrees
        current_heading = (current_heading + 90) % 360
        
        # Increase leg length every 2 legs
        if leg % 2 == 1:
            current_leg_length += leg_length_m
    
    return waypoints
