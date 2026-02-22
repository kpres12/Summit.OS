"""
Pre-built Simulation Scenarios for Summit.OS

Ready-to-run simulation scenarios for testing and demo.
Each scenario creates a WorldSimulator with entities and sensors.
"""
from __future__ import annotations

from packages.simulation.world import WorldSimulator, SimEntity, SimSensor


def border_patrol_scenario(seed: int = 42) -> WorldSimulator:
    """
    Border patrol scenario:
    - 2 friendly patrol drones flying a fence line
    - 1 ground radar at the observation post
    - 3 unknown intruders approaching from different angles
    """
    sim = WorldSimulator(dt=1.0, seed=seed)

    # Patrol drones
    sim.add_entity(SimEntity(
        entity_id="patrol-1",
        lat=32.0, lon=-110.0, alt=100,
        speed_mps=12.0, entity_type="friendly", domain="aerial",
        classification="UAS",
        waypoints=[
            (32.0, -110.0, 100), (32.05, -110.0, 100),
            (32.05, -109.95, 100), (32.0, -109.95, 100),
        ],
    ))
    sim.add_entity(SimEntity(
        entity_id="patrol-2",
        lat=32.025, lon=-109.975, alt=150,
        speed_mps=10.0, entity_type="friendly", domain="aerial",
        classification="UAS",
        waypoints=[
            (32.025, -109.975, 150), (32.025, -110.025, 150),
            (32.05, -110.025, 150), (32.05, -109.975, 150),
        ],
    ))

    # Ground radar
    sim.add_sensor(SimSensor(
        sensor_id="radar-op1",
        lat=32.025, lon=-110.0, alt=50,
        sensor_type="radar", max_range_m=15000,
        detection_probability=0.85, position_noise_m=20,
        update_interval_sec=2.0,
    ))

    # EO/IR on patrol-1
    sim.add_sensor(SimSensor(
        sensor_id="eoir-patrol1",
        sensor_type="eo_ir", max_range_m=5000,
        fov_deg=60, detection_probability=0.75,
        position_noise_m=5, update_interval_sec=0.5,
        attached_to="patrol-1",
    ))

    # Intruders
    sim.add_entity(SimEntity(
        entity_id="intruder-1",
        lat=31.95, lon=-109.98, alt=0,
        speed_mps=2.0, heading_deg=0, entity_type="unknown",
        domain="ground", classification="person",
        radar_cross_section_m2=0.5,
        waypoints=[(32.02, -109.98, 0)],
        loop_waypoints=False,
    ))
    sim.add_entity(SimEntity(
        entity_id="intruder-2",
        lat=31.96, lon=-110.02, alt=50,
        speed_mps=15.0, heading_deg=10, entity_type="unknown",
        domain="aerial", classification="small_uas",
        radar_cross_section_m2=0.1,
        waypoints=[(32.03, -110.01, 80)],
        loop_waypoints=False,
    ))
    sim.add_entity(SimEntity(
        entity_id="intruder-3",
        lat=31.97, lon=-109.95, alt=0,
        speed_mps=5.0, heading_deg=350, entity_type="unknown",
        domain="ground", classification="vehicle",
        radar_cross_section_m2=3.0,
        waypoints=[(32.01, -109.96, 0)],
        loop_waypoints=False,
    ))

    return sim


def airspace_monitoring_scenario(seed: int = 123) -> WorldSimulator:
    """
    Airspace monitoring scenario:
    - 1 ADS-B receiver at airport
    - 2 radars (primary + secondary)
    - 5 cooperative aircraft on known routes
    - 2 non-cooperative unknowns
    """
    sim = WorldSimulator(dt=2.0, seed=seed)

    # Sensors
    sim.add_sensor(SimSensor(
        sensor_id="adsb-1",
        lat=34.05, lon=-118.25, alt=30,
        sensor_type="adsb", max_range_m=200000,
        detection_probability=0.95, position_noise_m=7.5,
        update_interval_sec=1.0,
    ))
    sim.add_sensor(SimSensor(
        sensor_id="psr-1",
        lat=34.05, lon=-118.25, alt=30,
        sensor_type="radar", max_range_m=100000,
        detection_probability=0.8, position_noise_m=50,
        update_interval_sec=4.0,
    ))

    # Cooperative aircraft
    for i in range(5):
        sim.add_entity(SimEntity(
            entity_id=f"aircraft-{i+1}",
            lat=34.0 + i * 0.02, lon=-118.3 + i * 0.01,
            alt=3000 + i * 500, speed_mps=80 + i * 10,
            heading_deg=45 + i * 30,
            entity_type="friendly", domain="aerial",
            classification="commercial",
            radar_cross_section_m2=20.0,
        ))

    # Non-cooperative unknowns
    sim.add_entity(SimEntity(
        entity_id="unknown-1",
        lat=33.95, lon=-118.35, alt=500,
        speed_mps=30, heading_deg=45,
        entity_type="unknown", domain="aerial",
        classification="",
        radar_cross_section_m2=2.0,
    ))
    sim.add_entity(SimEntity(
        entity_id="unknown-2",
        lat=34.1, lon=-118.15, alt=200,
        speed_mps=15, heading_deg=225,
        entity_type="unknown", domain="aerial",
        classification="",
        radar_cross_section_m2=0.5,
    ))

    return sim


def convoy_escort_scenario(seed: int = 7) -> WorldSimulator:
    """
    Convoy escort scenario:
    - 3 ground vehicles in convoy
    - 1 aerial escort drone
    - 2 surveillance radars along route
    - 2 potential threats
    """
    sim = WorldSimulator(dt=1.0, seed=seed)

    route = [
        (33.0, -117.0, 0), (33.01, -116.98, 0),
        (33.02, -116.96, 0), (33.03, -116.94, 0),
    ]

    # Convoy vehicles
    for i in range(3):
        sim.add_entity(SimEntity(
            entity_id=f"convoy-{i+1}",
            lat=33.0 - i * 0.002, lon=-117.0,
            alt=0, speed_mps=8.0,
            entity_type="friendly", domain="ground",
            classification="vehicle",
            radar_cross_section_m2=10.0,
            waypoints=route, loop_waypoints=False,
        ))

    # Escort drone
    sim.add_entity(SimEntity(
        entity_id="escort-1",
        lat=33.0, lon=-117.0, alt=100,
        speed_mps=10.0, entity_type="friendly", domain="aerial",
        classification="UAS",
        waypoints=[(r[0], r[1], 100) for r in route],
        loop_waypoints=False,
    ))

    # Escort drone sensor
    sim.add_sensor(SimSensor(
        sensor_id="eoir-escort",
        sensor_type="eo_ir", max_range_m=3000,
        fov_deg=90, detection_probability=0.85,
        position_noise_m=3, update_interval_sec=0.5,
        attached_to="escort-1",
    ))

    # Road-side surveillance radars
    sim.add_sensor(SimSensor(
        sensor_id="radar-road1",
        lat=33.01, lon=-116.99, alt=10,
        sensor_type="radar", max_range_m=5000,
        detection_probability=0.9, position_noise_m=10,
    ))
    sim.add_sensor(SimSensor(
        sensor_id="radar-road2",
        lat=33.025, lon=-116.95, alt=10,
        sensor_type="radar", max_range_m=5000,
        detection_probability=0.9, position_noise_m=10,
    ))

    # Threats
    sim.add_entity(SimEntity(
        entity_id="threat-1",
        lat=33.015, lon=-116.97, alt=0,
        speed_mps=0, entity_type="hostile", domain="ground",
        classification="emplacement",
        radar_cross_section_m2=0.3,
    ))
    sim.add_entity(SimEntity(
        entity_id="threat-2",
        lat=33.028, lon=-116.945, alt=0,
        speed_mps=3.0, heading_deg=180,
        entity_type="hostile", domain="ground",
        classification="vehicle",
        radar_cross_section_m2=5.0,
        waypoints=[(33.02, -116.95, 0)],
        loop_waypoints=False,
    ))

    return sim
