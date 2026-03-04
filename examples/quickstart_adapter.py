#!/usr/bin/env python3
"""
Summit.OS Quickstart Adapter Template

Copy this file, implement get_telemetry() and handle_command(),
and your device will appear in the world model within 30 seconds.

Steps:
  1. Copy this file to your project
  2. pip install paho-mqtt requests
  3. Implement get_telemetry() — return {lat, lon, alt, battery, status}
  4. Implement handle_command() — handle "goto", "rtl", "land", etc.
  5. Run: python my_adapter.py

Your device will:
  - Register with the Summit.OS gateway
  - Publish heartbeats every 30s
  - Stream telemetry every 5s into the shared world model
  - Listen for commands on MQTT topics
"""
import asyncio
import random
from summit_os.adapter import SummitAdapter


class MyDeviceAdapter(SummitAdapter):
    """
    Example adapter — replace with your real hardware interface.

    For a real integration, you'd read from:
      - MAVLink (pymavlink) for drones
      - Serial/CAN for robots
      - GPIO/I2C for sensors
      - REST API for IP cameras
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize your hardware connection here
        self._sim_lat = 37.7749
        self._sim_lon = -122.4194
        self._sim_alt = 50.0
        self._sim_battery = 95.0

    async def get_telemetry(self) -> dict:
        """
        Return current device telemetry.

        REQUIRED fields: lat, lon, alt
        RECOMMENDED: battery, status, sensors
        """
        # Replace with real hardware reads
        self._sim_lat += random.uniform(-0.0001, 0.0001)
        self._sim_lon += random.uniform(-0.0001, 0.0001)
        self._sim_battery -= 0.01

        return {
            "lat": self._sim_lat,
            "lon": self._sim_lon,
            "alt": self._sim_alt,
            "battery": max(0, self._sim_battery),
            "status": "ACTIVE",
            "sensors": {
                "temperature": 22.5,
                "humidity": 45.0,
            },
        }

    async def handle_command(self, cmd: str, params: dict) -> bool:
        """
        Handle incoming commands from the platform.

        Common commands:
          "goto"     — Navigate to {lat, lon, alt, speed}
          "rtl"      — Return to launch
          "land"     — Land immediately
          "hold"     — Hold current position
          "set_mode" — Change flight/drive mode
        """
        if cmd == "goto":
            lat = params.get("lat", 0)
            lon = params.get("lon", 0)
            alt = params.get("alt", 50)
            print(f"  → Navigating to ({lat}, {lon}) at {alt}m")
            # Replace with: await self.vehicle.goto(lat, lon, alt)
            self._sim_lat = lat
            self._sim_lon = lon
            self._sim_alt = alt
            return True

        elif cmd == "rtl":
            print("  → Returning to launch")
            return True

        elif cmd == "land":
            print("  → Landing")
            return True

        elif cmd == "hold":
            print("  → Holding position")
            return True

        elif cmd == "ping":
            return True

        else:
            print(f"  → Unknown command: {cmd}")
            return False

    def get_capabilities(self) -> list:
        """List your device's capabilities."""
        return ["rgb_camera", "thermal", "gps"]

    async def on_connect(self):
        print(f"  Connected: {self.device_id}")

    async def on_disconnect(self):
        print(f"  Disconnected: {self.device_id}")


if __name__ == "__main__":
    adapter = MyDeviceAdapter(
        device_id="quickstart-drone-01",
        device_type="DRONE",
        # Uncomment to set MQTT credentials:
        # mqtt_host="localhost",
        # mqtt_port=1883,
        # mqtt_username="user",
        # mqtt_password="pass",
        # api_base="http://localhost:8000",
    )

    print("Summit.OS Quickstart Adapter")
    print(f"  Device: {adapter.device_id}")
    print(f"  Type:   {adapter.device_type}")
    print(f"  MQTT:   {adapter.mqtt_host}:{adapter.mqtt_port}")
    print()

    try:
        asyncio.run(adapter.start())
    except KeyboardInterrupt:
        print("\nShutting down...")
