# Summit.OS Python SDK

Bridges, clients, and utilities for integrating external systems with Summit.OS.

- License: MIT (SDK only). Optional extras may pull in other licenses (ROS 2, etc.).
- Install (core):
  ```bash
  pip install -e .[mqtt,websocket]
  ```
- ROS 2 bridge (optional):
  ```bash
  pip install -e .[ros2]
  ```

Features
- MQTT client helpers
- Optional ROS 2 -> Summit fabric bridge (topics to MQTT)

Notes
- The ROS 2 bridge will no-op if rclpy is not installed.
