"""
ROS 2 <-> Summit Fabric bridge

- Subscribes to selected ROS 2 topics and republishes to Summit via MQTT
- Optionally publishes Summit topics back to ROS 2

Safe to import even if ROS 2 is not installed; functions will no-op.
"""
from __future__ import annotations

import os
import json
from typing import Callable, Dict, Any, List, Optional

try:
    import rclpy  # type: ignore
    from rclpy.node import Node  # type: ignore
    from std_msgs.msg import String  # type: ignore
    ROS2_AVAILABLE = True
except Exception:
    ROS2_AVAILABLE = False

try:
    import paho.mqtt.client as mqtt  # type: ignore
    MQTT_AVAILABLE = True
except Exception:
    MQTT_AVAILABLE = False


def bridge_available() -> bool:
    return ROS2_AVAILABLE and MQTT_AVAILABLE


class _RosMqttBridge(Node):
    def __init__(self, name: str, mqtt_host: str, mqtt_port: int, topic_map: Dict[str, str]):
        super().__init__(name)
        self._topic_map = topic_map  # ROS topic -> MQTT topic
        self._mqtt = mqtt.Client() if MQTT_AVAILABLE else None
        if self._mqtt:
            self._mqtt.connect(mqtt_host, mqtt_port, 60)
            self._mqtt.loop_start()
        self._subs = []
        for ros_topic, mqtt_topic in topic_map.items():
            sub = self.create_subscription(String, ros_topic, self._make_cb(mqtt_topic), 10)
            self._subs.append(sub)
        self.get_logger().info(f"ROS<->MQTT bridge started; topics: {list(topic_map.keys())}")

    def _make_cb(self, mqtt_topic: str) -> Callable[[Any], None]:
        def _cb(msg: Any) -> None:
            try:
                payload = msg.data if hasattr(msg, "data") else str(msg)
                if isinstance(payload, (dict, list)):
                    data = json.dumps(payload)
                else:
                    data = str(payload)
                if self._mqtt:
                    self._mqtt.publish(mqtt_topic, data, qos=1)
            except Exception:
                pass
        return _cb

    def destroy(self) -> None:  # type: ignore[override]
        try:
            if self._mqtt:
                self._mqtt.loop_stop()
                self._mqtt.disconnect()
        finally:
            super().destroy_node()


def start_ros2_to_mqtt_bridge(
    topic_map: Dict[str, str],
    *,
    node_name: str = "summit_ros_bridge",
    mqtt_host: str | None = None,
    mqtt_port: int | None = None,
) -> None:
    """
    Start a simple ROS 2 -> MQTT bridge.

    topic_map: {ros_topic: mqtt_topic}
    """
    if not ROS2_AVAILABLE or not MQTT_AVAILABLE:
        return
    mqtt_host = mqtt_host or os.getenv("MQTT_BROKER", "localhost")
    mqtt_port = mqtt_port or int(os.getenv("MQTT_PORT", "1883"))
    rclpy.init()
    node = _RosMqttBridge(node_name, mqtt_host, mqtt_port, topic_map)
    try:
        rclpy.spin(node)
    finally:
        node.destroy()
        rclpy.shutdown()
