"""MQTT client for Summit.OS Data Fabric Service."""

import asyncio
import json
import logging
from typing import Callable, Optional, Dict, Any
import paho.mqtt.client as mqtt
from paho.mqtt.client import MQTTMessage


class MQTTClient:
    """Async MQTT client wrapper."""
    
    def __init__(
        self,
        broker: str,
        port: int = 1883,
        username: Optional[str] = None,
        password: Optional[str] = None,
        keepalive: int = 60
    ):
        self.broker = broker
        self.port = port
        self.username = username
        self.password = password
        self.keepalive = keepalive
        
        self.client = mqtt.Client()
        self.client.username_pw_set(username, password)
        
        # Set up callbacks
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message
        self.client.on_publish = self._on_publish
        
        self.connected = False
        self.subscriptions: Dict[str, Callable] = {}
        
    def _on_connect(self, client, userdata, flags, rc):
        """MQTT connection callback."""
        if rc == 0:
            self.connected = True
            logging.info(f"Connected to MQTT broker {self.broker}:{self.port}")
        else:
            logging.error(f"Failed to connect to MQTT broker: {rc}")
    
    def _on_disconnect(self, client, userdata, rc):
        """MQTT disconnection callback."""
        self.connected = False
        logging.info(f"Disconnected from MQTT broker: {rc}")
    
    def _on_message(self, client, userdata, msg: MQTTMessage):
        """MQTT message callback."""
        topic = msg.topic
        payload = msg.payload.decode('utf-8')
        
        # Find and call subscription handler
        for pattern, handler in self.subscriptions.items():
            if self._topic_matches(topic, pattern):
                try:
                    data = json.loads(payload)
                    asyncio.create_task(self._handle_message(handler, topic, data))
                except json.JSONDecodeError:
                    logging.error(f"Invalid JSON in MQTT message: {payload}")
                except Exception as e:
                    logging.error(f"Error handling MQTT message: {e}")
                break
    
    def _on_publish(self, client, userdata, mid):
        """MQTT publish callback."""
        logging.debug(f"Published message with mid: {mid}")
    
    async def _handle_message(self, handler: Callable, topic: str, data: Dict[str, Any]):
        """Handle incoming MQTT message."""
        try:
            if asyncio.iscoroutinefunction(handler):
                await handler(topic, data)
            else:
                handler(topic, data)
        except Exception as e:
            logging.error(f"Error in message handler: {e}")
    
    def _topic_matches(self, topic: str, pattern: str) -> bool:
        """Check if topic matches pattern (supports + and # wildcards)."""
        if pattern == topic:
            return True
        
        # Simple wildcard matching
        if '+' in pattern or '#' in pattern:
            pattern_parts = pattern.split('/')
            topic_parts = topic.split('/')
            
            if len(pattern_parts) != len(topic_parts) and '#' not in pattern:
                return False
            
            for i, pattern_part in enumerate(pattern_parts):
                if pattern_part == '#':
                    return True
                elif pattern_part == '+':
                    continue
                elif i >= len(topic_parts) or pattern_part != topic_parts[i]:
                    return False
            
            return True
        
        return False
    
    async def connect(self):
        """Connect to MQTT broker."""
        try:
            self.client.connect(self.broker, self.port, self.keepalive)
            self.client.loop_start()
            
            # Wait for connection
            timeout = 10
            while not self.connected and timeout > 0:
                await asyncio.sleep(0.1)
                timeout -= 0.1
            
            if not self.connected:
                raise ConnectionError("Failed to connect to MQTT broker")
                
        except Exception as e:
            logging.error(f"Error connecting to MQTT broker: {e}")
            raise
    
    async def disconnect(self):
        """Disconnect from MQTT broker."""
        try:
            self.client.loop_stop()
            self.client.disconnect()
            self.connected = False
            logging.info("Disconnected from MQTT broker")
        except Exception as e:
            logging.error(f"Error disconnecting from MQTT broker: {e}")
    
    async def publish(self, topic: str, message: str, qos: int = 0, retain: bool = False):
        """Publish message to MQTT topic."""
        if not self.connected:
            raise ConnectionError("Not connected to MQTT broker")
        
        try:
            result = self.client.publish(topic, message, qos=qos, retain=retain)
            if result.rc != mqtt.MQTT_ERR_SUCCESS:
                raise Exception(f"Failed to publish message: {result.rc}")
            
            logging.debug(f"Published to {topic}: {message[:100]}...")
            
        except Exception as e:
            logging.error(f"Error publishing to MQTT: {e}")
            raise
    
    async def subscribe(self, topic: str, handler: Callable):
        """Subscribe to MQTT topic with handler."""
        if not self.connected:
            raise ConnectionError("Not connected to MQTT broker")
        
        try:
            result = self.client.subscribe(topic)
            if result[0] != mqtt.MQTT_ERR_SUCCESS:
                raise Exception(f"Failed to subscribe to topic: {result[0]}")
            
            self.subscriptions[topic] = handler
            logging.info(f"Subscribed to topic: {topic}")
            
        except Exception as e:
            logging.error(f"Error subscribing to MQTT topic: {e}")
            raise
    
    async def unsubscribe(self, topic: str):
        """Unsubscribe from MQTT topic."""
        if not self.connected:
            return
        
        try:
            result = self.client.unsubscribe(topic)
            if result[0] != mqtt.MQTT_ERR_SUCCESS:
                logging.warning(f"Failed to unsubscribe from topic: {result[0]}")
            
            self.subscriptions.pop(topic, None)
            logging.info(f"Unsubscribed from topic: {topic}")
            
        except Exception as e:
            logging.error(f"Error unsubscribing from MQTT topic: {e}")
