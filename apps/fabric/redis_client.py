"""Redis client for Summit.OS Data Fabric Service."""

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
import redis.asyncio as redis
from redis.asyncio import Redis

from .models import TelemetryMessage, AlertMessage, MissionUpdate, SystemMetrics


class RedisClient:
    """Async Redis client for data fabric."""
    
    def __init__(self, redis_url: str):
        self.redis_url = redis_url
        self.redis: Optional[Redis] = None
        
    async def connect(self):
        """Connect to Redis."""
        try:
            self.redis = redis.from_url(self.redis_url, decode_responses=True)
            await self.redis.ping()
            logging.info("Connected to Redis")
        except Exception as e:
            logging.error(f"Error connecting to Redis: {e}")
            raise
    
    async def disconnect(self):
        """Disconnect from Redis."""
        if self.redis:
            await self.redis.close()
            logging.info("Disconnected from Redis")
    
    async def add_telemetry(self, telemetry: TelemetryMessage):
        """Add telemetry data to Redis Streams."""
        if not self.redis:
            raise ConnectionError("Not connected to Redis")
        
        try:
            # Add to telemetry stream
            stream_data = {
                "device_id": telemetry.device_id,
                "timestamp": telemetry.timestamp.isoformat(),
                "location": json.dumps(telemetry.location.model_dump()),
                "sensors": json.dumps(telemetry.sensors),
                "status": telemetry.status,
                "battery_level": str(telemetry.battery_level) if telemetry.battery_level else "",
                "signal_strength": str(telemetry.signal_strength) if telemetry.signal_strength else "",
                "metadata": json.dumps(telemetry.metadata)
            }
            
            await self.redis.xadd("telemetry_stream", stream_data)
            
            # Update device status
            await self.redis.hset(
                f"device:{telemetry.device_id}",
                mapping={
                    "status": telemetry.status,
                    "last_seen": telemetry.timestamp.isoformat(),
                    "location": json.dumps(telemetry.location.model_dump())
                }
            )
            
            # Set expiration for device status
            await self.redis.expire(f"device:{telemetry.device_id}", 3600)  # 1 hour
            
        except Exception as e:
            logging.error(f"Error adding telemetry to Redis: {e}")
            raise
    
    async def add_alert(self, alert: AlertMessage):
        """Add alert data to Redis Streams."""
        if not self.redis:
            raise ConnectionError("Not connected to Redis")
        
        try:
            stream_data = {
                "alert_id": alert.alert_id,
                "timestamp": alert.timestamp.isoformat(),
                "severity": alert.severity,
                "location": json.dumps(alert.location.model_dump()),
                "description": alert.description,
                "source": alert.source,
                "category": alert.category,
                "tags": json.dumps(alert.tags),
                "metadata": json.dumps(alert.metadata),
                "acknowledged": str(alert.acknowledged),
                "acknowledged_by": alert.acknowledged_by or "",
                "acknowledged_at": alert.acknowledged_at.isoformat() if alert.acknowledged_at else ""
            }
            
            await self.redis.xadd("alert_stream", stream_data)
            
            # Store alert details
            await self.redis.hset(
                f"alert:{alert.alert_id}",
                mapping=stream_data
            )
            
            # Add to severity-based sets
            await self.redis.sadd(f"alerts:{alert.severity}", alert.alert_id)
            await self.redis.sadd("alerts:active", alert.alert_id)
            
        except Exception as e:
            logging.error(f"Error adding alert to Redis: {e}")
            raise
    
    async def add_mission_update(self, mission: MissionUpdate):
        """Add mission update to Redis Streams."""
        if not self.redis:
            raise ConnectionError("Not connected to Redis")
        
        try:
            stream_data = {
                "mission_id": mission.mission_id,
                "timestamp": mission.timestamp.isoformat(),
                "status": mission.status,
                "assets": json.dumps(mission.assets),
                "objectives": json.dumps(mission.objectives),
                "progress": str(mission.progress),
                "estimated_completion": mission.estimated_completion.isoformat() if mission.estimated_completion else "",
                "metadata": json.dumps(mission.metadata)
            }
            
            await self.redis.xadd("mission_stream", stream_data)
            
            # Update mission status
            await self.redis.hset(
                f"mission:{mission.mission_id}",
                mapping=stream_data
            )
            
            # Update mission status sets
            await self.redis.sadd(f"missions:{mission.status}", mission.mission_id)
            
        except Exception as e:
            logging.error(f"Error adding mission update to Redis: {e}")
            raise
    
    async def process_telemetry_stream(self):
        """Process telemetry from Redis Streams."""
        if not self.redis:
            return
        
        try:
            # Read from telemetry stream
            messages = await self.redis.xread(
                {"telemetry_stream": "$"},
                count=100,
                block=1000
            )
            
            for stream, msgs in messages:
                for msg_id, fields in msgs:
                    # Process telemetry message
                    device_id = fields.get("device_id")
                    if device_id:
                        # Update device metrics
                        await self.redis.incr(f"metrics:telemetry:{device_id}")
                        await self.redis.incr("metrics:telemetry:total")
                        
                        # Update device location index
                        location = json.loads(fields.get("location", "{}"))
                        if "latitude" in location and "longitude" in location:
                            await self.redis.geoadd(
                                "device_locations",
                                location["longitude"],
                                location["latitude"],
                                device_id
                            )
            
        except Exception as e:
            logging.error(f"Error processing telemetry stream: {e}")
    
    async def process_alert_stream(self):
        """Process alerts from Redis Streams."""
        if not self.redis:
            return
        
        try:
            # Read from alert stream
            messages = await self.redis.xread(
                {"alert_stream": "$"},
                count=100,
                block=1000
            )
            
            for stream, msgs in messages:
                for msg_id, fields in msgs:
                    # Process alert message
                    alert_id = fields.get("alert_id")
                    severity = fields.get("severity")
                    
                    if alert_id and severity:
                        # Update alert metrics
                        await self.redis.incr(f"metrics:alerts:{severity}")
                        await self.redis.incr("metrics:alerts:total")
                        
                        # Check for alert escalation
                        if severity in ["high", "critical"]:
                            await self.redis.sadd("alerts:escalated", alert_id)
            
        except Exception as e:
            logging.error(f"Error processing alert stream: {e}")
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get system metrics from Redis."""
        if not self.redis:
            return {}
        
        try:
            metrics = {}
            
            # Get telemetry metrics
            total_telemetry = await self.redis.get("metrics:telemetry:total") or "0"
            metrics["telemetry_total"] = int(total_telemetry)
            
            # Get alert metrics
            total_alerts = await self.redis.get("metrics:alerts:total") or "0"
            metrics["alerts_total"] = int(total_alerts)
            
            # Get active devices
            device_keys = await self.redis.keys("device:*")
            metrics["active_devices"] = len(device_keys)
            
            # Get active alerts
            active_alerts = await self.redis.scard("alerts:active")
            metrics["active_alerts"] = active_alerts
            
            # Get escalated alerts
            escalated_alerts = await self.redis.scard("alerts:escalated")
            metrics["escalated_alerts"] = escalated_alerts
            
            return metrics
            
        except Exception as e:
            logging.error(f"Error getting metrics from Redis: {e}")
            return {}
    
    async def get_device_status(self, device_id: str) -> Optional[Dict[str, Any]]:
        """Get device status from Redis."""
        if not self.redis:
            return None
        
        try:
            device_data = await self.redis.hgetall(f"device:{device_id}")
            if device_data:
                # Parse JSON fields
                if "location" in device_data:
                    device_data["location"] = json.loads(device_data["location"])
                return device_data
            return None
            
        except Exception as e:
            logging.error(f"Error getting device status: {e}")
            return None
    
    async def get_active_alerts(self, severity: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get active alerts from Redis."""
        if not self.redis:
            return []
        
        try:
            if severity:
                alert_ids = await self.redis.smembers(f"alerts:{severity}")
            else:
                alert_ids = await self.redis.smembers("alerts:active")
            
            alerts = []
            for alert_id in alert_ids:
                alert_data = await self.redis.hgetall(f"alert:{alert_id}")
                if alert_data:
                    # Parse JSON fields
                    if "location" in alert_data:
                        alert_data["location"] = json.loads(alert_data["location"])
                    if "tags" in alert_data:
                        alert_data["tags"] = json.loads(alert_data["tags"])
                    if "metadata" in alert_data:
                        alert_data["metadata"] = json.loads(alert_data["metadata"])
                    alerts.append(alert_data)
            
            return alerts
            
        except Exception as e:
            logging.error(f"Error getting active alerts: {e}")
            return []
