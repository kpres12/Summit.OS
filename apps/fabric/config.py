"""Configuration settings for Summit.OS Data Fabric Service."""

from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings."""
    
    # Service configuration
    service_name: str = "data-fabric"
    debug: bool = False
    
    # Redis configuration
    redis_url: str = "redis://localhost:6379"
    redis_password: Optional[str] = None
    
    # MQTT configuration
    mqtt_broker: str = "localhost"
    mqtt_port: int = 1883
    mqtt_username: Optional[str] = None
    mqtt_password: Optional[str] = None
    mqtt_keepalive: int = 60
    
    # WebSocket configuration
    websocket_max_connections: int = 1000
    
    # Telemetry configuration
    telemetry_retention_hours: int = 24
    max_telemetry_rate: int = 1000  # messages per second
    
    # Alert configuration
    alert_retention_days: int = 30
    
    class Config:
        env_file = ".env"
        env_prefix = "FABRIC_"
