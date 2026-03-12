"""
AdapterManifest — the contract an adapter declares to Summit.OS.

Every adapter must declare a manifest. Summit.OS uses it to:
- Validate the adapter before it's allowed to publish
- Display capabilities in the DEV view Adapter Registry
- Enforce permission boundaries (a READ-only adapter cannot send actuator commands)
- Version-check compatibility
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class Protocol(str, Enum):
    """Physical/network protocol the adapter speaks."""
    MODBUS    = "modbus"
    OPCUA     = "opcua"
    MAVLINK   = "mavlink"
    ADSB      = "adsb"
    TLE       = "tle"          # satellite orbital elements
    MQTT      = "mqtt"
    HTTP      = "http"
    WEBSOCKET = "websocket"
    SERIAL    = "serial"
    CAN       = "can"
    ROS2      = "ros2"
    RTSP      = "rtsp"         # camera / video streams
    AIS       = "ais"          # maritime AIS
    CUSTOM    = "custom"


class Capability(str, Enum):
    """What the adapter can do."""
    READ       = "read"        # can read sensor/state data
    WRITE      = "write"       # can send commands to hardware (requires approval policy)
    SUBSCRIBE  = "subscribe"   # uses push/subscription model (not polling)
    STREAM     = "stream"      # produces continuous high-rate data
    DISCOVER   = "discover"    # can enumerate available devices/nodes


class EntityType(str, Enum):
    ASSET   = "ASSET"
    TRACK   = "TRACK"
    ALERT   = "ALERT"
    MISSION = "MISSION"


@dataclass
class AdapterManifest:
    """
    Declares what an adapter is and what it can do.

    This is the contract between an adapter and Summit.OS.
    The platform refuses to load an adapter without a valid manifest.
    """
    name: str
    """Unique adapter name. Used as source_id prefix in entities. Use kebab-case."""

    version: str
    """Semver version of this adapter (e.g. "1.0.0")."""

    protocol: Protocol
    """The physical/network protocol this adapter speaks."""

    capabilities: List[Capability]
    """What this adapter can do. Be conservative — request only what you need."""

    entity_types: List[str]
    """Entity types this adapter produces (e.g. ["ASSET", "TRACK"])."""

    description: str = ""
    """Human-readable description shown in the DEV Adapter Registry."""

    author: str = ""
    """Adapter author or organization."""

    min_summit_version: str = "1.0.0"
    """Minimum Summit.OS version required."""

    required_env: List[str] = field(default_factory=list)
    """
    Environment variables this adapter requires.
    Summit.OS will warn (not fail) if any are missing at startup.
    Example: ["MODBUS_HOST", "MODBUS_PORT"]
    """

    optional_env: List[str] = field(default_factory=list)
    """Environment variables this adapter uses but doesn't require."""

    homepage: str = ""
    """Link to adapter documentation or repository."""

    def validate(self) -> List[str]:
        """Return a list of validation errors. Empty list = valid."""
        errors = []
        if not self.name or not self.name.replace("-", "").replace("_", "").isalnum():
            errors.append("name must be alphanumeric (hyphens/underscores allowed)")
        if not self.version or len(self.version.split(".")) != 3:
            errors.append("version must be semver (e.g. '1.0.0')")
        if not self.capabilities:
            errors.append("capabilities must not be empty")
        if not self.entity_types:
            errors.append("entity_types must not be empty")
        if Capability.WRITE in self.capabilities and Capability.READ not in self.capabilities:
            errors.append("WRITE capability requires READ capability")
        return errors

    def requires_approval(self) -> bool:
        """True if this adapter can send commands to physical hardware."""
        return Capability.WRITE in self.capabilities

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "protocol": self.protocol.value,
            "capabilities": [c.value for c in self.capabilities],
            "entity_types": self.entity_types,
            "description": self.description,
            "author": self.author,
            "min_summit_version": self.min_summit_version,
            "required_env": self.required_env,
            "optional_env": self.optional_env,
            "homepage": self.homepage,
            "requires_approval": self.requires_approval(),
        }
