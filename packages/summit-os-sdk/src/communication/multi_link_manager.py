"""
Multi-Link Communication Manager for Summit.OS

Manages multiple communication links (radio mesh, cellular, satellite, WiFi)
with automatic failover, QoS, and autonomous operation capabilities.
"""

import asyncio
import logging
import time
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class LinkType(Enum):
    RADIO_MESH = "radio_mesh"
    CELLULAR = "cellular"
    SATELLITE = "satellite"
    WIFI = "wifi"


class LinkStatus(Enum):
    ACTIVE = "active"
    STANDBY = "standby"
    FAILED = "failed"
    DISABLED = "disabled"


@dataclass
class LinkMetrics:
    """Communication link quality metrics"""
    latency_ms: float = 0.0
    bandwidth_mbps: float = 0.0
    packet_loss_pct: float = 0.0
    signal_strength_dbm: Optional[float] = None
    reliability_score: float = 0.0
    last_update: float = field(default_factory=time.time)


@dataclass
class LinkConfiguration:
    """Configuration for a communication link"""
    link_type: LinkType
    enabled: bool = True
    priority: int = 0
    max_bandwidth_mbps: float = 10.0
    cost_per_mb: float = 0.0
    backup_only: bool = False
    config: Dict[str, Any] = field(default_factory=dict)


class CommunicationLink(ABC):
    """Abstract base class for communication links"""
    
    def __init__(self, config: LinkConfiguration):
        self.config = config
        self.status = LinkStatus.DISABLED
        self.metrics = LinkMetrics()
        self._callbacks: List[Callable] = []
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the communication link"""
        pass
    
    @abstractmethod
    async def send_data(self, data: bytes, priority: int = 0) -> bool:
        """Send data over the link"""
        pass
    
    @abstractmethod
    async def receive_data(self) -> Optional[bytes]:
        """Receive data from the link"""
        pass
    
    @abstractmethod
    async def get_metrics(self) -> LinkMetrics:
        """Get current link metrics"""
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the link"""
        pass
    
    def add_callback(self, callback: Callable):
        """Add status change callback"""
        self._callbacks.append(callback)
    
    def _notify_status_change(self, old_status: LinkStatus, new_status: LinkStatus):
        """Notify callbacks of status change"""
        for callback in self._callbacks:
            try:
                callback(self.config.link_type, old_status, new_status)
            except Exception as e:
                logger.error(f"Callback error: {e}")


class RadioMeshLink(CommunicationLink):
    """802.11s mesh networking link for low-latency local communication"""
    
    async def initialize(self) -> bool:
        try:
            # Initialize mesh interface
            mesh_config = self.config.config
            frequency = mesh_config.get('frequency', '900MHz')
            mesh_id = mesh_config.get('mesh_id', 'summit-mesh')
            
            logger.info(f"Initializing radio mesh on {frequency} (mesh_id: {mesh_id})")
            
            # TODO: Implement actual mesh setup
            # This would configure the wireless interface in mesh mode
            
            self.status = LinkStatus.ACTIVE
            self.metrics.reliability_score = 0.9  # High for local mesh
            return True
        except Exception as e:
            logger.error(f"Failed to initialize radio mesh: {e}")
            self.status = LinkStatus.FAILED
            return False
    
    async def send_data(self, data: bytes, priority: int = 0) -> bool:
        if self.status != LinkStatus.ACTIVE:
            return False
        
        try:
            # TODO: Implement mesh data transmission
            # This would send data via the mesh interface
            return True
        except Exception as e:
            logger.error(f"Mesh send error: {e}")
            return False
    
    async def receive_data(self) -> Optional[bytes]:
        if self.status != LinkStatus.ACTIVE:
            return None
        
        try:
            # TODO: Implement mesh data reception
            return None
        except Exception as e:
            logger.error(f"Mesh receive error: {e}")
            return None
    
    async def get_metrics(self) -> LinkMetrics:
        # TODO: Get actual metrics from mesh interface
        self.metrics.latency_ms = 10.0  # Low latency for local mesh
        self.metrics.bandwidth_mbps = 50.0
        self.metrics.packet_loss_pct = 0.1
        self.metrics.last_update = time.time()
        return self.metrics
    
    async def shutdown(self) -> None:
        # TODO: Shutdown mesh interface
        self.status = LinkStatus.DISABLED
        logger.info("Radio mesh link shutdown")


class CellularLink(CommunicationLink):
    """Cellular (LTE/5G) communication link"""
    
    async def initialize(self) -> bool:
        try:
            cellular_config = self.config.config
            apn = cellular_config.get('apn', '')
            bands = cellular_config.get('bands', [])
            
            logger.info(f"Initializing cellular link (APN: {apn})")
            
            # TODO: Implement cellular modem initialization
            
            self.status = LinkStatus.ACTIVE
            self.metrics.reliability_score = 0.8  # Good reliability
            return True
        except Exception as e:
            logger.error(f"Failed to initialize cellular: {e}")
            self.status = LinkStatus.FAILED
            return False
    
    async def send_data(self, data: bytes, priority: int = 0) -> bool:
        if self.status != LinkStatus.ACTIVE:
            return False
        
        try:
            # TODO: Send data via cellular modem
            return True
        except Exception as e:
            logger.error(f"Cellular send error: {e}")
            return False
    
    async def receive_data(self) -> Optional[bytes]:
        if self.status != LinkStatus.ACTIVE:
            return None
        
        try:
            # TODO: Receive data from cellular modem
            return None
        except Exception as e:
            logger.error(f"Cellular receive error: {e}")
            return None
    
    async def get_metrics(self) -> LinkMetrics:
        # TODO: Get actual metrics from cellular modem
        self.metrics.latency_ms = 50.0  # Moderate latency
        self.metrics.bandwidth_mbps = 20.0
        self.metrics.packet_loss_pct = 0.5
        self.metrics.signal_strength_dbm = -70.0
        self.metrics.last_update = time.time()
        return self.metrics
    
    async def shutdown(self) -> None:
        # TODO: Shutdown cellular modem
        self.status = LinkStatus.DISABLED
        logger.info("Cellular link shutdown")


class SatelliteLink(CommunicationLink):
    """Satellite communication link (Starlink, etc.)"""
    
    async def initialize(self) -> bool:
        try:
            sat_config = self.config.config
            provider = sat_config.get('provider', 'starlink')
            dish_type = sat_config.get('dish_type', 'mobile')
            
            logger.info(f"Initializing satellite link ({provider}, {dish_type})")
            
            # TODO: Implement satellite terminal initialization
            
            self.status = LinkStatus.STANDBY if self.config.backup_only else LinkStatus.ACTIVE
            self.metrics.reliability_score = 0.7  # Good but weather dependent
            return True
        except Exception as e:
            logger.error(f"Failed to initialize satellite: {e}")
            self.status = LinkStatus.FAILED
            return False
    
    async def send_data(self, data: bytes, priority: int = 0) -> bool:
        if self.status not in [LinkStatus.ACTIVE, LinkStatus.STANDBY]:
            return False
        
        # Only use satellite for critical data or when primary links are down
        if self.config.backup_only and priority < 5:
            return False
        
        try:
            # TODO: Send data via satellite terminal
            return True
        except Exception as e:
            logger.error(f"Satellite send error: {e}")
            return False
    
    async def receive_data(self) -> Optional[bytes]:
        if self.status not in [LinkStatus.ACTIVE, LinkStatus.STANDBY]:
            return None
        
        try:
            # TODO: Receive data from satellite terminal
            return None
        except Exception as e:
            logger.error(f"Satellite receive error: {e}")
            return None
    
    async def get_metrics(self) -> LinkMetrics:
        # TODO: Get actual metrics from satellite terminal
        self.metrics.latency_ms = 600.0  # High latency (GEO satellites)
        self.metrics.bandwidth_mbps = 100.0  # High bandwidth
        self.metrics.packet_loss_pct = 1.0
        self.metrics.last_update = time.time()
        return self.metrics
    
    async def shutdown(self) -> None:
        # TODO: Shutdown satellite terminal
        self.status = LinkStatus.DISABLED
        logger.info("Satellite link shutdown")


class MultiLinkManager:
    """
    Manages multiple communication links with automatic failover,
    load balancing, and autonomous operation capabilities.
    """
    
    def __init__(self, device_id: str):
        self.device_id = device_id
        self.links: Dict[LinkType, CommunicationLink] = {}
        self.active_links: List[LinkType] = []
        self.failover_order: List[LinkType] = []
        self.autonomous_mode = False
        self.message_buffer: List[Dict[str, Any]] = []
        self._running = False
        self._tasks: List[asyncio.Task] = []
    
    def add_link(self, link: CommunicationLink):
        """Add a communication link"""
        self.links[link.config.link_type] = link
        link.add_callback(self._on_link_status_change)
        logger.info(f"Added {link.config.link_type.value} link")
    
    def set_failover_order(self, order: List[LinkType]):
        """Set the failover priority order"""
        self.failover_order = order
        logger.info(f"Failover order set to: {[lt.value for lt in order]}")
    
    async def initialize_all_links(self) -> bool:
        """Initialize all configured links"""
        success_count = 0
        for link_type, link in self.links.items():
            try:
                if await link.initialize():
                    success_count += 1
                    if link.status == LinkStatus.ACTIVE:
                        self.active_links.append(link_type)
            except Exception as e:
                logger.error(f"Failed to initialize {link_type.value}: {e}")
        
        logger.info(f"Initialized {success_count}/{len(self.links)} links")
        return success_count > 0
    
    async def start(self):
        """Start the multi-link manager"""
        self._running = True
        
        # Start monitoring task
        monitor_task = asyncio.create_task(self._monitor_links())
        self._tasks.append(monitor_task)
        
        # Start autonomous sync task if enabled
        if self.autonomous_mode:
            sync_task = asyncio.create_task(self._autonomous_sync())
            self._tasks.append(sync_task)
        
        logger.info("Multi-link manager started")
    
    async def stop(self):
        """Stop the multi-link manager"""
        self._running = False
        
        # Cancel all tasks
        for task in self._tasks:
            task.cancel()
        
        # Shutdown all links
        for link in self.links.values():
            await link.shutdown()
        
        logger.info("Multi-link manager stopped")
    
    async def send_message(self, data: bytes, priority: int = 0, 
                          preferred_link: Optional[LinkType] = None) -> bool:
        """Send message with automatic link selection"""
        
        # Try preferred link first
        if preferred_link and preferred_link in self.active_links:
            link = self.links[preferred_link]
            if await link.send_data(data, priority):
                return True
        
        # Try active links in failover order
        for link_type in self.failover_order:
            if link_type in self.active_links:
                link = self.links[link_type]
                if await link.send_data(data, priority):
                    return True
        
        # If all active links failed, buffer for autonomous mode
        if self.autonomous_mode:
            self.message_buffer.append({
                'data': data,
                'priority': priority,
                'timestamp': time.time()
            })
            logger.warning("All links failed, buffered message for later sync")
            return True
        
        return False
    
    def enable_autonomous_mode(self, buffer_size_mb: int = 100):
        """Enable autonomous operation mode"""
        self.autonomous_mode = True
        # TODO: Set buffer size limit
        logger.info("Autonomous mode enabled")
    
    def disable_autonomous_mode(self):
        """Disable autonomous operation mode"""
        self.autonomous_mode = False
        logger.info("Autonomous mode disabled")
    
    async def get_best_link(self, criteria: Dict[str, float]) -> Optional[LinkType]:
        """Select best link based on criteria weights"""
        if not self.active_links:
            return None
        
        best_link = None
        best_score = -1.0
        
        for link_type in self.active_links:
            link = self.links[link_type]
            metrics = await link.get_metrics()
            
            # Calculate weighted score
            latency_score = max(0, 1.0 - (metrics.latency_ms / 1000.0))  # Normalize to 0-1
            bandwidth_score = min(1.0, metrics.bandwidth_mbps / 100.0)   # Normalize to 0-1
            reliability_score = metrics.reliability_score
            
            total_score = (
                latency_score * criteria.get('latency_weight', 0.33) +
                bandwidth_score * criteria.get('bandwidth_weight', 0.33) +
                reliability_score * criteria.get('reliability_weight', 0.34)
            )
            
            if total_score > best_score:
                best_score = total_score
                best_link = link_type
        
        return best_link
    
    async def _monitor_links(self):
        """Monitor link health and manage failover"""
        while self._running:
            try:
                failed_links = []
                
                for link_type in list(self.active_links):
                    link = self.links[link_type]
                    metrics = await link.get_metrics()
                    
                    # Check if link has failed
                    if (metrics.packet_loss_pct > 50.0 or 
                        time.time() - metrics.last_update > 60.0):
                        failed_links.append(link_type)
                
                # Remove failed links
                for link_type in failed_links:
                    if link_type in self.active_links:
                        self.active_links.remove(link_type)
                        logger.warning(f"Link {link_type.value} marked as failed")
                
                await asyncio.sleep(10)  # Monitor every 10 seconds
                
            except Exception as e:
                logger.error(f"Link monitoring error: {e}")
                await asyncio.sleep(10)
    
    async def _autonomous_sync(self):
        """Sync buffered data when connectivity is restored"""
        while self._running:
            try:
                if self.message_buffer and self.active_links:
                    logger.info(f"Syncing {len(self.message_buffer)} buffered messages")
                    
                    # Try to send buffered messages
                    sent_messages = []
                    for msg in self.message_buffer:
                        if await self.send_message(msg['data'], msg['priority']):
                            sent_messages.append(msg)
                    
                    # Remove successfully sent messages
                    for msg in sent_messages:
                        self.message_buffer.remove(msg)
                    
                    logger.info(f"Successfully synced {len(sent_messages)} messages")
                
                await asyncio.sleep(300)  # Sync every 5 minutes
                
            except Exception as e:
                logger.error(f"Autonomous sync error: {e}")
                await asyncio.sleep(300)
    
    def _on_link_status_change(self, link_type: LinkType, 
                              old_status: LinkStatus, new_status: LinkStatus):
        """Handle link status changes"""
        logger.info(f"Link {link_type.value} status: {old_status.value} -> {new_status.value}")
        
        if new_status == LinkStatus.ACTIVE and link_type not in self.active_links:
            self.active_links.append(link_type)
        elif new_status in [LinkStatus.FAILED, LinkStatus.DISABLED] and link_type in self.active_links:
            self.active_links.remove(link_type)