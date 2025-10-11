"""
Enhanced Offline Operation Manager for Summit.OS

Provides robust offline capabilities with local AI inference,
mission execution, and data synchronization for edge devices.
"""

import asyncio
import sqlite3
import json
import time
import logging
import threading
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import gzip
import pickle
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class SyncStatus(Enum):
    """Data synchronization status"""
    PENDING = "pending"
    SYNCING = "syncing"
    SYNCED = "synced"
    FAILED = "failed"
    CONFLICT = "conflict"


class OfflineMode(Enum):
    """Offline operation modes"""
    ONLINE = "online"
    OFFLINE = "offline"
    DEGRADED = "degraded"
    RECONNECTING = "reconnecting"


@dataclass
class DataMessage:
    """Data message for offline storage"""
    message_id: str
    message_type: str
    device_id: str
    timestamp: float
    data: Dict[str, Any]
    priority: int = 1
    sync_status: SyncStatus = SyncStatus.PENDING
    retry_count: int = 0
    max_retries: int = 3
    created_at: float = None
    synced_at: Optional[float] = None


@dataclass
class OfflineMission:
    """Offline mission definition"""
    mission_id: str
    device_id: str
    mission_type: str
    waypoints: List[Dict[str, Any]]
    parameters: Dict[str, Any]
    priority: int = 1
    created_at: float = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    status: str = "pending"


class OfflineDataManager:
    """
    Manages offline data storage and synchronization for Summit.OS edge devices.
    
    Provides local SQLite storage, compression, and conflict resolution.
    """
    
    def __init__(self, device_id: str, storage_path: str = "./offline_data"):
        self.device_id = device_id
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        # Database paths
        self.db_path = self.storage_path / f"{device_id}_offline.db"
        self.mission_db_path = self.storage_path / f"{device_id}_missions.db"
        
        # Initialize databases
        self._init_databases()
        
        # Storage limits
        self.max_storage_mb = 1000  # 1GB default
        self.max_message_age_hours = 168  # 1 week
        self.compression_enabled = True
        
        # Sync configuration
        self.sync_batch_size = 100
        self.sync_interval = 300  # 5 minutes
        self.retry_delay = 60  # 1 minute
        
        # Statistics
        self.stats = {
            'messages_stored': 0,
            'messages_synced': 0,
            'messages_failed': 0,
            'storage_used_mb': 0,
            'last_sync': None
        }
    
    def _init_databases(self):
        """Initialize SQLite databases"""
        # Main data database
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                message_id TEXT PRIMARY KEY,
                message_type TEXT NOT NULL,
                device_id TEXT NOT NULL,
                timestamp REAL NOT NULL,
                data BLOB NOT NULL,
                priority INTEGER DEFAULT 1,
                sync_status TEXT DEFAULT 'pending',
                retry_count INTEGER DEFAULT 0,
                max_retries INTEGER DEFAULT 3,
                created_at REAL DEFAULT (julianday('now')),
                synced_at REAL
            )
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_sync_status 
            ON messages(sync_status, priority, created_at)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_timestamp 
            ON messages(timestamp)
        ''')
        
        conn.commit()
        conn.close()
        
        # Mission database
        conn = sqlite3.connect(str(self.mission_db_path))
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS missions (
                mission_id TEXT PRIMARY KEY,
                device_id TEXT NOT NULL,
                mission_type TEXT NOT NULL,
                waypoints BLOB NOT NULL,
                parameters BLOB NOT NULL,
                priority INTEGER DEFAULT 1,
                created_at REAL DEFAULT (julianday('now')),
                started_at REAL,
                completed_at REAL,
                status TEXT DEFAULT 'pending'
            )
        ''')
        
        conn.commit()
        conn.close()
    
    async def store_message(self, message_type: str, data: Dict[str, Any], 
                          priority: int = 1) -> str:
        """Store a message for offline synchronization"""
        message_id = self._generate_message_id()
        timestamp = time.time()
        
        # Compress data if enabled
        if self.compression_enabled:
            data_bytes = gzip.compress(pickle.dumps(data))
        else:
            data_bytes = pickle.dumps(data)
        
        # Create message
        message = DataMessage(
            message_id=message_id,
            message_type=message_type,
            device_id=self.device_id,
            timestamp=timestamp,
            data=data,
            priority=priority,
            created_at=timestamp
        )
        
        # Store in database
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO messages 
            (message_id, message_type, device_id, timestamp, data, priority, sync_status, retry_count, max_retries, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            message.message_id,
            message.message_type,
            message.device_id,
            message.timestamp,
            data_bytes,
            message.priority,
            message.sync_status.value,
            message.retry_count,
            message.max_retries,
            message.created_at
        ))
        
        conn.commit()
        conn.close()
        
        # Update statistics
        self.stats['messages_stored'] += 1
        self._update_storage_stats()
        
        logger.info(f"Stored offline message: {message_id}")
        return message_id
    
    async def get_pending_messages(self, limit: int = None) -> List[DataMessage]:
        """Get messages pending synchronization"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        query = '''
            SELECT message_id, message_type, device_id, timestamp, data, priority, 
                   sync_status, retry_count, max_retries, created_at, synced_at
            FROM messages 
            WHERE sync_status IN ('pending', 'failed')
            ORDER BY priority DESC, created_at ASC
        '''
        
        if limit:
            query += f' LIMIT {limit}'
        
        cursor.execute(query)
        rows = cursor.fetchall()
        conn.close()
        
        messages = []
        for row in rows:
            # Decompress data
            data_bytes = row[4]
            if self.compression_enabled:
                data = pickle.loads(gzip.decompress(data_bytes))
            else:
                data = pickle.loads(data_bytes)
            
            message = DataMessage(
                message_id=row[0],
                message_type=row[1],
                device_id=row[2],
                timestamp=row[3],
                data=data,
                priority=row[5],
                sync_status=SyncStatus(row[6]),
                retry_count=row[7],
                max_retries=row[8],
                created_at=row[9],
                synced_at=row[10]
            )
            messages.append(message)
        
        return messages
    
    async def mark_message_synced(self, message_id: str) -> bool:
        """Mark a message as successfully synchronized"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE messages 
            SET sync_status = 'synced', synced_at = julianday('now')
            WHERE message_id = ?
        ''', (message_id,))
        
        updated = cursor.rowcount > 0
        conn.commit()
        conn.close()
        
        if updated:
            self.stats['messages_synced'] += 1
            logger.info(f"Marked message as synced: {message_id}")
        
        return updated
    
    async def mark_message_failed(self, message_id: str, increment_retry: bool = True) -> bool:
        """Mark a message as failed to sync"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        if increment_retry:
            cursor.execute('''
                UPDATE messages 
                SET sync_status = 'failed', retry_count = retry_count + 1
                WHERE message_id = ?
            ''', (message_id,))
        else:
            cursor.execute('''
                UPDATE messages 
                SET sync_status = 'failed'
                WHERE message_id = ?
            ''', (message_id,))
        
        updated = cursor.rowcount > 0
        conn.commit()
        conn.close()
        
        if updated:
            self.stats['messages_failed'] += 1
            logger.warning(f"Marked message as failed: {message_id}")
        
        return updated
    
    async def cleanup_old_messages(self):
        """Clean up old synchronized messages"""
        cutoff_time = time.time() - (self.max_message_age_hours * 3600)
        
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute('''
            DELETE FROM messages 
            WHERE sync_status = 'synced' AND synced_at < ?
        ''', (cutoff_time,))
        
        deleted_count = cursor.rowcount
        conn.commit()
        conn.close()
        
        if deleted_count > 0:
            logger.info(f"Cleaned up {deleted_count} old messages")
        
        self._update_storage_stats()
    
    def _generate_message_id(self) -> str:
        """Generate unique message ID"""
        timestamp = str(int(time.time() * 1000))
        random_part = str(hash(time.time()) % 10000)
        return f"{self.device_id}_{timestamp}_{random_part}"
    
    def _update_storage_stats(self):
        """Update storage statistics"""
        # Calculate storage usage
        if self.db_path.exists():
            size_bytes = self.db_path.stat().st_size
            self.stats['storage_used_mb'] = size_bytes / (1024 * 1024)
    
    async def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Count messages by status
        cursor.execute('''
            SELECT sync_status, COUNT(*) 
            FROM messages 
            GROUP BY sync_status
        ''')
        status_counts = dict(cursor.fetchall())
        
        # Get oldest pending message
        cursor.execute('''
            SELECT MIN(created_at) 
            FROM messages 
            WHERE sync_status = 'pending'
        ''')
        oldest_pending = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            **self.stats,
            'status_counts': status_counts,
            'oldest_pending': oldest_pending,
            'storage_limit_mb': self.max_storage_mb,
            'compression_enabled': self.compression_enabled
        }


class OfflineMissionManager:
    """
    Manages offline mission execution for edge devices.
    """
    
    def __init__(self, device_id: str, mission_executor: Callable):
        self.device_id = device_id
        self.mission_executor = mission_executor
        self.active_missions: Dict[str, OfflineMission] = {}
        self.mission_lock = threading.Lock()
        
    async def store_mission(self, mission: OfflineMission) -> bool:
        """Store a mission for offline execution"""
        try:
            conn = sqlite3.connect(f"./offline_data/{self.device_id}_missions.db")
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO missions 
                (mission_id, device_id, mission_type, waypoints, parameters, priority, created_at, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                mission.mission_id,
                mission.device_id,
                mission.mission_type,
                pickle.dumps(mission.waypoints),
                pickle.dumps(mission.parameters),
                mission.priority,
                mission.created_at or time.time(),
                mission.status
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Stored offline mission: {mission.mission_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store mission: {e}")
            return False
    
    async def execute_mission(self, mission_id: str) -> bool:
        """Execute a stored mission"""
        try:
            # Load mission from database
            conn = sqlite3.connect(f"./offline_data/{self.device_id}_missions.db")
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT mission_id, device_id, mission_type, waypoints, parameters, 
                       priority, created_at, started_at, completed_at, status
                FROM missions WHERE mission_id = ?
            ''', (mission_id,))
            
            row = cursor.fetchone()
            if not row:
                logger.error(f"Mission not found: {mission_id}")
                return False
            
            # Create mission object
            mission = OfflineMission(
                mission_id=row[0],
                device_id=row[1],
                mission_type=row[2],
                waypoints=pickle.loads(row[3]),
                parameters=pickle.loads(row[4]),
                priority=row[5],
                created_at=row[6],
                started_at=row[7],
                completed_at=row[8],
                status=row[9]
            )
            
            # Update mission status
            cursor.execute('''
                UPDATE missions 
                SET status = 'executing', started_at = julianday('now')
                WHERE mission_id = ?
            ''', (mission_id,))
            
            conn.commit()
            conn.close()
            
            # Execute mission
            with self.mission_lock:
                self.active_missions[mission_id] = mission
            
            success = await self.mission_executor(mission)
            
            # Update completion status
            conn = sqlite3.connect(f"./offline_data/{self.device_id}_missions.db")
            cursor = conn.cursor()
            
            status = 'completed' if success else 'failed'
            cursor.execute('''
                UPDATE missions 
                SET status = ?, completed_at = julianday('now')
                WHERE mission_id = ?
            ''', (status, mission_id))
            
            conn.commit()
            conn.close()
            
            # Remove from active missions
            with self.mission_lock:
                if mission_id in self.active_missions:
                    del self.active_missions[mission_id]
            
            logger.info(f"Mission {mission_id} {'completed' if success else 'failed'}")
            return success
            
        except Exception as e:
            logger.error(f"Failed to execute mission {mission_id}: {e}")
            return False
    
    async def get_pending_missions(self) -> List[OfflineMission]:
        """Get missions pending execution"""
        conn = sqlite3.connect(f"./offline_data/{self.device_id}_missions.db")
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT mission_id, device_id, mission_type, waypoints, parameters, 
                   priority, created_at, started_at, completed_at, status
            FROM missions 
            WHERE status = 'pending'
            ORDER BY priority DESC, created_at ASC
        ''')
        
        rows = cursor.fetchall()
        conn.close()
        
        missions = []
        for row in rows:
            mission = OfflineMission(
                mission_id=row[0],
                device_id=row[1],
                mission_type=row[2],
                waypoints=pickle.loads(row[3]),
                parameters=pickle.loads(row[4]),
                priority=row[5],
                created_at=row[6],
                started_at=row[7],
                completed_at=row[8],
                status=row[9]
            )
            missions.append(mission)
        
        return missions


class OfflineManager:
    """
    Main offline operation manager for Summit.OS edge devices.
    
    Coordinates data storage, mission execution, and synchronization.
    """
    
    def __init__(self, device_id: str, summit_client: Any = None):
        self.device_id = device_id
        self.summit_client = summit_client
        
        # Initialize managers
        self.data_manager = OfflineDataManager(device_id)
        self.mission_manager = OfflineMissionManager(device_id, self._execute_mission)
        
        # Offline mode tracking
        self.offline_mode = OfflineMode.ONLINE
        self.last_connection_check = time.time()
        self.connection_timeout = 30  # seconds
        
        # Background tasks
        self.sync_task = None
        self.cleanup_task = None
        self.mission_task = None
        
        # Callbacks
        self.connection_callbacks: List[Callable] = []
        self.sync_callbacks: List[Callable] = []
    
    async def start(self):
        """Start offline manager"""
        logger.info(f"Starting offline manager for device {self.device_id}")
        
        # Start background tasks
        self.sync_task = asyncio.create_task(self._sync_loop())
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        self.mission_task = asyncio.create_task(self._mission_loop())
        
        logger.info("Offline manager started")
    
    async def stop(self):
        """Stop offline manager"""
        logger.info("Stopping offline manager")
        
        # Cancel background tasks
        if self.sync_task:
            self.sync_task.cancel()
        if self.cleanup_task:
            self.cleanup_task.cancel()
        if self.mission_task:
            self.mission_task.cancel()
        
        logger.info("Offline manager stopped")
    
    async def store_telemetry(self, telemetry_data: Dict[str, Any], priority: int = 1) -> str:
        """Store telemetry data for offline sync"""
        return await self.data_manager.store_message("telemetry", telemetry_data, priority)
    
    async def store_alert(self, alert_data: Dict[str, Any], priority: int = 5) -> str:
        """Store alert data for offline sync"""
        return await self.data_manager.store_message("alert", alert_data, priority)
    
    async def store_mission(self, mission: OfflineMission) -> bool:
        """Store mission for offline execution"""
        return await self.mission_manager.store_mission(mission)
    
    async def check_connection(self) -> bool:
        """Check if connection to Summit.OS is available"""
        if not self.summit_client:
            return False
        
        try:
            # Try to get system health
            await self.summit_client.get_system_health()
            return True
        except:
            return False
    
    async def _sync_loop(self):
        """Background sync loop"""
        while True:
            try:
                # Check connection
                is_connected = await self.check_connection()
                
                if is_connected and self.offline_mode != OfflineMode.ONLINE:
                    self.offline_mode = OfflineMode.ONLINE
                    await self._notify_connection_callbacks(True)
                    logger.info("Connection restored")
                
                elif not is_connected and self.offline_mode == OfflineMode.ONLINE:
                    self.offline_mode = OfflineMode.OFFLINE
                    await self._notify_connection_callbacks(False)
                    logger.warning("Connection lost, entering offline mode")
                
                # Sync pending messages if connected
                if is_connected:
                    await self._sync_pending_messages()
                
                await asyncio.sleep(self.data_manager.sync_interval)
                
            except Exception as e:
                logger.error(f"Sync loop error: {e}")
                await asyncio.sleep(60)
    
    async def _cleanup_loop(self):
        """Background cleanup loop"""
        while True:
            try:
                await self.data_manager.cleanup_old_messages()
                await asyncio.sleep(3600)  # 1 hour
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")
                await asyncio.sleep(3600)
    
    async def _mission_loop(self):
        """Background mission execution loop"""
        while True:
            try:
                if self.offline_mode == OfflineMode.OFFLINE:
                    # Execute pending missions in offline mode
                    pending_missions = await self.mission_manager.get_pending_missions()
                    
                    for mission in pending_missions:
                        logger.info(f"Executing offline mission: {mission.mission_id}")
                        await self.mission_manager.execute_mission(mission.mission_id)
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Mission loop error: {e}")
                await asyncio.sleep(30)
    
    async def _sync_pending_messages(self):
        """Sync pending messages to Summit.OS"""
        try:
            pending_messages = await self.data_manager.get_pending_messages(
                limit=self.data_manager.sync_batch_size
            )
            
            for message in pending_messages:
                try:
                    # Send message to Summit.OS
                    if message.message_type == "telemetry":
                        await self.summit_client.publish_telemetry(message.data)
                    elif message.message_type == "alert":
                        await self.summit_client.publish_alert(message.data)
                    
                    # Mark as synced
                    await self.data_manager.mark_message_synced(message.message_id)
                    
                except Exception as e:
                    logger.error(f"Failed to sync message {message.message_id}: {e}")
                    await self.data_manager.mark_message_failed(message.message_id)
            
            if pending_messages:
                await self._notify_sync_callbacks(len(pending_messages))
                
        except Exception as e:
            logger.error(f"Sync error: {e}")
    
    async def _execute_mission(self, mission: OfflineMission) -> bool:
        """Execute a mission (to be implemented by specific device)"""
        # This would be implemented by the specific device type
        # For example, drone missions, robot missions, etc.
        logger.info(f"Executing mission: {mission.mission_id}")
        
        # Simulate mission execution
        await asyncio.sleep(1)
        return True
    
    async def _notify_connection_callbacks(self, connected: bool):
        """Notify connection status callbacks"""
        for callback in self.connection_callbacks:
            try:
                await callback(connected)
            except Exception as e:
                logger.error(f"Connection callback error: {e}")
    
    async def _notify_sync_callbacks(self, message_count: int):
        """Notify sync status callbacks"""
        for callback in self.sync_callbacks:
            try:
                await callback(message_count)
            except Exception as e:
                logger.error(f"Sync callback error: {e}")
    
    def add_connection_callback(self, callback: Callable):
        """Add connection status callback"""
        self.connection_callbacks.append(callback)
    
    def add_sync_callback(self, callback: Callable):
        """Add sync status callback"""
        self.sync_callbacks.append(callback)
    
    async def get_status(self) -> Dict[str, Any]:
        """Get offline manager status"""
        storage_stats = await self.data_manager.get_storage_stats()
        
        return {
            'device_id': self.device_id,
            'offline_mode': self.offline_mode.value,
            'storage_stats': storage_stats,
            'active_missions': len(self.mission_manager.active_missions),
            'last_connection_check': self.last_connection_check
        }
