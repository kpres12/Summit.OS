"""
Summit.OS Python SDK - Main Client

Provides the main SummitClient for integrating with Summit.OS.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Union
import aiohttp
import requests
from .exceptions import SummitOSError, AuthenticationError, ConnectionError

logger = logging.getLogger(__name__)


class SummitClient:
    """
    Main client for integrating with Summit.OS.
    
    Provides high-level APIs for telemetry, alerts, missions, and intelligence.
    """
    
    def __init__(
        self,
        api_key: str,
        base_url: str = "http://localhost:8000",
        timeout: int = 30,
        retry_attempts: int = 3
    ):
        """
        Initialize Summit.OS client.
        
        Args:
            api_key: Summit.OS API key
            base_url: Base URL for Summit.OS API
            timeout: Request timeout in seconds
            retry_attempts: Number of retry attempts for failed requests
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.retry_attempts = retry_attempts
        
        # Initialize HTTP session
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json',
            'User-Agent': f'SummitOS-SDK-Python/{__version__}'
        })
        
        # Initialize async session
        self._async_session = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self._ensure_async_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._async_session:
            await self._async_session.close()
    
    async def _ensure_async_session(self):
        """Ensure async session is initialized."""
        if not self._async_session:
            self._async_session = aiohttp.ClientSession(
                headers={
                    'Authorization': f'Bearer {self.api_key}',
                    'Content-Type': 'application/json',
                    'User-Agent': f'SummitOS-SDK-Python/{__version__}'
                },
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )
    
    # Telemetry Methods
    
    async def publish_telemetry(self, telemetry: Dict[str, Any]) -> Dict[str, Any]:
        """
        Publish telemetry data to Summit.OS.
        
        Args:
            telemetry: Telemetry data dictionary
            
        Returns:
            Response from Summit.OS
            
        Raises:
            SummitOSError: If request fails
        """
        await self._ensure_async_session()
        
        # Add timestamp if not present
        if 'timestamp' not in telemetry:
            telemetry['timestamp'] = datetime.now(timezone.utc).isoformat()
        
        url = f"{self.base_url}/api/v1/telemetry"
        
        try:
            async with self._async_session.post(url, json=telemetry) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    raise SummitOSError(f"Failed to publish telemetry: {response.status} - {error_text}")
        except aiohttp.ClientError as e:
            raise ConnectionError(f"Failed to connect to Summit.OS: {e}")
    
    async def get_telemetry(
        self, 
        device_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get telemetry data from Summit.OS.
        
        Args:
            device_id: Filter by device ID
            start_time: Start time for data range
            end_time: End time for data range
            limit: Maximum number of records to return
            
        Returns:
            List of telemetry records
        """
        await self._ensure_async_session()
        
        params = {'limit': limit}
        if device_id:
            params['device_id'] = device_id
        if start_time:
            params['start_time'] = start_time.isoformat()
        if end_time:
            params['end_time'] = end_time.isoformat()
        
        url = f"{self.base_url}/api/v1/telemetry"
        
        try:
            async with self._async_session.get(url, params=params) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    raise SummitOSError(f"Failed to get telemetry: {response.status} - {error_text}")
        except aiohttp.ClientError as e:
            raise ConnectionError(f"Failed to connect to Summit.OS: {e}")
    
    # Alert Methods
    
    async def publish_alert(self, alert: Dict[str, Any]) -> Dict[str, Any]:
        """
        Publish alert to Summit.OS.
        
        Args:
            alert: Alert data dictionary
            
        Returns:
            Response from Summit.OS
        """
        await self._ensure_async_session()
        
        # Add timestamp if not present
        if 'timestamp' not in alert:
            alert['timestamp'] = datetime.now(timezone.utc).isoformat()
        
        url = f"{self.base_url}/api/v1/alerts"
        
        try:
            async with self._async_session.post(url, json=alert) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    raise SummitOSError(f"Failed to publish alert: {response.status} - {error_text}")
        except aiohttp.ClientError as e:
            raise ConnectionError(f"Failed to connect to Summit.OS: {e}")
    
    async def get_alerts(
        self,
        severity: Optional[str] = None,
        category: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get alerts from Summit.OS.
        
        Args:
            severity: Filter by severity level
            category: Filter by category
            status: Filter by status
            limit: Maximum number of records to return
            
        Returns:
            List of alert records
        """
        await self._ensure_async_session()
        
        params = {'limit': limit}
        if severity:
            params['severity'] = severity
        if category:
            params['category'] = category
        if status:
            params['status'] = status
        
        url = f"{self.base_url}/api/v1/alerts"
        
        try:
            async with self._async_session.get(url, params=params) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    raise SummitOSError(f"Failed to get alerts: {response.status} - {error_text}")
        except aiohttp.ClientError as e:
            raise ConnectionError(f"Failed to connect to Summit.OS: {e}")
    
    async def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> Dict[str, Any]:
        """
        Acknowledge an alert.
        
        Args:
            alert_id: ID of alert to acknowledge
            acknowledged_by: User or system acknowledging the alert
            
        Returns:
            Response from Summit.OS
        """
        await self._ensure_async_session()
        
        url = f"{self.base_url}/api/v1/alerts/{alert_id}/acknowledge"
        data = {
            'acknowledged_by': acknowledged_by,
            'acknowledged_at': datetime.now(timezone.utc).isoformat()
        }
        
        try:
            async with self._async_session.post(url, json=data) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    raise SummitOSError(f"Failed to acknowledge alert: {response.status} - {error_text}")
        except aiohttp.ClientError as e:
            raise ConnectionError(f"Failed to connect to Summit.OS: {e}")
    
    # Mission Methods
    
    async def create_mission(self, mission: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new mission.
        
        Args:
            mission: Mission data dictionary
            
        Returns:
            Created mission data
        """
        await self._ensure_async_session()
        
        url = f"{self.base_url}/api/v1/missions"
        
        try:
            async with self._async_session.post(url, json=mission) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    raise SummitOSError(f"Failed to create mission: {response.status} - {error_text}")
        except aiohttp.ClientError as e:
            raise ConnectionError(f"Failed to connect to Summit.OS: {e}")
    
    async def get_mission(self, mission_id: str) -> Dict[str, Any]:
        """
        Get mission by ID.
        
        Args:
            mission_id: Mission ID
            
        Returns:
            Mission data
        """
        await self._ensure_async_session()
        
        url = f"{self.base_url}/api/v1/missions/{mission_id}"
        
        try:
            async with self._async_session.get(url) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    raise SummitOSError(f"Failed to get mission: {response.status} - {error_text}")
        except aiohttp.ClientError as e:
            raise ConnectionError(f"Failed to connect to Summit.OS: {e}")
    
    async def update_mission(self, mission_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update mission.
        
        Args:
            mission_id: Mission ID
            updates: Updates to apply
            
        Returns:
            Updated mission data
        """
        await self._ensure_async_session()
        
        url = f"{self.base_url}/api/v1/missions/{mission_id}"
        
        try:
            async with self._async_session.put(url, json=updates) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    raise SummitOSError(f"Failed to update mission: {response.status} - {error_text}")
        except aiohttp.ClientError as e:
            raise ConnectionError(f"Failed to connect to Summit.OS: {e}")
    
    # Intelligence Methods
    
    async def get_intelligence_alerts(
        self,
        severity: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get intelligence alerts.
        
        Args:
            severity: Filter by severity
            limit: Maximum number of records
            
        Returns:
            List of intelligence alerts
        """
        await self._ensure_async_session()
        
        params = {'limit': limit}
        if severity:
            params['severity'] = severity
        
        url = f"{self.base_url}/api/v1/intelligence/alerts"
        
        try:
            async with self._async_session.get(url, params=params) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    raise SummitOSError(f"Failed to get intelligence alerts: {response.status} - {error_text}")
        except aiohttp.ClientError as e:
            raise ConnectionError(f"Failed to connect to Summit.OS: {e}")
    
    async def get_risk_assessment(
        self,
        location: Dict[str, float],
        radius: float = 5000.0
    ) -> Dict[str, Any]:
        """
        Get risk assessment for a location.
        
        Args:
            location: Location dictionary with lat/lon
            radius: Assessment radius in meters
            
        Returns:
            Risk assessment data
        """
        await self._ensure_async_session()
        
        params = {
            'latitude': location['latitude'],
            'longitude': location['longitude'],
            'radius': radius
        }
        
        url = f"{self.base_url}/api/v1/intelligence/risk"
        
        try:
            async with self._async_session.get(url, params=params) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    raise SummitOSError(f"Failed to get risk assessment: {response.status} - {error_text}")
        except aiohttp.ClientError as e:
            raise ConnectionError(f"Failed to connect to Summit.OS: {e}")
    
    # World Model Methods
    
    async def get_world_model(self) -> Dict[str, Any]:
        """
        Get current world model state.
        
        Returns:
            World model data
        """
        await self._ensure_async_session()
        
        url = f"{self.base_url}/api/v1/world-model"
        
        try:
            async with self._async_session.get(url) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    raise SummitOSError(f"Failed to get world model: {response.status} - {error_text}")
        except aiohttp.ClientError as e:
            raise ConnectionError(f"Failed to connect to Summit.OS: {e}")
    
    # Health and Status Methods
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Check Summit.OS health.
        
        Returns:
            Health status
        """
        await self._ensure_async_session()
        
        url = f"{self.base_url}/health"
        
        try:
            async with self._async_session.get(url) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    raise SummitOSError(f"Health check failed: {response.status} - {error_text}")
        except aiohttp.ClientError as e:
            raise ConnectionError(f"Failed to connect to Summit.OS: {e}")
    
    # Utility Methods
    
    def close(self):
        """Close the client and cleanup resources."""
        if hasattr(self, 'session'):
            self.session.close()
    
    async def aclose(self):
        """Async close the client and cleanup resources."""
        if self._async_session:
            await self._async_session.close()
