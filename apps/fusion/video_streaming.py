"""
Video Streaming Service for Summit.OS

Provides real-time video streaming capabilities for sentry towers, drones,
and other edge devices with RTSP, WebRTC, and video analytics support.
"""

import asyncio
import cv2
import numpy as np
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timezone
import logging
import json
import base64
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
import uvicorn
from pydantic import BaseModel
import threading
import queue
import time

logger = logging.getLogger(__name__)


class VideoStream(BaseModel):
    """Video stream configuration"""
    stream_id: str
    device_id: str
    stream_type: str  # "rtsp", "webrtc", "http"
    resolution: tuple = (1920, 1080)
    fps: int = 30
    quality: int = 80
    enabled: bool = True


class VideoAnalytics(BaseModel):
    """Video analytics result"""
    timestamp: float
    stream_id: str
    detections: List[Dict[str, Any]]
    confidence: float
    metadata: Dict[str, Any] = {}


class VideoStreamingService:
    """
    Real-time video streaming service for Summit.OS edge devices.
    
    Supports RTSP, WebRTC, and HTTP streaming with real-time analytics.
    """
    
    def __init__(self):
        self.active_streams: Dict[str, VideoStream] = {}
        self.stream_connections: Dict[str, List[WebSocket]] = {}
        self.analytics_callbacks: Dict[str, List[Callable]] = {}
        self.video_buffers: Dict[str, queue.Queue] = {}
        self.running = False
        
    async def start_stream(self, stream_config: VideoStream) -> bool:
        """Start a video stream"""
        try:
            self.active_streams[stream_config.stream_id] = stream_config
            self.stream_connections[stream_config.stream_id] = []
            self.video_buffers[stream_config.stream_id] = queue.Queue(maxsize=100)
            
            # Start stream processing thread
            if stream_config.stream_type == "rtsp":
                await self._start_rtsp_stream(stream_config)
            elif stream_config.stream_type == "webrtc":
                await self._start_webrtc_stream(stream_config)
            elif stream_config.stream_type == "http":
                await self._start_http_stream(stream_config)
            
            logger.info(f"Started video stream: {stream_config.stream_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start stream {stream_config.stream_id}: {e}")
            return False
    
    async def stop_stream(self, stream_id: str) -> bool:
        """Stop a video stream"""
        try:
            if stream_id in self.active_streams:
                del self.active_streams[stream_id]
                del self.stream_connections[stream_id]
                del self.video_buffers[stream_id]
                logger.info(f"Stopped video stream: {stream_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to stop stream {stream_id}: {e}")
            return False
    
    async def add_connection(self, stream_id: str, websocket: WebSocket):
        """Add a WebSocket connection to a stream"""
        if stream_id in self.stream_connections:
            self.stream_connections[stream_id].append(websocket)
            logger.info(f"Added connection to stream {stream_id}")
    
    async def remove_connection(self, stream_id: str, websocket: WebSocket):
        """Remove a WebSocket connection from a stream"""
        if stream_id in self.stream_connections:
            try:
                self.stream_connections[stream_id].remove(websocket)
                logger.info(f"Removed connection from stream {stream_id}")
            except ValueError:
                pass
    
    async def _start_rtsp_stream(self, stream_config: VideoStream):
        """Start RTSP stream processing"""
        def process_rtsp():
            cap = cv2.VideoCapture(stream_config.stream_id)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, stream_config.resolution[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, stream_config.resolution[1])
            cap.set(cv2.CAP_PROP_FPS, stream_config.fps)
            
            while stream_config.stream_id in self.active_streams:
                ret, frame = cap.read()
                if ret:
                    # Process frame for analytics
                    analytics = self._analyze_frame(frame, stream_config.stream_id)
                    
                    # Add to buffer
                    if stream_config.stream_id in self.video_buffers:
                        try:
                            self.video_buffers[stream_config.stream_id].put_nowait({
                                'frame': frame,
                                'timestamp': time.time(),
                                'analytics': analytics
                            })
                        except queue.Full:
                            # Remove oldest frame if buffer is full
                            try:
                                self.video_buffers[stream_config.stream_id].get_nowait()
                                self.video_buffers[stream_config.stream_id].put_nowait({
                                    'frame': frame,
                                    'timestamp': time.time(),
                                    'analytics': analytics
                                })
                            except queue.Empty:
                                pass
                
                time.sleep(1.0 / stream_config.fps)
            
            cap.release()
        
        # Start processing in separate thread
        thread = threading.Thread(target=process_rtsp, daemon=True)
        thread.start()
    
    async def _start_webrtc_stream(self, stream_config: VideoStream):
        """Start WebRTC stream processing"""
        # WebRTC implementation would go here
        # For now, use HTTP streaming as fallback
        await self._start_http_stream(stream_config)
    
    async def _start_http_stream(self, stream_config: VideoStream):
        """Start HTTP stream processing"""
        def process_http():
            # Simulate video stream for HTTP
            while stream_config.stream_id in self.active_streams:
                # Generate test frame
                frame = np.random.randint(0, 255, (stream_config.resolution[1], stream_config.resolution[0], 3), dtype=np.uint8)
                
                # Process frame for analytics
                analytics = self._analyze_frame(frame, stream_config.stream_id)
                
                # Add to buffer
                if stream_config.stream_id in self.video_buffers:
                    try:
                        self.video_buffers[stream_config.stream_id].put_nowait({
                            'frame': frame,
                            'timestamp': time.time(),
                            'analytics': analytics
                        })
                    except queue.Full:
                        pass
                
                time.sleep(1.0 / stream_config.fps)
        
        # Start processing in separate thread
        thread = threading.Thread(target=process_http, daemon=True)
        thread.start()
    
    def _analyze_frame(self, frame: np.ndarray, stream_id: str) -> VideoAnalytics:
        """Analyze video frame for detections"""
        # Basic frame analysis
        detections = []
        
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Simple motion detection
        if hasattr(self, '_prev_frame'):
            diff = cv2.absdiff(gray, self._prev_frame)
            motion = np.sum(diff) / (frame.shape[0] * frame.shape[1])
            
            if motion > 1000:  # Threshold for motion detection
                detections.append({
                    'type': 'motion',
                    'confidence': min(motion / 10000, 1.0),
                    'bbox': [0, 0, frame.shape[1], frame.shape[0]]
                })
        
        self._prev_frame = gray.copy()
        
        return VideoAnalytics(
            timestamp=time.time(),
            stream_id=stream_id,
            detections=detections,
            confidence=0.8,
            metadata={'frame_size': frame.shape}
        )
    
    async def get_frame(self, stream_id: str) -> Optional[Dict[str, Any]]:
        """Get latest frame from stream"""
        if stream_id in self.video_buffers:
            try:
                return self.video_buffers[stream_id].get_nowait()
            except queue.Empty:
                return None
        return None
    
    async def broadcast_frame(self, stream_id: str, frame_data: Dict[str, Any]):
        """Broadcast frame to all connected clients"""
        if stream_id in self.stream_connections:
            # Encode frame as JPEG
            frame = frame_data['frame']
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            frame_bytes = buffer.tobytes()
            frame_b64 = base64.b64encode(frame_bytes).decode('utf-8')
            
            message = {
                'type': 'video_frame',
                'stream_id': stream_id,
                'timestamp': frame_data['timestamp'],
                'frame': frame_b64,
                'analytics': frame_data['analytics'].dict() if 'analytics' in frame_data else None
            }
            
            # Send to all connected clients
            disconnected = []
            for websocket in self.stream_connections[stream_id]:
                try:
                    await websocket.send_text(json.dumps(message))
                except:
                    disconnected.append(websocket)
            
            # Remove disconnected clients
            for ws in disconnected:
                await self.remove_connection(stream_id, ws)
    
    def add_analytics_callback(self, stream_id: str, callback: Callable):
        """Add analytics callback for stream"""
        if stream_id not in self.analytics_callbacks:
            self.analytics_callbacks[stream_id] = []
        self.analytics_callbacks[stream_id].append(callback)
    
    async def process_analytics(self, stream_id: str, analytics: VideoAnalytics):
        """Process analytics results"""
        if stream_id in self.analytics_callbacks:
            for callback in self.analytics_callbacks[stream_id]:
                try:
                    await callback(analytics)
                except Exception as e:
                    logger.error(f"Analytics callback error: {e}")


# Global video streaming service
video_service = VideoStreamingService()


async def websocket_video_endpoint(websocket: WebSocket, stream_id: str):
    """WebSocket endpoint for video streaming"""
    await websocket.accept()
    await video_service.add_connection(stream_id, websocket)
    
    try:
        while True:
            # Get latest frame
            frame_data = await video_service.get_frame(stream_id)
            if frame_data:
                await video_service.broadcast_frame(stream_id, frame_data)
            
            await asyncio.sleep(1.0 / 30)  # 30 FPS
            
    except WebSocketDisconnect:
        await video_service.remove_connection(stream_id, websocket)
    except Exception as e:
        logger.error(f"Video WebSocket error: {e}")
        await video_service.remove_connection(stream_id, websocket)


def generate_frames(stream_id: str):
    """Generate video frames for HTTP streaming"""
    while True:
        frame_data = video_service.get_frame(stream_id)
        if frame_data:
            frame = frame_data['frame']
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(1.0 / 30)


async def start_video_streaming():
    """Start video streaming service"""
    video_service.running = True
    logger.info("Video streaming service started")


async def stop_video_streaming():
    """Stop video streaming service"""
    video_service.running = False
    logger.info("Video streaming service stopped")
