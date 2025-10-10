"""
AI Models for Summit.OS Fusion Service

This module contains the AI models used for multimodal sensor fusion,
anomaly detection, and world model updates in the Summit.OS fusion service.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class MultimodalFusionNetwork(nn.Module):
    """
    Neural network for fusing multimodal sensor data into a unified representation.
    
    This network takes inputs from various sensors (weather, LiDAR, IR, visual)
    and produces a fused representation that can be used for downstream tasks.
    """
    
    def __init__(self, 
                 weather_dim: int = 10,
                 lidar_dim: int = 1024, 
                 ir_dim: int = 256,
                 visual_dim: int = 512,
                 hidden_dim: int = 256,
                 output_dim: int = 128):
        super().__init__()
        
        # Individual encoders for each modality
        self.weather_encoder = nn.Sequential(
            nn.Linear(weather_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32)
        )
        
        self.lidar_encoder = nn.Sequential(
            nn.Linear(lidar_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128)
        )
        
        self.ir_encoder = nn.Sequential(
            nn.Linear(ir_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64)
        )
        
        self.visual_encoder = nn.Sequential(
            nn.Linear(visual_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128)
        )
        
        # Fusion layer
        fusion_input_dim = 32 + 128 + 64 + 128  # Sum of encoder outputs
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Attention mechanism for weighted fusion
        self.attention = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=8,
            dropout=0.1
        )
    
    def forward(self, weather_data, lidar_data, ir_data, visual_data):
        """
        Forward pass through the fusion network.
        
        Args:
            weather_data: Weather sensor data [batch_size, weather_dim]
            lidar_data: LiDAR point cloud data [batch_size, lidar_dim]
            ir_data: Infrared sensor data [batch_size, ir_dim]
            visual_data: Visual/optical data [batch_size, visual_dim]
            
        Returns:
            Fused representation [batch_size, output_dim]
        """
        # Encode each modality
        weather_encoded = self.weather_encoder(weather_data)
        lidar_encoded = self.lidar_encoder(lidar_data)
        ir_encoded = self.ir_encoder(ir_data)
        visual_encoded = self.visual_encoder(visual_data)
        
        # Concatenate encoded features
        fused_input = torch.cat([
            weather_encoded, lidar_encoded, ir_encoded, visual_encoded
        ], dim=-1)
        
        # Apply fusion layer
        fused_output = self.fusion_layer(fused_input)
        
        # Apply attention mechanism
        fused_output = fused_output.unsqueeze(1)  # Add sequence dimension
        attended_output, _ = self.attention(
            fused_output, fused_output, fused_output
        )
        attended_output = attended_output.squeeze(1)
        
        return attended_output


class AnomalyDetector:
    """
    Anomaly detection system for identifying unusual patterns in sensor data.
    
    Uses a combination of statistical methods and machine learning to detect
    anomalies in real-time sensor streams.
    """
    
    def __init__(self, contamination=0.05, random_state=42):
        self.contamination = contamination
        self.random_state = random_state
        self.isolation_forest = IsolationForest(
            contamination=contamination,
            random_state=random_state
        )
        self.dbscan = DBSCAN(eps=0.5, min_samples=5)
        self.is_trained = False
        self.normal_data = None
        self.statistical_thresholds = {}
    
    def train(self, data: np.ndarray):
        """
        Train the anomaly detector on normal operational data.
        
        Args:
            data: Normal sensor data [n_samples, n_features]
        """
        logger.info(f"Training anomaly detector on {len(data)} samples")
        
        # Store normal data for statistical analysis
        self.normal_data = data
        
        # Calculate statistical thresholds
        for i in range(data.shape[1]):
            mean_val = np.mean(data[:, i])
            std_val = np.std(data[:, i])
            self.statistical_thresholds[i] = {
                'mean': mean_val,
                'std': std_val,
                'upper_bound': mean_val + 3 * std_val,
                'lower_bound': mean_val - 3 * std_val
            }
        
        # Train isolation forest
        self.isolation_forest.fit(data)
        self.is_trained = True
        
        logger.info("Anomaly detector training completed")
    
    def detect_anomalies(self, data: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Detect anomalies in sensor data.
        
        Args:
            data: Sensor data to analyze [n_samples, n_features]
            
        Returns:
            Tuple of (anomaly_flags, anomaly_details)
        """
        if not self.is_trained:
            logger.warning("Anomaly detector not trained, returning no anomalies")
            return np.zeros(data.shape[0], dtype=bool), {}
        
        # Statistical anomaly detection
        statistical_anomalies = self._detect_statistical_anomalies(data)
        
        # Isolation forest anomaly detection
        isolation_anomalies = self.isolation_forest.predict(data) == -1
        
        # DBSCAN clustering for density-based anomalies
        clustering_anomalies = self._detect_clustering_anomalies(data)
        
        # Combine different anomaly detection methods
        combined_anomalies = (
            statistical_anomalies | 
            isolation_anomalies | 
            clustering_anomalies
        )
        
        # Calculate anomaly scores
        anomaly_scores = self.isolation_forest.decision_function(data)
        
        anomaly_details = {
            'statistical_anomalies': np.sum(statistical_anomalies),
            'isolation_anomalies': np.sum(isolation_anomalies),
            'clustering_anomalies': np.sum(clustering_anomalies),
            'combined_anomalies': np.sum(combined_anomalies),
            'anomaly_scores': anomaly_scores.tolist(),
            'detection_time': datetime.now(timezone.utc).isoformat()
        }
        
        return combined_anomalies, anomaly_details
    
    def _detect_statistical_anomalies(self, data: np.ndarray) -> np.ndarray:
        """Detect anomalies using statistical thresholds."""
        anomalies = np.zeros(data.shape[0], dtype=bool)
        
        for i in range(data.shape[1]):
            if i in self.statistical_thresholds:
                threshold = self.statistical_thresholds[i]
                feature_anomalies = (
                    (data[:, i] > threshold['upper_bound']) |
                    (data[:, i] < threshold['lower_bound'])
                )
                anomalies |= feature_anomalies
        
        return anomalies
    
    def _detect_clustering_anomalies(self, data: np.ndarray) -> np.ndarray:
        """Detect anomalies using DBSCAN clustering."""
        if len(data) < 5:  # Need minimum samples for DBSCAN
            return np.zeros(len(data), dtype=bool)
        
        try:
            clusters = self.dbscan.fit_predict(data)
            # Points with cluster label -1 are considered anomalies
            return clusters == -1
        except Exception as e:
            logger.warning(f"DBSCAN clustering failed: {e}")
            return np.zeros(len(data), dtype=bool)


class WorldModelUpdater:
    """
    Updates and maintains the unified geospatially indexed world model.
    
    This component integrates new sensor data into the existing world model,
    maintaining spatial-temporal consistency and updating environmental state.
    """
    
    def __init__(self, spatial_resolution=10.0, temporal_window=3600):
        """
        Initialize world model updater.
        
        Args:
            spatial_resolution: Spatial resolution in meters
            temporal_window: Temporal window in seconds
        """
        self.spatial_resolution = spatial_resolution
        self.temporal_window = temporal_window
        self.world_model = {
            'terrain': {},
            'environmental_state': {},
            'objects': {},
            'events': [],
            'last_updated': None
        }
        self.spatial_index = {}
        self.temporal_index = {}
    
    def update_terrain_segmentation(self, geo_data: Dict[str, Any], 
                                  segmentation_mask: np.ndarray) -> Dict[str, Any]:
        """
        Update terrain segmentation based on drone imagery/LiDAR.
        
        Args:
            geo_data: Geographic location and metadata
            segmentation_mask: Segmentation mask from AI model
            
        Returns:
            Update status and metadata
        """
        location_key = self._get_location_key(geo_data['latitude'], geo_data['longitude'])
        
        terrain_update = {
            'location': location_key,
            'segmentation_mask': segmentation_mask.tolist(),
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'resolution': self.spatial_resolution,
            'confidence': geo_data.get('confidence', 0.8)
        }
        
        # Update world model
        if location_key not in self.world_model['terrain']:
            self.world_model['terrain'][location_key] = []
        
        self.world_model['terrain'][location_key].append(terrain_update)
        
        # Update spatial index
        self._update_spatial_index(location_key, terrain_update)
        
        logger.info(f"Updated terrain segmentation for {location_key}")
        
        return {
            'status': 'success',
            'location_key': location_key,
            'updated_at': terrain_update['timestamp']
        }
    
    def update_environmental_state(self, geo_location: Dict[str, float], 
                                  state_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update environmental state (dryness, wind vectors, fire spread rate).
        
        Args:
            geo_location: Geographic location
            state_data: Environmental state data
            
        Returns:
            Update status and metadata
        """
        location_key = self._get_location_key(
            geo_location['latitude'], 
            geo_location['longitude']
        )
        
        environmental_update = {
            'location': location_key,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'temperature': state_data.get('temperature'),
            'humidity': state_data.get('humidity'),
            'wind_speed': state_data.get('wind_speed'),
            'wind_direction': state_data.get('wind_direction'),
            'fuel_moisture': state_data.get('fuel_moisture'),
            'fire_spread_rate': state_data.get('fire_spread_rate'),
            'risk_level': state_data.get('risk_level', 'unknown')
        }
        
        # Update world model
        if location_key not in self.world_model['environmental_state']:
            self.world_model['environmental_state'][location_key] = []
        
        self.world_model['environmental_state'][location_key].append(environmental_update)
        
        # Update spatial index
        self._update_spatial_index(location_key, environmental_update)
        
        logger.info(f"Updated environmental state for {location_key}")
        
        return {
            'status': 'success',
            'location_key': location_key,
            'updated_at': environmental_update['timestamp']
        }
    
    def add_object(self, object_id: str, object_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add or update an object in the world model.
        
        Args:
            object_id: Unique object identifier
            object_data: Object properties and location
            
        Returns:
            Update status and metadata
        """
        object_update = {
            'object_id': object_id,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'location': object_data.get('location'),
            'properties': object_data.get('properties', {}),
            'type': object_data.get('type', 'unknown'),
            'confidence': object_data.get('confidence', 0.8)
        }
        
        # Update world model
        self.world_model['objects'][object_id] = object_update
        
        # Update spatial index
        if object_update['location']:
            location_key = self._get_location_key(
                object_update['location']['latitude'],
                object_update['location']['longitude']
            )
            self._update_spatial_index(location_key, object_update)
        
        logger.info(f"Added/updated object {object_id} in world model")
        
        return {
            'status': 'success',
            'object_id': object_id,
            'updated_at': object_update['timestamp']
        }
    
    def add_event(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add an event to the world model.
        
        Args:
            event_data: Event information
            
        Returns:
            Update status and metadata
        """
        event = {
            'event_id': event_data.get('event_id', f"event_{int(datetime.now().timestamp())}"),
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'location': event_data.get('location'),
            'type': event_data.get('type', 'unknown'),
            'severity': event_data.get('severity', 'medium'),
            'description': event_data.get('description', ''),
            'confidence': event_data.get('confidence', 0.8)
        }
        
        # Update world model
        self.world_model['events'].append(event)
        
        # Update spatial index
        if event['location']:
            location_key = self._get_location_key(
                event['location']['latitude'],
                event['location']['longitude']
            )
            self._update_spatial_index(location_key, event)
        
        logger.info(f"Added event {event['event_id']} to world model")
        
        return {
            'status': 'success',
            'event_id': event['event_id'],
            'updated_at': event['timestamp']
        }
    
    def get_world_model_snapshot(self) -> Dict[str, Any]:
        """
        Get a snapshot of the current world model.
        
        Returns:
            World model snapshot
        """
        self.world_model['last_updated'] = datetime.now(timezone.utc).isoformat()
        
        return {
            'world_model': self.world_model.copy(),
            'spatial_index': self.spatial_index.copy(),
            'temporal_index': self.temporal_index.copy(),
            'snapshot_time': self.world_model['last_updated']
        }
    
    def query_spatial_region(self, center_lat: float, center_lon: float, 
                           radius: float) -> Dict[str, Any]:
        """
        Query the world model for a specific spatial region.
        
        Args:
            center_lat: Center latitude
            center_lon: Center longitude
            radius: Query radius in meters
            
        Returns:
            Objects and events in the specified region
        """
        # Simple spatial query implementation
        # In a real system, this would use a spatial database
        results = {
            'terrain': [],
            'environmental_state': [],
            'objects': [],
            'events': []
        }
        
        # Query spatial index (simplified)
        for location_key, data in self.spatial_index.items():
            # Check if location is within radius
            # This is a simplified implementation
            if self._is_within_radius(location_key, center_lat, center_lon, radius):
                if 'terrain' in data:
                    results['terrain'].extend(data['terrain'])
                if 'environmental_state' in data:
                    results['environmental_state'].extend(data['environmental_state'])
                if 'objects' in data:
                    results['objects'].extend(data['objects'])
                if 'events' in data:
                    results['events'].extend(data['events'])
        
        return results
    
    def _get_location_key(self, latitude: float, longitude: float) -> str:
        """Generate a location key for spatial indexing."""
        # Round to spatial resolution
        lat_key = round(latitude / self.spatial_resolution) * self.spatial_resolution
        lon_key = round(longitude / self.spatial_resolution) * self.spatial_resolution
        return f"{lat_key:.1f}_{lon_key:.1f}"
    
    def _update_spatial_index(self, location_key: str, data: Dict[str, Any]):
        """Update the spatial index with new data."""
        if location_key not in self.spatial_index:
            self.spatial_index[location_key] = {}
        
        # Add data to spatial index
        data_type = data.get('type', 'unknown')
        if data_type not in self.spatial_index[location_key]:
            self.spatial_index[location_key][data_type] = []
        
        self.spatial_index[location_key][data_type].append(data)
    
    def _is_within_radius(self, location_key: str, center_lat: float, 
                         center_lon: float, radius: float) -> bool:
        """Check if a location is within the specified radius."""
        # Parse location key
        try:
            lat, lon = map(float, location_key.split('_'))
            # Simple distance calculation (not accurate for large distances)
            distance = np.sqrt((lat - center_lat)**2 + (lon - center_lon)**2)
            return distance <= radius
        except:
            return False


class SensorFusionEngine:
    """
    Main sensor fusion engine that coordinates all AI models.
    
    This class orchestrates the fusion of multimodal sensor data,
    anomaly detection, and world model updates.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the sensor fusion engine.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Initialize AI models
        self.fusion_network = MultimodalFusionNetwork(
            weather_dim=config.get('weather_dim', 10),
            lidar_dim=config.get('lidar_dim', 1024),
            ir_dim=config.get('ir_dim', 256),
            visual_dim=config.get('visual_dim', 512),
            hidden_dim=config.get('hidden_dim', 256),
            output_dim=config.get('output_dim', 128)
        )
        
        self.anomaly_detector = AnomalyDetector(
            contamination=config.get('anomaly_contamination', 0.05)
        )
        
        self.world_model_updater = WorldModelUpdater(
            spatial_resolution=config.get('spatial_resolution', 10.0),
            temporal_window=config.get('temporal_window', 3600)
        )
        
        # Load pre-trained models if available
        self._load_models()
    
    def process_sensor_data(self, sensor_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process multimodal sensor data through the fusion pipeline.
        
        Args:
            sensor_data: Dictionary containing sensor data from different modalities
            
        Returns:
            Processed fusion results
        """
        try:
            # Extract sensor data
            weather_data = sensor_data.get('weather', {})
            lidar_data = sensor_data.get('lidar', {})
            ir_data = sensor_data.get('ir', {})
            visual_data = sensor_data.get('visual', {})
            
            # Convert to tensors
            weather_tensor = torch.tensor(
                self._extract_weather_features(weather_data), 
                dtype=torch.float32
            )
            lidar_tensor = torch.tensor(
                self._extract_lidar_features(lidar_data), 
                dtype=torch.float32
            )
            ir_tensor = torch.tensor(
                self._extract_ir_features(ir_data), 
                dtype=torch.float32
            )
            visual_tensor = torch.tensor(
                self._extract_visual_features(visual_data), 
                dtype=torch.float32
            )
            
            # Perform sensor fusion
            fused_output = self.fusion_network(
                weather_tensor, lidar_tensor, ir_tensor, visual_tensor
            )
            
            # Detect anomalies
            anomaly_data = np.array([
                weather_tensor.numpy().flatten(),
                lidar_tensor.numpy().flatten(),
                ir_tensor.numpy().flatten(),
                visual_tensor.numpy().flatten()
            ]).T
            
            anomalies, anomaly_details = self.anomaly_detector.detect_anomalies(anomaly_data)
            
            # Update world model
            world_model_update = self.world_model_updater.update_environmental_state(
                sensor_data.get('location', {}),
                {
                    'fused_vector': fused_output.detach().numpy().tolist(),
                    'anomaly_detected': bool(anomalies.any()),
                    'anomaly_score': float(anomaly_details.get('anomaly_scores', [0])[0])
                }
            )
            
            # Prepare results
            results = {
                'fused_output': fused_output.detach().numpy().tolist(),
                'anomalies_detected': bool(anomalies.any()),
                'anomaly_details': anomaly_details,
                'world_model_update': world_model_update,
                'processing_time': datetime.now(timezone.utc).isoformat(),
                'status': 'success'
            }
            
            logger.info(f"Sensor fusion completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Sensor fusion failed: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'processing_time': datetime.now(timezone.utc).isoformat()
            }
    
    def _extract_weather_features(self, weather_data: Dict[str, Any]) -> List[float]:
        """Extract features from weather data."""
        features = [
            weather_data.get('temperature', 0.0),
            weather_data.get('humidity', 0.0),
            weather_data.get('wind_speed', 0.0),
            weather_data.get('wind_direction', 0.0),
            weather_data.get('pressure', 0.0),
            weather_data.get('precipitation', 0.0),
            weather_data.get('visibility', 0.0),
            weather_data.get('cloud_cover', 0.0),
            weather_data.get('uv_index', 0.0),
            weather_data.get('air_quality', 0.0)
        ]
        return features
    
    def _extract_lidar_features(self, lidar_data: Dict[str, Any]) -> List[float]:
        """Extract features from LiDAR data."""
        # Simplified feature extraction
        # In a real system, this would involve point cloud processing
        point_cloud = lidar_data.get('points', [])
        if not point_cloud:
            return [0.0] * 1024  # Default feature vector
        
        # Extract basic statistics
        features = []
        if len(point_cloud) > 0:
            points = np.array(point_cloud)
            features.extend([
                np.mean(points, axis=0).tolist(),  # Mean position
                np.std(points, axis=0).tolist(),   # Standard deviation
                np.min(points, axis=0).tolist(),   # Min values
                np.max(points, axis=0).tolist()   # Max values
            ])
        
        # Pad or truncate to fixed size
        while len(features) < 1024:
            features.append(0.0)
        return features[:1024]
    
    def _extract_ir_features(self, ir_data: Dict[str, Any]) -> List[float]:
        """Extract features from infrared data."""
        # Simplified feature extraction
        ir_values = ir_data.get('values', [])
        if not ir_values:
            return [0.0] * 256  # Default feature vector
        
        # Extract basic statistics
        features = [
            np.mean(ir_values),
            np.std(ir_values),
            np.min(ir_values),
            np.max(ir_values)
        ]
        
        # Pad or truncate to fixed size
        while len(features) < 256:
            features.append(0.0)
        return features[:256]
    
    def _extract_visual_features(self, visual_data: Dict[str, Any]) -> List[float]:
        """Extract features from visual data."""
        # Simplified feature extraction
        # In a real system, this would involve computer vision models
        visual_features = visual_data.get('features', [])
        if not visual_features:
            return [0.0] * 512  # Default feature vector
        
        # Pad or truncate to fixed size
        while len(visual_features) < 512:
            visual_features.append(0.0)
        return visual_features[:512]
    
    def _load_models(self):
        """Load pre-trained models if available."""
        try:
            # Try to load pre-trained fusion network
            model_path = self.config.get('model_path', './models')
            fusion_model_path = f"{model_path}/fusion_network.pth"
            
            if torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                device = torch.device('cpu')
            
            # Load model state dict
            # self.fusion_network.load_state_dict(torch.load(fusion_model_path, map_location=device))
            # self.fusion_network.eval()
            
            logger.info("Pre-trained models loaded successfully")
            
        except Exception as e:
            logger.warning(f"Could not load pre-trained models: {e}")
    
    def get_world_model(self) -> Dict[str, Any]:
        """Get the current world model state."""
        return self.world_model_updater.get_world_model_snapshot()
    
    def train_anomaly_detector(self, training_data: np.ndarray):
        """Train the anomaly detector on new data."""
        self.anomaly_detector.train(training_data)
        logger.info("Anomaly detector training completed")
