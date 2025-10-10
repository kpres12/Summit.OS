"""
Summit.OS AI Models for Sensor Fusion

Implements the AI models for multimodal sensor fusion, anomaly detection,
object detection, and environmental state estimation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import cv2
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
import onnxruntime as ort
from datetime import datetime, timezone


class MultimodalFusionNetwork(nn.Module):
    """
    Neural network for fusing multiple sensor modalities into unified world model.
    
    Inputs: weather, LiDAR, IR, visual, acoustic data
    Outputs: unified spatial context with confidence scores
    """
    
    def __init__(self, input_dims: Dict[str, int], hidden_dim: int = 512):
        super().__init__()
        
        # Encoder for each modality
        self.weather_encoder = nn.Sequential(
            nn.Linear(input_dims['weather'], 64),
            nn.ReLU(),
            nn.Linear(64, 128)
        )
        
        self.lidar_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128)
        )
        
        self.thermal_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128)
        )
        
        self.visual_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256)
        )
        
        # Fusion network
        total_dim = 128 + 128 + 128 + 256  # weather + lidar + thermal + visual
        self.fusion_network = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 256)
        )
        
        # Output heads for different tasks
        self.fire_detection_head = nn.Linear(256, 2)  # fire/no-fire
        self.smoke_detection_head = nn.Linear(256, 2)  # smoke/no-smoke
        self.terrain_classification_head = nn.Linear(256, 5)  # brush/road/water/rock/unknown
        self.anomaly_detection_head = nn.Linear(256, 1)  # anomaly score
        
    def forward(self, weather: torch.Tensor, lidar: torch.Tensor, 
                thermal: torch.Tensor, visual: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through multimodal fusion network."""
        
        # Encode each modality
        weather_features = self.weather_encoder(weather)
        lidar_features = self.lidar_encoder(lidar)
        thermal_features = self.thermal_encoder(thermal)
        visual_features = self.visual_encoder(visual)
        
        # Fuse all modalities
        fused_features = torch.cat([
            weather_features, lidar_features, 
            thermal_features, visual_features
        ], dim=1)
        
        # Pass through fusion network
        fused_output = self.fusion_network(fused_features)
        
        # Generate predictions for each task
        fire_pred = self.fire_detection_head(fused_output)
        smoke_pred = self.smoke_detection_head(fused_output)
        terrain_pred = self.terrain_classification_head(fused_output)
        anomaly_score = self.anomaly_detection_head(fused_output)
        
        return {
            'fire_detection': F.softmax(fire_pred, dim=1),
            'smoke_detection': F.softmax(smoke_pred, dim=1),
            'terrain_classification': F.softmax(terrain_pred, dim=1),
            'anomaly_score': torch.sigmoid(anomaly_score)
        }


class FireSpreadPredictor:
    """
    Physics + ML hybrid model for fire spread prediction.
    Combines Rothermel fire behavior model with ML calibration.
    """
    
    def __init__(self):
        self.rothermel_params = {
            'fuel_moisture': 0.05,  # 5% moisture content
            'wind_speed': 0.0,     # m/s
            'slope': 0.0,          # degrees
            'fuel_load': 2.0       # tons/hectare
        }
        
        # ML calibration factors
        self.ml_calibration = {
            'temperature_factor': 1.0,
            'humidity_factor': 1.0,
            'terrain_factor': 1.0,
            'vegetation_factor': 1.0
        }
    
    def predict_spread(self, fire_location: Tuple[float, float], 
                      weather_data: Dict[str, float],
                      terrain_data: Dict[str, float]) -> Dict[str, Any]:
        """Predict fire spread using physics + ML hybrid model."""
        
        # Base Rothermel calculation
        base_rate = self._calculate_rothermel_rate(
            weather_data['wind_speed'],
            terrain_data['slope'],
            self.rothermel_params['fuel_moisture']
        )
        
        # ML calibration factors
        temp_factor = self._get_temperature_factor(weather_data['temperature'])
        humidity_factor = self._get_humidity_factor(weather_data['humidity'])
        terrain_factor = self._get_terrain_factor(terrain_data)
        
        # Apply ML corrections
        calibrated_rate = base_rate * temp_factor * humidity_factor * terrain_factor
        
        # Predict spread pattern
        spread_pattern = self._calculate_spread_pattern(
            fire_location, calibrated_rate, weather_data['wind_direction']
        )
        
        return {
            'spread_rate': calibrated_rate,
            'spread_pattern': spread_pattern,
            'confidence': self._calculate_confidence(weather_data, terrain_data),
            'time_to_containment': self._estimate_containment_time(calibrated_rate)
        }
    
    def _calculate_rothermel_rate(self, wind_speed: float, slope: float, 
                                 fuel_moisture: float) -> float:
        """Calculate base fire spread rate using Rothermel model."""
        # Simplified Rothermel calculation
        wind_factor = 1.0 + (wind_speed * 0.1)
        slope_factor = 1.0 + (slope * 0.01)
        moisture_factor = max(0.1, 1.0 - fuel_moisture * 10)
        
        return 0.1 * wind_factor * slope_factor * moisture_factor
    
    def _get_temperature_factor(self, temperature: float) -> float:
        """ML-based temperature correction factor."""
        # Higher temperatures increase fire spread
        return 1.0 + (temperature - 25.0) * 0.01
    
    def _get_humidity_factor(self, humidity: float) -> float:
        """ML-based humidity correction factor."""
        # Higher humidity decreases fire spread
        return 1.0 - (humidity - 50.0) * 0.005
    
    def _get_terrain_factor(self, terrain_data: Dict[str, float]) -> float:
        """ML-based terrain correction factor."""
        # Complex terrain increases fire spread unpredictability
        roughness = terrain_data.get('roughness', 0.0)
        return 1.0 + roughness * 0.1


class AnomalyDetector:
    """
    Anomaly detection system for identifying unusual patterns in sensor data.
    Uses isolation forest and custom anomaly detection algorithms.
    """
    
    def __init__(self):
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.anomaly_threshold = 0.8
        self.historical_data = []
        
    def detect_anomalies(self, sensor_data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect anomalies in sensor data."""
        
        # Extract features for anomaly detection
        features = self._extract_features(sensor_data)
        
        # Check for statistical anomalies
        statistical_anomalies = self._detect_statistical_anomalies(features)
        
        # Check for pattern anomalies
        pattern_anomalies = self._detect_pattern_anomalies(features)
        
        # Check for temporal anomalies
        temporal_anomalies = self._detect_temporal_anomalies(features)
        
        # Combine all anomaly types
        total_anomaly_score = max(
            statistical_anomalies['score'],
            pattern_anomalies['score'],
            temporal_anomalies['score']
        )
        
        return {
            'anomaly_detected': total_anomaly_score > self.anomaly_threshold,
            'anomaly_score': total_anomaly_score,
            'anomaly_type': self._classify_anomaly_type(
                statistical_anomalies, pattern_anomalies, temporal_anomalies
            ),
            'confidence': total_anomaly_score,
            'recommendations': self._generate_anomaly_recommendations(total_anomaly_score)
        }
    
    def _extract_features(self, sensor_data: Dict[str, Any]) -> np.ndarray:
        """Extract numerical features from sensor data."""
        features = []
        
        # Temperature features
        if 'temperature' in sensor_data:
            features.append(sensor_data['temperature'])
            features.append(sensor_data.get('temperature_change', 0.0))
        
        # Humidity features
        if 'humidity' in sensor_data:
            features.append(sensor_data['humidity'])
            features.append(sensor_data.get('humidity_change', 0.0))
        
        # Wind features
        if 'wind_speed' in sensor_data:
            features.append(sensor_data['wind_speed'])
            features.append(sensor_data.get('wind_direction', 0.0))
        
        # Pressure features
        if 'pressure' in sensor_data:
            features.append(sensor_data['pressure'])
            features.append(sensor_data.get('pressure_change', 0.0))
        
        return np.array(features).reshape(1, -1)
    
    def _detect_statistical_anomalies(self, features: np.ndarray) -> Dict[str, Any]:
        """Detect statistical anomalies using isolation forest."""
        if len(self.historical_data) < 100:
            return {'score': 0.0, 'type': 'insufficient_data'}
        
        # Fit isolation forest on historical data
        self.isolation_forest.fit(self.historical_data)
        
        # Predict anomaly score
        anomaly_score = self.isolation_forest.decision_function(features)[0]
        anomaly_score = (anomaly_score + 1) / 2  # Normalize to [0, 1]
        
        return {
            'score': anomaly_score,
            'type': 'statistical'
        }
    
    def _detect_pattern_anomalies(self, features: np.ndarray) -> Dict[str, Any]:
        """Detect pattern-based anomalies."""
        # Check for sudden changes in sensor values
        if len(self.historical_data) > 0:
            recent_avg = np.mean(self.historical_data[-10:], axis=0)
            current_diff = np.abs(features[0] - recent_avg)
            max_diff = np.max(current_diff)
            
            if max_diff > 3.0:  # 3 standard deviations
                return {
                    'score': min(1.0, max_diff / 10.0),
                    'type': 'pattern_change'
                }
        
        return {'score': 0.0, 'type': 'normal'}
    
    def _detect_temporal_anomalies(self, features: np.ndarray) -> Dict[str, Any]:
        """Detect temporal anomalies in sensor data."""
        # Check for unusual timing patterns
        current_hour = datetime.now().hour
        
        # Fire risk is higher during certain hours
        if 10 <= current_hour <= 16:  # Peak fire risk hours
            time_factor = 1.2
        else:
            time_factor = 0.8
        
        # Check for unusual sensor readings for the time of day
        if 'temperature' in features and features[0][0] > 35.0 and current_hour < 8:
            return {
                'score': 0.8,
                'type': 'temporal_unusual'
            }
        
        return {'score': 0.0, 'type': 'normal'}
    
    def _classify_anomaly_type(self, statistical: Dict, pattern: Dict, 
                              temporal: Dict) -> str:
        """Classify the type of anomaly detected."""
        if statistical['score'] > pattern['score'] and statistical['score'] > temporal['score']:
            return 'statistical'
        elif pattern['score'] > temporal['score']:
            return 'pattern'
        else:
            return 'temporal'
    
    def _generate_anomaly_recommendations(self, anomaly_score: float) -> List[str]:
        """Generate recommendations based on anomaly score."""
        recommendations = []
        
        if anomaly_score > 0.9:
            recommendations.extend([
                "Immediate investigation required",
                "Deploy additional monitoring assets",
                "Alert emergency services"
            ])
        elif anomaly_score > 0.7:
            recommendations.extend([
                "Increase monitoring frequency",
                "Prepare response resources",
                "Notify operations team"
            ])
        elif anomaly_score > 0.5:
            recommendations.extend([
                "Continue monitoring",
                "Document observations",
                "Check sensor calibration"
            ])
        
        return recommendations


class EdgeInferenceEngine:
    """
    Lightweight inference engine for edge devices.
    Runs ONNX models for real-time detection and classification.
    """
    
    def __init__(self, model_path: str):
        self.session = ort.InferenceSession(model_path)
        self.input_names = [input.name for input in self.session.get_inputs()]
        self.output_names = [output.name for output in self.session.get_outputs()]
    
    def detect_fire(self, thermal_image: np.ndarray, 
                   rgb_image: np.ndarray) -> Dict[str, Any]:
        """Detect fire using thermal and RGB images."""
        
        # Preprocess images
        thermal_processed = self._preprocess_thermal(thermal_image)
        rgb_processed = self._preprocess_rgb(rgb_image)
        
        # Run inference
        inputs = {
            'thermal': thermal_processed,
            'rgb': rgb_processed
        }
        
        outputs = self.session.run(self.output_names, inputs)
        
        # Parse results
        fire_confidence = outputs[0][0][1]  # Fire class probability
        fire_location = outputs[1][0] if len(outputs) > 1 else None
        
        return {
            'fire_detected': fire_confidence > 0.5,
            'confidence': float(fire_confidence),
            'location': fire_location,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    
    def detect_smoke(self, image: np.ndarray) -> Dict[str, Any]:
        """Detect smoke in image."""
        
        # Preprocess image
        processed = self._preprocess_image(image)
        
        # Run inference
        inputs = {'image': processed}
        outputs = self.session.run(self.output_names, inputs)
        
        smoke_confidence = outputs[0][0][1]  # Smoke class probability
        
        return {
            'smoke_detected': smoke_confidence > 0.3,
            'confidence': float(smoke_confidence),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    
    def classify_terrain(self, image: np.ndarray) -> Dict[str, Any]:
        """Classify terrain type from image."""
        
        # Preprocess image
        processed = self._preprocess_image(image)
        
        # Run inference
        inputs = {'image': processed}
        outputs = self.session.run(self.output_names, inputs)
        
        # Get class probabilities
        class_probs = outputs[0][0]
        class_names = ['brush', 'road', 'water', 'rock', 'unknown']
        
        predicted_class = class_names[np.argmax(class_probs)]
        confidence = float(np.max(class_probs))
        
        return {
            'terrain_type': predicted_class,
            'confidence': confidence,
            'class_probabilities': dict(zip(class_names, class_probs.tolist()))
        }
    
    def _preprocess_thermal(self, image: np.ndarray) -> np.ndarray:
        """Preprocess thermal image for inference."""
        # Normalize to [0, 1]
        normalized = (image - image.min()) / (image.max() - image.min())
        
        # Resize to model input size
        resized = cv2.resize(normalized, (224, 224))
        
        # Add batch dimension
        return np.expand_dims(resized, axis=0)
    
    def _preprocess_rgb(self, image: np.ndarray) -> np.ndarray:
        """Preprocess RGB image for inference."""
        # Convert BGR to RGB
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        normalized = rgb.astype(np.float32) / 255.0
        
        # Resize to model input size
        resized = cv2.resize(normalized, (224, 224))
        
        # Add batch dimension and transpose to CHW format
        return np.expand_dims(resized.transpose(2, 0, 1), axis=0)
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Generic image preprocessing."""
        # Normalize to [0, 1]
        normalized = image.astype(np.float32) / 255.0
        
        # Resize to model input size
        resized = cv2.resize(normalized, (224, 224))
        
        # Add batch dimension
        return np.expand_dims(resized, axis=0)


class WorldModelUpdater:
    """
    Updates the unified world model with new sensor data and AI predictions.
    Maintains spatial and temporal consistency across all data sources.
    """
    
    def __init__(self):
        self.world_model = {}
        self.spatial_grid = {}
        self.temporal_buffer = []
        self.confidence_threshold = 0.7
    
    def update_world_model(self, sensor_data: Dict[str, Any], 
                          ai_predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Update world model with new sensor data and AI predictions."""
        
        # Extract location and timestamp
        location = sensor_data.get('location', {})
        timestamp = sensor_data.get('timestamp', datetime.now(timezone.utc).isoformat())
        
        # Update spatial grid
        grid_cell = self._get_grid_cell(location)
        if grid_cell not in self.spatial_grid:
            self.spatial_grid[grid_cell] = {}
        
        # Update cell with new data
        self.spatial_grid[grid_cell].update({
            'timestamp': timestamp,
            'sensor_data': sensor_data,
            'ai_predictions': ai_predictions,
            'confidence': self._calculate_confidence(ai_predictions)
        })
        
        # Update temporal buffer
        self.temporal_buffer.append({
            'timestamp': timestamp,
            'location': location,
            'data': sensor_data,
            'predictions': ai_predictions
        })
        
        # Keep only recent data (last 24 hours)
        cutoff_time = datetime.now(timezone.utc).timestamp() - 86400
        self.temporal_buffer = [
            item for item in self.temporal_buffer 
            if datetime.fromisoformat(item['timestamp'].replace('Z', '+00:00')).timestamp() > cutoff_time
        ]
        
        # Generate updated world model
        return self._generate_world_model()
    
    def _get_grid_cell(self, location: Dict[str, float]) -> Tuple[int, int]:
        """Get grid cell coordinates for location."""
        # 100m grid cells
        lat_cell = int(location['latitude'] * 1000) // 100
        lon_cell = int(location['longitude'] * 1000) // 100
        return (lat_cell, lon_cell)
    
    def _calculate_confidence(self, predictions: Dict[str, Any]) -> float:
        """Calculate overall confidence for predictions."""
        confidences = []
        
        for key, value in predictions.items():
            if isinstance(value, dict) and 'confidence' in value:
                confidences.append(value['confidence'])
            elif isinstance(value, (int, float)) and 0 <= value <= 1:
                confidences.append(value)
        
        return np.mean(confidences) if confidences else 0.0
    
    def _generate_world_model(self) -> Dict[str, Any]:
        """Generate current world model state."""
        
        # Aggregate spatial data
        spatial_summary = {}
        for cell, data in self.spatial_grid.items():
            if data.get('confidence', 0) > self.confidence_threshold:
                spatial_summary[cell] = {
                    'location': data['sensor_data'].get('location', {}),
                    'predictions': data['ai_predictions'],
                    'confidence': data['confidence'],
                    'timestamp': data['timestamp']
                }
        
        # Generate temporal trends
        temporal_trends = self._analyze_temporal_trends()
        
        # Calculate overall risk score
        risk_score = self._calculate_risk_score()
        
        return {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'spatial_data': spatial_summary,
            'temporal_trends': temporal_trends,
            'risk_score': risk_score,
            'confidence': np.mean([data.get('confidence', 0) for data in self.spatial_grid.values()])
        }
    
    def _analyze_temporal_trends(self) -> Dict[str, Any]:
        """Analyze temporal trends in the data."""
        if len(self.temporal_buffer) < 2:
            return {'trend': 'insufficient_data'}
        
        # Analyze temperature trends
        temperatures = [item['data'].get('temperature', 0) for item in self.temporal_buffer]
        if temperatures:
            temp_trend = 'increasing' if temperatures[-1] > temperatures[0] else 'decreasing'
        else:
            temp_trend = 'unknown'
        
        # Analyze fire detection trends
        fire_detections = [item['predictions'].get('fire_detected', False) for item in self.temporal_buffer]
        fire_trend = 'increasing' if sum(fire_detections[-5:]) > sum(fire_detections[:5]) else 'stable'
        
        return {
            'temperature_trend': temp_trend,
            'fire_detection_trend': fire_trend,
            'data_points': len(self.temporal_buffer)
        }
    
    def _calculate_risk_score(self) -> float:
        """Calculate overall risk score for the area."""
        risk_factors = []
        
        for cell, data in self.spatial_grid.items():
            predictions = data.get('ai_predictions', {})
            confidence = data.get('confidence', 0)
            
            # Fire risk
            if predictions.get('fire_detected', False):
                risk_factors.append(0.9 * confidence)
            
            # Smoke risk
            if predictions.get('smoke_detected', False):
                risk_factors.append(0.7 * confidence)
            
            # Anomaly risk
            if predictions.get('anomaly_score', 0) > 0.5:
                risk_factors.append(predictions['anomaly_score'] * confidence)
        
        return min(1.0, np.mean(risk_factors)) if risk_factors else 0.0
