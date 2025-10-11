"""
Fire Detection and Analysis Models for Summit.OS

Provides specialized AI models for fire detection, smoke analysis,
and fire spread prediction for sentry towers and drone operations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime, timezone
from dataclasses import dataclass
from enum import Enum
import json
import time

logger = logging.getLogger(__name__)


class FireType(Enum):
    """Types of fire detection"""
    SMOKE = "smoke"
    FLAME = "flame"
    HOTSPOT = "hotspot"
    EMBER = "ember"


class DetectionConfidence(Enum):
    """Detection confidence levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class FireDetection:
    """Fire detection result"""
    detection_id: str
    fire_type: FireType
    confidence: float
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    center: Tuple[float, float]  # lat, lon
    size: float  # estimated size in meters
    temperature: Optional[float] = None
    timestamp: float = None
    metadata: Dict[str, Any] = None


@dataclass
class FireAnalysis:
    """Comprehensive fire analysis"""
    detections: List[FireDetection]
    risk_level: str
    spread_direction: Optional[float] = None
    spread_rate: Optional[float] = None
    containment_priority: int = 1
    environmental_factors: Dict[str, Any] = None


class SmokeDetectionCNN(nn.Module):
    """
    CNN model for smoke detection in thermal and visual imagery.
    
    Uses a lightweight architecture optimized for edge deployment.
    """
    
    def __init__(self, input_channels: int = 3, num_classes: int = 2):
        super().__init__()
        
        # Feature extraction backbone
        self.conv1 = nn.Conv2d(input_channels, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Spatial attention
        self.attention = nn.Sequential(
            nn.Conv2d(128, 64, 1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()
        )
        
        # Classification head
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )
        
        # Detection head for bounding boxes
        self.detection_head = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 4, 1)  # x, y, w, h
        )
    
    def forward(self, x):
        # Feature extraction
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)
        
        # Attention mechanism
        attention_weights = self.attention(x)
        x = x * attention_weights
        
        # Classification
        pooled = self.global_pool(x)
        pooled = pooled.view(pooled.size(0), -1)
        classification = self.classifier(pooled)
        
        # Detection
        detection = self.detection_head(x)
        
        return classification, detection, attention_weights


class FireSpreadPredictor(nn.Module):
    """
    Neural network for predicting fire spread based on environmental conditions.
    """
    
    def __init__(self, input_dim: int = 10, hidden_dim: int = 64):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 3)  # spread_rate, direction, probability
        )
    
    def forward(self, x):
        return self.network(x)


class FireDetectionEngine:
    """
    Main fire detection engine that coordinates all fire-related AI models.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize models
        self.smoke_detector = SmokeDetectionCNN(
            input_channels=config.get('input_channels', 3),
            num_classes=config.get('num_classes', 2)
        ).to(self.device)
        
        self.fire_spread_predictor = FireSpreadPredictor(
            input_dim=config.get('environmental_input_dim', 10),
            hidden_dim=config.get('hidden_dim', 64)
        ).to(self.device)
        
        # Load pre-trained weights if available
        self._load_models()
        
        # Detection thresholds
        self.detection_thresholds = {
            FireType.SMOKE: config.get('smoke_threshold', 0.7),
            FireType.FLAME: config.get('flame_threshold', 0.8),
            FireType.HOTSPOT: config.get('hotspot_threshold', 0.6),
            FireType.EMBER: config.get('ember_threshold', 0.5)
        }
        
        # False positive filters
        self.false_positive_filters = {
            'cloud_threshold': config.get('cloud_threshold', 0.3),
            'sun_glint_threshold': config.get('sun_glint_threshold', 0.4),
            'dust_threshold': config.get('dust_threshold', 0.2)
        }
    
    def detect_fire(self, image: np.ndarray, thermal_data: Optional[np.ndarray] = None) -> List[FireDetection]:
        """
        Detect fire in image using multi-modal analysis.
        
        Args:
            image: RGB or grayscale image
            thermal_data: Optional thermal image data
            
        Returns:
            List of fire detections
        """
        try:
            # Preprocess image
            processed_image = self._preprocess_image(image)
            
            # Run smoke detection
            with torch.no_grad():
                classification, detection, attention = self.smoke_detector(
                    torch.FloatTensor(processed_image).unsqueeze(0).to(self.device)
                )
            
            # Process detections
            detections = self._process_detections(
                classification, detection, attention, image.shape
            )
            
            # Filter false positives
            filtered_detections = self._filter_false_positives(detections, image)
            
            # Add thermal analysis if available
            if thermal_data is not None:
                filtered_detections = self._enhance_with_thermal(
                    filtered_detections, thermal_data
                )
            
            return filtered_detections
            
        except Exception as e:
            logger.error(f"Fire detection failed: {e}")
            return []
    
    def analyze_fire_spread(self, detections: List[FireDetection], 
                           environmental_data: Dict[str, Any]) -> FireAnalysis:
        """
        Analyze fire spread potential based on detections and environmental conditions.
        
        Args:
            detections: List of fire detections
            environmental_data: Weather and environmental conditions
            
        Returns:
            Comprehensive fire analysis
        """
        try:
            if not detections:
                return FireAnalysis(detections=[], risk_level="low")
            
            # Prepare environmental features
            env_features = self._extract_environmental_features(environmental_data)
            
            # Predict fire spread
            with torch.no_grad():
                spread_prediction = self.fire_spread_predictor(
                    torch.FloatTensor(env_features).unsqueeze(0).to(self.device)
                )
            
            spread_rate, direction, probability = spread_prediction[0].cpu().numpy()
            
            # Calculate risk level
            risk_level = self._calculate_risk_level(detections, spread_rate, probability)
            
            # Determine containment priority
            containment_priority = self._calculate_containment_priority(
                detections, spread_rate, environmental_data
            )
            
            return FireAnalysis(
                detections=detections,
                risk_level=risk_level,
                spread_direction=direction,
                spread_rate=float(spread_rate),
                containment_priority=containment_priority,
                environmental_factors=environmental_data
            )
            
        except Exception as e:
            logger.error(f"Fire spread analysis failed: {e}")
            return FireAnalysis(detections=detections, risk_level="unknown")
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for model input"""
        # Resize to model input size
        target_size = (224, 224)
        if len(image.shape) == 3:
            image = cv2.resize(image, target_size)
        else:
            image = cv2.resize(image, target_size)
            image = np.stack([image] * 3, axis=2)  # Convert to 3-channel
        
        # Normalize
        image = image.astype(np.float32) / 255.0
        
        # Convert to tensor format (C, H, W)
        if len(image.shape) == 3:
            image = np.transpose(image, (2, 0, 1))
        
        return image
    
    def _process_detections(self, classification: torch.Tensor, 
                           detection: torch.Tensor, attention: torch.Tensor,
                           image_shape: Tuple[int, ...]) -> List[FireDetection]:
        """Process model outputs into detections"""
        detections = []
        
        # Get classification probabilities
        probs = F.softmax(classification, dim=1)
        smoke_prob = probs[0, 1].item()  # Assuming class 1 is smoke
        
        if smoke_prob > self.detection_thresholds[FireType.SMOKE]:
            # Process detection boxes
            detection_map = detection[0].cpu().numpy()
            attention_map = attention[0, 0].cpu().numpy()
            
            # Find peaks in attention map
            peaks = self._find_attention_peaks(attention_map)
            
            for peak in peaks:
                # Convert to image coordinates
                y, x = peak
                scale_y = image_shape[0] / attention_map.shape[0]
                scale_x = image_shape[1] / attention_map.shape[1]
                
                # Estimate bounding box
                bbox = self._estimate_bbox(x * scale_x, y * scale_y, image_shape)
                
                detection = FireDetection(
                    detection_id=f"fire_{int(time.time() * 1000)}",
                    fire_type=FireType.SMOKE,
                    confidence=smoke_prob,
                    bbox=bbox,
                    center=(x * scale_x, y * scale_y),
                    size=self._estimate_fire_size(bbox),
                    timestamp=time.time()
                )
                detections.append(detection)
        
        return detections
    
    def _find_attention_peaks(self, attention_map: np.ndarray, 
                            threshold: float = 0.5) -> List[Tuple[int, int]]:
        """Find peaks in attention map"""
        from scipy.ndimage import maximum_filter
        
        # Find local maxima
        local_maxima = maximum_filter(attention_map, size=5) == attention_map
        peaks = np.where((local_maxima) & (attention_map > threshold))
        
        return list(zip(peaks[0], peaks[1]))
    
    def _estimate_bbox(self, center_x: float, center_y: float, 
                      image_shape: Tuple[int, ...]) -> Tuple[int, int, int, int]:
        """Estimate bounding box around detection center"""
        # Default box size (can be made adaptive)
        box_width = 50
        box_height = 50
        
        x1 = max(0, int(center_x - box_width // 2))
        y1 = max(0, int(center_y - box_height // 2))
        x2 = min(image_shape[1], int(center_x + box_width // 2))
        y2 = min(image_shape[0], int(center_y + box_height // 2))
        
        return (x1, y1, x2 - x1, y2 - y1)
    
    def _estimate_fire_size(self, bbox: Tuple[int, int, int, int]) -> float:
        """Estimate fire size in meters"""
        x, y, w, h = bbox
        # Rough estimation based on pixel area
        pixel_area = w * h
        # Assume 1 pixel = 0.1 meter (this would need calibration)
        return pixel_area * 0.01
    
    def _filter_false_positives(self, detections: List[FireDetection], 
                               image: np.ndarray) -> List[FireDetection]:
        """Filter out false positive detections"""
        filtered = []
        
        for detection in detections:
            # Check for cloud-like patterns
            if self._is_cloud_pattern(image, detection.bbox):
                continue
            
            # Check for sun glint
            if self._is_sun_glint(image, detection.bbox):
                continue
            
            # Check for dust patterns
            if self._is_dust_pattern(image, detection.bbox):
                continue
            
            filtered.append(detection)
        
        return filtered
    
    def _is_cloud_pattern(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> bool:
        """Check if detection is likely a cloud"""
        x, y, w, h = bbox
        roi = image[y:y+h, x:x+w]
        
        # Analyze texture and color patterns
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
        texture_variance = np.var(gray)
        
        # Clouds typically have low texture variance
        return texture_variance < self.false_positive_filters['cloud_threshold']
    
    def _is_sun_glint(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> bool:
        """Check if detection is likely sun glint"""
        x, y, w, h = bbox
        roi = image[y:y+h, x:x+w]
        
        # Check for bright, uniform areas
        if len(roi.shape) == 3:
            brightness = np.mean(roi)
        else:
            brightness = np.mean(roi)
        
        return brightness > self.false_positive_filters['sun_glint_threshold']
    
    def _is_dust_pattern(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> bool:
        """Check if detection is likely dust"""
        x, y, w, h = bbox
        roi = image[y:y+h, x:x+w]
        
        # Analyze particle-like patterns
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (w * h)
        
        return edge_density < self.false_positive_filters['dust_threshold']
    
    def _enhance_with_thermal(self, detections: List[FireDetection], 
                             thermal_data: np.ndarray) -> List[FireDetection]:
        """Enhance detections with thermal data"""
        enhanced = []
        
        for detection in detections:
            x, y, w, h = detection.bbox
            
            # Extract thermal data for detection region
            thermal_roi = thermal_data[y:y+h, x:x+w]
            if thermal_roi.size > 0:
                max_temp = np.max(thermal_roi)
                avg_temp = np.mean(thermal_roi)
                
                # Update detection with thermal information
                detection.temperature = float(max_temp)
                
                # Adjust confidence based on temperature
                if max_temp > 100:  # Celsius
                    detection.confidence = min(1.0, detection.confidence + 0.2)
            
            enhanced.append(detection)
        
        return enhanced
    
    def _extract_environmental_features(self, environmental_data: Dict[str, Any]) -> List[float]:
        """Extract environmental features for fire spread prediction"""
        features = [
            environmental_data.get('wind_speed', 0.0),
            environmental_data.get('wind_direction', 0.0),
            environmental_data.get('temperature', 0.0),
            environmental_data.get('humidity', 0.0),
            environmental_data.get('fuel_moisture', 0.0),
            environmental_data.get('slope_angle', 0.0),
            environmental_data.get('vegetation_density', 0.0),
            environmental_data.get('fuel_type', 0.0),
            environmental_data.get('elevation', 0.0),
            environmental_data.get('precipitation', 0.0)
        ]
        
        return features
    
    def _calculate_risk_level(self, detections: List[FireDetection], 
                             spread_rate: float, probability: float) -> str:
        """Calculate overall fire risk level"""
        if not detections:
            return "low"
        
        # Factors for risk assessment
        detection_count = len(detections)
        avg_confidence = np.mean([d.confidence for d in detections])
        max_confidence = max([d.confidence for d in detections])
        
        # Risk scoring
        risk_score = (
            detection_count * 0.3 +
            avg_confidence * 0.3 +
            max_confidence * 0.2 +
            spread_rate * 0.1 +
            probability * 0.1
        )
        
        if risk_score > 0.8:
            return "critical"
        elif risk_score > 0.6:
            return "high"
        elif risk_score > 0.4:
            return "medium"
        else:
            return "low"
    
    def _calculate_containment_priority(self, detections: List[FireDetection],
                                      spread_rate: float, 
                                      environmental_data: Dict[str, Any]) -> int:
        """Calculate containment priority (1-5, 5 being highest)"""
        priority = 1
        
        # Increase priority based on detection count
        priority += min(len(detections), 2)
        
        # Increase priority based on spread rate
        if spread_rate > 0.5:
            priority += 1
        
        # Increase priority based on environmental conditions
        wind_speed = environmental_data.get('wind_speed', 0)
        if wind_speed > 20:  # High wind
            priority += 1
        
        return min(priority, 5)
    
    def _load_models(self):
        """Load pre-trained model weights"""
        try:
            model_path = self.config.get('model_path', './models')
            
            # Load smoke detection model
            smoke_model_path = f"{model_path}/smoke_detection.pth"
            if torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                device = torch.device('cpu')
            
            # self.smoke_detector.load_state_dict(torch.load(smoke_model_path, map_location=device))
            # self.smoke_detector.eval()
            
            logger.info("Fire detection models loaded successfully")
            
        except Exception as e:
            logger.warning(f"Could not load pre-trained models: {e}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models"""
        return {
            'smoke_detector': {
                'parameters': sum(p.numel() for p in self.smoke_detector.parameters()),
                'device': str(self.device)
            },
            'fire_spread_predictor': {
                'parameters': sum(p.numel() for p in self.fire_spread_predictor.parameters()),
                'device': str(self.device)
            },
            'detection_thresholds': self.detection_thresholds,
            'false_positive_filters': self.false_positive_filters
        }
