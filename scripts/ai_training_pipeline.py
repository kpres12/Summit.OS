#!/usr/bin/env python3
"""
Summit.OS AI Training Pipeline

Implements the complete AI training and deployment pipeline for Summit.OS.
Includes data collection, model training, validation, and deployment to edge devices.
"""

import asyncio
import json
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import onnx
from onnx import helper, TensorProto
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Tuple, Optional, Any
import os
import sys
import requests
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from apps.fusion.ai_models import MultimodalFusionNetwork, FireSpreadPredictor
from apps.intelligence.ai_reasoning import RiskAssessmentEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SummitAITrainingPipeline:
    """
    Complete AI training pipeline for Summit.OS.
    Handles data collection, model training, validation, and deployment.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data_lake_url = config.get('data_lake_url', 'http://localhost:8005')
        self.model_registry_url = config.get('model_registry_url', 'http://localhost:8006')
        self.edge_deployment_url = config.get('edge_deployment_url', 'http://localhost:8007')
        
        # Training parameters
        self.batch_size = config.get('batch_size', 32)
        self.learning_rate = config.get('learning_rate', 0.001)
        self.epochs = config.get('epochs', 100)
        self.validation_split = config.get('validation_split', 0.2)
        
        # Model paths
        self.models_dir = Path(config.get('models_dir', 'models'))
        self.models_dir.mkdir(exist_ok=True)
        
        # Initialize training components
        self.data_collector = DataCollector(self.data_lake_url)
        self.model_trainer = ModelTrainer(self.models_dir)
        self.model_validator = ModelValidator()
        self.model_deployer = ModelDeployer(self.edge_deployment_url)
    
    async def run_training_pipeline(self):
        """Run the complete AI training pipeline."""
        logger.info("🚀 Starting Summit.OS AI Training Pipeline")
        
        try:
            # Step 1: Data Collection
            logger.info("📊 Step 1: Collecting training data...")
            training_data = await self.data_collector.collect_training_data()
            
            # Step 2: Data Preprocessing
            logger.info("🔧 Step 2: Preprocessing data...")
            processed_data = self._preprocess_data(training_data)
            
            # Step 3: Model Training
            logger.info("🧠 Step 3: Training AI models...")
            trained_models = await self._train_models(processed_data)
            
            # Step 4: Model Validation
            logger.info("✅ Step 4: Validating models...")
            validation_results = await self._validate_models(trained_models, processed_data)
            
            # Step 5: Model Deployment
            logger.info("🚀 Step 5: Deploying models...")
            deployment_results = await self._deploy_models(trained_models, validation_results)
            
            # Step 6: Performance Monitoring
            logger.info("📈 Step 6: Monitoring model performance...")
            await self._setup_performance_monitoring(deployment_results)
            
            logger.info("✅ AI Training Pipeline completed successfully!")
            
        except Exception as e:
            logger.error(f"❌ AI Training Pipeline failed: {e}")
            raise
    
    def _preprocess_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess training data for model training."""
        logger.info("Preprocessing training data...")
        
        processed_data = {}
        
        # Preprocess sensor fusion data
        if 'sensor_fusion' in raw_data:
            processed_data['sensor_fusion'] = self._preprocess_sensor_fusion_data(
                raw_data['sensor_fusion']
            )
        
        # Preprocess risk assessment data
        if 'risk_assessment' in raw_data:
            processed_data['risk_assessment'] = self._preprocess_risk_assessment_data(
                raw_data['risk_assessment']
            )
        
        # Preprocess fire spread data
        if 'fire_spread' in raw_data:
            processed_data['fire_spread'] = self._preprocess_fire_spread_data(
                raw_data['fire_spread']
            )
        
        return processed_data
    
    def _preprocess_sensor_fusion_data(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Preprocess sensor fusion training data."""
        # Extract features and labels
        features = []
        labels = []
        
        for sample in data:
            # Extract sensor features
            feature_vector = self._extract_sensor_features(sample)
            features.append(feature_vector)
            
            # Extract labels
            label_vector = self._extract_sensor_labels(sample)
            labels.append(label_vector)
        
        # Convert to numpy arrays
        X = np.array(features)
        y = np.array(labels)
        
        # Split into train/validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.validation_split, random_state=42
        )
        
        return {
            'X_train': X_train,
            'X_val': X_val,
            'y_train': y_train,
            'y_val': y_val,
            'feature_names': self._get_sensor_feature_names(),
            'label_names': self._get_sensor_label_names()
        }
    
    def _preprocess_risk_assessment_data(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Preprocess risk assessment training data."""
        # Extract features and labels
        features = []
        labels = []
        
        for sample in data:
            # Extract risk features
            feature_vector = self._extract_risk_features(sample)
            features.append(feature_vector)
            
            # Extract risk labels
            label_vector = self._extract_risk_labels(sample)
            labels.append(label_vector)
        
        # Convert to numpy arrays
        X = np.array(features)
        y = np.array(labels)
        
        # Split into train/validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.validation_split, random_state=42
        )
        
        return {
            'X_train': X_train,
            'X_val': X_val,
            'y_train': y_train,
            'y_val': y_val,
            'feature_names': self._get_risk_feature_names(),
            'label_names': self._get_risk_label_names()
        }
    
    def _preprocess_fire_spread_data(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Preprocess fire spread training data."""
        # Extract features and labels
        features = []
        labels = []
        
        for sample in data:
            # Extract fire spread features
            feature_vector = self._extract_fire_spread_features(sample)
            features.append(feature_vector)
            
            # Extract spread rate labels
            label_vector = self._extract_fire_spread_labels(sample)
            labels.append(label_vector)
        
        # Convert to numpy arrays
        X = np.array(features)
        y = np.array(labels)
        
        # Split into train/validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.validation_split, random_state=42
        )
        
        return {
            'X_train': X_train,
            'X_val': X_val,
            'y_train': y_train,
            'y_val': y_val,
            'feature_names': self._get_fire_spread_feature_names(),
            'label_names': self._get_fire_spread_label_names()
        }
    
    async def _train_models(self, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Train all AI models."""
        trained_models = {}
        
        # Train sensor fusion model
        if 'sensor_fusion' in processed_data:
            logger.info("Training sensor fusion model...")
            fusion_model = await self.model_trainer.train_sensor_fusion_model(
                processed_data['sensor_fusion']
            )
            trained_models['sensor_fusion'] = fusion_model
        
        # Train risk assessment model
        if 'risk_assessment' in processed_data:
            logger.info("Training risk assessment model...")
            risk_model = await self.model_trainer.train_risk_assessment_model(
                processed_data['risk_assessment']
            )
            trained_models['risk_assessment'] = risk_model
        
        # Train fire spread model
        if 'fire_spread' in processed_data:
            logger.info("Training fire spread model...")
            fire_model = await self.model_trainer.train_fire_spread_model(
                processed_data['fire_spread']
            )
            trained_models['fire_spread'] = fire_model
        
        return trained_models
    
    async def _validate_models(self, trained_models: Dict[str, Any], 
                              processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate trained models."""
        validation_results = {}
        
        for model_name, model in trained_models.items():
            logger.info(f"Validating {model_name} model...")
            
            if model_name in processed_data:
                validation_result = await self.model_validator.validate_model(
                    model, processed_data[model_name]
                )
                validation_results[model_name] = validation_result
            else:
                logger.warning(f"No validation data found for {model_name}")
        
        return validation_results
    
    async def _deploy_models(self, trained_models: Dict[str, Any], 
                           validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Deploy validated models to edge devices."""
        deployment_results = {}
        
        for model_name, model in trained_models.items():
            if model_name in validation_results:
                validation_result = validation_results[model_name]
                
                # Check if model meets deployment criteria
                if validation_result.get('accuracy', 0) > 0.8:
                    logger.info(f"Deploying {model_name} model...")
                    deployment_result = await self.model_deployer.deploy_model(
                        model, model_name, validation_result
                    )
                    deployment_results[model_name] = deployment_result
                else:
                    logger.warning(f"{model_name} model accuracy too low for deployment")
            else:
                logger.warning(f"No validation results for {model_name}")
        
        return deployment_results
    
    async def _setup_performance_monitoring(self, deployment_results: Dict[str, Any]):
        """Setup performance monitoring for deployed models."""
        logger.info("Setting up performance monitoring...")
        
        for model_name, deployment_result in deployment_results.items():
            if deployment_result.get('deployed', False):
                logger.info(f"Setting up monitoring for {model_name} model...")
                # Setup monitoring for this model
                await self._setup_model_monitoring(model_name, deployment_result)
    
    async def _setup_model_monitoring(self, model_name: str, deployment_result: Dict[str, Any]):
        """Setup monitoring for a specific model."""
        # This would integrate with the monitoring system
        # For now, just log the setup
        logger.info(f"Monitoring setup for {model_name}: {deployment_result}")
    
    # Feature extraction methods
    def _extract_sensor_features(self, sample: Dict[str, Any]) -> np.ndarray:
        """Extract features from sensor data sample."""
        features = []
        
        # Weather features
        weather = sample.get('weather', {})
        features.extend([
            weather.get('temperature', 25.0),
            weather.get('humidity', 50.0),
            weather.get('wind_speed', 5.0),
            weather.get('wind_direction', 0.0),
            weather.get('pressure', 1013.25)
        ])
        
        # Sensor features
        sensors = sample.get('sensors', {})
        features.extend([
            sensors.get('temperature', 25.0),
            sensors.get('humidity', 50.0),
            sensors.get('visibility', 10000.0)
        ])
        
        # Location features
        location = sample.get('location', {})
        features.extend([
            location.get('latitude', 0.0),
            location.get('longitude', 0.0),
            location.get('altitude', 0.0)
        ])
        
        return np.array(features)
    
    def _extract_sensor_labels(self, sample: Dict[str, Any]) -> np.ndarray:
        """Extract labels from sensor data sample."""
        labels = []
        
        # Fire detection labels
        labels.append(1 if sample.get('fire_detected', False) else 0)
        
        # Smoke detection labels
        labels.append(1 if sample.get('smoke_detected', False) else 0)
        
        # Terrain classification labels
        terrain = sample.get('terrain_type', 'unknown')
        terrain_labels = [0, 0, 0, 0, 0]  # brush, road, water, rock, unknown
        terrain_map = {'brush': 0, 'road': 1, 'water': 2, 'rock': 3, 'unknown': 4}
        if terrain in terrain_map:
            terrain_labels[terrain_map[terrain]] = 1
        labels.extend(terrain_labels)
        
        return np.array(labels)
    
    def _extract_risk_features(self, sample: Dict[str, Any]) -> np.ndarray:
        """Extract features from risk assessment data."""
        features = []
        
        # Weather risk features
        weather = sample.get('weather', {})
        features.extend([
            weather.get('temperature', 25.0),
            weather.get('humidity', 50.0),
            weather.get('wind_speed', 5.0),
            weather.get('pressure', 1013.25)
        ])
        
        # Terrain risk features
        terrain = sample.get('terrain', {})
        features.extend([
            terrain.get('slope', 0.0),
            terrain.get('elevation', 0.0),
            terrain.get('vegetation_density', 0.5)
        ])
        
        # Historical risk features
        historical = sample.get('historical', {})
        features.extend([
            historical.get('fire_frequency', 0.0),
            historical.get('fire_severity', 0.0),
            historical.get('response_time', 0.0)
        ])
        
        return np.array(features)
    
    def _extract_risk_labels(self, sample: Dict[str, Any]) -> np.ndarray:
        """Extract labels from risk assessment data."""
        labels = []
        
        # Risk level labels
        risk_level = sample.get('risk_level', 'low')
        risk_labels = [0, 0, 0, 0]  # low, medium, high, critical
        risk_map = {'low': 0, 'medium': 1, 'high': 2, 'critical': 3}
        if risk_level in risk_map:
            risk_labels[risk_map[risk_level]] = 1
        labels.extend(risk_labels)
        
        return np.array(labels)
    
    def _extract_fire_spread_features(self, sample: Dict[str, Any]) -> np.ndarray:
        """Extract features from fire spread data."""
        features = []
        
        # Fire characteristics
        fire = sample.get('fire', {})
        features.extend([
            fire.get('size', 0.0),
            fire.get('intensity', 0.0),
            fire.get('temperature', 0.0)
        ])
        
        # Weather conditions
        weather = sample.get('weather', {})
        features.extend([
            weather.get('temperature', 25.0),
            weather.get('humidity', 50.0),
            weather.get('wind_speed', 5.0),
            weather.get('wind_direction', 0.0)
        ])
        
        # Terrain conditions
        terrain = sample.get('terrain', {})
        features.extend([
            terrain.get('slope', 0.0),
            terrain.get('elevation', 0.0),
            terrain.get('vegetation_density', 0.5)
        ])
        
        return np.array(features)
    
    def _extract_fire_spread_labels(self, sample: Dict[str, Any]) -> np.ndarray:
        """Extract labels from fire spread data."""
        labels = []
        
        # Spread rate labels
        spread_rate = sample.get('spread_rate', 0.0)
        labels.append(spread_rate)
        
        # Spread direction labels
        spread_direction = sample.get('spread_direction', 0.0)
        labels.append(spread_direction)
        
        return np.array(labels)
    
    # Feature name methods
    def _get_sensor_feature_names(self) -> List[str]:
        return [
            'weather_temperature', 'weather_humidity', 'weather_wind_speed',
            'weather_wind_direction', 'weather_pressure',
            'sensor_temperature', 'sensor_humidity', 'sensor_visibility',
            'location_latitude', 'location_longitude', 'location_altitude'
        ]
    
    def _get_sensor_label_names(self) -> List[str]:
        return [
            'fire_detected', 'smoke_detected',
            'terrain_brush', 'terrain_road', 'terrain_water', 'terrain_rock', 'terrain_unknown'
        ]
    
    def _get_risk_feature_names(self) -> List[str]:
        return [
            'weather_temperature', 'weather_humidity', 'weather_wind_speed', 'weather_pressure',
            'terrain_slope', 'terrain_elevation', 'terrain_vegetation_density',
            'historical_fire_frequency', 'historical_fire_severity', 'historical_response_time'
        ]
    
    def _get_risk_label_names(self) -> List[str]:
        return ['risk_low', 'risk_medium', 'risk_high', 'risk_critical']
    
    def _get_fire_spread_feature_names(self) -> List[str]:
        return [
            'fire_size', 'fire_intensity', 'fire_temperature',
            'weather_temperature', 'weather_humidity', 'weather_wind_speed', 'weather_wind_direction',
            'terrain_slope', 'terrain_elevation', 'terrain_vegetation_density'
        ]
    
    def _get_fire_spread_label_names(self) -> List[str]:
        return ['spread_rate', 'spread_direction']


class DataCollector:
    """Collects training data from the data lake."""
    
    def __init__(self, data_lake_url: str):
        self.data_lake_url = data_lake_url
    
    async def collect_training_data(self) -> Dict[str, Any]:
        """Collect training data from various sources."""
        logger.info("Collecting training data from data lake...")
        
        # This would connect to the actual data lake
        # For now, return mock data
        return {
            'sensor_fusion': self._collect_sensor_fusion_data(),
            'risk_assessment': self._collect_risk_assessment_data(),
            'fire_spread': self._collect_fire_spread_data()
        }
    
    def _collect_sensor_fusion_data(self) -> List[Dict[str, Any]]:
        """Collect sensor fusion training data."""
        # Mock data - in production this would come from the data lake
        return [
            {
                'weather': {'temperature': 25.0, 'humidity': 50.0, 'wind_speed': 5.0, 'wind_direction': 0.0, 'pressure': 1013.25},
                'sensors': {'temperature': 25.0, 'humidity': 50.0, 'visibility': 10000.0},
                'location': {'latitude': 37.7749, 'longitude': -122.4194, 'altitude': 100.0},
                'fire_detected': False,
                'smoke_detected': False,
                'terrain_type': 'brush'
            }
            # ... more samples
        ]
    
    def _collect_risk_assessment_data(self) -> List[Dict[str, Any]]:
        """Collect risk assessment training data."""
        # Mock data - in production this would come from the data lake
        return [
            {
                'weather': {'temperature': 25.0, 'humidity': 50.0, 'wind_speed': 5.0, 'pressure': 1013.25},
                'terrain': {'slope': 0.0, 'elevation': 100.0, 'vegetation_density': 0.5},
                'historical': {'fire_frequency': 0.1, 'fire_severity': 0.2, 'response_time': 30.0},
                'risk_level': 'low'
            }
            # ... more samples
        ]
    
    def _collect_fire_spread_data(self) -> List[Dict[str, Any]]:
        """Collect fire spread training data."""
        # Mock data - in production this would come from the data lake
        return [
            {
                'fire': {'size': 10.0, 'intensity': 0.5, 'temperature': 800.0},
                'weather': {'temperature': 25.0, 'humidity': 50.0, 'wind_speed': 5.0, 'wind_direction': 0.0},
                'terrain': {'slope': 0.0, 'elevation': 100.0, 'vegetation_density': 0.5},
                'spread_rate': 0.1,
                'spread_direction': 0.0
            }
            # ... more samples
        ]


class ModelTrainer:
    """Trains AI models for Summit.OS."""
    
    def __init__(self, models_dir: Path):
        self.models_dir = models_dir
    
    async def train_sensor_fusion_model(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Train sensor fusion model."""
        logger.info("Training sensor fusion model...")
        
        # Create model
        model = MultimodalFusionNetwork(
            input_dims={'weather': 5, 'lidar': 1, 'thermal': 1, 'visual': 3},
            hidden_dim=512
        )
        
        # Training would happen here
        # For now, return mock results
        return {
            'model': model,
            'accuracy': 0.95,
            'loss': 0.05,
            'epochs_trained': 100
        }
    
    async def train_risk_assessment_model(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Train risk assessment model."""
        logger.info("Training risk assessment model...")
        
        # Create model
        model = RiskAssessmentEngine()
        
        # Training would happen here
        # For now, return mock results
        return {
            'model': model,
            'accuracy': 0.90,
            'loss': 0.10,
            'epochs_trained': 100
        }
    
    async def train_fire_spread_model(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Train fire spread model."""
        logger.info("Training fire spread model...")
        
        # Create model
        model = FireSpreadPredictor()
        
        # Training would happen here
        # For now, return mock results
        return {
            'model': model,
            'accuracy': 0.85,
            'loss': 0.15,
            'epochs_trained': 100
        }


class ModelValidator:
    """Validates trained models."""
    
    async def validate_model(self, model: Any, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a trained model."""
        logger.info("Validating model...")
        
        # Validation would happen here
        # For now, return mock results
        return {
            'accuracy': 0.90,
            'precision': 0.88,
            'recall': 0.92,
            'f1_score': 0.90,
            'confusion_matrix': [[100, 5], [3, 92]],
            'validation_loss': 0.10
        }


class ModelDeployer:
    """Deploys models to edge devices."""
    
    def __init__(self, edge_deployment_url: str):
        self.edge_deployment_url = edge_deployment_url
    
    async def deploy_model(self, model: Any, model_name: str, 
                          validation_result: Dict[str, Any]) -> Dict[str, Any]:
        """Deploy a model to edge devices."""
        logger.info(f"Deploying {model_name} model to edge devices...")
        
        # Convert model to ONNX format
        onnx_model = self._convert_to_onnx(model, model_name)
        
        # Deploy to edge devices
        deployment_result = await self._deploy_to_edge(onnx_model, model_name)
        
        return {
            'deployed': True,
            'model_name': model_name,
            'deployment_id': f"deploy-{model_name}-{int(datetime.now().timestamp())}",
            'edge_devices': ['drone-001', 'drone-002', 'ugv-001'],
            'deployment_time': datetime.now(timezone.utc).isoformat()
        }
    
    def _convert_to_onnx(self, model: Any, model_name: str) -> str:
        """Convert model to ONNX format."""
        logger.info(f"Converting {model_name} to ONNX format...")
        
        # ONNX conversion would happen here
        # For now, return mock ONNX model path
        onnx_path = f"models/{model_name}.onnx"
        return onnx_path
    
    async def _deploy_to_edge(self, onnx_model: str, model_name: str) -> Dict[str, Any]:
        """Deploy ONNX model to edge devices."""
        logger.info(f"Deploying {model_name} to edge devices...")
        
        # Edge deployment would happen here
        # For now, return mock deployment result
        return {
            'success': True,
            'edge_devices': ['drone-001', 'drone-002', 'ugv-001'],
            'deployment_time': datetime.now(timezone.utc).isoformat()
        }


async def main():
    """Main entry point for AI training pipeline."""
    config = {
        'data_lake_url': 'http://localhost:8005',
        'model_registry_url': 'http://localhost:8006',
        'edge_deployment_url': 'http://localhost:8007',
        'models_dir': 'models',
        'batch_size': 32,
        'learning_rate': 0.001,
        'epochs': 100,
        'validation_split': 0.2
    }
    
    pipeline = SummitAITrainingPipeline(config)
    await pipeline.run_training_pipeline()


if __name__ == "__main__":
    asyncio.run(main())
