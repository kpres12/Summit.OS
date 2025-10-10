"""
Summit.OS AI Reasoning Engine

Implements contextual reasoning, risk assessment, and advisory generation
for the intelligence layer of Summit.OS.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timezone, timedelta
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class RiskAssessmentEngine:
    """
    AI-powered risk assessment engine that evaluates fire risk, environmental hazards,
    and operational risks based on multi-modal sensor data and historical patterns.
    """
    
    def __init__(self):
        self.risk_models = {
            'fire_risk': FireRiskModel(),
            'environmental_risk': EnvironmentalRiskModel(),
            'operational_risk': OperationalRiskModel()
        }
        
        self.historical_data = []
        self.risk_thresholds = {
            'low': 0.3,
            'medium': 0.6,
            'high': 0.8,
            'critical': 0.9
        }
    
    def assess_risk(self, sensor_data: Dict[str, Any], 
                   world_model: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive risk assessment across all risk categories."""
        
        # Fire risk assessment
        fire_risk = self.risk_models['fire_risk'].assess(
            sensor_data, world_model
        )
        
        # Environmental risk assessment
        environmental_risk = self.risk_models['environmental_risk'].assess(
            sensor_data, world_model
        )
        
        # Operational risk assessment
        operational_risk = self.risk_models['operational_risk'].assess(
            sensor_data, world_model
        )
        
        # Calculate overall risk score
        overall_risk = self._calculate_overall_risk(
            fire_risk, environmental_risk, operational_risk
        )
        
        # Generate risk recommendations
        recommendations = self._generate_risk_recommendations(
            fire_risk, environmental_risk, operational_risk
        )
        
        return {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'overall_risk': overall_risk,
            'fire_risk': fire_risk,
            'environmental_risk': environmental_risk,
            'operational_risk': operational_risk,
            'risk_level': self._classify_risk_level(overall_risk),
            'recommendations': recommendations,
            'confidence': self._calculate_confidence(fire_risk, environmental_risk, operational_risk)
        }
    
    def _calculate_overall_risk(self, fire_risk: Dict, environmental_risk: Dict, 
                              operational_risk: Dict) -> float:
        """Calculate weighted overall risk score."""
        weights = {
            'fire': 0.5,      # Fire risk is most critical
            'environmental': 0.3,
            'operational': 0.2
        }
        
        overall_score = (
            fire_risk['score'] * weights['fire'] +
            environmental_risk['score'] * weights['environmental'] +
            operational_risk['score'] * weights['operational']
        )
        
        return min(1.0, overall_score)
    
    def _classify_risk_level(self, risk_score: float) -> str:
        """Classify risk level based on score."""
        if risk_score >= self.risk_thresholds['critical']:
            return 'critical'
        elif risk_score >= self.risk_thresholds['high']:
            return 'high'
        elif risk_score >= self.risk_thresholds['medium']:
            return 'medium'
        else:
            return 'low'
    
    def _generate_risk_recommendations(self, fire_risk: Dict, environmental_risk: Dict, 
                                     operational_risk: Dict) -> List[str]:
        """Generate contextual recommendations based on risk assessment."""
        recommendations = []
        
        # Fire risk recommendations
        if fire_risk['score'] > 0.7:
            recommendations.extend([
                "Deploy additional fire monitoring assets",
                "Prepare suppression resources",
                "Issue fire weather warning"
            ])
        
        # Environmental risk recommendations
        if environmental_risk['score'] > 0.6:
            recommendations.extend([
                "Monitor weather conditions closely",
                "Prepare for adverse weather conditions",
                "Secure loose equipment and materials"
            ])
        
        # Operational risk recommendations
        if operational_risk['score'] > 0.5:
            recommendations.extend([
                "Review operational procedures",
                "Increase safety monitoring",
                "Consider operational adjustments"
            ])
        
        return recommendations
    
    def _calculate_confidence(self, fire_risk: Dict, environmental_risk: Dict, 
                            operational_risk: Dict) -> float:
        """Calculate confidence in risk assessment."""
        confidences = [
            fire_risk.get('confidence', 0.5),
            environmental_risk.get('confidence', 0.5),
            operational_risk.get('confidence', 0.5)
        ]
        return np.mean(confidences)


class FireRiskModel:
    """AI model for fire risk assessment."""
    
    def __init__(self):
        self.weather_factors = ['temperature', 'humidity', 'wind_speed', 'pressure']
        self.terrain_factors = ['slope', 'elevation', 'vegetation_density']
        self.historical_fires = []
        
    def assess(self, sensor_data: Dict[str, Any], 
               world_model: Dict[str, Any]) -> Dict[str, Any]:
        """Assess fire risk based on current conditions."""
        
        # Extract weather factors
        weather_risk = self._assess_weather_risk(sensor_data)
        
        # Extract terrain factors
        terrain_risk = self._assess_terrain_risk(world_model)
        
        # Check for existing fires
        fire_presence_risk = self._assess_fire_presence_risk(world_model)
        
        # Calculate combined risk score
        risk_score = (
            weather_risk * 0.4 +
            terrain_risk * 0.3 +
            fire_presence_risk * 0.3
        )
        
        # Generate fire-specific recommendations
        recommendations = self._generate_fire_recommendations(risk_score, sensor_data)
        
        return {
            'score': min(1.0, risk_score),
            'weather_risk': weather_risk,
            'terrain_risk': terrain_risk,
            'fire_presence_risk': fire_presence_risk,
            'recommendations': recommendations,
            'confidence': self._calculate_fire_confidence(sensor_data, world_model)
        }
    
    def _assess_weather_risk(self, sensor_data: Dict[str, Any]) -> float:
        """Assess fire risk based on weather conditions."""
        risk_factors = []
        
        # Temperature risk (higher = more risk)
        temp = sensor_data.get('temperature', 25.0)
        temp_risk = min(1.0, (temp - 20.0) / 30.0)  # 20-50°C range
        risk_factors.append(temp_risk)
        
        # Humidity risk (lower = more risk)
        humidity = sensor_data.get('humidity', 50.0)
        humidity_risk = max(0.0, (50.0 - humidity) / 50.0)  # 0-50% range
        risk_factors.append(humidity_risk)
        
        # Wind speed risk (higher = more risk)
        wind_speed = sensor_data.get('wind_speed', 5.0)
        wind_risk = min(1.0, wind_speed / 30.0)  # 0-30 m/s range
        risk_factors.append(wind_risk)
        
        # Pressure risk (lower = more risk)
        pressure = sensor_data.get('pressure', 1013.25)
        pressure_risk = max(0.0, (1013.25 - pressure) / 50.0)  # 50 hPa range
        risk_factors.append(pressure_risk)
        
        return np.mean(risk_factors)
    
    def _assess_terrain_risk(self, world_model: Dict[str, Any]) -> float:
        """Assess fire risk based on terrain characteristics."""
        terrain_data = world_model.get('spatial_data', {})
        
        if not terrain_data:
            return 0.5  # Default moderate risk
        
        # Analyze terrain features
        slope_risks = []
        vegetation_risks = []
        
        for cell_data in terrain_data.values():
            predictions = cell_data.get('predictions', {})
            
            # Slope risk (steeper = more risk)
            slope = predictions.get('slope', 0.0)
            slope_risk = min(1.0, slope / 45.0)  # 0-45 degree range
            slope_risks.append(slope_risk)
            
            # Vegetation risk (denser = more risk)
            vegetation = predictions.get('vegetation_density', 0.5)
            vegetation_risks.append(vegetation)
        
        slope_risk = np.mean(slope_risks) if slope_risks else 0.5
        vegetation_risk = np.mean(vegetation_risks) if vegetation_risks else 0.5
        
        return (slope_risk + vegetation_risk) / 2
    
    def _assess_fire_presence_risk(self, world_model: Dict[str, Any]) -> float:
        """Assess risk based on existing fires in the area."""
        spatial_data = world_model.get('spatial_data', {})
        
        fire_detections = 0
        total_cells = len(spatial_data)
        
        for cell_data in spatial_data.values():
            predictions = cell_data.get('predictions', {})
            if predictions.get('fire_detected', False):
                fire_detections += 1
        
        if total_cells == 0:
            return 0.0
        
        fire_density = fire_detections / total_cells
        return min(1.0, fire_density * 2)  # Scale up fire density impact
    
    def _generate_fire_recommendations(self, risk_score: float, 
                                     sensor_data: Dict[str, Any]) -> List[str]:
        """Generate fire-specific recommendations."""
        recommendations = []
        
        if risk_score > 0.8:
            recommendations.extend([
                "CRITICAL: Immediate fire suppression required",
                "Evacuate personnel from high-risk areas",
                "Deploy all available suppression assets",
                "Issue emergency fire warning"
            ])
        elif risk_score > 0.6:
            recommendations.extend([
                "HIGH: Increase fire monitoring frequency",
                "Prepare suppression resources",
                "Consider pre-positioning assets",
                "Issue fire weather watch"
            ])
        elif risk_score > 0.4:
            recommendations.extend([
                "MEDIUM: Monitor conditions closely",
                "Prepare fire response plan",
                "Check equipment readiness"
            ])
        else:
            recommendations.extend([
                "LOW: Continue routine monitoring",
                "Maintain standard fire safety protocols"
            ])
        
        return recommendations
    
    def _calculate_fire_confidence(self, sensor_data: Dict[str, Any], 
                                 world_model: Dict[str, Any]) -> float:
        """Calculate confidence in fire risk assessment."""
        # Check data quality
        required_fields = ['temperature', 'humidity', 'wind_speed']
        data_completeness = sum(1 for field in required_fields if field in sensor_data) / len(required_fields)
        
        # Check world model quality
        spatial_data = world_model.get('spatial_data', {})
        model_confidence = np.mean([
            data.get('confidence', 0.5) for data in spatial_data.values()
        ]) if spatial_data else 0.5
        
        return (data_completeness + model_confidence) / 2


class EnvironmentalRiskModel:
    """AI model for environmental risk assessment."""
    
    def __init__(self):
        self.weather_patterns = []
        self.seasonal_factors = {}
        
    def assess(self, sensor_data: Dict[str, Any], 
               world_model: Dict[str, Any]) -> Dict[str, Any]:
        """Assess environmental risk factors."""
        
        # Weather pattern analysis
        weather_risk = self._analyze_weather_patterns(sensor_data)
        
        # Seasonal risk factors
        seasonal_risk = self._assess_seasonal_risk()
        
        # Extreme weather risk
        extreme_weather_risk = self._assess_extreme_weather_risk(sensor_data)
        
        # Calculate combined risk
        risk_score = (
            weather_risk * 0.4 +
            seasonal_risk * 0.3 +
            extreme_weather_risk * 0.3
        )
        
        return {
            'score': min(1.0, risk_score),
            'weather_risk': weather_risk,
            'seasonal_risk': seasonal_risk,
            'extreme_weather_risk': extreme_weather_risk,
            'confidence': 0.8  # Environmental models are generally reliable
        }
    
    def _analyze_weather_patterns(self, sensor_data: Dict[str, Any]) -> float:
        """Analyze weather patterns for risk factors."""
        # Check for rapid weather changes
        if len(self.weather_patterns) > 0:
            recent_temp = sensor_data.get('temperature', 25.0)
            recent_humidity = sensor_data.get('humidity', 50.0)
            
            # Calculate change rates
            temp_change = abs(recent_temp - self.weather_patterns[-1].get('temperature', 25.0))
            humidity_change = abs(recent_humidity - self.weather_patterns[-1].get('humidity', 50.0))
            
            # Rapid changes indicate instability
            instability_risk = min(1.0, (temp_change + humidity_change) / 20.0)
        else:
            instability_risk = 0.5
        
        # Update weather patterns
        self.weather_patterns.append(sensor_data)
        if len(self.weather_patterns) > 100:  # Keep last 100 readings
            self.weather_patterns = self.weather_patterns[-100:]
        
        return instability_risk
    
    def _assess_seasonal_risk(self) -> float:
        """Assess seasonal risk factors."""
        current_month = datetime.now().month
        
        # Fire season risk (higher in summer months)
        if 6 <= current_month <= 9:  # Summer months
            seasonal_risk = 0.8
        elif 4 <= current_month <= 5 or 10 <= current_month <= 11:  # Spring/Fall
            seasonal_risk = 0.5
        else:  # Winter
            seasonal_risk = 0.2
        
        return seasonal_risk
    
    def _assess_extreme_weather_risk(self, sensor_data: Dict[str, Any]) -> float:
        """Assess risk from extreme weather conditions."""
        risk_factors = []
        
        # High temperature risk
        temp = sensor_data.get('temperature', 25.0)
        if temp > 35.0:
            risk_factors.append(0.9)
        elif temp > 30.0:
            risk_factors.append(0.6)
        
        # Low humidity risk
        humidity = sensor_data.get('humidity', 50.0)
        if humidity < 20.0:
            risk_factors.append(0.9)
        elif humidity < 30.0:
            risk_factors.append(0.6)
        
        # High wind risk
        wind_speed = sensor_data.get('wind_speed', 5.0)
        if wind_speed > 20.0:
            risk_factors.append(0.8)
        elif wind_speed > 15.0:
            risk_factors.append(0.5)
        
        return np.mean(risk_factors) if risk_factors else 0.2


class OperationalRiskModel:
    """AI model for operational risk assessment."""
    
    def __init__(self):
        self.asset_status = {}
        self.operational_history = []
        
    def assess(self, sensor_data: Dict[str, Any], 
               world_model: Dict[str, Any]) -> Dict[str, Any]:
        """Assess operational risk factors."""
        
        # Asset availability risk
        asset_risk = self._assess_asset_risk()
        
        # Communication risk
        communication_risk = self._assess_communication_risk(sensor_data)
        
        # Operational complexity risk
        complexity_risk = self._assess_complexity_risk(world_model)
        
        # Calculate combined risk
        risk_score = (
            asset_risk * 0.4 +
            communication_risk * 0.3 +
            complexity_risk * 0.3
        )
        
        return {
            'score': min(1.0, risk_score),
            'asset_risk': asset_risk,
            'communication_risk': communication_risk,
            'complexity_risk': complexity_risk,
            'confidence': 0.7  # Operational risk is harder to predict
        }
    
    def _assess_asset_risk(self) -> float:
        """Assess risk based on asset availability and status."""
        if not self.asset_status:
            return 0.5  # Default moderate risk
        
        # Check asset availability
        total_assets = len(self.asset_status)
        available_assets = sum(1 for status in self.asset_status.values() 
                              if status.get('status') == 'online')
        
        availability_ratio = available_assets / total_assets if total_assets > 0 else 0.5
        
        # Lower availability = higher risk
        return 1.0 - availability_ratio
    
    def _assess_communication_risk(self, sensor_data: Dict[str, Any]) -> float:
        """Assess communication risk based on signal strength."""
        signal_strength = sensor_data.get('signal_strength', -70.0)
        
        # Signal strength thresholds (dBm)
        if signal_strength > -50:
            return 0.1  # Excellent signal
        elif signal_strength > -70:
            return 0.3  # Good signal
        elif signal_strength > -80:
            return 0.6  # Fair signal
        else:
            return 0.9  # Poor signal
    
    def _assess_complexity_risk(self, world_model: Dict[str, Any]) -> float:
        """Assess risk based on operational complexity."""
        spatial_data = world_model.get('spatial_data', {})
        
        # More active areas = higher complexity
        active_areas = sum(1 for data in spatial_data.values() 
                          if data.get('confidence', 0) > 0.5)
        
        complexity_ratio = active_areas / 10.0  # Normalize to 0-1
        return min(1.0, complexity_ratio)


class AdvisoryGenerator:
    """
    Generates human-readable advisory messages and recommendations
    based on AI reasoning and risk assessment.
    """
    
    def __init__(self):
        self.template_engine = AdvisoryTemplateEngine()
        self.context_analyzer = ContextAnalyzer()
        
    def generate_advisory(self, risk_assessment: Dict[str, Any], 
                         world_model: Dict[str, Any]) -> Dict[str, Any]:
        """Generate contextual advisory message."""
        
        # Analyze context
        context = self.context_analyzer.analyze(risk_assessment, world_model)
        
        # Generate advisory message
        advisory_message = self.template_engine.generate_message(
            risk_assessment, context
        )
        
        # Generate specific recommendations
        recommendations = self._generate_specific_recommendations(
            risk_assessment, context
        )
        
        # Generate priority actions
        priority_actions = self._generate_priority_actions(
            risk_assessment, context
        )
        
        return {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'advisory_message': advisory_message,
            'recommendations': recommendations,
            'priority_actions': priority_actions,
            'context': context,
            'confidence': risk_assessment.get('confidence', 0.5)
        }
    
    def _generate_specific_recommendations(self, risk_assessment: Dict[str, Any], 
                                         context: Dict[str, Any]) -> List[str]:
        """Generate specific, actionable recommendations."""
        recommendations = []
        
        # Fire risk recommendations
        fire_risk = risk_assessment.get('fire_risk', {})
        if fire_risk.get('score', 0) > 0.7:
            recommendations.extend([
                "Deploy thermal monitoring drones to high-risk sectors",
                "Pre-position fire suppression equipment",
                "Establish communication with local fire departments",
                "Prepare evacuation routes and procedures"
            ])
        
        # Environmental risk recommendations
        env_risk = risk_assessment.get('environmental_risk', {})
        if env_risk.get('score', 0) > 0.6:
            recommendations.extend([
                "Monitor weather radar for approaching storms",
                "Secure all loose equipment and materials",
                "Prepare for potential power outages",
                "Establish backup communication protocols"
            ])
        
        # Operational risk recommendations
        op_risk = risk_assessment.get('operational_risk', {})
        if op_risk.get('score', 0) > 0.5:
            recommendations.extend([
                "Conduct equipment status check",
                "Verify communication links with all assets",
                "Review operational procedures",
                "Prepare contingency plans"
            ])
        
        return recommendations
    
    def _generate_priority_actions(self, risk_assessment: Dict[str, Any], 
                                 context: Dict[str, Any]) -> List[str]:
        """Generate priority actions based on risk level."""
        priority_actions = []
        
        overall_risk = risk_assessment.get('overall_risk', 0)
        risk_level = risk_assessment.get('risk_level', 'low')
        
        if risk_level == 'critical':
            priority_actions.extend([
                "IMMEDIATE: Activate emergency response protocols",
                "IMMEDIATE: Notify all personnel and stakeholders",
                "IMMEDIATE: Deploy all available resources",
                "IMMEDIATE: Establish incident command center"
            ])
        elif risk_level == 'high':
            priority_actions.extend([
                "URGENT: Increase monitoring frequency",
                "URGENT: Prepare response resources",
                "URGENT: Brief operations team",
                "URGENT: Update risk assessment"
            ])
        elif risk_level == 'medium':
            priority_actions.extend([
                "Monitor conditions closely",
                "Prepare response plan",
                "Check equipment status",
                "Review procedures"
            ])
        else:
            priority_actions.extend([
                "Continue routine monitoring",
                "Maintain standard protocols",
                "Document observations"
            ])
        
        return priority_actions


class AdvisoryTemplateEngine:
    """Generates human-readable advisory messages using templates."""
    
    def __init__(self):
        self.templates = {
            'fire_risk': {
                'critical': "CRITICAL FIRE RISK: Immediate action required. Fire conditions are extreme with high probability of ignition and rapid spread.",
                'high': "HIGH FIRE RISK: Fire conditions are dangerous. Increased monitoring and preparation required.",
                'medium': "MODERATE FIRE RISK: Fire conditions are elevated. Monitor closely and prepare for potential response.",
                'low': "LOW FIRE RISK: Fire conditions are normal. Continue routine monitoring."
            },
            'environmental_risk': {
                'critical': "CRITICAL ENVIRONMENTAL CONDITIONS: Extreme weather conditions pose significant risk to operations.",
                'high': "HIGH ENVIRONMENTAL RISK: Adverse weather conditions require increased caution.",
                'medium': "MODERATE ENVIRONMENTAL RISK: Weather conditions are changing. Monitor closely.",
                'low': "LOW ENVIRONMENTAL RISK: Weather conditions are stable."
            },
            'operational_risk': {
                'critical': "CRITICAL OPERATIONAL RISK: System integrity compromised. Immediate intervention required.",
                'high': "HIGH OPERATIONAL RISK: System performance degraded. Increased monitoring required.",
                'medium': "MODERATE OPERATIONAL RISK: System performance normal with minor issues.",
                'low': "LOW OPERATIONAL RISK: System performance optimal."
            }
        }
    
    def generate_message(self, risk_assessment: Dict[str, Any], 
                        context: Dict[str, Any]) -> str:
        """Generate advisory message using templates."""
        
        # Get risk levels
        fire_level = self._get_risk_level(risk_assessment.get('fire_risk', {}).get('score', 0))
        env_level = self._get_risk_level(risk_assessment.get('environmental_risk', {}).get('score', 0))
        op_level = self._get_risk_level(risk_assessment.get('operational_risk', {}).get('score', 0))
        
        # Generate messages for each risk type
        messages = []
        
        fire_msg = self.templates['fire_risk'].get(fire_level, "Fire risk assessment unavailable.")
        messages.append(f"FIRE RISK: {fire_msg}")
        
        env_msg = self.templates['environmental_risk'].get(env_level, "Environmental risk assessment unavailable.")
        messages.append(f"ENVIRONMENTAL: {env_msg}")
        
        op_msg = self.templates['operational_risk'].get(op_level, "Operational risk assessment unavailable.")
        messages.append(f"OPERATIONAL: {op_msg}")
        
        # Combine messages
        combined_message = " | ".join(messages)
        
        # Add context-specific information
        if context.get('trend', '') == 'deteriorating':
            combined_message += " | TREND: Conditions are deteriorating."
        elif context.get('trend', '') == 'improving':
            combined_message += " | TREND: Conditions are improving."
        
        return combined_message
    
    def _get_risk_level(self, risk_score: float) -> str:
        """Convert risk score to risk level."""
        if risk_score >= 0.8:
            return 'critical'
        elif risk_score >= 0.6:
            return 'high'
        elif risk_score >= 0.4:
            return 'medium'
        else:
            return 'low'


class ContextAnalyzer:
    """Analyzes context for advisory generation."""
    
    def analyze(self, risk_assessment: Dict[str, Any], 
                world_model: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze context for advisory generation."""
        
        # Analyze trends
        trend = self._analyze_trends(world_model)
        
        # Analyze spatial distribution
        spatial_distribution = self._analyze_spatial_distribution(world_model)
        
        # Analyze temporal patterns
        temporal_patterns = self._analyze_temporal_patterns(world_model)
        
        return {
            'trend': trend,
            'spatial_distribution': spatial_distribution,
            'temporal_patterns': temporal_patterns,
            'overall_context': self._synthesize_context(trend, spatial_distribution, temporal_patterns)
        }
    
    def _analyze_trends(self, world_model: Dict[str, Any]) -> str:
        """Analyze trends in the data."""
        temporal_trends = world_model.get('temporal_trends', {})
        
        if temporal_trends.get('fire_detection_trend') == 'increasing':
            return 'deteriorating'
        elif temporal_trends.get('fire_detection_trend') == 'decreasing':
            return 'improving'
        else:
            return 'stable'
    
    def _analyze_spatial_distribution(self, world_model: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze spatial distribution of risks."""
        spatial_data = world_model.get('spatial_data', {})
        
        if not spatial_data:
            return {'distribution': 'unknown', 'concentration': 'low'}
        
        # Count high-risk areas
        high_risk_areas = sum(1 for data in spatial_data.values() 
                             if data.get('confidence', 0) > 0.7)
        
        total_areas = len(spatial_data)
        risk_concentration = high_risk_areas / total_areas if total_areas > 0 else 0
        
        return {
            'distribution': 'clustered' if risk_concentration > 0.5 else 'dispersed',
            'concentration': 'high' if risk_concentration > 0.7 else 'low'
        }
    
    def _analyze_temporal_patterns(self, world_model: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze temporal patterns in the data."""
        temporal_trends = world_model.get('temporal_trends', {})
        
        return {
            'temperature_trend': temporal_trends.get('temperature_trend', 'unknown'),
            'fire_detection_trend': temporal_trends.get('fire_detection_trend', 'unknown'),
            'data_points': temporal_trends.get('data_points', 0)
        }
    
    def _synthesize_context(self, trend: str, spatial_distribution: Dict[str, Any], 
                           temporal_patterns: Dict[str, Any]) -> str:
        """Synthesize overall context from all factors."""
        if trend == 'deteriorating' and spatial_distribution['concentration'] == 'high':
            return 'critical_situation'
        elif trend == 'deteriorating' or spatial_distribution['concentration'] == 'high':
            return 'elevated_concern'
        elif trend == 'improving':
            return 'improving_situation'
        else:
            return 'normal_operations'
