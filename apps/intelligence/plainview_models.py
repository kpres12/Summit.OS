"""
Plainview-specific AI models for Intelligence service.

Includes:
- Flow anomaly detection
- Valve health prediction
- Pipeline integrity assessment
"""

import os
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone

try:
    import numpy as np
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

MODEL_REGISTRY = os.getenv("MODEL_REGISTRY", "/models")


class FlowAnomalyDetector:
    """Detect anomalies in flow, pressure, and temperature metrics."""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.baseline = {
            "flow_rate_lpm": 150.0,
            "pressure_pa": 2500000.0,
            "temperature_c": 45.0
        }
        
        if ONNX_AVAILABLE and model_path and os.path.exists(model_path):
            try:
                self.model = ort.InferenceSession(model_path)
            except Exception as e:
                print(f"Failed to load flow anomaly model: {e}")
    
    def detect(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Detect anomalies in flow metrics.
        
        Args:
            metrics: {"flow_rate_lpm": float, "pressure_pa": float, "temperature_c": float}
        
        Returns:
            {
                "is_anomaly": bool,
                "confidence": float,
                "anomaly_type": str,
                "severity": str,
                "details": dict
            }
        """
        if self.model:
            return self._detect_ml(metrics)
        else:
            return self._detect_rules(metrics)
    
    def _detect_ml(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """ML-based anomaly detection using ONNX model."""
        try:
            # Prepare features: [flow_rate, pressure, temperature, hour_of_day]
            hour = datetime.now(timezone.utc).hour
            features = np.array([[
                metrics.get("flow_rate_lpm", 0.0),
                metrics.get("pressure_pa", 0.0),
                metrics.get("temperature_c", 0.0),
                float(hour)
            ]], dtype=np.float32)
            
            # Run inference
            input_name = self.model.get_inputs()[0].name
            outputs = self.model.run(None, {input_name: features})
            
            # Assume model outputs [anomaly_score, anomaly_type_idx]
            anomaly_score = float(outputs[0][0])
            is_anomaly = anomaly_score > 0.5
            
            anomaly_types = ["normal", "flow_deviation", "pressure_spike", "temperature_anomaly"]
            anomaly_type = anomaly_types[int(outputs[1][0])] if len(outputs) > 1 else "unknown"
            
            severity = "high" if anomaly_score > 0.8 else "medium" if anomaly_score > 0.5 else "low"
            
            return {
                "is_anomaly": is_anomaly,
                "confidence": anomaly_score,
                "anomaly_type": anomaly_type,
                "severity": severity,
                "details": {"ml_score": anomaly_score}
            }
        except Exception as e:
            print(f"ML detection failed: {e}, falling back to rules")
            return self._detect_rules(metrics)
    
    def _detect_rules(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Rule-based anomaly detection as fallback."""
        flow = metrics.get("flow_rate_lpm", 0.0)
        pressure = metrics.get("pressure_pa", 0.0)
        temperature = metrics.get("temperature_c", 0.0)
        
        anomalies = []
        max_confidence = 0.0
        
        # Flow rate check
        flow_dev = abs(flow - self.baseline["flow_rate_lpm"]) / self.baseline["flow_rate_lpm"]
        if flow_dev > 0.25:
            confidence = min(flow_dev, 1.0)
            anomalies.append({
                "type": "flow_deviation",
                "confidence": confidence,
                "severity": "high" if flow_dev > 0.5 else "medium"
            })
            max_confidence = max(max_confidence, confidence)
        
        # Pressure check
        pressure_dev = abs(pressure - self.baseline["pressure_pa"])
        if pressure_dev > 100000:
            confidence = min(pressure_dev / 500000, 1.0)
            anomalies.append({
                "type": "pressure_spike",
                "confidence": confidence,
                "severity": "high" if pressure_dev > 200000 else "medium"
            })
            max_confidence = max(max_confidence, confidence)
        
        # Temperature check
        temp_dev = abs(temperature - self.baseline["temperature_c"])
        if temp_dev > 10:
            confidence = min(temp_dev / 30, 1.0)
            anomalies.append({
                "type": "temperature_anomaly",
                "confidence": confidence,
                "severity": "high" if temp_dev > 20 else "medium"
            })
            max_confidence = max(max_confidence, confidence)
        
        if anomalies:
            primary = max(anomalies, key=lambda x: x["confidence"])
            return {
                "is_anomaly": True,
                "confidence": primary["confidence"],
                "anomaly_type": primary["type"],
                "severity": primary["severity"],
                "details": {"all_anomalies": anomalies}
            }
        else:
            return {
                "is_anomaly": False,
                "confidence": 0.0,
                "anomaly_type": "normal",
                "severity": "low",
                "details": {}
            }


class ValveHealthPredictor:
    """Predict valve health and maintenance needs."""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        if ONNX_AVAILABLE and model_path and os.path.exists(model_path):
            try:
                self.model = ort.InferenceSession(model_path)
            except Exception as e:
                print(f"Failed to load valve health model: {e}")
    
    def predict_health(self, valve_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict valve health score and maintenance window.
        
        Args:
            valve_data: {
                "cycles_count": int,
                "last_torque_nm": float,
                "age_days": int,
                "avg_temp_c": float
            }
        
        Returns:
            {
                "health_score": float,  # 0-100
                "maintenance_days": int,  # days until maintenance needed
                "risk_level": str,
                "recommendations": list[str]
            }
        """
        if self.model:
            return self._predict_ml(valve_data)
        else:
            return self._predict_rules(valve_data)
    
    def _predict_ml(self, valve_data: Dict[str, Any]) -> Dict[str, Any]:
        """ML-based health prediction."""
        try:
            features = np.array([[
                float(valve_data.get("cycles_count", 0)),
                float(valve_data.get("last_torque_nm", 0)),
                float(valve_data.get("age_days", 0)),
                float(valve_data.get("avg_temp_c", 0))
            ]], dtype=np.float32)
            
            input_name = self.model.get_inputs()[0].name
            outputs = self.model.run(None, {input_name: features})
            
            health_score = float(outputs[0][0]) * 100  # Normalize to 0-100
            maintenance_days = int(outputs[1][0]) if len(outputs) > 1 else 30
            
            risk_level = "critical" if health_score < 30 else "high" if health_score < 60 else "medium" if health_score < 80 else "low"
            
            recommendations = self._generate_recommendations(health_score, maintenance_days)
            
            return {
                "health_score": health_score,
                "maintenance_days": maintenance_days,
                "risk_level": risk_level,
                "recommendations": recommendations
            }
        except Exception as e:
            print(f"ML prediction failed: {e}, falling back to rules")
            return self._predict_rules(valve_data)
    
    def _predict_rules(self, valve_data: Dict[str, Any]) -> Dict[str, Any]:
        """Rule-based health prediction."""
        cycles = valve_data.get("cycles_count", 0)
        torque = valve_data.get("last_torque_nm", 50.0)
        age_days = valve_data.get("age_days", 0)
        
        # Simple scoring
        health_score = 100.0
        
        # Degrade based on cycles (assume 10k cycles is lifecycle)
        if cycles > 10000:
            health_score -= 50
        elif cycles > 5000:
            health_score -= 30
        elif cycles > 2000:
            health_score -= 15
        
        # Degrade based on age (5 years = 1825 days)
        if age_days > 1825:
            health_score -= 30
        elif age_days > 1000:
            health_score -= 15
        
        # Torque anomalies
        if torque < 30 or torque > 70:
            health_score -= 20
        
        health_score = max(0, health_score)
        
        # Estimate maintenance window
        if health_score < 40:
            maintenance_days = 7
        elif health_score < 60:
            maintenance_days = 30
        elif health_score < 80:
            maintenance_days = 90
        else:
            maintenance_days = 180
        
        risk_level = "critical" if health_score < 30 else "high" if health_score < 60 else "medium" if health_score < 80 else "low"
        
        recommendations = self._generate_recommendations(health_score, maintenance_days)
        
        return {
            "health_score": health_score,
            "maintenance_days": maintenance_days,
            "risk_level": risk_level,
            "recommendations": recommendations
        }
    
    def _generate_recommendations(self, health_score: float, maintenance_days: int) -> List[str]:
        """Generate actionable recommendations."""
        recs = []
        
        if health_score < 40:
            recs.append("URGENT: Schedule immediate inspection")
            recs.append("Reduce operational load on valve")
        elif health_score < 60:
            recs.append("Schedule maintenance within 30 days")
            recs.append("Increase monitoring frequency")
        elif health_score < 80:
            recs.append("Plan preventive maintenance")
            recs.append("Monitor torque readings weekly")
        else:
            recs.append("Valve health is good")
            recs.append("Continue routine monitoring")
        
        return recs


class PipelineIntegrityAssessor:
    """Assess pipeline integrity and leak risk."""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        if ONNX_AVAILABLE and model_path and os.path.exists(model_path):
            try:
                self.model = ort.InferenceSession(model_path)
            except Exception as e:
                print(f"Failed to load pipeline model: {e}")
    
    def assess(self, pipeline_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess pipeline integrity.
        
        Args:
            pipeline_data: {
                "section_id": str,
                "pressure_variance": float,
                "flow_consistency": float,
                "age_years": float,
                "inspection_score": float
            }
        
        Returns:
            {
                "integrity_score": float,  # 0-100
                "leak_probability": float,  # 0-1
                "risk_level": str,
                "action_required": str
            }
        """
        if self.model:
            return self._assess_ml(pipeline_data)
        else:
            return self._assess_rules(pipeline_data)
    
    def _assess_ml(self, pipeline_data: Dict[str, Any]) -> Dict[str, Any]:
        """ML-based integrity assessment."""
        try:
            features = np.array([[
                float(pipeline_data.get("pressure_variance", 0)),
                float(pipeline_data.get("flow_consistency", 1.0)),
                float(pipeline_data.get("age_years", 0)),
                float(pipeline_data.get("inspection_score", 100))
            ]], dtype=np.float32)
            
            input_name = self.model.get_inputs()[0].name
            outputs = self.model.run(None, {input_name: features})
            
            integrity_score = float(outputs[0][0]) * 100
            leak_probability = float(outputs[1][0]) if len(outputs) > 1 else (100 - integrity_score) / 100
            
            risk_level = "critical" if leak_probability > 0.7 else "high" if leak_probability > 0.5 else "medium" if leak_probability > 0.3 else "low"
            
            action = self._determine_action(integrity_score, leak_probability)
            
            return {
                "integrity_score": integrity_score,
                "leak_probability": leak_probability,
                "risk_level": risk_level,
                "action_required": action
            }
        except Exception as e:
            print(f"ML assessment failed: {e}, falling back to rules")
            return self._assess_rules(pipeline_data)
    
    def _assess_rules(self, pipeline_data: Dict[str, Any]) -> Dict[str, Any]:
        """Rule-based integrity assessment."""
        pressure_var = pipeline_data.get("pressure_variance", 0.0)
        flow_consistency = pipeline_data.get("flow_consistency", 1.0)
        age_years = pipeline_data.get("age_years", 0.0)
        inspection_score = pipeline_data.get("inspection_score", 100.0)
        
        # Calculate integrity score
        integrity = inspection_score
        
        # Pressure variance penalty
        if pressure_var > 0.2:
            integrity -= 30
        elif pressure_var > 0.1:
            integrity -= 15
        
        # Flow consistency bonus/penalty
        if flow_consistency < 0.8:
            integrity -= 20
        
        # Age penalty
        if age_years > 20:
            integrity -= 25
        elif age_years > 10:
            integrity -= 10
        
        integrity = max(0, min(100, integrity))
        
        # Leak probability
        leak_prob = (100 - integrity) / 100 * 0.8  # Scale to 0-0.8
        
        risk_level = "critical" if leak_prob > 0.6 else "high" if leak_prob > 0.4 else "medium" if leak_prob > 0.2 else "low"
        
        action = self._determine_action(integrity, leak_prob)
        
        return {
            "integrity_score": integrity,
            "leak_probability": leak_prob,
            "risk_level": risk_level,
            "action_required": action
        }
    
    def _determine_action(self, integrity: float, leak_prob: float) -> str:
        """Determine required action based on assessment."""
        if leak_prob > 0.6:
            return "IMMEDIATE: Deploy inspection team and reduce pressure"
        elif leak_prob > 0.4:
            return "Deploy drone inspection within 24 hours"
        elif leak_prob > 0.2:
            return "Schedule inspection within 7 days"
        else:
            return "Continue routine monitoring"


# Singleton instances
_flow_detector: Optional[FlowAnomalyDetector] = None
_valve_predictor: Optional[ValveHealthPredictor] = None
_pipeline_assessor: Optional[PipelineIntegrityAssessor] = None


def get_flow_detector() -> FlowAnomalyDetector:
    """Get or create FlowAnomalyDetector instance."""
    global _flow_detector
    if _flow_detector is None:
        model_path = os.path.join(MODEL_REGISTRY, "flow_anomaly.onnx")
        _flow_detector = FlowAnomalyDetector(model_path if os.path.exists(model_path) else None)
    return _flow_detector


def get_valve_predictor() -> ValveHealthPredictor:
    """Get or create ValveHealthPredictor instance."""
    global _valve_predictor
    if _valve_predictor is None:
        model_path = os.path.join(MODEL_REGISTRY, "valve_health.onnx")
        _valve_predictor = ValveHealthPredictor(model_path if os.path.exists(model_path) else None)
    return _valve_predictor


def get_pipeline_assessor() -> PipelineIntegrityAssessor:
    """Get or create PipelineIntegrityAssessor instance."""
    global _pipeline_assessor
    if _pipeline_assessor is None:
        model_path = os.path.join(MODEL_REGISTRY, "pipeline_integrity.onnx")
        _pipeline_assessor = PipelineIntegrityAssessor(model_path if os.path.exists(model_path) else None)
    return _pipeline_assessor
