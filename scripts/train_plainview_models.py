"""
Train mock ONNX models for Plainview domain.

This script creates simple synthetic models for:
- Flow anomaly detection
- Valve health prediction
- Pipeline integrity assessment

In production, replace with real training data and proper ML pipelines.
"""

import os
import numpy as np

try:
    import onnx
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    DEPS_AVAILABLE = True
except ImportError:
    print("ERROR: Required dependencies not installed")
    print("Install: pip install onnx scikit-learn skl2onnx")
    DEPS_AVAILABLE = False
    exit(1)


def train_flow_anomaly_model(output_path: str):
    """
    Train flow anomaly detection model.
    
    Input features: [flow_rate, pressure, temperature, hour_of_day]
    Output: anomaly_score (0-1)
    """
    print("Training flow anomaly model...")
    
    # Generate synthetic training data
    np.random.seed(42)
    n_samples = 1000
    
    # Normal samples
    normal_flow = np.random.normal(150, 10, n_samples // 2)
    normal_pressure = np.random.normal(2500000, 50000, n_samples // 2)
    normal_temp = np.random.normal(45, 3, n_samples // 2)
    normal_hour = np.random.randint(0, 24, n_samples // 2)
    normal_labels = np.zeros(n_samples // 2)
    
    # Anomalous samples
    anom_flow = np.concatenate([
        np.random.normal(100, 15, n_samples // 4),  # Low flow
        np.random.normal(200, 15, n_samples // 4),   # High flow
    ])
    anom_pressure = np.concatenate([
        np.random.normal(2300000, 100000, n_samples // 4),
        np.random.normal(2700000, 100000, n_samples // 4),
    ])
    anom_temp = np.concatenate([
        np.random.normal(30, 5, n_samples // 4),
        np.random.normal(60, 5, n_samples // 4),
    ])
    anom_hour = np.random.randint(0, 24, n_samples // 2)
    anom_labels = np.ones(n_samples // 2)
    
    # Combine
    X = np.column_stack([
        np.concatenate([normal_flow, anom_flow]),
        np.concatenate([normal_pressure, anom_pressure]),
        np.concatenate([normal_temp, anom_temp]),
        np.concatenate([normal_hour, anom_hour]),
    ])
    y = np.concatenate([normal_labels, anom_labels])
    
    # Train model
    model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
    model.fit(X, y)
    
    # Convert to ONNX
    initial_type = [('float_input', FloatTensorType([None, 4]))]
    onnx_model = convert_sklearn(model, initial_types=initial_type, target_opset=12)
    
    # Save
    onnx.save(onnx_model, output_path)
    print(f"✓ Flow anomaly model saved to {output_path}")


def train_valve_health_model(output_path: str):
    """
    Train valve health prediction model.
    
    Input features: [cycles_count, last_torque_nm, age_days, avg_temp_c]
    Output: health_score (0-1), maintenance_days
    """
    print("Training valve health model...")
    
    np.random.seed(42)
    n_samples = 800
    
    # Generate synthetic data
    cycles = np.random.randint(0, 15000, n_samples)
    torque = np.random.normal(50, 10, n_samples)
    age_days = np.random.randint(0, 2000, n_samples)
    temp = np.random.normal(45, 10, n_samples)
    
    # Synthetic health score (inversely proportional to cycles and age)
    health = 100 - (cycles / 150) - (age_days / 20) - np.abs(torque - 50) * 2
    health = np.clip(health / 100, 0, 1)
    
    X = np.column_stack([cycles, torque, age_days, temp])
    
    # Train model (regression for health score)
    model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
    model.fit(X, health)
    
    # Convert to ONNX
    initial_type = [('float_input', FloatTensorType([None, 4]))]
    onnx_model = convert_sklearn(model, initial_types=initial_type, target_opset=12)
    
    # Save
    onnx.save(onnx_model, output_path)
    print(f"✓ Valve health model saved to {output_path}")


def train_pipeline_integrity_model(output_path: str):
    """
    Train pipeline integrity assessment model.
    
    Input features: [pressure_variance, flow_consistency, age_years, inspection_score]
    Output: integrity_score (0-1)
    """
    print("Training pipeline integrity model...")
    
    np.random.seed(42)
    n_samples = 600
    
    # Generate synthetic data
    pressure_var = np.random.uniform(0, 0.5, n_samples)
    flow_cons = np.random.uniform(0.5, 1.0, n_samples)
    age_years = np.random.uniform(0, 30, n_samples)
    inspection = np.random.uniform(60, 100, n_samples)
    
    # Synthetic integrity (higher is better)
    integrity = inspection / 100 - pressure_var * 0.5 + (flow_cons - 0.5) * 0.3 - (age_years / 30) * 0.2
    integrity = np.clip(integrity, 0, 1)
    
    X = np.column_stack([pressure_var, flow_cons, age_years, inspection])
    
    # Train model
    model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
    model.fit(X, integrity)
    
    # Convert to ONNX
    initial_type = [('float_input', FloatTensorType([None, 4]))]
    onnx_model = convert_sklearn(model, initial_types=initial_type, target_opset=12)
    
    # Save
    onnx.save(onnx_model, output_path)
    print(f"✓ Pipeline integrity model saved to {output_path}")


if __name__ == "__main__":
    # Create models directory
    models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    os.makedirs(models_dir, exist_ok=True)
    
    print("=" * 60)
    print("Training Plainview ONNX Models")
    print("=" * 60)
    print()
    
    # Train all models
    train_flow_anomaly_model(os.path.join(models_dir, "flow_anomaly.onnx"))
    train_valve_health_model(os.path.join(models_dir, "valve_health.onnx"))
    train_pipeline_integrity_model(os.path.join(models_dir, "pipeline_integrity.onnx"))
    
    print()
    print("=" * 60)
    print("✓ All models trained successfully")
    print(f"Models saved to: {models_dir}")
    print("=" * 60)
