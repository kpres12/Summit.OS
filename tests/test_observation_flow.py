"""
Integration test for end-to-end observation flow:
MQTT → Fabric → Redis Stream → Fusion → Postgres → API Gateway
"""
import json
import time
import pytest
import paho.mqtt.client as mqtt
import httpx
from datetime import datetime, timezone


def test_observation_end_to_end():
    """
    Test the complete data path:
    1. Publish observation to MQTT
    2. Fabric forwards to Redis Stream
    3. Fusion consumes, validates, persists to Postgres
    4. Query via API Gateway
    5. Assert data matches
    """
    # Sample observation
    observation = {
        "class": "smoke",
        "ts_iso": datetime.now(timezone.utc).isoformat(),
        "confidence": 0.87,
        "lat": 37.7749,
        "lon": -122.4194,
        "source": "test-integration"
    }
    
    # Step 1: Publish to MQTT
    client = mqtt.Client()
    client.connect("localhost", 1883, 60)
    client.loop_start()
    result = client.publish("observations/smoke", json.dumps(observation), qos=1)
    result.wait_for_publish()
    client.loop_stop()
    client.disconnect()
    
    # Step 2-3: Wait for processing (Fabric → Redis → Fusion → DB)
    time.sleep(3)
    
    # Step 4: Query via API Gateway
    response = httpx.get("http://localhost:8000/v1/observations?cls=smoke&limit=10")
    assert response.status_code == 200
    
    data = response.json()
    observations = data.get("observations", [])
    
    # Step 5: Assert
    assert len(observations) > 0, "No observations found"
    
    # Find our test observation
    test_obs = next((o for o in observations if o.get("source") == "test-integration"), None)
    assert test_obs is not None, "Test observation not found"
    assert test_obs["cls"] == "smoke"
    assert test_obs["confidence"] == 0.87
    assert abs(test_obs["lat"] - 37.7749) < 0.001
    assert abs(test_obs["lon"] + 122.4194) < 0.001


def test_legacy_detection_topic():
    """Test backward compatibility with detections/* topics."""
    observation = {
        "ts_iso": datetime.now(timezone.utc).isoformat(),
        "confidence": 0.75,
        "lat": 37.8,
        "lon": -122.5,
        "bbox_xywh": [100, 200, 50, 60],
        "source": "test-legacy"
    }
    
    # Publish to legacy topic (class should be derived from topic)
    client = mqtt.Client()
    client.connect("localhost", 1883, 60)
    client.loop_start()
    result = client.publish("detections/smoke", json.dumps(observation), qos=1)
    result.wait_for_publish()
    client.loop_stop()
    client.disconnect()
    
    time.sleep(3)
    
    # Query
    response = httpx.get("http://localhost:8000/v1/observations?cls=smoke&limit=10")
    assert response.status_code == 200
    
    data = response.json()
    observations = data.get("observations", [])
    
    # Find legacy observation
    legacy_obs = next((o for o in observations if o.get("source") == "test-legacy"), None)
    assert legacy_obs is not None, "Legacy observation not found"
    assert legacy_obs["cls"] == "smoke"  # Class derived from topic


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
