import json
import os
from jsonschema import validate


def test_observation_schema_valid():
    schema_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "jsonschemas", "observation.schema.json"))
    with open(schema_path, 'r') as f:
        schema = json.load(f)
    sample = {
        "class": "smoke",
        "confidence": 0.92,
        "ts_iso": "2024-01-01T00:00:00Z",
        "lat": 37.0,
        "lon": -122.0,
        "attributes": {"bbox": [0,0,100,100]}
    }
    validate(instance=sample, schema=schema)
