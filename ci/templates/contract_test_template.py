"""
Contract test template for Summit Sentinel Platform
- Validates sample payloads against canonical JSON Schemas
- Optionally validates live API responses if API_BASE_URL is set
Usage:
  pip install -r ci/templates/requirements.txt
  pytest ci/templates/contract_test_template.py -q
"""
from __future__ import annotations
import json
import os
from pathlib import Path

import pytest
import requests
from jsonschema import Draft202012Validator, RefResolver

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_DIR = ROOT / "docs/platform/schemas"
EXAMPLES_DIR = ROOT / "docs/platform/examples"
API_BASE_URL = os.getenv("API_BASE_URL")  # e.g., http://localhost:8000

SCHEMAS = {
    "detection_event": SCHEMA_DIR / "detection_event.json",
    "track": SCHEMA_DIR / "track.json",
    "mission_intent": SCHEMA_DIR / "mission_intent.json",
    "task_assignment": SCHEMA_DIR / "task_assignment.json",
    "vehicle_telemetry": SCHEMA_DIR / "vehicle_telemetry.json",
    "action_ack": SCHEMA_DIR / "action_ack.json",
}

EXAMPLES = {
    name: EXAMPLES_DIR / f"{name}.json" for name in SCHEMAS.keys()
}

@pytest.fixture(scope="session")
def resolver():
    # Allow $ref to common.json via absolute/remote IDs used in schemas
    store = {}
    for p in SCHEMA_DIR.glob("*.json"):
        with p.open() as f:
            data = json.load(f)
            if "$id" in data:
                store[data["$id"]] = data
    return RefResolver.from_schema(store.get("https://bigmt.dev/schemas/common.json", {}), store=store)


def load_json(path: Path):
    with path.open() as f:
        return json.load(f)


@pytest.mark.parametrize("name", list(SCHEMAS.keys()))
def test_examples_conform_to_schema(name: str, resolver):
    schema = load_json(SCHEMAS[name])
    validator = Draft202012Validator(schema, resolver=resolver)
    example = load_json(EXAMPLES[name])
    errors = sorted(validator.iter_errors(example), key=lambda e: e.path)
    assert not errors, f"Schema validation failed for {name}:\n" + "\n".join(str(e) for e in errors)


@pytest.mark.skipif(not API_BASE_URL, reason="API_BASE_URL not set")
@pytest.mark.parametrize(
    "name,endpoint",
    [
        ("detection_event", "/contracts/example/detection_event"),
        ("track", "/contracts/example/track"),
        ("mission_intent", "/contracts/example/mission_intent"),
        ("task_assignment", "/contracts/example/task_assignment"),
        ("vehicle_telemetry", "/contracts/example/vehicle_telemetry"),
        ("action_ack", "/contracts/example/action_ack"),
    ],
)
def test_live_contract_examples(name: str, endpoint: str, resolver):
    """
    Optional: API gateway exposes contract example endpoints for CI validation.
    Endpoint should return a minimal valid JSON instance of the schema.
    """
    url = API_BASE_URL.rstrip("/") + endpoint
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    payload = resp.json()
    schema = load_json(SCHEMAS[name])
    validator = Draft202012Validator(schema, resolver=resolver)
    errors = sorted(validator.iter_errors(payload), key=lambda e: e.path)
    assert not errors, f"Live schema validation failed for {name} at {url}:\n" + "\n".join(str(e) for e in errors)
