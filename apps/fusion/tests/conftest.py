import os
import sys

# Disable fusion startup side-effects (MQTT/DB) during unit tests
os.environ["FUSION_DISABLE_STARTUP"] = "1"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from fastapi.testclient import TestClient
from main import app


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c
