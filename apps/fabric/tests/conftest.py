import os
import sys

# Must set env BEFORE importing app
os.environ["FABRIC_TEST_MODE"] = "true"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from fastapi.testclient import TestClient
from main import app


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c
