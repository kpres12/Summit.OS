import os
import sys

os.environ["TASKING_TEST_MODE"] = "true"
os.environ["PYTHONPATH"] = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
)

import pytest
from fastapi.testclient import TestClient
from main import app


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c
