"""
Import shim for the API Gateway app.

The gateway lives at apps/api-gateway/main.py but the hyphenated path
can't be imported directly. This module re-exports the FastAPI `app`.
"""

import importlib.util
import os

_gateway_path = os.path.join(os.path.dirname(__file__), "api-gateway", "main.py")
_spec = importlib.util.spec_from_file_location("api_gateway_main", _gateway_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

app = _mod.app
