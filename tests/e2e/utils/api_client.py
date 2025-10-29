import httpx
from typing import Any, Dict

class ApiClient:
    def __init__(self, base_url: str) -> None:
        self.base = base_url.rstrip("/")
        self._client = httpx.Client(timeout=10.0)

    def get(self, path: str, **kwargs) -> httpx.Response:
        url = path if path.startswith("http") else f"{self.base}{path}"
        return self._client.get(url, **kwargs)

    def post(self, path: str, json: Dict[str, Any] | None = None, **kwargs) -> httpx.Response:
        url = path if path.startswith("http") else f"{self.base}{path}"
        return self._client.post(url, json=json, **kwargs)

    def close(self):
        self._client.close()
