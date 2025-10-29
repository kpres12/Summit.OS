import httpx
import os
from typing import Any, Dict, Optional

DEFAULT_OPA_URL = os.getenv("OPA_URL", "http://opa:8181")

class OPAClient:
    def __init__(self, base_url: str | None = None, policy_path: str = "/v1/data/policy") -> None:
        self.base_url = base_url or DEFAULT_OPA_URL
        self.policy_path = policy_path

    async def evaluate(self, rule: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate an OPA policy rule.

        Args:
            rule: e.g., "missions/allow"
            input_data: input JSON for policy
        Returns: result dict; expected keys: allow(bool), deny_reasons(list[str])
        """
        url = f"{self.base_url}{self.policy_path}/{rule.lstrip('/')}"
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                r = await client.post(url, json={"input": input_data})
                r.raise_for_status()
                data = r.json() or {}
                # OPA returns {"result": {...}} or a raw value
                if isinstance(data, dict) and "result" in data:
                    return data["result"] or {}
                return data
        except Exception:
            # Fail open with empty result
            return {"allow": True, "deny_reasons": []}
