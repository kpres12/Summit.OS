"""
Summit.OS Physical Tool Definitions — LLM callable actions.

These are the "hands" of the AI brain. Every physical action goes through
the OPA safety gate before hardware dispatch. The LLM calls these by name
and the agent runtime executes them against the tasking/fabric services.

Tool schema follows the Ollama/OpenAI function-calling format so it can be
parsed by brain.py and sent to the local Llama model.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger("summit.intelligence.tools")


# ── Tool definitions (sent to LLM as function schemas) ───────────────────────

TOOL_DEFINITIONS: List[Dict] = [
    {
        "type": "function",
        "function": {
            "name": "deploy_asset",
            "description": (
                "Dispatch an available asset (drone, ground robot) to a mission task. "
                "Specify what you want done and where — the assignment engine will pick "
                "the best-fit asset and route it. All dispatches pass OPA safety checks."
            ),
            "parameters": {
                "type": "object",
                "required": ["task_type", "lat", "lon"],
                "properties": {
                    "task_type": {
                        "type": "string",
                        "enum": ["SURVEY", "MONITOR", "SEARCH", "DELIVER", "PERIMETER", "ORBIT"],
                        "description": "Type of task to assign",
                    },
                    "lat": {"type": "number", "description": "Target latitude"},
                    "lon": {"type": "number", "description": "Target longitude"},
                    "altitude_m": {"type": "number", "description": "Target altitude in metres (optional)"},
                    "asset_id": {"type": "string", "description": "Specific asset ID to use (optional — omit to let engine choose)"},
                    "priority": {"type": "integer", "minimum": 1, "maximum": 5, "description": "Task priority 1-5 (5=highest)"},
                    "radius_m": {"type": "number", "description": "Area radius in metres (for SURVEY/PERIMETER tasks)"},
                    "reason": {"type": "string", "description": "Brief rationale for this dispatch"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_alert",
            "description": "Raise an alert for operator attention. Use for anomalies, threats, or situations requiring human decision.",
            "parameters": {
                "type": "object",
                "required": ["severity", "description"],
                "properties": {
                    "severity": {
                        "type": "string",
                        "enum": ["INFO", "WARNING", "CRITICAL"],
                    },
                    "description": {"type": "string", "description": "Clear description of the situation"},
                    "entity_id": {"type": "string", "description": "Entity this alert is about (optional)"},
                    "lat": {"type": "number"},
                    "lon": {"type": "number"},
                    "recommended_action": {"type": "string", "description": "What you recommend the operator do"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_geofence",
            "description": "Define a geographic boundary — exclusion zone, inclusion zone, or alert trigger area.",
            "parameters": {
                "type": "object",
                "required": ["name", "fence_type", "center_lat", "center_lon", "radius_m"],
                "properties": {
                    "name": {"type": "string"},
                    "fence_type": {
                        "type": "string",
                        "enum": ["EXCLUSION", "INCLUSION", "ALERT"],
                        "description": "EXCLUSION = keep assets out. INCLUSION = keep assets in. ALERT = trigger alert on entry/exit.",
                    },
                    "center_lat": {"type": "number"},
                    "center_lon": {"type": "number"},
                    "radius_m": {"type": "number"},
                    "max_altitude_m": {"type": "number", "description": "Altitude ceiling (optional)"},
                    "reason": {"type": "string"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_command",
            "description": "Send a direct command to a specific asset (drone, robot, actuator).",
            "parameters": {
                "type": "object",
                "required": ["asset_id", "action"],
                "properties": {
                    "asset_id": {"type": "string"},
                    "action": {
                        "type": "string",
                        "enum": ["RETURN_HOME", "LAND", "HOVER", "EMERGENCY_STOP", "RESUME", "GOTO"],
                        "description": "Command action to send",
                    },
                    "lat": {"type": "number", "description": "Required for GOTO"},
                    "lon": {"type": "number", "description": "Required for GOTO"},
                    "altitude_m": {"type": "number", "description": "Required for GOTO"},
                    "reason": {"type": "string"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "open_actuator",
            "description": (
                "Actuate an industrial device (valve, pump, relay). "
                "Requires OPA pre-flight safety check to pass before execution."
            ),
            "parameters": {
                "type": "object",
                "required": ["device_id", "action"],
                "properties": {
                    "device_id": {"type": "string", "description": "Modbus/OPC-UA device entity ID"},
                    "action": {"type": "string", "enum": ["OPEN", "CLOSE", "SET", "RESET"]},
                    "value": {"type": "number", "description": "Setpoint value (for SET action)"},
                    "reason": {"type": "string"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_world",
            "description": "Query the world model for entities matching a filter. Use to check current state before acting.",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_type": {
                        "type": "string",
                        "enum": ["ASSET", "TRACK", "ALERT", "MISSION", "SENSOR", "GEOFENCE"],
                    },
                    "state": {
                        "type": "string",
                        "enum": ["ACTIVE", "WARNING", "CRITICAL", "INACTIVE", "LOST"],
                    },
                    "domain": {"type": "string", "enum": ["AERIAL", "GROUND", "MARITIME", "SPACE"]},
                    "near_lat": {"type": "number", "description": "Filter by proximity — centre lat"},
                    "near_lon": {"type": "number", "description": "Filter by proximity — centre lon"},
                    "radius_m": {"type": "number", "description": "Proximity radius in metres"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "request_human_input",
            "description": (
                "Surface a decision to a human operator when you are uncertain, "
                "when the situation requires authorisation, or when policy mandates human approval. "
                "Execution pauses until the operator responds."
            ),
            "parameters": {
                "type": "object",
                "required": ["question", "context"],
                "properties": {
                    "question": {"type": "string", "description": "What you need the human to decide"},
                    "context": {"type": "string", "description": "Situation summary for the operator"},
                    "options": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Suggested response options (optional)",
                    },
                    "urgency": {"type": "string", "enum": ["LOW", "MEDIUM", "HIGH", "CRITICAL"]},
                },
            },
        },
    },
]


# ── Tool executor ─────────────────────────────────────────────────────────────

class ToolExecutor:
    """
    Dispatches LLM-requested tool calls to the appropriate Summit.OS services.
    All state-changing tools route through the OPA safety gate.
    """

    def __init__(
        self,
        tasking_url: str = "http://localhost:8004",
        fabric_url: str = "http://localhost:8001",
        gateway_url: str = "http://localhost:8000",
    ):
        self.tasking_url = tasking_url.rstrip("/")
        self.fabric_url = fabric_url.rstrip("/")
        self.gateway_url = gateway_url.rstrip("/")
        self._pending_human_inputs: List[Dict] = []

    async def execute(self, tool_name: str, args: Dict) -> Dict[str, Any]:
        """Execute a tool call. Returns a result dict with 'ok' and 'result'/'error' keys."""
        try:
            handler = getattr(self, f"_tool_{tool_name}", None)
            if handler is None:
                return {"ok": False, "error": f"Unknown tool: {tool_name}"}
            return await handler(args)
        except Exception as e:
            logger.error(f"Tool {tool_name} failed: {e}")
            return {"ok": False, "error": str(e)}

    async def _post(self, url: str, payload: Dict) -> Dict:
        import httpx
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.post(url, json=payload)
            r.raise_for_status()
            return r.json()

    async def _get(self, url: str, params: Optional[Dict] = None) -> Dict:
        import httpx
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(url, params=params or {})
            r.raise_for_status()
            return r.json()

    async def _tool_deploy_asset(self, args: Dict) -> Dict:
        payload = {
            "task_type": args["task_type"],
            "target_lat": args["lat"],
            "target_lon": args["lon"],
            "priority": args.get("priority", 3),
            "description": args.get("reason", "AI-initiated task"),
        }
        if args.get("asset_id"):
            payload["asset_id"] = args["asset_id"]
        if args.get("altitude_m") is not None:
            payload["target_altitude"] = args["altitude_m"]
        if args.get("radius_m") is not None:
            payload["radius_m"] = args["radius_m"]

        result = await self._post(f"{self.tasking_url}/missions", payload)
        return {"ok": True, "result": result}

    async def _tool_create_alert(self, args: Dict) -> Dict:
        payload = {
            "severity": args["severity"],
            "description": args["description"],
            "source": "summit-ai",
        }
        if args.get("entity_id"):
            payload["entity_id"] = args["entity_id"]
        if args.get("lat") is not None:
            payload["lat"] = args["lat"]
            payload["lon"] = args["lon"]
        if args.get("recommended_action"):
            payload["recommended_action"] = args["recommended_action"]

        result = await self._post(f"{self.fabric_url}/alerts", payload)
        return {"ok": True, "result": result}

    async def _tool_create_geofence(self, args: Dict) -> Dict:
        payload = {
            "name": args["name"],
            "fence_type": args["fence_type"],
            "center": {"lat": args["center_lat"], "lon": args["center_lon"]},
            "radius_m": args["radius_m"],
        }
        if args.get("max_altitude_m") is not None:
            payload["max_altitude_m"] = args["max_altitude_m"]
        if args.get("reason"):
            payload["description"] = args["reason"]

        result = await self._post(f"{self.fabric_url}/geofences", payload)
        return {"ok": True, "result": result}

    async def _tool_send_command(self, args: Dict) -> Dict:
        payload = {
            "action": args["action"],
            "reason": args.get("reason", "AI command"),
        }
        if args["action"] == "GOTO":
            payload["lat"] = args["lat"]
            payload["lon"] = args["lon"]
            if args.get("altitude_m") is not None:
                payload["altitude_m"] = args["altitude_m"]

        result = await self._post(
            f"{self.tasking_url}/assets/{args['asset_id']}/command", payload
        )
        return {"ok": True, "result": result}

    async def _tool_open_actuator(self, args: Dict) -> Dict:
        payload = {
            "action": args["action"],
            "reason": args.get("reason", "AI actuator command"),
        }
        if args.get("value") is not None:
            payload["value"] = args["value"]

        result = await self._post(
            f"{self.tasking_url}/actuators/{args['device_id']}/command", payload
        )
        return {"ok": True, "result": result}

    async def _tool_query_world(self, args: Dict) -> Dict:
        params = {k: v for k, v in args.items() if v is not None}
        result = await self._get(f"{self.fabric_url}/entities", params)
        entities = result if isinstance(result, list) else result.get("entities", [])
        return {"ok": True, "result": {"entity_count": len(entities), "entities": entities}}

    async def _tool_request_human_input(self, args: Dict) -> Dict:
        """Queue a human decision request. Returns immediately — response comes async."""
        request = {
            "question": args["question"],
            "context": args["context"],
            "options": args.get("options", []),
            "urgency": args.get("urgency", "MEDIUM"),
            "status": "PENDING",
        }
        self._pending_human_inputs.append(request)
        logger.info(f"Human input requested [{args.get('urgency','MEDIUM')}]: {args['question']}")

        # Create a WARNING alert so the operator sees it in the console
        try:
            await self._tool_create_alert({
                "severity": "WARNING",
                "description": f"AI requesting operator input: {args['question']}",
                "recommended_action": args["context"],
            })
        except Exception:
            pass

        return {
            "ok": True,
            "result": {
                "status": "PENDING",
                "message": "Human input request queued for operator review",
            },
        }
