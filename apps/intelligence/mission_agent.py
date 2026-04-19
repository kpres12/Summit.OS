"""
Heli.OS Mission Agent

Hosts the AI agent loop for a single mission objective. Wires together:
  - Brain (Ollama LLM — perceive/plan/act)
  - ContextBuilder (WorldStore → LLM prompt)
  - ToolExecutor (physical actions via tasking/fabric APIs)

Multiple MissionAgent instances can run concurrently — one per active mission.
Lifecycle: PENDING → RUNNING → COMPLETED / FAILED / CANCELLED

Environment variables:
    AGENT_TICK_INTERVAL     - seconds between brain steps (default: 30)
    AGENT_MAX_STEPS         - max steps before agent self-terminates (default: 20)
    OLLAMA_URL              - Ollama base URL
    OLLAMA_MODEL            - Llama model to use
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from brain import Brain

logger = logging.getLogger("heli.intelligence.agent")

AGENT_TICK_INTERVAL = float(os.getenv("AGENT_TICK_INTERVAL", "30"))
AGENT_MAX_STEPS = int(os.getenv("AGENT_MAX_STEPS", "20"))


class AgentStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


@dataclass
class AgentStep:
    step_num: int
    ts: float
    summary: str
    tool_calls: List[Dict]
    available: bool
    context_chars: int

    def to_dict(self) -> Dict:
        return {
            "step_num": self.step_num,
            "ts": self.ts,
            "summary": self.summary,
            "tool_call_count": len(self.tool_calls),
            "available": self.available,
            "context_chars": self.context_chars,
        }


class MissionAgent:
    """
    Autonomous agent that runs the perceive/plan/act loop for one mission.
    """

    def __init__(
        self,
        mission_objective: str,
        mission_id: Optional[str] = None,
        tick_interval: float = AGENT_TICK_INTERVAL,
        max_steps: int = AGENT_MAX_STEPS,
        world_url: str = "http://localhost:8001",
        tasking_url: str = "http://localhost:8004",
        fabric_url: str = "http://localhost:8001",
        ollama_url: str = os.getenv("OLLAMA_URL", "http://localhost:11434"),
        model: str = os.getenv("OLLAMA_MODEL", "llama3.1"),
    ):
        self.mission_objective = mission_objective
        self.mission_id = mission_id or f"agent-{uuid.uuid4().hex[:8]}"
        self.tick_interval = tick_interval
        self.max_steps = max_steps

        self.brain = Brain(
            world_url=world_url,
            tasking_url=tasking_url,
            fabric_url=fabric_url,
            ollama_url=ollama_url,
            model=model,
        )

        self.status = AgentStatus.PENDING
        self.steps: List[AgentStep] = []
        self.created_at = time.time()
        self.started_at: Optional[float] = None
        self.finished_at: Optional[float] = None
        self._task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()

    async def start(self) -> None:
        """Start the agent loop as a background task."""
        if self.status != AgentStatus.PENDING:
            logger.warning(
                f"Agent {self.mission_id} already started (status={self.status})"
            )
            return
        self.status = AgentStatus.RUNNING
        self.started_at = time.time()
        self._task = asyncio.create_task(self._run())
        logger.info(f"Agent {self.mission_id} started: {self.mission_objective[:80]}")

    async def stop(self) -> None:
        """Cancel the agent loop gracefully."""
        self._stop_event.set()
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        if self.status == AgentStatus.RUNNING:
            self.status = AgentStatus.CANCELLED
            self.finished_at = time.time()
        logger.info(f"Agent {self.mission_id} stopped")

    async def _run(self) -> None:
        try:
            for step_num in range(1, self.max_steps + 1):
                if self._stop_event.is_set():
                    self.status = AgentStatus.CANCELLED
                    break

                result = await self.brain.step(self.mission_objective)

                step = AgentStep(
                    step_num=step_num,
                    ts=time.time(),
                    summary=result.get("summary", ""),
                    tool_calls=result.get("tool_calls", []),
                    available=result.get("available", False),
                    context_chars=result.get("context_chars", 0),
                )
                self.steps.append(step)

                logger.info(
                    f"[{self.mission_id}] step {step_num}: "
                    f"{len(step.tool_calls)} actions | {step.summary[:80]}"
                )

                # Natural completion: brain takes no actions for 2 consecutive steps
                if step_num >= 3:
                    recent_steps = self.steps[-2:]
                    if all(not s.tool_calls for s in recent_steps):
                        logger.info(
                            f"Agent {self.mission_id}: idle for 2 steps — marking complete"
                        )
                        self.status = AgentStatus.COMPLETED
                        break

                # Wait for next tick (or stop signal)
                try:
                    await asyncio.wait_for(
                        asyncio.shield(self._stop_event.wait()),
                        timeout=self.tick_interval,
                    )
                    # Stop event fired
                    self.status = AgentStatus.CANCELLED
                    break
                except asyncio.TimeoutError:
                    pass  # Normal — tick interval elapsed

            else:
                # Exhausted max_steps
                self.status = AgentStatus.COMPLETED

        except asyncio.CancelledError:
            self.status = AgentStatus.CANCELLED
            raise
        except Exception as e:
            logger.error(f"Agent {self.mission_id} failed: {e}", exc_info=True)
            self.status = AgentStatus.FAILED
        finally:
            self.finished_at = time.time()

    def to_dict(self) -> Dict[str, Any]:
        duration = None
        if self.started_at and self.finished_at:
            duration = round(self.finished_at - self.started_at, 1)
        return {
            "mission_id": self.mission_id,
            "mission_objective": self.mission_objective,
            "status": self.status.value,
            "step_count": len(self.steps),
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_s": duration,
            "steps": [s.to_dict() for s in self.steps[-5:]],  # last 5 steps
        }


class AgentRegistry:
    """Manages all running MissionAgent instances."""

    def __init__(self):
        self._agents: Dict[str, MissionAgent] = {}

    async def create(
        self,
        mission_objective: str,
        mission_id: Optional[str] = None,
        **kwargs,
    ) -> MissionAgent:
        agent = MissionAgent(
            mission_objective=mission_objective,
            mission_id=mission_id,
            **kwargs,
        )
        self._agents[agent.mission_id] = agent
        await agent.start()
        return agent

    async def cancel(self, mission_id: str) -> bool:
        agent = self._agents.get(mission_id)
        if agent is None:
            return False
        await agent.stop()
        return True

    def get(self, mission_id: str) -> Optional[MissionAgent]:
        return self._agents.get(mission_id)

    def list_agents(self, status: Optional[str] = None) -> List[Dict]:
        agents = list(self._agents.values())
        if status:
            agents = [a for a in agents if a.status.value == status]
        return [a.to_dict() for a in agents]

    def prune_finished(self, keep: int = 50) -> int:
        """Remove old finished agents, keeping the most recent `keep` active ones."""
        finished = [
            a
            for a in self._agents.values()
            if a.status
            in (AgentStatus.COMPLETED, AgentStatus.FAILED, AgentStatus.CANCELLED)
        ]
        # Sort by finish time, remove oldest
        finished.sort(key=lambda a: a.finished_at or 0)
        remove_n = max(0, len(finished) - keep)
        for agent in finished[:remove_n]:
            del self._agents[agent.mission_id]
        return remove_n
