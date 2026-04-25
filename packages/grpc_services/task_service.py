"""
gRPC Task Service for Heli.OS

Implements the Heli.OS Task API:
- CreateTask / GetTask / ListTasks / CancelTask
- AssignTask / CompleteTask / FailTask
- Task state machine: PENDING → ASSIGNED → RUNNING → COMPLETED/FAILED/CANCELLED
- Task dependencies and priority scheduling
"""

from __future__ import annotations

import asyncio
import time
import uuid
import logging
import heapq
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional
from enum import Enum

logger = logging.getLogger("grpc.task_service")


class TaskState(str, Enum):
    PENDING = "pending"
    ASSIGNED = "assigned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskPriority(int, Enum):
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3


@dataclass
class TaskRecord:
    """A task in the mission planning system."""

    task_id: str
    task_type: str  # "navigate", "observe", "patrol", "strike", "recon"
    state: TaskState = TaskState.PENDING
    priority: TaskPriority = TaskPriority.NORMAL
    # Assignment
    assignee_id: str = ""  # Entity ID of assigned asset
    assignee_type: str = ""  # "drone", "vehicle", "sensor"
    # Target/Objective
    target_lat: float = 0.0
    target_lon: float = 0.0
    target_alt: float = 0.0
    target_entity_id: str = ""
    # Parameters
    params: Dict[str, Any] = field(default_factory=dict)
    # Dependencies
    depends_on: List[str] = field(default_factory=list)
    # Results
    result: Dict[str, Any] = field(default_factory=dict)
    error: str = ""
    # Timing
    created_at: float = field(default_factory=time.time)
    assigned_at: float = 0.0
    started_at: float = 0.0
    completed_at: float = 0.0
    deadline: float = 0.0  # 0 = no deadline
    # Metadata
    mission_id: str = ""
    created_by: str = ""
    version: int = 0

    def to_dict(self) -> Dict:
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "state": self.state.value,
            "priority": self.priority.value,
            "assignee_id": self.assignee_id,
            "target": {
                "lat": self.target_lat,
                "lon": self.target_lon,
                "alt": self.target_alt,
            },
            "params": self.params,
            "depends_on": self.depends_on,
            "result": self.result,
            "error": self.error,
            "mission_id": self.mission_id,
            "created_at": self.created_at,
            "version": self.version,
        }

    @property
    def is_terminal(self) -> bool:
        return self.state in (
            TaskState.COMPLETED,
            TaskState.FAILED,
            TaskState.CANCELLED,
        )

    @property
    def is_overdue(self) -> bool:
        return (
            self.deadline > 0 and time.time() > self.deadline and not self.is_terminal
        )


class TaskStore:
    """Task store with priority queue and dependency tracking."""

    def __init__(self):
        self._tasks: Dict[str, TaskRecord] = {}
        self._priority_queue: List[tuple] = []  # (priority, created_at, task_id)
        self._watchers: List[asyncio.Queue] = []
        self._version = 0

    def create(self, task: TaskRecord) -> TaskRecord:
        if not task.task_id:
            task.task_id = str(uuid.uuid4())
        task.created_at = time.time()
        self._version += 1
        task.version = self._version
        self._tasks[task.task_id] = task
        heapq.heappush(
            self._priority_queue, (task.priority.value, task.created_at, task.task_id)
        )
        self._notify("created", task)
        return task

    def get(self, task_id: str) -> Optional[TaskRecord]:
        return self._tasks.get(task_id)

    def list(
        self,
        state: Optional[TaskState] = None,
        mission_id: Optional[str] = None,
        assignee_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[TaskRecord]:
        results = []
        for t in self._tasks.values():
            if state and t.state != state:
                continue
            if mission_id and t.mission_id != mission_id:
                continue
            if assignee_id and t.assignee_id != assignee_id:
                continue
            results.append(t)
            if len(results) >= limit:
                break
        return results

    def assign(
        self, task_id: str, assignee_id: str, assignee_type: str = ""
    ) -> Optional[TaskRecord]:
        task = self._tasks.get(task_id)
        if not task or task.state != TaskState.PENDING:
            return None

        # Check dependencies are met
        for dep_id in task.depends_on:
            dep = self._tasks.get(dep_id)
            if dep and dep.state != TaskState.COMPLETED:
                logger.warning(f"Task {task_id} dependency {dep_id} not completed")
                return None

        task.state = TaskState.ASSIGNED
        task.assignee_id = assignee_id
        task.assignee_type = assignee_type
        task.assigned_at = time.time()
        self._version += 1
        task.version = self._version
        self._notify("assigned", task)
        return task

    def start(self, task_id: str) -> Optional[TaskRecord]:
        task = self._tasks.get(task_id)
        if not task or task.state != TaskState.ASSIGNED:
            return None

        task.state = TaskState.RUNNING
        task.started_at = time.time()
        self._version += 1
        task.version = self._version
        self._notify("started", task)
        return task

    def complete(
        self, task_id: str, result: Optional[Dict] = None
    ) -> Optional[TaskRecord]:
        task = self._tasks.get(task_id)
        if not task or task.is_terminal:
            return None

        task.state = TaskState.COMPLETED
        task.completed_at = time.time()
        task.result = result or {}
        self._version += 1
        task.version = self._version
        self._notify("completed", task)
        return task

    def fail(self, task_id: str, error: str = "") -> Optional[TaskRecord]:
        task = self._tasks.get(task_id)
        if not task or task.is_terminal:
            return None

        task.state = TaskState.FAILED
        task.completed_at = time.time()
        task.error = error
        self._version += 1
        task.version = self._version
        self._notify("failed", task)
        return task

    def cancel(self, task_id: str) -> Optional[TaskRecord]:
        task = self._tasks.get(task_id)
        if not task or task.is_terminal:
            return None

        task.state = TaskState.CANCELLED
        task.completed_at = time.time()
        self._version += 1
        task.version = self._version
        self._notify("cancelled", task)
        return task

    def next_pending(self) -> Optional[TaskRecord]:
        """Get the highest-priority pending task with met dependencies."""
        while self._priority_queue:
            _, _, task_id = self._priority_queue[0]
            task = self._tasks.get(task_id)
            if task and task.state == TaskState.PENDING:
                # Check dependencies
                deps_met = all(
                    (self._tasks.get(d) and self._tasks[d].state == TaskState.COMPLETED)
                    for d in task.depends_on
                )
                if deps_met:
                    return task
            heapq.heappop(self._priority_queue)
        return None

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=1000)
        self._watchers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._watchers = [w for w in self._watchers if w is not q]

    def _notify(self, event: str, task: TaskRecord) -> None:
        msg = {"event": event, "task": task.to_dict(), "timestamp": time.time()}
        for q in self._watchers:
            try:
                q.put_nowait(msg)
            except asyncio.QueueFull:
                pass

    def stats(self) -> Dict:
        states = {}
        for t in self._tasks.values():
            states[t.state.value] = states.get(t.state.value, 0) + 1
        return {
            "total": len(self._tasks),
            "by_state": states,
            "overdue": sum(1 for t in self._tasks.values() if t.is_overdue),
            "version": self._version,
        }


class TaskServicer:
    """gRPC Task service for Heli.OS."""

    def __init__(self, store: Optional[TaskStore] = None):
        self.store = store or TaskStore()

    async def CreateTask(self, request: Dict) -> Dict:
        task = TaskRecord(
            task_id=request.get("task_id", ""),
            task_type=request.get("task_type", "navigate"),
            priority=TaskPriority(request.get("priority", TaskPriority.NORMAL.value)),
            target_lat=request.get("target_lat", 0.0),
            target_lon=request.get("target_lon", 0.0),
            target_alt=request.get("target_alt", 0.0),
            target_entity_id=request.get("target_entity_id", ""),
            params=request.get("params", {}),
            depends_on=request.get("depends_on", []),
            deadline=request.get("deadline", 0.0),
            mission_id=request.get("mission_id", ""),
            created_by=request.get("created_by", ""),
        )
        created = self.store.create(task)
        return {"task": created.to_dict()}

    async def GetTask(self, request: Dict) -> Dict:
        task = self.store.get(request.get("task_id", ""))
        if not task:
            return {"error": "not_found"}
        return {"task": task.to_dict()}

    async def ListTasks(self, request: Dict) -> Dict:
        state = TaskState(request["state"]) if "state" in request else None
        tasks = self.store.list(
            state=state,
            mission_id=request.get("mission_id"),
            assignee_id=request.get("assignee_id"),
        )
        return {"tasks": [t.to_dict() for t in tasks], "total": len(tasks)}

    async def AssignTask(self, request: Dict) -> Dict:
        task = self.store.assign(
            request.get("task_id", ""),
            request.get("assignee_id", ""),
            request.get("assignee_type", ""),
        )
        if not task:
            return {"error": "cannot_assign"}
        return {"task": task.to_dict()}

    async def StartTask(self, request: Dict) -> Dict:
        task = self.store.start(request.get("task_id", ""))
        if not task:
            return {"error": "cannot_start"}
        return {"task": task.to_dict()}

    async def CompleteTask(self, request: Dict) -> Dict:
        task = self.store.complete(
            request.get("task_id", ""),
            request.get("result"),
        )
        if not task:
            return {"error": "cannot_complete"}
        return {"task": task.to_dict()}

    async def FailTask(self, request: Dict) -> Dict:
        task = self.store.fail(
            request.get("task_id", ""),
            request.get("error", ""),
        )
        if not task:
            return {"error": "cannot_fail"}
        return {"task": task.to_dict()}

    async def CancelTask(self, request: Dict) -> Dict:
        task = self.store.cancel(request.get("task_id", ""))
        if not task:
            return {"error": "cannot_cancel"}
        return {"task": task.to_dict()}

    async def WatchTasks(self, request: Dict) -> AsyncIterator[Dict]:
        q = self.store.subscribe()
        try:
            while True:
                msg = await q.get()
                yield msg
        finally:
            self.store.unsubscribe(q)

    async def GetStats(self, request: Dict) -> Dict:
        return self.store.stats()
