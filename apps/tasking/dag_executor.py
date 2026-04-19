"""
apps/tasking/dag_executor.py — DAG-based task dependency execution.

Adds ordered, dependency-aware execution to mission workflows.
Tasks declare depends_on=[task_id, ...]; the executor resolves a topological
order and runs independent tasks concurrently within each wave.

CANVAS TA2: distributed workflow execution with explicit handoff logic.

    graph = TaskGraph()
    graph.add_task(DAGTask("relay_up",   "Comms relay",  relay_fn,   depends_on=[]))
    graph.add_task(DAGTask("aerial_s1",  "Aerial sweep", sweep_fn,   depends_on=["relay_up"]))
    graph.add_task(DAGTask("ground_cfm", "Ground check", confirm_fn, depends_on=["aerial_s1"]))
    result = await DAGExecutor().run(graph)
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set

logger = logging.getLogger("tasking.dag")


class TaskStatus(Enum):
    PENDING  = "PENDING"
    RUNNING  = "RUNNING"
    SUCCESS  = "SUCCESS"
    FAILED   = "FAILED"
    SKIPPED  = "SKIPPED"


@dataclass
class DAGTask:
    task_id:    str
    name:       str
    execute:    Callable[[Dict, Dict], Coroutine]
    depends_on: List[str] = field(default_factory=list)
    on_failure: str = "abort"   # "abort" | "skip" | "continue"
    metadata:   Dict[str, Any] = field(default_factory=dict)

    # Set by executor during run
    status: TaskStatus = field(default=TaskStatus.PENDING, init=False)
    result: Optional[Any] = field(default=None, init=False)
    error:  Optional[str] = field(default=None, init=False)


@dataclass
class DAGResult:
    success:         bool
    task_results:    Dict[str, Any]
    failed_tasks:    List[str]
    skipped_tasks:   List[str]
    execution_order: List[str]


class TaskGraph:
    """Directed acyclic graph of tasks."""

    def __init__(self) -> None:
        self._tasks: Dict[str, DAGTask] = {}

    def add_task(self, task: DAGTask) -> "TaskGraph":
        self._tasks[task.task_id] = task
        return self

    def tasks(self) -> Dict[str, DAGTask]:
        return self._tasks

    def validate(self) -> List[str]:
        errors: List[str] = []
        for task in self._tasks.values():
            for dep in task.depends_on:
                if dep not in self._tasks:
                    errors.append(
                        f"Task '{task.task_id}' depends on unknown task '{dep}'"
                    )
        # Cycle detection via DFS
        visited: Set[str] = set()
        rec: Set[str] = set()

        def _has_cycle(tid: str) -> bool:
            visited.add(tid)
            rec.add(tid)
            for dep in self._tasks.get(tid, DAGTask("", "", _noop)).depends_on:
                if dep not in visited:
                    if _has_cycle(dep):
                        return True
                elif dep in rec:
                    return True
            rec.discard(tid)
            return False

        for tid in self._tasks:
            if tid not in visited and _has_cycle(tid):
                errors.append(f"Cycle detected involving task '{tid}'")
                break

        return errors

    def execution_waves(self) -> List[List[str]]:
        """
        Group tasks into parallel execution waves via Kahn's algorithm.
        Wave 0 has no dependencies; wave N depends only on waves < N.
        """
        completed: Set[str] = set()
        remaining = set(self._tasks.keys())
        waves: List[List[str]] = []

        while remaining:
            wave = [
                tid for tid in remaining
                if all(dep in completed for dep in self._tasks[tid].depends_on)
            ]
            if not wave:
                logger.error("DAG stalled — possible unresolved dependency in: %s", remaining)
                break
            waves.append(sorted(wave))
            for tid in wave:
                remaining.discard(tid)
                completed.add(tid)

        return waves


class DAGExecutor:
    """Execute a TaskGraph, running each wave concurrently."""

    async def run(
        self,
        graph: TaskGraph,
        context: Optional[Dict[str, Any]] = None,
    ) -> DAGResult:
        errors = graph.validate()
        if errors:
            raise ValueError(f"Invalid task graph: {errors}")

        ctx      = context or {}
        tasks    = graph.tasks()
        waves    = graph.execution_waves()
        results: Dict[str, Any] = {}
        failed:  List[str] = []
        skipped: List[str] = []
        order:   List[str] = []
        abort = False

        for wave_idx, wave in enumerate(waves):
            if abort:
                for tid in wave:
                    tasks[tid].status = TaskStatus.SKIPPED
                    skipped.append(tid)
                continue

            logger.info("DAG wave %d: %s", wave_idx, wave)
            await asyncio.gather(*[
                self._run_one(tasks[tid], ctx, results)
                for tid in wave
            ])

            for tid in wave:
                order.append(tid)
                results[tid] = tasks[tid].result
                if tasks[tid].status == TaskStatus.FAILED:
                    failed.append(tid)
                    if tasks[tid].on_failure == "abort":
                        logger.warning("Task '%s' aborted DAG", tid)
                        abort = True

        success = not failed
        logger.info(
            "DAG done — success=%s failed=%s skipped=%s order=%s",
            success, failed, skipped, order,
        )
        return DAGResult(
            success=success,
            task_results=results,
            failed_tasks=failed,
            skipped_tasks=skipped,
            execution_order=order,
        )

    async def _run_one(
        self, task: DAGTask, ctx: Dict, results: Dict
    ) -> None:
        task.status = TaskStatus.RUNNING
        try:
            task.result = await task.execute(ctx, results)
            task.status = TaskStatus.SUCCESS
            logger.debug("Task '%s' succeeded", task.task_id)
        except Exception as exc:
            task.status = TaskStatus.FAILED
            task.error  = str(exc)
            logger.warning("Task '%s' failed: %s", task.task_id, exc)


# ── Mission DAG builder ───────────────────────────────────────────────────────

def build_mission_dag(
    mission_id: str,
    assignments_map: Dict[str, Any],
    mqtt_publish_fn: Optional[Callable] = None,
) -> TaskGraph:
    """
    Convert a mission's role assignments into a dependency-ordered TaskGraph.

    Ordering rule: role priority (from RoleDecomposer) determines wave.
      priority=0  (mesh/relay)  → must run first — comms before assets launch
      priority=1  (aerial)      → depends on all priority-0 tasks completing
      priority=2  (ground/sub)  → depends on all priority-1 tasks completing

    This ensures the relay network is up before drones launch,
    and aerial recon completes before ground confirmation begins.
    """
    graph = TaskGraph()

    # Group by priority
    by_priority: Dict[int, List[str]] = {}
    for asset_id, plan in assignments_map.items():
        p = int(plan.get("priority", 3))
        by_priority.setdefault(p, []).append(asset_id)

    prev_task_ids: List[str] = []

    for priority in sorted(by_priority.keys()):
        wave_ids: List[str] = []

        for asset_id in by_priority[priority]:
            plan = assignments_map[asset_id]
            task_id = f"{mission_id}:{asset_id}"

            def _make_execute(aid: str, p: Dict) -> Callable:
                async def execute(ctx: Dict, results: Dict) -> Dict:
                    if mqtt_publish_fn:
                        try:
                            await mqtt_publish_fn(aid, p)
                        except Exception as exc:
                            raise RuntimeError(
                                f"MQTT dispatch failed for {aid}: {exc}"
                            ) from exc
                    logger.info("DAG dispatched asset=%s role=%s", aid, p.get("role"))
                    return {"asset_id": aid, "plan": p, "status": "DISPATCHED"}
                return execute

            graph.add_task(DAGTask(
                task_id    = task_id,
                name       = f"{plan.get('role', 'task')} — {asset_id}",
                execute    = _make_execute(asset_id, plan),
                depends_on = list(prev_task_ids),
                on_failure = "skip",
                metadata   = {"asset_id": asset_id, "priority": priority},
            ))
            wave_ids.append(task_id)

        prev_task_ids = wave_ids

    return graph


async def _noop(ctx: Dict, results: Dict) -> None:
    pass
