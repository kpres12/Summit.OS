"""
Swarm Planner — Summit.OS

Allocates tasks to assets using the Hungarian algorithm (scipy.optimize.linear_sum_assignment).
Cost matrix: distance from asset to task location (haversine).
Falls back to greedy nearest-neighbor if scipy not available.

Re-runs allocation whenever:
  - A new task is added
  - An asset fails/leaves the swarm
  - A task is completed
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("swarm.planner")

try:
    import numpy as np
    from scipy.optimize import linear_sum_assignment  # type: ignore

    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False
    logger.warning(
        "scipy/numpy not available — SwarmPlanner will use greedy nearest-neighbor allocation"
    )


# ── Internal data types ──────────────────────────────────────

class _TaskRecord:
    __slots__ = ("task_id", "lat", "lon", "priority", "required_capabilities")

    def __init__(
        self,
        task_id: str,
        lat: float,
        lon: float,
        priority: int = 1,
        required_capabilities: Optional[List[str]] = None,
    ) -> None:
        self.task_id = task_id
        self.lat = lat
        self.lon = lon
        self.priority = priority
        self.required_capabilities: List[str] = required_capabilities or []


class _AssetRecord:
    __slots__ = ("asset_id", "lat", "lon", "capabilities", "battery")

    def __init__(
        self,
        asset_id: str,
        lat: float,
        lon: float,
        capabilities: Optional[List[str]] = None,
        battery: float = 100.0,
    ) -> None:
        self.asset_id = asset_id
        self.lat = lat
        self.lon = lon
        self.capabilities: List[str] = capabilities or []
        self.battery = battery


class SwarmPlanner:
    """
    Optimal task-to-asset allocator for a swarm of autonomous vehicles.

    Uses the Hungarian algorithm when scipy is available; falls back to
    greedy nearest-neighbor otherwise. Capability filtering is applied
    before constructing the cost matrix so that tasks are only assigned
    to assets that can perform them.
    """

    def __init__(self, task_crdt: "TaskCRDT") -> None:  # noqa: F821
        self._crdt = task_crdt
        self._tasks: Dict[str, _TaskRecord] = {}
        self._assets: Dict[str, _AssetRecord] = {}

    # ── Mutation ─────────────────────────────────────────────

    def add_task(
        self,
        task_id: str,
        lat: float,
        lon: float,
        priority: int = 1,
        required_capabilities: Optional[List[str]] = None,
    ) -> None:
        """Register a new task location."""
        self._tasks[task_id] = _TaskRecord(
            task_id=task_id,
            lat=lat,
            lon=lon,
            priority=priority,
            required_capabilities=required_capabilities,
        )
        logger.debug("Task added: %s (%.5f, %.5f)", task_id, lat, lon)

    def remove_task(self, task_id: str) -> None:
        """Remove a task (e.g. it was completed or cancelled)."""
        self._tasks.pop(task_id, None)
        logger.debug("Task removed: %s", task_id)

    def update_asset(
        self,
        asset_id: str,
        lat: float,
        lon: float,
        capabilities: Optional[List[str]] = None,
        battery: float = 100.0,
    ) -> None:
        """Upsert an asset's current position and capabilities."""
        existing = self._assets.get(asset_id)
        caps = capabilities if capabilities is not None else (existing.capabilities if existing else [])
        self._assets[asset_id] = _AssetRecord(
            asset_id=asset_id,
            lat=lat,
            lon=lon,
            capabilities=caps,
            battery=battery,
        )

    def remove_asset(self, asset_id: str) -> None:
        """Remove an asset from the pool (failure / departure)."""
        self._assets.pop(asset_id, None)
        logger.debug("Asset removed from planner: %s", asset_id)

    # ── Planning ─────────────────────────────────────────────

    def replan(self) -> Dict[str, str]:
        """
        Run the allocation algorithm and update the TaskCRDT.

        Returns a task_id → asset_id mapping for all assigned tasks.
        Unassignable tasks (no capable asset available) are left unassigned.
        """
        if not self._tasks or not self._assets:
            return {}

        task_ids, asset_ids, cost_mat = self._cost_matrix()

        if not task_ids or not asset_ids:
            return {}

        if _SCIPY_AVAILABLE:
            assignment = self._hungarian(task_ids, asset_ids, cost_mat)
        else:
            assignment = self._greedy(task_ids, asset_ids, cost_mat)

        # Write results into CRDT
        for task_id, asset_id in assignment.items():
            self._crdt.assign(task_id, asset_id)

        logger.info("Replan complete: %d tasks assigned", len(assignment))
        return assignment

    def _cost_matrix(
        self,
    ) -> Tuple[List[str], List[str], Any]:
        """
        Build the (task_ids, asset_ids, cost_matrix) triple.

        Cost is haversine distance in km divided by asset battery fraction
        (so low-battery assets are de-prioritised). Tasks requiring
        capabilities not present on an asset are given a very high cost
        (1e9) to effectively block that pairing.
        """
        task_ids = list(self._tasks.keys())
        asset_ids = list(self._assets.keys())

        n_tasks = len(task_ids)
        n_assets = len(asset_ids)

        if _SCIPY_AVAILABLE:
            import numpy as np  # local import for clarity
            cost_mat = np.zeros((n_tasks, n_assets), dtype=float)
        else:
            cost_mat = [[0.0] * n_assets for _ in range(n_tasks)]

        for i, tid in enumerate(task_ids):
            task = self._tasks[tid]
            for j, aid in enumerate(asset_ids):
                asset = self._assets[aid]

                # Capability filter
                if task.required_capabilities:
                    missing = set(task.required_capabilities) - set(asset.capabilities)
                    if missing:
                        val = 1e9
                        if _SCIPY_AVAILABLE:
                            cost_mat[i][j] = val
                        else:
                            cost_mat[i][j] = val
                        continue

                dist_km = self._haversine(task.lat, task.lon, asset.lat, asset.lon)
                # Penalise low-battery assets: divide by battery fraction clamped to [0.1, 1.0]
                battery_factor = max(0.1, min(1.0, asset.battery / 100.0))
                # Scale by priority (higher priority → lower effective cost for the planner)
                priority_scale = 1.0 / max(1, task.priority)
                cost = (dist_km / battery_factor) * priority_scale

                if _SCIPY_AVAILABLE:
                    cost_mat[i][j] = cost
                else:
                    cost_mat[i][j] = cost

        if _SCIPY_AVAILABLE:
            return task_ids, asset_ids, cost_mat
        return task_ids, asset_ids, cost_mat

    def _hungarian(
        self,
        task_ids: List[str],
        asset_ids: List[str],
        cost_mat: Any,
    ) -> Dict[str, str]:
        """Run scipy's Hungarian algorithm."""
        import numpy as np

        n_tasks = len(task_ids)
        n_assets = len(asset_ids)

        # Pad to square if needed
        size = max(n_tasks, n_assets)
        padded = np.full((size, size), 1e9)
        padded[:n_tasks, :n_assets] = cost_mat

        row_ind, col_ind = linear_sum_assignment(padded)

        result: Dict[str, str] = {}
        for r, c in zip(row_ind, col_ind):
            if r < n_tasks and c < n_assets and padded[r, c] < 1e9:
                result[task_ids[r]] = asset_ids[c]
        return result

    def _greedy(
        self,
        task_ids: List[str],
        asset_ids: List[str],
        cost_mat: List[List[float]],
    ) -> Dict[str, str]:
        """Greedy nearest-neighbour fallback."""
        assigned_assets: set = set()
        result: Dict[str, str] = {}

        for i, tid in enumerate(task_ids):
            best_j: Optional[int] = None
            best_cost = float("inf")
            for j, aid in enumerate(asset_ids):
                if aid in assigned_assets:
                    continue
                c = cost_mat[i][j]
                if c < best_cost:
                    best_cost = c
                    best_j = j
            if best_j is not None and best_cost < 1e9:
                result[tid] = asset_ids[best_j]
                assigned_assets.add(asset_ids[best_j])

        return result

    # ── Geometry ─────────────────────────────────────────────

    def _haversine(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Great-circle distance between two points in kilometres."""
        R = 6371.0
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)
        a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
