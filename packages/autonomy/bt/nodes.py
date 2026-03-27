"""
Behavior Tree Node Types for Summit.OS Autonomy Engine

Implements the standard BT node taxonomy:

Control Flow:
  - Sequence: runs children left-to-right, fails on first failure
  - Selector (Fallback): runs children left-to-right, succeeds on first success
  - Parallel: runs all children, policy-based success/failure

Decorators:
  - Inverter: negates child result
  - Repeat: runs child N times
  - RetryUntilSuccess: retries child on failure
  - Timeout: fails child after duration
  - RateLimit: throttles child execution

Leaf Nodes:
  - Action: executes a callable
  - Condition: evaluates a predicate
  - Wait: succeeds after delay
"""

from __future__ import annotations

import time
import logging
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
from abc import ABC, abstractmethod

logger = logging.getLogger("autonomy.bt")


class NodeStatus(str, Enum):
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    RUNNING = "RUNNING"


class Blackboard:
    """
    Shared key-value store for behavior tree nodes.

    Provides scoped access (global, tree, subtree) for data exchange
    between nodes without tight coupling.
    """

    def __init__(self):
        self._data: Dict[str, Any] = {}

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def set(self, key: str, value: Any):
        self._data[key] = value

    def has(self, key: str) -> bool:
        return key in self._data

    def delete(self, key: str):
        self._data.pop(key, None)

    def clear(self):
        self._data.clear()

    def keys(self) -> List[str]:
        return list(self._data.keys())

    def to_dict(self) -> Dict[str, Any]:
        return dict(self._data)


class BTNode(ABC):
    """Base class for all behavior tree nodes."""

    def __init__(self, name: str = ""):
        self.name = name or self.__class__.__name__
        self.status: NodeStatus = NodeStatus.FAILURE
        self.parent: Optional[BTNode] = None
        self.blackboard: Optional[Blackboard] = None

    @abstractmethod
    def tick(self) -> NodeStatus:
        """Execute one tick of this node. Returns status."""
        ...

    def setup(self, blackboard: Blackboard):
        """Called once before the tree starts running."""
        self.blackboard = blackboard

    def reset(self):
        """Reset node state for re-execution."""
        self.status = NodeStatus.FAILURE


# ═══════════════════════════════════════════════════════════
# Control Flow Nodes
# ═══════════════════════════════════════════════════════════


class Composite(BTNode):
    """Base class for nodes with children."""

    def __init__(self, name: str = "", children: List[BTNode] | None = None):
        super().__init__(name)
        self.children: List[BTNode] = children or []

    def add_child(self, child: BTNode) -> "Composite":
        child.parent = self
        self.children.append(child)
        return self

    def setup(self, blackboard: Blackboard):
        super().setup(blackboard)
        for child in self.children:
            child.setup(blackboard)

    def reset(self):
        super().reset()
        for child in self.children:
            child.reset()


class Sequence(Composite):
    """
    Sequence (AND): Execute children left to right.
    - Returns FAILURE immediately if any child fails.
    - Returns RUNNING if a child is running.
    - Returns SUCCESS only if all children succeed.

    Memory variant: remembers which child was running and resumes from there.
    """

    def __init__(
        self,
        name: str = "Sequence",
        children: List[BTNode] | None = None,
        memory: bool = True,
    ):
        super().__init__(name, children)
        self.memory = memory
        self._running_idx: int = 0

    def tick(self) -> NodeStatus:
        start = self._running_idx if self.memory else 0

        for i in range(start, len(self.children)):
            status = self.children[i].tick()

            if status == NodeStatus.RUNNING:
                self._running_idx = i
                self.status = NodeStatus.RUNNING
                return NodeStatus.RUNNING

            if status == NodeStatus.FAILURE:
                self._running_idx = 0
                self.status = NodeStatus.FAILURE
                return NodeStatus.FAILURE

        self._running_idx = 0
        self.status = NodeStatus.SUCCESS
        return NodeStatus.SUCCESS

    def reset(self):
        super().reset()
        self._running_idx = 0


class Selector(Composite):
    """
    Selector (OR / Fallback): Execute children left to right.
    - Returns SUCCESS immediately if any child succeeds.
    - Returns RUNNING if a child is running.
    - Returns FAILURE only if all children fail.

    Memory variant: remembers which child was running.
    """

    def __init__(
        self,
        name: str = "Selector",
        children: List[BTNode] | None = None,
        memory: bool = True,
    ):
        super().__init__(name, children)
        self.memory = memory
        self._running_idx: int = 0

    def tick(self) -> NodeStatus:
        start = self._running_idx if self.memory else 0

        for i in range(start, len(self.children)):
            status = self.children[i].tick()

            if status == NodeStatus.RUNNING:
                self._running_idx = i
                self.status = NodeStatus.RUNNING
                return NodeStatus.RUNNING

            if status == NodeStatus.SUCCESS:
                self._running_idx = 0
                self.status = NodeStatus.SUCCESS
                return NodeStatus.SUCCESS

        self._running_idx = 0
        self.status = NodeStatus.FAILURE
        return NodeStatus.FAILURE

    def reset(self):
        super().reset()
        self._running_idx = 0


class Parallel(Composite):
    """
    Parallel: Tick all children every tick.

    Success policy:
      - "all": succeeds when ALL children succeed
      - "one": succeeds when ANY child succeeds

    Failure policy:
      - "all": fails when ALL children fail
      - "one": fails when ANY child fails
    """

    def __init__(
        self,
        name: str = "Parallel",
        children: List[BTNode] | None = None,
        success_policy: str = "all",
        failure_policy: str = "one",
    ):
        super().__init__(name, children)
        self.success_policy = success_policy
        self.failure_policy = failure_policy

    def tick(self) -> NodeStatus:
        successes = 0
        failures = 0

        for child in self.children:
            status = child.tick()
            if status == NodeStatus.SUCCESS:
                successes += 1
            elif status == NodeStatus.FAILURE:
                failures += 1

        # Check failure
        if self.failure_policy == "one" and failures > 0:
            self.status = NodeStatus.FAILURE
            return NodeStatus.FAILURE
        if self.failure_policy == "all" and failures == len(self.children):
            self.status = NodeStatus.FAILURE
            return NodeStatus.FAILURE

        # Check success
        if self.success_policy == "all" and successes == len(self.children):
            self.status = NodeStatus.SUCCESS
            return NodeStatus.SUCCESS
        if self.success_policy == "one" and successes > 0:
            self.status = NodeStatus.SUCCESS
            return NodeStatus.SUCCESS

        self.status = NodeStatus.RUNNING
        return NodeStatus.RUNNING


# ═══════════════════════════════════════════════════════════
# Decorator Nodes
# ═══════════════════════════════════════════════════════════


class Decorator(BTNode):
    """Base class for decorator nodes (single child)."""

    def __init__(self, name: str = "", child: BTNode | None = None):
        super().__init__(name)
        self.child = child
        if child:
            child.parent = self

    def setup(self, blackboard: Blackboard):
        super().setup(blackboard)
        if self.child:
            self.child.setup(blackboard)

    def reset(self):
        super().reset()
        if self.child:
            self.child.reset()


class Inverter(Decorator):
    """Inverts child result: SUCCESS ↔ FAILURE. RUNNING passes through."""

    def tick(self) -> NodeStatus:
        if not self.child:
            return NodeStatus.FAILURE
        status = self.child.tick()
        if status == NodeStatus.SUCCESS:
            self.status = NodeStatus.FAILURE
            return NodeStatus.FAILURE
        if status == NodeStatus.FAILURE:
            self.status = NodeStatus.SUCCESS
            return NodeStatus.SUCCESS
        self.status = NodeStatus.RUNNING
        return NodeStatus.RUNNING


class Repeat(Decorator):
    """Repeats child N times. Fails immediately if child fails."""

    def __init__(
        self, name: str = "Repeat", child: BTNode | None = None, count: int = 3
    ):
        super().__init__(name, child)
        self.count = count
        self._current = 0

    def tick(self) -> NodeStatus:
        if not self.child:
            return NodeStatus.FAILURE

        while self._current < self.count:
            status = self.child.tick()
            if status == NodeStatus.RUNNING:
                self.status = NodeStatus.RUNNING
                return NodeStatus.RUNNING
            if status == NodeStatus.FAILURE:
                self._current = 0
                self.status = NodeStatus.FAILURE
                return NodeStatus.FAILURE
            self._current += 1
            self.child.reset()

        self._current = 0
        self.status = NodeStatus.SUCCESS
        return NodeStatus.SUCCESS

    def reset(self):
        super().reset()
        self._current = 0


class RetryUntilSuccess(Decorator):
    """Retries child up to max_retries times on failure."""

    def __init__(
        self, name: str = "Retry", child: BTNode | None = None, max_retries: int = 3
    ):
        super().__init__(name, child)
        self.max_retries = max_retries
        self._attempts = 0

    def tick(self) -> NodeStatus:
        if not self.child:
            return NodeStatus.FAILURE

        status = self.child.tick()

        if status == NodeStatus.SUCCESS:
            self._attempts = 0
            self.status = NodeStatus.SUCCESS
            return NodeStatus.SUCCESS

        if status == NodeStatus.RUNNING:
            self.status = NodeStatus.RUNNING
            return NodeStatus.RUNNING

        # Failure — retry
        self._attempts += 1
        if self._attempts >= self.max_retries:
            self._attempts = 0
            self.status = NodeStatus.FAILURE
            return NodeStatus.FAILURE

        self.child.reset()
        self.status = NodeStatus.RUNNING
        return NodeStatus.RUNNING

    def reset(self):
        super().reset()
        self._attempts = 0


class Timeout(Decorator):
    """Fails child if it runs longer than duration_sec."""

    def __init__(
        self,
        name: str = "Timeout",
        child: BTNode | None = None,
        duration_sec: float = 10.0,
    ):
        super().__init__(name, child)
        self.duration_sec = duration_sec
        self._start_time: Optional[float] = None

    def tick(self) -> NodeStatus:
        if not self.child:
            return NodeStatus.FAILURE

        if self._start_time is None:
            self._start_time = time.time()

        if time.time() - self._start_time > self.duration_sec:
            self._start_time = None
            self.status = NodeStatus.FAILURE
            return NodeStatus.FAILURE

        status = self.child.tick()
        if status != NodeStatus.RUNNING:
            self._start_time = None
        self.status = status
        return status

    def reset(self):
        super().reset()
        self._start_time = None


class RateLimit(Decorator):
    """Throttles child execution to at most once per interval."""

    def __init__(
        self,
        name: str = "RateLimit",
        child: BTNode | None = None,
        interval_sec: float = 1.0,
    ):
        super().__init__(name, child)
        self.interval_sec = interval_sec
        self._last_tick: float = 0.0
        self._last_status: NodeStatus = NodeStatus.FAILURE

    def tick(self) -> NodeStatus:
        if not self.child:
            return NodeStatus.FAILURE

        now = time.time()
        if now - self._last_tick < self.interval_sec:
            return self._last_status

        self._last_tick = now
        self._last_status = self.child.tick()
        self.status = self._last_status
        return self._last_status


# ═══════════════════════════════════════════════════════════
# Leaf Nodes
# ═══════════════════════════════════════════════════════════


class Action(BTNode):
    """
    Execute a callable action.

    The action receives the blackboard and returns a NodeStatus.
    """

    def __init__(
        self,
        name: str = "Action",
        action: Callable[[Blackboard], NodeStatus] | None = None,
    ):
        super().__init__(name)
        self.action = action

    def tick(self) -> NodeStatus:
        if not self.action or not self.blackboard:
            return NodeStatus.FAILURE
        try:
            self.status = self.action(self.blackboard)
            return self.status
        except Exception as e:
            logger.error(f"Action '{self.name}' failed: {e}")
            self.status = NodeStatus.FAILURE
            return NodeStatus.FAILURE


class Condition(BTNode):
    """
    Evaluate a predicate.

    The predicate receives the blackboard and returns True/False.
    Never returns RUNNING.
    """

    def __init__(
        self,
        name: str = "Condition",
        predicate: Callable[[Blackboard], bool] | None = None,
    ):
        super().__init__(name)
        self.predicate = predicate

    def tick(self) -> NodeStatus:
        if not self.predicate or not self.blackboard:
            return NodeStatus.FAILURE
        try:
            result = self.predicate(self.blackboard)
            self.status = NodeStatus.SUCCESS if result else NodeStatus.FAILURE
            return self.status
        except Exception as e:
            logger.error(f"Condition '{self.name}' failed: {e}")
            self.status = NodeStatus.FAILURE
            return NodeStatus.FAILURE


class Wait(BTNode):
    """Wait for a specified duration, then succeed."""

    def __init__(self, name: str = "Wait", duration_sec: float = 1.0):
        super().__init__(name)
        self.duration_sec = duration_sec
        self._start_time: Optional[float] = None

    def tick(self) -> NodeStatus:
        if self._start_time is None:
            self._start_time = time.time()

        if time.time() - self._start_time >= self.duration_sec:
            self._start_time = None
            self.status = NodeStatus.SUCCESS
            return NodeStatus.SUCCESS

        self.status = NodeStatus.RUNNING
        return NodeStatus.RUNNING

    def reset(self):
        super().reset()
        self._start_time = None


class SetBlackboard(BTNode):
    """Set a blackboard value and succeed."""

    def __init__(self, name: str = "SetBB", key: str = "", value: Any = None):
        super().__init__(name)
        self.key = key
        self.value = value

    def tick(self) -> NodeStatus:
        if self.blackboard and self.key:
            self.blackboard.set(self.key, self.value)
            self.status = NodeStatus.SUCCESS
            return NodeStatus.SUCCESS
        self.status = NodeStatus.FAILURE
        return NodeStatus.FAILURE


class CheckBlackboard(BTNode):
    """Check if a blackboard key exists and optionally matches a value."""

    def __init__(
        self,
        name: str = "CheckBB",
        key: str = "",
        expected: Any = None,
        check_exists_only: bool = False,
    ):
        super().__init__(name)
        self.key = key
        self.expected = expected
        self.check_exists_only = check_exists_only

    def tick(self) -> NodeStatus:
        if not self.blackboard:
            return NodeStatus.FAILURE

        if self.check_exists_only:
            result = self.blackboard.has(self.key)
        else:
            result = self.blackboard.get(self.key) == self.expected

        self.status = NodeStatus.SUCCESS if result else NodeStatus.FAILURE
        return self.status
