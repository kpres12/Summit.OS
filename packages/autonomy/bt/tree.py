"""
Behavior Tree Executor for Summit.OS

Manages the tick loop, tree lifecycle, and provides introspection
for debugging and visualization.
"""
from __future__ import annotations

import time
import logging
from typing import Any, Callable, Dict, List, Optional

from packages.autonomy.bt.nodes import BTNode, Blackboard, NodeStatus, Composite, Decorator

logger = logging.getLogger("autonomy.bt.tree")


class BehaviorTree:
    """
    Manages a behavior tree: setup, tick loop, introspection.

    Usage:
        tree = BehaviorTree(root_node)
        tree.setup()
        while True:
            status = tree.tick()
            if status != NodeStatus.RUNNING:
                break
    """

    def __init__(self, root: BTNode, name: str = "BehaviorTree"):
        self.root = root
        self.name = name
        self.blackboard = Blackboard()
        self.tick_count: int = 0
        self.status: NodeStatus = NodeStatus.FAILURE
        self._setup_done: bool = False

        # Timing
        self.last_tick_time: float = 0.0
        self.total_tick_time: float = 0.0

        # Callbacks
        self._pre_tick: List[Callable] = []
        self._post_tick: List[Callable] = []

    def setup(self, initial_data: Dict[str, Any] | None = None):
        """Initialize the tree. Must be called before first tick."""
        if initial_data:
            for k, v in initial_data.items():
                self.blackboard.set(k, v)
        self.root.setup(self.blackboard)
        self._setup_done = True

    def tick(self) -> NodeStatus:
        """Execute one tick of the entire tree."""
        if not self._setup_done:
            self.setup()

        # Pre-tick callbacks
        for cb in self._pre_tick:
            cb(self)

        start = time.time()
        self.status = self.root.tick()
        elapsed = time.time() - start

        self.tick_count += 1
        self.last_tick_time = elapsed
        self.total_tick_time += elapsed

        # Post-tick callbacks
        for cb in self._post_tick:
            cb(self, self.status)

        return self.status

    def reset(self):
        """Reset the tree for re-execution."""
        self.root.reset()
        self.tick_count = 0
        self.status = NodeStatus.FAILURE
        self.total_tick_time = 0.0

    def on_pre_tick(self, callback: Callable):
        self._pre_tick.append(callback)

    def on_post_tick(self, callback: Callable):
        self._post_tick.append(callback)

    # ── Introspection ───────────────────────────────────────

    def get_tree_structure(self) -> Dict:
        """Get a serializable representation of the tree structure."""
        return self._node_to_dict(self.root)

    def _node_to_dict(self, node: BTNode, depth: int = 0) -> Dict:
        result = {
            "name": node.name,
            "type": node.__class__.__name__,
            "status": node.status.value,
            "depth": depth,
        }

        if isinstance(node, Composite):
            result["children"] = [
                self._node_to_dict(c, depth + 1) for c in node.children
            ]
        elif isinstance(node, Decorator):
            if node.child:
                result["child"] = self._node_to_dict(node.child, depth + 1)

        return result

    def get_stats(self) -> Dict:
        """Get execution statistics."""
        avg = self.total_tick_time / max(1, self.tick_count)
        return {
            "name": self.name,
            "status": self.status.value,
            "tick_count": self.tick_count,
            "avg_tick_ms": round(avg * 1000, 3),
            "last_tick_ms": round(self.last_tick_time * 1000, 3),
            "blackboard_keys": self.blackboard.keys(),
        }

    def print_tree(self, node: BTNode | None = None, indent: int = 0) -> str:
        """Return a text visualization of the tree."""
        if node is None:
            node = self.root
        lines = []

        status_icon = {"SUCCESS": "✓", "FAILURE": "✗", "RUNNING": "→"}.get(
            node.status.value, "?"
        )
        prefix = "  " * indent
        lines.append(f"{prefix}{status_icon} {node.__class__.__name__}({node.name})")

        if isinstance(node, Composite):
            for child in node.children:
                lines.append(self.print_tree(child, indent + 1))
        elif isinstance(node, Decorator) and node.child:
            lines.append(self.print_tree(node.child, indent + 1))

        return "\n".join(lines)
