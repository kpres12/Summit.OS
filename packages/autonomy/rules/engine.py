"""
Reactive Rule Engine for Heli.OS Autonomy

Provides a lightweight reactive layer that runs alongside behavior trees.
Rules are condition → action pairs with:
- Priority ordering
- Cooldown timers (prevent rapid re-firing)
- Context matching (evaluate against entity/track state)

Use cases:
- Geofence violations → emergency RTB
- Battery critical → abort mission
- Hostile track detected → alert + reposition
- Comms lost → switch to autonomous mode
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from enum import IntEnum

logger = logging.getLogger("autonomy.rules")


class RulePriority(IntEnum):
    """Rule priorities (higher = evaluated first)."""

    SAFETY = 100
    TACTICAL = 50
    OPERATIONAL = 25
    INFORMATIONAL = 10


@dataclass
class Rule:
    """A condition-action rule."""

    name: str
    condition: Callable[[Dict[str, Any]], bool]
    action: Callable[[Dict[str, Any]], Any]
    priority: int = RulePriority.OPERATIONAL
    cooldown_sec: float = 5.0
    enabled: bool = True
    description: str = ""

    # Runtime state
    last_fired: float = 0.0
    fire_count: int = 0
    last_result: Any = None


class RuleEngine:
    """
    Reactive rule engine that evaluates rules against context.

    Rules are sorted by priority and evaluated in order.
    Multiple rules can fire per evaluation cycle.
    """

    def __init__(self):
        self.rules: List[Rule] = []
        self.evaluation_count: int = 0
        self._context: Dict[str, Any] = {}

    def add_rule(self, rule: Rule) -> "RuleEngine":
        """Add a rule to the engine."""
        self.rules.append(rule)
        self.rules.sort(key=lambda r: r.priority, reverse=True)
        return self

    def remove_rule(self, name: str):
        """Remove a rule by name."""
        self.rules = [r for r in self.rules if r.name != name]

    def enable_rule(self, name: str):
        for r in self.rules:
            if r.name == name:
                r.enabled = True

    def disable_rule(self, name: str):
        for r in self.rules:
            if r.name == name:
                r.enabled = False

    def update_context(self, **kwargs):
        """Update the evaluation context."""
        self._context.update(kwargs)

    def set_context(self, context: Dict[str, Any]):
        """Replace the evaluation context."""
        self._context = dict(context)

    def evaluate(
        self, extra_context: Dict[str, Any] | None = None
    ) -> List[Dict[str, Any]]:
        """
        Evaluate all enabled rules against current context.

        Returns list of fired rule results.
        """
        now = time.time()
        self.evaluation_count += 1

        ctx = dict(self._context)
        if extra_context:
            ctx.update(extra_context)

        fired = []
        for rule in self.rules:
            if not rule.enabled:
                continue

            # Cooldown check
            if now - rule.last_fired < rule.cooldown_sec:
                continue

            try:
                if rule.condition(ctx):
                    result = rule.action(ctx)
                    rule.last_fired = now
                    rule.fire_count += 1
                    rule.last_result = result
                    fired.append(
                        {
                            "rule": rule.name,
                            "priority": rule.priority,
                            "result": result,
                            "timestamp": now,
                        }
                    )
                    logger.info(f"Rule fired: {rule.name} (priority={rule.priority})")
            except Exception as e:
                logger.error(f"Rule '{rule.name}' error: {e}")

        return fired

    def get_stats(self) -> Dict:
        return {
            "evaluation_count": self.evaluation_count,
            "rules": [
                {
                    "name": r.name,
                    "priority": r.priority,
                    "enabled": r.enabled,
                    "fire_count": r.fire_count,
                    "cooldown_sec": r.cooldown_sec,
                }
                for r in self.rules
            ],
        }


# ═══════════════════════════════════════════════════════════
# Pre-built Safety Rules
# ═══════════════════════════════════════════════════════════


def build_safety_rules() -> List[Rule]:
    """Standard safety rules for all vehicles."""
    return [
        Rule(
            name="battery_critical",
            description="Emergency RTB when battery drops below 10%",
            condition=lambda ctx: ctx.get("battery_percent", 100) < 10,
            action=lambda ctx: {
                "command": "RTB",
                "reason": "battery_critical",
                "priority": "emergency",
            },
            priority=RulePriority.SAFETY,
            cooldown_sec=30.0,
        ),
        Rule(
            name="battery_low",
            description="Warning when battery drops below 25%",
            condition=lambda ctx: ctx.get("battery_percent", 100) < 25,
            action=lambda ctx: {
                "command": "WARN",
                "reason": "battery_low",
                "priority": "high",
            },
            priority=RulePriority.SAFETY,
            cooldown_sec=60.0,
        ),
        Rule(
            name="geofence_violation",
            description="Emergency stop when outside geofence",
            condition=lambda ctx: ctx.get("geofence_violated", False),
            action=lambda ctx: {
                "command": "HOLD",
                "reason": "geofence_violation",
                "priority": "emergency",
            },
            priority=RulePriority.SAFETY,
            cooldown_sec=5.0,
        ),
        Rule(
            name="comms_lost",
            description="Switch to autonomous mode on comms loss",
            condition=lambda ctx: ctx.get("comms_status") == "lost",
            action=lambda ctx: {
                "command": "AUTONOMOUS",
                "reason": "comms_lost",
                "priority": "high",
            },
            priority=RulePriority.SAFETY,
            cooldown_sec=10.0,
        ),
        Rule(
            name="altitude_limit",
            description="Enforce maximum altitude",
            condition=lambda ctx: ctx.get("vehicle_alt", 0)
            > ctx.get("max_altitude_m", 400),
            action=lambda ctx: {
                "command": "DESCEND",
                "target_alt": ctx.get("max_altitude_m", 400),
                "reason": "altitude_limit",
            },
            priority=RulePriority.SAFETY,
            cooldown_sec=5.0,
        ),
    ]


def build_tactical_rules() -> List[Rule]:
    """Tactical rules for threat response."""
    return [
        Rule(
            name="hostile_detected",
            description="Alert on hostile track detection",
            condition=lambda ctx: any(
                t.get("classification") == "hostile" for t in ctx.get("tracks", [])
            ),
            action=lambda ctx: {
                "command": "ALERT",
                "reason": "hostile_detected",
                "tracks": [
                    t
                    for t in ctx.get("tracks", [])
                    if t.get("classification") == "hostile"
                ],
            },
            priority=RulePriority.TACTICAL,
            cooldown_sec=15.0,
        ),
        Rule(
            name="high_density_area",
            description="Increase scan rate in high-track-density areas",
            condition=lambda ctx: len(ctx.get("tracks", [])) > 10,
            action=lambda ctx: {
                "command": "INCREASE_SCAN_RATE",
                "reason": "high_density",
            },
            priority=RulePriority.TACTICAL,
            cooldown_sec=30.0,
        ),
    ]
