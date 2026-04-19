"""
Mission State Machine for Heli.OS Tasking

Formal state machine governing mission lifecycle:

    PLANNING → POLICY_CHECK → DISPATCHED → ACTIVE → COMPLETING → COMPLETED
                    ↓                        ↓          ↓
                  DENIED                   FAILED     FAILED

Each transition:
1. Validates the transition is legal
2. Updates mission state
3. Emits an event (MQTT + WorldStore)
4. Returns the new state

Illegal transitions raise MissionStateError.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("tasking.state_machine")


class MissionState(str, Enum):
    PLANNING = "PLANNING"
    POLICY_CHECK = "POLICY_CHECK"
    DENIED = "DENIED"
    DISPATCHED = "DISPATCHED"
    ACTIVE = "ACTIVE"
    COMPLETING = "COMPLETING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class MissionStateError(Exception):
    """Raised when an illegal state transition is attempted."""

    def __init__(self, current: MissionState, target: MissionState):
        self.current = current
        self.target = target
        super().__init__(
            f"Illegal mission state transition: {current.value} → {target.value}"
        )


# Legal transitions: current_state -> set of allowed next states
TRANSITIONS = {
    MissionState.PLANNING: {MissionState.POLICY_CHECK, MissionState.CANCELLED},
    MissionState.POLICY_CHECK: {
        MissionState.DISPATCHED,
        MissionState.DENIED,
        MissionState.FAILED,
    },
    MissionState.DENIED: {MissionState.PLANNING, MissionState.CANCELLED},  # Can retry
    MissionState.DISPATCHED: {
        MissionState.ACTIVE,
        MissionState.FAILED,
        MissionState.CANCELLED,
    },
    MissionState.ACTIVE: {
        MissionState.COMPLETING,
        MissionState.FAILED,
        MissionState.CANCELLED,
    },
    MissionState.COMPLETING: {MissionState.COMPLETED, MissionState.FAILED},
    MissionState.COMPLETED: set(),  # Terminal
    MissionState.FAILED: {MissionState.PLANNING},  # Can retry
    MissionState.CANCELLED: set(),  # Terminal
}


@dataclass
class MissionEvent:
    """Event emitted on state transition."""

    mission_id: str
    from_state: MissionState
    to_state: MissionState
    timestamp: float = field(default_factory=time.time)
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event": "MISSION_STATE_CHANGE",
            "mission_id": self.mission_id,
            "from_state": self.from_state.value,
            "to_state": self.to_state.value,
            "timestamp": self.timestamp,
            "reason": self.reason,
            "metadata": self.metadata,
        }


class MissionStateMachine:
    """
    State machine for a single mission.

    Usage:
        sm = MissionStateMachine("mission-123")
        sm.on_transition(my_callback)

        sm.transition(MissionState.POLICY_CHECK)
        sm.transition(MissionState.DISPATCHED)
        sm.transition(MissionState.ACTIVE)
        sm.transition(MissionState.COMPLETING)
        sm.transition(MissionState.COMPLETED)
    """

    def __init__(
        self, mission_id: str, initial_state: MissionState = MissionState.PLANNING
    ):
        self.mission_id = mission_id
        self.state = initial_state
        self.history: List[MissionEvent] = []
        self._callbacks: List[Callable[[MissionEvent], None]] = []

    def transition(
        self,
        target: MissionState,
        reason: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MissionEvent:
        """
        Attempt a state transition.

        Raises MissionStateError if the transition is illegal.
        Returns the emitted MissionEvent.
        """
        allowed = TRANSITIONS.get(self.state, set())
        if target not in allowed:
            raise MissionStateError(self.state, target)

        event = MissionEvent(
            mission_id=self.mission_id,
            from_state=self.state,
            to_state=target,
            reason=reason,
            metadata=metadata or {},
        )

        old_state = self.state
        self.state = target
        self.history.append(event)

        logger.info(
            f"Mission {self.mission_id}: {old_state.value} → {target.value}"
            + (f" ({reason})" if reason else "")
        )

        # Notify callbacks
        for cb in self._callbacks:
            try:
                cb(event)
            except Exception as e:
                logger.error(f"State transition callback error: {e}")

        return event

    def can_transition(self, target: MissionState) -> bool:
        """Check if a transition to target is legal from current state."""
        return target in TRANSITIONS.get(self.state, set())

    def on_transition(self, callback: Callable[[MissionEvent], None]):
        """Register a callback for state transitions."""
        self._callbacks.append(callback)

    @property
    def is_terminal(self) -> bool:
        """Check if mission is in a terminal state."""
        return self.state in {MissionState.COMPLETED, MissionState.CANCELLED}

    @property
    def is_active(self) -> bool:
        """Check if mission is currently executing."""
        return self.state in {
            MissionState.DISPATCHED,
            MissionState.ACTIVE,
            MissionState.COMPLETING,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mission_id": self.mission_id,
            "state": self.state.value,
            "is_terminal": self.is_terminal,
            "is_active": self.is_active,
            "history": [e.to_dict() for e in self.history],
            "allowed_transitions": [
                s.value for s in TRANSITIONS.get(self.state, set())
            ],
        }


class MissionStateMachineRegistry:
    """
    Manages state machines for all active missions.
    """

    def __init__(self):
        self._machines: Dict[str, MissionStateMachine] = {}
        self._global_callbacks: List[Callable[[MissionEvent], None]] = []

    def create(
        self,
        mission_id: str,
        initial_state: MissionState = MissionState.PLANNING,
    ) -> MissionStateMachine:
        """Create and register a new mission state machine."""
        sm = MissionStateMachine(mission_id, initial_state)

        # Wire global callbacks
        for cb in self._global_callbacks:
            sm.on_transition(cb)

        self._machines[mission_id] = sm
        return sm

    def get(self, mission_id: str) -> Optional[MissionStateMachine]:
        return self._machines.get(mission_id)

    def remove(self, mission_id: str):
        self._machines.pop(mission_id, None)

    def on_any_transition(self, callback: Callable[[MissionEvent], None]):
        """Register callback for transitions on ANY mission."""
        self._global_callbacks.append(callback)
        # Also wire into existing machines
        for sm in self._machines.values():
            sm.on_transition(callback)

    def active_missions(self) -> List[MissionStateMachine]:
        return [sm for sm in self._machines.values() if sm.is_active]

    def stats(self) -> Dict[str, Any]:
        by_state: Dict[str, int] = {}
        for sm in self._machines.values():
            s = sm.state.value
            by_state[s] = by_state.get(s, 0) + 1
        return {
            "total": len(self._machines),
            "by_state": by_state,
            "active": len(self.active_missions()),
        }
