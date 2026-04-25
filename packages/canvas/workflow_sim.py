"""
CANVAS TA1 Workflow Simulator
=================================
The TA1 deliverable: a virtual C2 layer that simulates dynamic workflows
under operationally relevant conditions, lets a COCOM-level operator
edit business rules + conditional authorities, runs them against a
synthetic ACE scenario, and surfaces the propagation graph (which
downstream authority/state changes happen) before pushing the bundle to
the decentralized TA2 framework.

Design (mirrors BAA language):
  Scenario              — a set of nodes (AOC / Wing / FOB / Tactical),
                          each with a comms-state trajectory and operator
                          identities, plus a sequence of incoming
                          engagement-decision requests (e.g. counter-UAS
                          tracks, force-protection alerts, ISR cues).
  IntentPolicy          — the editable bundle: commander intent id,
                          engagement permits, delegation thresholds.
                          The COCOM edits this; we sign it; we evaluate.
  TraceStep             — one (request × node × time) decision evaluation,
                          recording the authority pathway taken (baseline /
                          conditional_delegation / denied) plus the
                          immediate downstream effects (which next-step
                          requests get unblocked or held).
  PropagationGraph      — DAG of TraceStep transitions across nodes/time.
                          Render as DOT or JSON for the operator UI.

The simulator uses the same DSL evaluator (`authority_dsl.py`) the live
gate uses, so what the COCOM sees in TA1 is what TA2 will execute.

Usage:
    from packages.canvas.workflow_sim import (
        Scenario, Node, EngagementRequest, IntentPolicy, run_simulation,
    )
    sim = run_simulation(scenario, policy)
    print(sim.summary())
    sim.to_dot("scenario_run.dot")
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Optional

from .authority_dsl import (
    CommanderIntent, CommsState, DecisionAuthority,
    evaluate_authority, required_role,
)

logger = logging.getLogger("canvas.workflow_sim")


# ---------------------------------------------------------------------------
# Scenario model
# ---------------------------------------------------------------------------


@dataclass
class CommsTrajectory:
    """Per-node comms-state schedule. Each entry is (t_seconds, state)."""
    schedule: list[tuple[int, CommsState]] = field(default_factory=list)

    def state_at(self, t: int) -> CommsState:
        active = self.schedule[0][1] if self.schedule else CommsState()
        for ts, st in self.schedule:
            if ts <= t:
                active = st
            else:
                break
        return active


@dataclass
class Node:
    node_id:    str
    tier:       str   # "aoc" | "coc" | "wing" | "fob" | "tactical"
    operator_id: str
    operator_role: str   # operator / mission_commander / joint_force_commander
    comms:      CommsTrajectory


@dataclass
class EngagementRequest:
    request_id:        str
    t_seconds:         int
    routed_to_node:    str
    engagement_class:  str
    weapon_class:      str
    track_id:          str
    description:       str = ""


@dataclass
class Scenario:
    name:    str
    nodes:   list[Node]
    requests: list[EngagementRequest]
    duration_seconds: int = 3600


@dataclass
class IntentPolicy:
    """The editable + signable TA1 bundle."""
    intent: CommanderIntent
    description: str = ""

    def to_serializable(self) -> dict:
        return {
            "intent_id":     self.intent.id,
            "permits":       list(self.intent.permits),
            "thresholds":    {
                "uplink_seconds_since":
                    self.intent.delegated_thresholds_uplink_seconds,
                "intent_age_seconds":
                    self.intent.delegated_thresholds_intent_age_seconds,
            },
            "signed_by":     self.intent.signed_by,
            "description":   self.description,
        }


# ---------------------------------------------------------------------------
# Simulation result
# ---------------------------------------------------------------------------


@dataclass
class TraceStep:
    request_id:       str
    node_id:          str
    t_seconds:        int
    decision:         DecisionAuthority
    operator_id:      str
    operator_role:    str
    engagement_class: str
    weapon_class:     str

    def to_dict(self) -> dict:
        return {
            "request_id":       self.request_id,
            "node_id":          self.node_id,
            "t_seconds":        self.t_seconds,
            "operator_id":      self.operator_id,
            "operator_role":    self.operator_role,
            "engagement_class": self.engagement_class,
            "weapon_class":     self.weapon_class,
            "allowed":          self.decision.allowed,
            "pathway":          self.decision.pathway,
            "reason":           self.decision.reason,
            "delegated_from":   self.decision.delegated_from,
        }


@dataclass
class SimulationResult:
    scenario:  Scenario
    policy:    IntentPolicy
    trace:     list[TraceStep] = field(default_factory=list)

    # Aggregates
    n_total:     int = 0
    n_allowed:   int = 0
    n_denied:    int = 0
    n_baseline:  int = 0
    n_delegated: int = 0

    def summary(self) -> dict:
        return {
            "scenario":     self.scenario.name,
            "intent":       self.policy.intent.id,
            "n_requests":   self.n_total,
            "n_allowed":    self.n_allowed,
            "n_denied":     self.n_denied,
            "n_baseline_authority":      self.n_baseline,
            "n_conditional_delegation":  self.n_delegated,
            "delegation_rate":
                round(self.n_delegated / max(self.n_allowed, 1), 4),
            "denial_rate":
                round(self.n_denied / max(self.n_total, 1), 4),
            "denial_codes": _aggregate_codes(self),
        }

    def to_dot(self, path: str) -> None:
        """Emit a Graphviz DOT of the propagation graph (one cluster per node)."""
        lines = ["digraph CANVAS {", "  rankdir=LR;", '  fontname="Helvetica";']
        by_node: dict[str, list[TraceStep]] = {}
        for step in self.trace:
            by_node.setdefault(step.node_id, []).append(step)
        for node_id, steps in by_node.items():
            lines.append(f'  subgraph cluster_{node_id} {{')
            lines.append(f'    label="{node_id}";')
            for s in steps:
                color = ("green" if s.decision.allowed and s.decision.pathway == "baseline"
                         else "orange" if s.decision.allowed
                         else "red")
                lines.append(
                    f'    "{s.request_id}@{s.node_id}@{s.t_seconds}" '
                    f'[label="{s.engagement_class}:{s.weapon_class}\\n'
                    f'{s.decision.pathway}\\n'
                    f'@t={s.t_seconds}s",color={color},style=filled,'
                    f'fillcolor=light{color}];')
            lines.append("  }")
        # Connect within-node steps in time order
        for node_id, steps in by_node.items():
            for a, b in zip(steps, steps[1:]):
                lines.append(
                    f'  "{a.request_id}@{a.node_id}@{a.t_seconds}" -> '
                    f'"{b.request_id}@{b.node_id}@{b.t_seconds}";')
        lines.append("}")
        with open(path, "w") as f:
            f.write("\n".join(lines))

    def to_json(self) -> str:
        return json.dumps({
            "summary": self.summary(),
            "policy":  self.policy.to_serializable(),
            "trace":   [s.to_dict() for s in self.trace],
        }, indent=2)


def _aggregate_codes(sim: SimulationResult) -> dict[str, int]:
    out: dict[str, int] = {}
    for step in sim.trace:
        for code in step.decision.deny_codes:
            out[code] = out.get(code, 0) + 1
    return out


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------


def run_simulation(scenario: Scenario, policy: IntentPolicy) -> SimulationResult:
    """Run the scenario through the authority DSL, producing a trace."""
    by_node: dict[str, Node] = {n.node_id: n for n in scenario.nodes}
    result = SimulationResult(scenario=scenario, policy=policy)

    for req in sorted(scenario.requests, key=lambda r: r.t_seconds):
        node = by_node.get(req.routed_to_node)
        if node is None:
            logger.warning("[canvas-sim] request %s routed to unknown node %s",
                           req.request_id, req.routed_to_node)
            continue
        comms = node.comms.state_at(req.t_seconds)
        decision = evaluate_authority(
            engagement_class=req.engagement_class,
            weapon_class=req.weapon_class,
            operator_role=node.operator_role,
            node_tier=node.tier,
            signature_verified=True,
            comms=comms,
            intent=policy.intent,
        )
        step = TraceStep(
            request_id=req.request_id, node_id=node.node_id,
            t_seconds=req.t_seconds,
            decision=decision,
            operator_id=node.operator_id, operator_role=node.operator_role,
            engagement_class=req.engagement_class,
            weapon_class=req.weapon_class,
        )
        result.trace.append(step)
        result.n_total += 1
        if decision.allowed:
            result.n_allowed += 1
            if decision.pathway == "baseline":
                result.n_baseline += 1
            elif decision.pathway == "conditional_delegation":
                result.n_delegated += 1
        else:
            result.n_denied += 1

    logger.info("[canvas-sim] %s: %d requests, %d allowed (%d baseline + "
                "%d delegated), %d denied",
                scenario.name, result.n_total, result.n_allowed,
                result.n_baseline, result.n_delegated, result.n_denied)
    return result


# ---------------------------------------------------------------------------
# Demo scenario builder (used by the CANVAS white paper's worked example)
# ---------------------------------------------------------------------------


def demo_ace_scenario() -> tuple[Scenario, IntentPolicy]:
    """The canonical ACE scenario in the CANVAS white paper:

      - 2 wing-level nodes (W1, W2) and 3 FOB nodes (F1, F2, F3)
      - Adversary jamming starts at t=600s, isolating W2 + F2/F3
      - 12 engagement requests (counter-UAS soft + hard kill, force
        protection, ISR cues) flow in over 2400 s
      - Without conditional delegation, F2/F3 cannot act once isolated
      - With the policy under test, lower-tier operators acquire
        authority after uplink degraded for >90 s
    """
    intent = CommanderIntent(
        id="OPLAN-2026-AOR3-INTENT-007",
        permits=[
            "counter_uas:soft_kill",
            "counter_uas:hard_kill",
            "force_protection_perimeter:soft_kill",
            "force_protection_perimeter:hard_kill",
            "ace_strike:any",
            "base_defense:any",
        ],
        delegated_thresholds_uplink_seconds=90,
        delegated_thresholds_intent_age_seconds=900,
        signed_by="cocom-1",
    )
    policy = IntentPolicy(intent=intent,
                          description="Demo ACE OPLAN with delegation enabled")

    # Node comms trajectories
    healthy = CommsState(uplink_seconds_since=5, pace_active="primary",
                         intent_age_seconds=120)
    degraded_at = lambda t: CommsState(uplink_seconds_since=t,
                                       pace_active="alternate",
                                       intent_age_seconds=120)

    w1 = Node("W1", "wing", "wing-cdr-1", "joint_force_commander",
              CommsTrajectory([(0, healthy)]))
    w2 = Node("W2", "wing", "wing-cdr-2", "joint_force_commander",
              CommsTrajectory([(0, healthy),
                               (600, degraded_at(120)),
                               (1200, degraded_at(720)),
                               (2400, healthy)]))
    f1 = Node("F1", "fob", "fob-cmd-1", "mission_commander",
              CommsTrajectory([(0, healthy)]))
    f2 = Node("F2", "fob", "fob-cmd-2", "mission_commander",
              CommsTrajectory([(0, healthy),
                               (600, degraded_at(120)),
                               (1500, degraded_at(1020))]))
    f3 = Node("F3", "fob", "fob-cmd-3", "mission_commander",
              CommsTrajectory([(0, healthy),
                               (600, degraded_at(150)),
                               (1800, degraded_at(1350))]))

    # Mix of requests routed to wing-level (where ACE strike authority
    # normally lives) and FOB-level (where counter_uas + force_protection
    # authority lives). The interesting cases are FOB-routed ACE strikes
    # during jam: those normally need wing JFC authority, so they only
    # succeed when conditional delegation kicks in.
    requests = [
        EngagementRequest("R1",   60,  "F1", "counter_uas",                "soft_kill", "T1"),
        EngagementRequest("R2",  180,  "F2", "counter_uas",                "soft_kill", "T2"),
        EngagementRequest("R3",  300,  "W1", "ace_strike",                 "any",       "T3"),
        EngagementRequest("R4",  720,  "F2", "counter_uas",                "hard_kill", "T4"),
        EngagementRequest("R5",  900,  "F3", "force_protection_perimeter", "soft_kill", "T5"),
        # ACE strikes during jam — normally need W2 (JFC); when uplink is
        # degraded, F2/F3 mission_commanders acquire delegation
        EngagementRequest("R6", 1200,  "F2", "ace_strike",                 "any",       "T6"),
        EngagementRequest("R7", 1320,  "F2", "force_protection_perimeter", "hard_kill", "T7"),
        EngagementRequest("R8", 1500,  "F3", "counter_uas",                "soft_kill", "T8"),
        EngagementRequest("R9", 1800,  "F1", "base_defense",               "any",       "T9"),
        EngagementRequest("R10", 2100, "F2", "counter_uas",                "soft_kill", "T10"),
        EngagementRequest("R11", 2400, "F3", "ace_strike",                 "any",       "T11"),
        EngagementRequest("R12", 2700, "F3", "counter_uas",                "hard_kill", "T12"),
    ]

    scenario = Scenario(
        name="ACE-Jamming-Test-1",
        nodes=[w1, w2, f1, f2, f3],
        requests=requests,
        duration_seconds=3000,
    )
    return scenario, policy


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    scenario, policy = demo_ace_scenario()
    result = run_simulation(scenario, policy)
    print(json.dumps(result.summary(), indent=2))
