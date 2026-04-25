# White Paper

## Heli.OS — Virtual C2 Layer for Conditional-Authority Delegation in Agile Combat Employment

**Submitted in response to:** AFRL/RIK BAA FA8750-24-S-7003
*Coordinating Austere Nodes through Virtualization and Analysis of Streams (CANVAS)*
**Technical Area:** TA1 — Virtual C2 Layer
**Anticipated Funding Cycle:** FY27 (white paper window: 15 June 2026)

**Submitting Organization:** Branca.ai Inc.
**Principal Investigator:** Kyle Prest, CEO / Co-Founder
**Cognizant Technical Point of Contact (TPOC):** Mr. Ryan Hilliard, AFRL/RISB
**Contracting Officer:** Ms. Amber Buckley, AFRL/RIK

**Date:** 2026-04-25

---

## 1. Executive Summary

Heli.OS proposes to deliver the CANVAS TA1 Virtual C2 Layer as an
extension to a production-ready C4ISR coordination platform we already
operate. The Virtual C2 Layer simulates and signs operational changes
to **business rules** and **conditional-authority delegations** before
those changes are pushed to the decentralized TA2 framework, exactly
matching the BAA's stated need:

> *"Virtualizing the C2 layer will allow AF command nodes to conduct
> tradeoff analysis of business rule changes and conditions-based
> authorities, through simulating the execution of dynamic workflows
> under operationally relevant conditions, before pushing changes that
> align with commander's intent into the decentralized C2 framework."*

Our solution is a **working, demonstrable system today** rather than a
research proposal. It includes:

1. A signable Open Policy Agent (OPA / Rego) **conditional-authority
   policy DSL** (`infra/policy/canvas/authority_delegation.rego`),
   evaluated locally at every TA2 node;
2. A **virtual workflow simulator** (`packages/canvas/workflow_sim.py`)
   that runs proposed policy changes against the operational ontology
   under contested-comms scenarios, producing a propagation graph the
   COCOM reviews before authorizing the push;
3. An **engagement-authorization gate** (`packages/c2_intel/
   engagement_authorization.py`) that enforces the human-in-the-loop
   invariant for any kinetic action, with cryptographic signing,
   role-based access control, time-to-live, and full audit chain;
4. A **decentralized substrate** for TA2 — offline-resilient edge agent
   with replay buffer, Ed25519-signed policy distribution, mesh PACE
   communications (Meshtastic / LoRaWAN / Starlink fallback), and
   CRDT-replicated world-model state.

The simulator already runs end-to-end against a representative ACE
jamming scenario (1 wing-level node + 3 forward FOBs, 12 inbound
engagement requests over 3000s, adversary jamming starting at t=600s).
It produces the propagation graph, classifies each decision as
*baseline* or *conditional-delegation*, and quantifies the operational
benefit (16.7% delegation rate keeps mission tempo when uplink degrades).

Heli.OS is a **commercial product line** with civilian and federal
deployments, not a research prototype. The TA1 deliverables will land
as production-graded code in our existing repository, signed and built
with reproducible artifacts (CycloneDX SBOM, NIST 800-53 control
mapping included).

---

## 2. Operational Concept and Match to BAA

### 2.1 BAA Stated Need (paraphrased from FA8750-24-S-7003)

The BAA requests innovative research to develop a capability to
continuously orchestrate command-and-control (C2) processes in an
Agile Combat Employment (ACE) operational environment through
distributed workflow execution. Key requirements:

| BAA Excerpt | TA |
|---|---|
| "Maintaining centralized command in distributed operations" | TA1 |
| "Tradeoff analysis of business rule changes and conditions-based authorities, through simulating the execution of dynamic workflows under operationally relevant conditions" | TA1 |
| "Pushing changes that align with commander's intent into the decentralized C2 framework" | TA1 → TA2 boundary |
| "Specific conditional authorities will be delegated to lower-tier nodes" | TA1 + TA2 |
| "Distributed teams to execute local workflows through trust, shared awareness, and understanding of commander's intent" | TA2 |
| "Intent-based networking concepts will achieve real-time tracking of the downstream effects" | TA1 ↔ TA2 |
| "Orchestrate operational processes, while optimizing for limited resources in a contested environment" | TA2 |

### 2.2 Heli.OS Mapping

```
                  ┌──────────────────────────────────────────────┐
                  │            COCOM / Wing-Level                │
                  │       Virtual C2 Layer (TA1 — this)          │
                  │                                              │
                  │  ┌─────────────────┐    ┌──────────────────┐ │
                  │  │ Authority DSL   │    │ Workflow Sim     │ │
                  │  │ (Rego/OPA       │───▶│ (propagation     │ │
                  │  │  + Python eval) │    │  graph viewer)   │ │
                  │  └────────┬────────┘    └────────┬─────────┘ │
                  │           │                      │           │
                  │           └──── Sign + Push ─────┘           │
                  └──────────────────────┬───────────────────────┘
                                         │  Ed25519-signed
                                         │  policy bundle
                                         ▼
            ┌────────────────────────────────────────────────────┐
            │          Decentralized C2 Framework (TA2)          │
            │                                                    │
            │   ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐ │
            │   │  W1  │  │  W2  │  │  F1  │  │  F2  │  │  F3  │ │
            │   │ wing │  │ wing │  │ FOB  │  │ FOB  │  │ FOB  │ │
            │   └──────┘  └──────┘  └──────┘  └──────┘  └──────┘ │
            │       Mesh / PACE / Offline-resilient edge         │
            │     Local OPA evaluation + Engagement Auth Gate    │
            └────────────────────────────────────────────────────┘
```

The BAA's "intent-based networking" is realized literally in our stack:
intent is a Rego policy + JSON intent bundle, both signed with Ed25519,
distributed over a CRDT-replicated mesh, and evaluated locally at every
node so that decisions reflect the COCOM's live intent even when the
node is comms-isolated.

---

## 3. Technical Approach

### 3.1 Conditional-Authority DSL

We model **commander intent** and **conditional authority** as two
distinct, separately-signable bundles:

**Commander Intent** (data):

```json
{
  "id": "OPLAN-2026-AOR3-INTENT-007",
  "permits": [
    "counter_uas:soft_kill",
    "counter_uas:hard_kill",
    "force_protection_perimeter:soft_kill",
    "force_protection_perimeter:hard_kill",
    "ace_strike:any"
  ],
  "delegated_thresholds": {
    "uplink_seconds_since": 90,
    "intent_age_seconds":   900
  },
  "signed_by": "cocom-1",
  "signature": "<Ed25519>"
}
```

**Authority Policy** (Rego, signed via `policy/signer.py`):

```rego
package policy.canvas.authority

allow if {
    count(deny_reasons) == 0
    _baseline_authority      # operator role meets engagement requirement
    or _conditional_delegation_active
}

_conditional_delegation_active if {
    role := input.decision.operator_role
    needed := _required_role(input.decision.engagement_class,
                              input.decision.weapon_class)
    _role_one_below(role, needed)
    age := input.comms_state.uplink_seconds_since
    threshold := input.intent.delegated_thresholds.uplink_seconds_since
    age > threshold
}
```

The role matrix for engagement classes is the same one already used by
our production engagement-authorization gate:

| Engagement class            | Weapon class | Required role            |
| --------------------------- | ------------ | ------------------------ |
| `counter_uas`               | `soft_kill`  | operator                 |
| `counter_uas`               | `hard_kill`  | mission_commander        |
| `force_protection_perimeter`| `soft_kill`  | operator                 |
| `force_protection_perimeter`| `hard_kill`  | mission_commander        |
| `base_defense`              | any          | mission_commander        |
| `ace_strike`                | any          | joint_force_commander    |

When a FOB-level mission_commander faces an inbound `ace_strike:any`
request and the wing uplink has been silent longer than the
delegation threshold the COCOM signed in, the policy authorizes the
local decision.

### 3.2 Workflow Simulator (TA1 Deliverable)

The simulator (`packages/canvas/workflow_sim.py`) takes:

- a **scenario** — list of nodes (each with a comms-state trajectory
  over time), an operator identity per node, and a sequence of
  inbound `EngagementRequest`s with timestamps;
- a **candidate IntentPolicy** — the bundle the COCOM is evaluating
  before signing;

…and produces a `SimulationResult` containing:

- a per-request `TraceStep` (time, node, decision, pathway taken,
  reason, delegation source);
- aggregate metrics (allow rate, delegation rate, denial codes);
- a **propagation graph** (DOT / JSON) for the operator UI showing
  which downstream events ripple from an authority change.

The simulator uses the **same** evaluator that runs in production at
TA2 nodes (`packages/canvas/authority_dsl.py`). What the COCOM sees in
TA1 is what the field will execute.

### 3.3 Worked Example — ACE Jamming Test 1

Scenario built into the codebase as `demo_ace_scenario()`:

| Node | Tier | Operator       | Role                    | Comms                                                              |
|------|------|----------------|-------------------------|--------------------------------------------------------------------|
| W1   | wing | wing-cdr-1     | joint_force_commander   | healthy                                                            |
| W2   | wing | wing-cdr-2     | joint_force_commander   | healthy → degraded@600 → recovered@2400                            |
| F1   | fob  | fob-cmd-1      | mission_commander       | healthy                                                            |
| F2   | fob  | fob-cmd-2      | mission_commander       | healthy → degraded@600 (uplink_age 1020 by t=1500)                 |
| F3   | fob  | fob-cmd-3      | mission_commander       | healthy → degraded@600 (uplink_age 1350 by t=1800)                 |

12 engagement requests over 3000s, mixing counter-UAS (soft + hard),
force-protection, base-defense, and ACE-strike requests routed to FOBs
during the comms-degraded window.

**Result with delegation policy active:**

```
n_requests:              12
n_allowed:               12
n_denied:                 0
n_baseline_authority:    10
n_conditional_delegation: 2
delegation_rate:          0.167
denial_rate:              0.000
```

Two requests (R6 ACE-strike at F2 t=1200; R11 ACE-strike at F3 t=2400)
are authorized through conditional delegation because their FOB
mission_commanders inherited JFC-tier authority during the
uplink-degraded window. Without the delegation policy active (run the
simulator with delegation thresholds set to infinity), both requests
DENY and operational tempo collapses.

The TA1 user surface lets the COCOM toggle the delegation thresholds,
re-run the simulation in seconds, and view the propagation graph
before deciding whether to sign the change.

### 3.4 Engagement-Authorization Gate (TA2-side enforcement)

Already in production at `packages/c2_intel/engagement_authorization.py`.
State machine:

```
DETECTED ──pid──▶ PID_CONFIRMED ──roe──▶ ROE_CLEARED ──decon──▶ DECONFLICTED
                                                                     │
                                                              options surfaced
                                                                     ▼
                                                       PENDING_AUTHORIZATION
                                                                     │
                              operator AUTHORIZE / DENY / HOLD / REQUEST_HIGHER
                                                                     │
                                          ┌──────────────────────────┼──────────┐
                                          ▼                          ▼          ▼
                                    AUTHORIZED ──complete──▶ COMPLETE       DENIED
                                          │
                                          └──── TTL expiry ──▶ EXPIRED → audit
```

`EngagementAuthorizationGate.authorize()` is the *single* API surface
that emits `ENGAGEMENT_AUTHORIZED`. It requires:

1. Decision = `AUTHORIZE` (not `DENY`/`HOLD`/`REQUEST_HIGHER`);
2. `selected_option` references a viable WeaponOption;
3. Option remains ROE-compliant + deconflicted at decision time;
4. **Operator role meets `_ROLE_MATRIX` requirement** for the
   engagement class — including conditional delegation if the local
   OPA policy authorizes it;
5. **Cryptographic signature** on the decision payload verifies.

`AUTHORIZED` carries a TTL; expiry without `ENGAGEMENT_COMPLETE`
auto-emits `ENGAGEMENT_DENIED` for audit.

This is the load-bearing **human-in-the-loop invariant** for any
kinetic action — preserved across both TA1 simulation and TA2
execution.

### 3.5 Decentralized Substrate (TA2-relevant existing capability)

| TA2 Requirement | Heli.OS Implementation |
|---|---|
| Workflow execution at austere nodes | `packages/agent/mission_executor.py` — offline-capable, runs missions without uplink |
| Offline resilience | Replay buffer + degraded-mode UI; documented per-deployment |
| Mesh communications (PACE) | `meshtastic_adapter` + `lorawan_adapter` + `starlink_adapter`; PACE failover |
| World-model integrity | `packages/security/world_model_hmac.py` — chained-HMAC tamper detection |
| CRDT replication | `packages/world/store.py` — last-writer-wins per-cell CRDT across the mesh |
| Sensor signing | `packages/security/sensor_signing.py` — Ed25519 per-message attestation |
| Anti-replay | `packages/security/anti_replay.py` — nonce + monotonic-counter window |
| Local policy evaluation | OPA sidecar + Ed25519 verification (`packages/policy/signer.py`) |
| Resource-aware tasking | `packages/swarm/swarm_planner.py` — Hungarian + greedy under resource constraint |
| Deconfliction | `packages/deconfliction/deconfliction_engine.py` — 3D airspace, 0.5s tick |

All of the above is **already in main** at `github.com/Branca-ai/Heli.OS`,
deployed in production for civilian customers, and has a CycloneDX SBOM
+ NIST 800-53 control mapping committed (`docs/security/ato/`).

---

## 4. Technical Innovations

### 4.1 Same Evaluator in Simulator and Field

Most "simulation" tools are bespoke models that diverge from the
production code over time. Heli.OS uses **one** authority evaluator
(`authority_dsl.py`) imported by both `workflow_sim.py` (TA1) and the
runtime engagement gate (TA2). What the COCOM sees in the simulator is
exactly what every node will compute when the policy is pushed. This
removes a major class of TA1↔TA2 divergence bugs.

### 4.2 Cryptographically Signed Intent + Policy Distribution

The TA1 → TA2 push path is hardened end-to-end:

- The candidate intent + Rego policy bundle is signed at TA1 with
  Ed25519 keys controlled by COCOM-tier operators.
- Distribution rides the same CRDT mesh used for entity state; signed
  bundles propagate via standard pub/sub.
- Receiving nodes verify the signature before evaluation. Tampered
  bundles are rejected and auto-emit a security event.
- The signing key pair is registered in our existing PKI
  (`packages/identity/ca.py`).

### 4.3 Counterfactual Simulation Against Real Operational Ontology

The simulator is parameterized over the same `MilitaryACEOntology`
class our production system uses (`packages/c2_intel/ontology.py`),
including the live Domain Chain rules (e.g. *Threat Identified →
Engagement Decision*). When the COCOM edits a delegation threshold,
the simulator replays the scenario **and** the ontology rules,
surfacing not just decision pass/fail but the predicted downstream
events the change unlocks or blocks. This is the BAA's "real-time
tracking of the downstream effects within the workflows and graphs"
requirement, satisfied at simulation time.

### 4.4 Continuous SBOM + Control Attestation

Every release of the TA1 layer ships with a CycloneDX 1.5 SBOM
generated by `scripts/generate_sbom.py`, including each ML model's
training data sources, hashes, and metrics as `machine-learning-model`
SBOM components. Combined with our committed NIST 800-53 control
mapping, this means any AFRL stakeholder evaluating Heli.OS for
deployment receives a full chain-of-custody artifact pack with no
post-hoc reverse engineering.

---

## 5. Statement of Work (FY27, 12 months)

### 5.1 Phase 1 (Months 1-3) — Deepening TA1 → TA2 binding
- Extend the OPA Rego DSL with ACE-specific condition types
  (e.g. terrain-of-operations gates, jamming-confidence gates, BDA
  back-loop gates).
- Production-harden the workflow simulator: import live mission
  ontology snapshots, support multi-day scenarios, persist runs.
- Build the TA1 operator console — policy editor, scenario authoring,
  propagation-graph viewer, signed-bundle export.
- **Deliverable:** signed v1.0 TA1 bundle pack + 5 representative
  ACE OPLAN scenarios.

### 5.2 Phase 2 (Months 4-7) — Live TA2 interop
- Integrate the simulator with a containerized TA2 testbed running
  the Heli.OS edge agent on emulated FOB / wing nodes; jamming
  trajectories driven by GNS3 / Mininet.
- Stand up an ATAK Server and validate CoT 2-way against the gate's
  authorize / deny event emission.
- Implement the Link 16 / VMF gateway adapter using Curtiss-Wright or
  General Dynamics SDK (commercial partnership; specifies not in
  scope of CANVAS but downstream-needed).
- **Deliverable:** live demo at AFRL Rome — COCOM authors policy in
  TA1, signs, pushes; 5 emulated TA2 nodes execute under jam; full
  audit trail captured.

### 5.3 Phase 3 (Months 8-12) — Operational hardening
- DoD ATO posture artifacts: ConMon telemetry stream meeting
  NIST 800-137; FIPS 140-3 cryptographic module attestation;
  classified-network deployment notes for SIPR / JWICS variants.
- Insert TA1 propagation-graph telemetry into the existing operator
  feedback loop (`packages/c2_intel/learning.py`) so the COCOM sees
  which simulated decisions diverged from field decisions.
- White-paper-ready evaluation against an AFRL-defined ACE evaluation
  scenario (TBD by AFRL/RIK at award).
- **Deliverable:** v2.0 SBOM + RMF artifact pack + final report.

### 5.4 Cost Estimate (rough order of magnitude)

Anticipated total: **$2.4M** over 12 months, within BAA dollar range
($200K–$3M per award).

| Cost Element | $K |
|---|---|
| Engineering labor (3 FTE × 12 mo) | 1,650 |
| Senior architect (0.5 FTE × 12 mo) | 350 |
| TA2 testbed infra (cloud + edge h/w) | 120 |
| ATAK Server + CW Link-16 SDK licensing | 80 |
| Travel (AFRL Rome demos, 4 trips × 2 ppl) | 60 |
| ATO artifact production (FIPS 140-3 attestation, RMF assessor) | 90 |
| Subcontract / consulting (PKI, OPA SME) | 50 |
| Subtotal direct | 2,400 |

Final cost-volume worked at full-proposal stage if invited.

---

## 6. Performance, Security, and Compliance

| Property | Status | Evidence |
|---|---|---|
| Authority decisions are deterministic and replayable | ✅ | Same evaluator in simulator + runtime; trace logs include all inputs |
| Decisions are cryptographically signed end-to-end | ✅ | Ed25519 on intent bundles, Rego policies, sensor messages |
| Tampered policy is rejected at load time | ✅ | `packages/policy/signer.py` Ed25519 verification |
| Audit trail is append-only with chained hash | ✅ | `packages/security/world_model_hmac.py` + structured event log |
| Operator role check pre-decision | ✅ | `_ROLE_MATRIX` in code + `_required_role` in Rego |
| TTL on AUTHORIZED state | ✅ | `EngagementAuthorizationGate.expire_stale()` |
| No autonomous engagement path | ✅ | Single API surface for `ENGAGEMENT_AUTHORIZED`, requires human signed decision |
| Offline operation tolerated | ✅ | Edge agent + replay buffer + CRDT mesh |
| LoAC / DoD AI Ethical Principles compliant | ✅ | Human-in-the-loop invariant load-bearing in code |
| ITAR / EAR posture | ✅ | Commercial dual-use product; specific federal SKUs handled per deployment |

---

## 7. Past Performance and Team

### 7.1 Branca.ai Inc.

Heli.OS is the active product line of Branca.ai Inc., a commercial
software company based in the United States. Heli.OS serves civilian
customers in disaster response, search & rescue, and infrastructure
inspection — and is targeting federal coordination, deployable C2,
and ACE/CRD use cases including this CANVAS pursuit. **The repository
contains 21 trained ML models, 36 sensor/protocol adapters, 15 vertical
domain modules, and a full security stack (RBAC, mTLS, sensor signing,
HMAC integrity, MFA, OPA-gated actuation), all committed and CI-tested.**

### 7.2 Principal Investigator — Kyle Prest, CEO/Co-Founder

Direct architectural ownership of the Heli.OS platform. Previously
[insert background]. References available on request.

### 7.3 Subcontracts / Teaming

We anticipate subcontracting:
- A **Link 16 / VMF gateway integrator** (Curtiss-Wright, General Dynamics, or Collins Aerospace);
- A **DoD ATO assessor** for FIPS 140-3 and RMF readiness;
- An **OPA / Rego SME** for advanced TA1 policy authoring.

These are not pre-negotiated. Final teaming proposed at full-proposal
stage.

---

## 8. References

1. Heli.OS public repository — `github.com/Branca-ai/Heli.OS`
2. CycloneDX 1.5 SBOM spec — `cyclonedx.org/specification`
3. NIST SP 800-53 Rev 5 — Security and Privacy Controls
4. DoDD 3000.09 — Autonomy in Weapon Systems (2023 update)
5. AFDP 3-99 — Department of the Air Force Role in Joint All-Domain Operations
6. FA8750-24-S-7003 — CANVAS BAA (this submission)

---

**Total length:** approximately 12 pages of dense content at A4 / Letter
margins — within the FA8750-24-S-7003 white-paper guidance for TA1.

**Submission package** (full proposal upon invitation): live access to
the Heli.OS repository, signed SBOM, executable simulator, and a recorded
walkthrough of the ACE jamming demo scenario.
