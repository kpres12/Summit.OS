# Honest Gaps — What's Real vs Scaffolding

Last full audit: 2026-04-25 (commit `67a6661`).

This doc exists to prevent README/white-paper claims from drifting from
reality. It is intentionally short — most things below ✅ are real and
verified, and only items that legitimately can't be closed in code
remain.

---

## Verified real (this audit)

- **Engagement Authorization Gate** — full state machine, 35 unit tests
  + 7 live HTTP API E2E tests, all passing. Pass-through defaults
  removed; constructor refuses missing dependencies. Single API surface
  for `ENGAGEMENT_AUTHORIZED` proven by grep.
- **Sensor signing** — fail-closed when `cryptography` is missing
  (was fail-open). Real Ed25519 wired to the gate.
- **Field encryption** — fail-closed when `FIELD_ENCRYPTION_KEY` is
  missing (was silently storing PII as plaintext).
- **RBAC** — doctrine→RBAC role mapping with rank-aware inheritance.
- **Chained-HMAC audit sink** — append-only, tamper-detected, 24k+
  writes/s sustained.
- **MQTT bridge** — auto-opens engagement cases from sensor-fusion
  confirmed tracks. 7 tests passing.
- **Operator UI** — `apps/console/components/ops/EngagementQueue.tsx`
  + `app/engagement/page.tsx` for surfacing PENDING_AUTHORIZATION cases
  and capturing signed AUTHORIZE / DENY / HOLD decisions.
- **CANVAS TA1** — `infra/policy/canvas/authority_delegation.rego` +
  `packages/canvas/{authority_dsl, workflow_sim}.py` + white paper at
  `docs/canvas/CANVAS_TA1_WHITE_PAPER.md`. Demo scenario produces
  12 requests / 10 baseline + 2 conditional delegation / 0 denied.
- **All 31 trained models load** (11 .pt + 20 .joblib). All 9 missing
  `data_source` provenances backfilled.
- **All 39 sensor/protocol adapters import** cleanly.
- **HTTP frontend security** — CSP, HSTS, X-Frame-Options DENY,
  X-Content-Type-Options nosniff, Referrer-Policy, Permissions-Policy,
  X-DNS-Prefetch-Control off. Source-level XSS scan: clean.
- **K8s pod security (all 7 deployments)** — runAsNonRoot, runAsUser
  65534, readOnlyRootFilesystem, allowPrivilegeEscalation false,
  capabilities drop ALL, seccompProfile RuntimeDefault.
- **Container security** — 13/13 Dockerfiles run as non-root user.
- **gRPC services** — refuse to bind to `0.0.0.0` without TLS; support
  mTLS + auth interceptor with token validator.
- **Alembic migrations** — both have proper upgrade/downgrade, multi-
  service defensive.
- **CVEs** — 11 of 12 patched. Only `pip` itself remains (no upstream
  fix yet).
- **Performance** — engagement gate benchmarked: 141k case-opens/s,
  0.26 ms p99 full-workflow latency, 24k audit writes/s.

## Test suite

```
312 passed, 34 skipped (correctly — require live MQTT/Redis/Postgres/TAK or
Docker stack), 0 failed
```

Breakdown of new tests since last audit:
- 95 mission simulation tests (`tests/test_mission_simulations.py`) — 8 domains,
  17 sensor types, 14 asset types, adversarial conditions
- 3 TAK interop tests passing (`tests/integration/test_atak_interop.py`)
- 2 retrained sklearn models (damage_classifier, flood_classifier)

Run `pytest -m perf` to add the 3 perf benchmarks (141k case-opens/s, 0.26ms p99, 24k audit writes/s).

---

## Genuinely still open (cannot be closed by code alone)

### ✅ TAK Server live interop validation — CLOSED (2026-04-26)

**Status:** 3 passed, 1 skipped (correctly). Docker stub (`infra/docker/tak_stub/`)
replaced missing `defcontracting/taky` image; `pytest_asyncio.fixture` decorator
fixed for strict mode. Adapter connect / publish / waypoint / disconnect paths
validated against a real TCP CoT stream. The receive-inbound-CoT test correctly
skips when the stub sends via TCP and the adapter's recv loop listens on UDP
(correct production posture — real ATAK networks use UDP multicast for SA).
**Evidence:** `3 passed, 1 skipped in 30.11s` against live Docker container.
**Remaining for production:** Swap stub for official TAK Server (tak.gov) with
real certs before AFRL Rome demo.

### ⚠️ Live integration stack (MQTT → Fabric → Redis → Fusion → API)

**Status:** Each component has its own tests; the wired E2E stack does
not.
**Why still open:** Same — needs Docker compose stack running.
**Action to close:**
```
docker compose up
HELI_RUN_INTEGRATION=1 pytest tests/test_observation_flow.py
```

### ⚠️ pip CVE-2026-3219

**Status:** No upstream fix published.
**Action to close:** Watch upstream pip releases; bump when a patched
version ships.

### ✅ sklearn version skew — CLOSED (2026-04-26)

**Status:** 0 remaining warnings. Only 2 of the 5 listed models actually
fired version-skew warnings; the rest were already current. Both retrained:
- `damage_classifier.joblib` — CV F1-macro **0.896** (3,000 samples, xBD + USGS)
- `flood_classifier.joblib` — CV F1 **0.983** (Harvey + physics-synthetic)
All 155 tests pass, 0 sklearn version warnings on model load.

### ❌ Third-party penetration test

**Status:** Has not happened.
**Why still open:** Definitionally requires an external assessor.
**Action to close:** Engage a CREST / OSCP-staffed pentest firm before
the AFRL Rome demo or any production federal customer go-live.

---

## Update cadence

Re-run the audit and update this file:

- After every significant module addition
- Before every external customer demo or proposal submission
- At least quarterly during the CANVAS proposal window

The audit script is just the static-analysis sweep + test suite + a
handful of greps. No excuse for the README and white paper drifting
from reality again.
