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
155 passed, 7 skipped (correctly — require live MQTT/Redis/Postgres/TAK), 0 failed
```

Run `pytest -m perf` to add the 3 perf benchmarks.

---

## Genuinely still open (cannot be closed by code alone)

### ⚠️ TAK Server live interop validation

**Status:** Harness fully ready — `infra/docker/docker-compose.tak.yml`
+ `tests/integration/test_atak_interop.py` + `scripts/run_tak_interop_test.sh`.
**Why still open:** Requires Docker daemon running. Docker wasn't
available on the audit machine.
**Action to close:** From any workstation with Docker:
```
./scripts/run_tak_interop_test.sh
```
This brings up the TAK Server, runs the 4 integration tests, tears
down. ~2 minutes end-to-end. After running, paste the output into a
PR or commit message as evidence.

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

### ⚠️ Older sklearn models

**Status:** 5 `.joblib` files were trained on scikit-learn 1.6.1 and
load with 1.8.0. Version-skew warnings fire but model output is
correct.
**Action to close:** Re-run their respective `train_*.py` scripts on
current scikit-learn. Each takes 1–10 minutes.

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
