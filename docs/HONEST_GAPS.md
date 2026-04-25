# Honest Gaps — What's Real vs Scaffolding

Last audit: 2026-04-25 (commit `8b9d44d`).

This document is the source of truth for what is **production-quality
implementation** vs **scaffolding / synthetic / unwired**. The README
and white papers reference capabilities at a higher level; this is
where reviewers (federal customers, code auditors, due-diligence teams)
should look first.

The standard for "real" used here:

- ✅ **Real** — class + tests + at least one production caller wired in,
  no fail-open defaults
- 🟡 **Scaffold** — class with correct API, but no production caller or
  unsafe default in some path
- 🔴 **Synthetic** — model trained on procedurally-generated data, or
  external dataset not actually downloaded
- ⚪ **Documented** — design exists in code comments / docs only

---

## Engagement Authorization Gate (the load-bearing one)

| Aspect | Status | Notes |
|---|---|---|
| State machine | ✅ | `packages/c2_intel/engagement_authorization.py` — DETECTED → PID → ROE → DECONFLICTED → PENDING_AUTH → AUTHORIZED → COMPLETE, with DENIED/HELD/EXPIRED branches |
| Constructor refuses pass-through defaults | ✅ | None or non-callable in any of 4 deps → ValueError. `for_testing()` helper is the only documented escape, easy to grep |
| Single emission of ENGAGEMENT_AUTHORIZED | ✅ | Verified by grep: only `gate.authorize()` emits this event |
| Ed25519 signature verification (real) | ✅ | `packages/c2_intel/engagement_wiring.make_ed25519_verifier()` calls `sensor_signing.verify_frame` — fails closed when crypto unavailable |
| RBAC role check with inheritance | ✅ | `make_rbac_role_check(rbac_engine)` consults RBACEngine, normalizes doctrine → RBAC role names, walks rank hierarchy |
| Chained-HMAC append-only audit log | ✅ | `ChainedHMACAuditSink` — per-row HMAC, tamper-evident, restart-safe |
| TTL auto-deny on expiry | ✅ | `expire_stale()` walks AUTHORIZED cases, emits ENGAGEMENT_DENIED |
| Production HTTP API surface | ✅ | `apps/api-gateway/routers/engagement.py` — 10 endpoints, the AUTHORIZE endpoint requires X-Operator-Signature header |
| Test coverage | ✅ | `tests/test_engagement_authorization.py` — 35 tests, all passing |
| MQTT subscriber `summit/engagement/track_confirmed` → open_case | 🟡 | API endpoint exists; MQTT bridge that auto-opens cases from sensor fusion not yet wired |
| Operator UI for surfacing PENDING_AUTHORIZATION cases + signing | 🟡 | API endpoints ready; React component in `apps/console/` not yet built |
| `db_logger.engagement_audit` enterprise sink | 🟡 | `engagement_wiring.make_audit_sink` will use it if present; default is the chained-HMAC file sink |

**Bottom line for DoDD 3000.09**: the human-in-the-loop invariant is
real, signed, role-checked, audited, end-to-end tested, and exposed
through a single production HTTP endpoint. Open work: MQTT auto-open
bridge + operator UI for the actual signing workflow.

---

## ML Models — Real Data vs Synthetic

### ✅ Trained on real federal/civilian datasets

| Model | Real source | Real samples | Metric |
|---|---|---|---|
| `aftershock_lstm.pt` | USGS ComCat | 54,411 events | AUC 0.81 |
| `aircraft_anomaly_lstm.pt` | OpenSky Network | 5,601 aircraft × 18 snapshots | val MSE 0.003 |
| `ais_vessel_anomaly_lstm.pt` | MarineCadastre.gov AIS | 10,626 vessels (200k records) | val MSE 0.00997 |
| `corrosion_vision_classifier.pt` | CodeBrim concrete defects | ~3,000 patches | trained |
| `damage_classifier.joblib` | xBD (Maxar) labels | thousands | trained |
| `drought_severity_classifier.joblib` | USDM 50-state + Open-Meteo | 5,250 state-weeks | F1 0.687 |
| `eonet_hazard_classifier.joblib` | NASA EONET | 5,000 events | F1 0.999 (caveat: imbalanced classes) |
| `eurosat_lulc_classifier.pt` | Sentinel-2 (Zenodo mirror) | 27,000 patches × 10 LULC | val acc 0.9554 |
| `fire_danger_classifier.joblib` | NASA FIRMS VIIRS + Open-Meteo | thousands | F1 0.982 |
| `fire_intensity_regressor.joblib` | NASA FIRMS VIIRS + Open-Meteo | thousands | trained |
| `compound_hazard_scorer.joblib` | FIRMS + USGS + Open-Meteo | 1,600 mixed | MAE 0.020 |
| `heatwave_transformer.pt` | Open-Meteo, 30 cities × 2 yr | 20,619 sequences | AUC 0.96 avg |
| `lightning_ignition_classifier.joblib` | NOAA Storm + 1.08M FIRMS hotspots + Open-Meteo | 716 events × 1.08M hotspots | AUC 0.667 |
| `maritime_sar_classifier.joblib` | NDBC buoys + NOAA marine events | 30k buoy-hours | F1 0.899, AUC 0.891 |
| `storm_severity_classifier.joblib` | NOAA Storm Events 2022-2024 | 215,281 rows | F1 0.843 |

### 🔴 Synthetic-fallback only (real data not yet on disk)

| Model | Synthetic-fallback reason | What unlocks real |
|---|---|---|
| `xview3_vessel_classifier.pt` | xView3 portal-only, no API | Manual download → `packages/training/data/xview3/` |
| `radioml_modulation_classifier.pt` | DeepSig portal-only | Manual download from deepsig.ai → `packages/training/data/radioml/` |
| `tornado_nowcast_cnn.pt` | Needs `pyart` library installed for NEXRAD parse | `pip install pyart-mch` + run with `--stations KOUN,KFWS` |
| `wildfire_lstm.pt` | FIRMS public CSV only spans 8 days | Use FIRMS_MAP_KEY (in `.env`) for archive — not yet rewired in trainer |

### 🟡 Pre-existing models (older, source mix unclear from meta)

`corrosion_classifier.joblib`, `counter_uas_classifier.joblib`, `crowd_estimator.joblib`, `damage_vision_classifier.pt`, `deforestation_classifier.joblib`, `flood_classifier.joblib`, `flood_risk_regressor.joblib`, `pipeline_anomaly_classifier.joblib`, `slope_stability_classifier.joblib`, `vehicle_classifier.joblib`, `seismic_risk_regressor.joblib`, `fire_risk_regressor.joblib` — meta files don't always record data_source. Check the corresponding `train_*.py` to confirm before claiming real-data provenance.

---

## Adapters — Wired vs Scaffold

39 adapters total. Of those:

### ✅ Fully working (live data, tested)

OpenSky, AISStream, NMEA, NMEA-2000, NEXRAD (via `datasets/nexrad.py`), MarineCadastre (via `datasets/marinecadastre.py`), FIRMS public + keyed archive, Open-Meteo, USDM, NASA EONET, NOAA NDBC, NOAA Storm Events, GDELT, Sentinel Hub (with creds), Copernicus Data Space (with creds), Microsoft Planetary Computer, AWS Open Data anonymous S3, Global Fishing Watch (with token), Space-Track (with creds), CelesTrak.

### 🟡 Adapter exists, unverified end-to-end

ATAK / CoT — `packages/adapters/atak_adapter.py` has `publish_entity` and `send_command`. Validated against a TAK Server via `infra/docker/docker-compose.tak.yml` + `tests/integration/test_atak_interop.py` — but the integration test has not been run end-to-end against a live TAK Server during this session. Ship-blocker if not validated before CANVAS Phase 2.

MAVLink, ROS2, Spot, UR, DJI, Tesla, Modbus, BACnet, Kraken, Zigbee, LoRaWAN, Meshtastic, Starlink, dump1090 — adapter classes exist, follow the BaseAdapter contract, but require physical hardware or a SITL stack to fully verify.

### 🔴 Scaffold only (need vendor SDK or hardware partnership)

`link16_vmf_adapter.py` — Link 16 / VMF. Has J/K-series message decoder skeleton, JREAP-C XML parsing, but **needs a paired MIDS terminal + Type-1 crypto + a vendor message gateway** (Curtiss-Wright / General Dynamics / Collins) for production. The `send_command` outbound path is gated on `engagement_authorized` flag → will refuse without a real signed authorization, but actual radio emission is logged-only in scaffold mode.

`sentinel_hub_adapter.py` — works fully when `SH_CLIENT_ID` / `SH_CLIENT_SECRET` are in `.env`. Without creds, it errors at first call (does NOT silently fail-open).

---

## CANVAS TA1 Substrate

| Component | Status |
|---|---|
| `infra/policy/canvas/authority_delegation.rego` | ✅ — Rego policy, signable via `policy/signer.py` |
| `packages/canvas/authority_dsl.py` | ✅ — Pure-Python evaluator mirroring Rego, used by simulator |
| `packages/canvas/workflow_sim.py` | ✅ — `run_simulation()` works; `demo_ace_scenario()` produces 12 requests / 10 baseline + 2 conditional delegation / 0 denied; DOT graph export working |
| Integration with engagement gate | 🟡 — Both pieces exist; the simulator does not currently call into the gate to verify the same evaluator path. They share the DSL but the gate doesn't yet evaluate the OPA policy at decision time |
| TA1 operator console UI | ⚪ — Designed only. The simulator runs from the Python CLI; no React-side editor exists yet |
| Signed-bundle distribution | ✅ — `policy/signer.py` Ed25519 signs Rego files; CRDT mesh push via `world/store.py` exists for entity state, would need a parallel pub/sub for policy bundles |

---

## ATO Posture

| Artifact | Status |
|---|---|
| NIST 800-53 control mapping (`docs/security/ato/RMF_CONTROL_MAPPING.md`) | ✅ |
| ConMon design + classified deployment notes (`CONMON_AND_CLASSIFIED_DEPLOYMENT.md`) | ✅ design ⚪ implementation |
| CycloneDX SBOM | ✅ — generated from `scripts/generate_sbom.py`, 169 components, ML provenance included |
| FIPS 140-3 cryptographic module attestation | ⚪ — Pursued in CANVAS Phase 3; uses OpenSSL FIPS provider when container is FIPS-built |
| RMF SSP per deployment | ⚪ — Per-deployment artifact; codebase produces evidence but the SSP is operator-side |
| ConMon telemetry stream | 🟡 — MQTT topic and event format defined; SIEM integration sample configs exist; no production deployment running it yet |

---

## Test Coverage

```
tests/test_engagement_authorization.py    35 tests passing
tests/unit/                              106 tests passing (adapter/identity/policy_signer/rate_limiter/sdk/secrets_client)
tests/test_observation_flow.py            2 tests, skip unless HELI_RUN_INTEGRATION=1
tests/integration/                       skip unless integration stack is up
tests/e2e/                                skip unless live deployment is up
tests/perf/                               skip unless explicit perf run
tests/test_integration.py                 SKIP at module level (stale post-rebrand SDK imports)
```

Total green when running unit + engagement: **141 passed, 30 skipped, 0 failed**.

---

## Stale / Disabled

- `tests/test_integration.py` — references the old `heli_os` SDK module name (pre-rebrand). Module-level skip in place; rewrite required.
- Various `train_*.py` for older models — meta files don't always record `data_source`. Provenance check required before claiming real-data training.

---

## Punch list — to make every CANVAS-relevant claim airtight

These are the gaps a federal evaluator would catch:

1. **MQTT bridge** — auto-open engagement cases from confirmed-track events emitted by the fusion service
2. **Operator console UI** for the engagement workflow — `apps/console/` React component for surfacing PENDING_AUTHORIZATION + capturing the signed AUTHORIZE
3. **Live TAK Server validation** — actually run `docker compose -f infra/docker/docker-compose.tak.yml up` and the `test_atak_interop.py` test against it; record the run as evidence
4. **Wire OPA policy evaluation into the gate** — currently the `_role_matrix` Python check covers RBAC; the Rego policy is a parallel path the simulator uses. Make the gate consult the OPA policy at decision time so TA1 sim and TA2 runtime are bit-for-bit the same
5. **Provenance backfill** — read every older `train_*.py`, populate the `data_source` field in their `_meta.json`, document any model trained on synthetic data
6. **Real-data retrain on the four synthetic-fallback models** — get xView3, RadioML, NEXRAD pyart, FIRMS archive datasets on disk and retrain
7. **FIPS 140-3 attestation** — pursue in Phase 3 of CANVAS; deliverable is a third-party-attested OpenSSL FIPS build of the runtime container
8. **ConMon telemetry stream live** — stand up the MQTT security topic + SIEM forwarder against a real SIEM (Splunk, Elastic) for at least one deployment, capture the events as evidence

Items 1-4 are <1 week of focused work each. Items 5-6 are 1-3 weeks. Items 7-8 are operator-side / Phase-3.

---

## Update cadence for this document

Re-audit and update this file:

- After every significant module addition
- Before every external customer demo or proposal submission
- At least quarterly during the CANVAS proposal window

The audit script is just `grep`s, mostly. There's no excuse for the README and white paper claims drifting from reality again.
