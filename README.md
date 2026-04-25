# Heli.OS

**Autonomous coordination platform for the physical world.**
Sensor fusion → world model → inference → tasking → human-in-the-loop.

A [Branca.ai](https://branca.ai) product. Civilian and federal/DoD use cases.
Proprietary software — see [LICENSE](LICENSE).

---

## What it does

Heli.OS is the coordination layer that sits between physical sensors / robots / vehicles and the operator. It ingests live signals from 30+ sensor and protocol types, fuses them into a unified world model, runs domain-specific inference (16 trained ML models — wildfire risk, marine SAR demand, aircraft anomaly, drought severity, storm severity, aftershock probability, heat-wave forecasting, structural damage, counter-UAS, more), and surfaces ranked decisions to a human operator who authorizes execution.

The same core platform serves civilian disaster response, search-and-rescue, infrastructure monitoring, and commercial UAV fleets — and federal Combat Readiness Deployment, Agile Combat Employment, force protection, counter-UAS, CASEVAC, and battle damage assessment workflows. The architecture is domain-agnostic; the ontology and decision-support layers adapt by context.

### Example flow — wildfire smoke detection

At 14:33 on a Tuesday, a ridge camera detects smoke. In the next 90 seconds, with no human at the keyboard:

1. **Fusion** ingests `smoke, confidence 0.91, lat 34.12, lon -118.34`.
2. **Intelligence** runs the trained `fire_danger_classifier` (F1 0.98 on real FIRMS+OpenMeteo data), scores the cell CRITICAL, and dispatches the nearest UAV to SURVEY using ML, no LLM required.
3. **Tasking** creates a mission, terrain-corrects altitude, sends MAVLink waypoints over MQTT.
4. **Fabric** raises an alert in the OPS console; escalation kicks in if no operator acknowledges within the configured timeout.
5. **Console** shows the operator a live map with the smoke location, the drone en route, and a live HLS feed the moment the drone is on station.

The operator's job: watch, verify, decide. The software handles everything before that decision.

### Example flow — counter-UAS engagement (federal)

A radar plus RF cue produces a track over a protected asset:

1. **Track formation** — sensor fusion confirms a rotary UAS with `confidence 0.92`.
2. **PID** — `counter_uas_classifier` (trained) returns `is_rogue: True, prob: 0.94`. ISR asset confirms visual.
3. **ROE check** — `engagement_authorization` gate evaluates current ROE; collateral estimate `low`; proportionality passes.
4. **Deconfliction** — `deconfliction/deconfliction_engine.py` confirms blue-force clear, airspace clear.
5. **Options ranked** — `weapon_target_ranker` surfaces ranked options (soft-kill, hard-kill, kinetic intercept) with PK, TOF, collateral, and blue-force margin.
6. **Operator authorizes** — Mission Commander signs the AUTHORIZE decision; gate verifies role + signature, emits `ENGAGEMENT_AUTHORIZED` with a TTL.
7. **Tasking** dispatches the authorized command via authorized C2.
8. **BDA** — track loss confirms effect; `ENGAGEMENT_COMPLETE` emitted; full LoAC audit trail in the chained-HMAC audit log.

There is no API path that emits `ENGAGEMENT_AUTHORIZED` without a human authorization step that passes all of the above checks. That invariant is load-bearing.

---

## Architecture

```
                  ┌────────────────── Operators ──────────────────┐
                  │   OPS view  │  COMMAND view  │  DEV view       │
                  │   Map / alerts / mission console               │
                  └────────────────────────┬───────────────────────┘
                                           │
                  ┌────────────────────────▼───────────────────────┐
                  │            Tasking & Mission Orchestrator      │
                  │   schemas/missions.py · agent/mission_executor │
                  └─┬────────────────┬─────────────────┬──────────┬┘
                    │                │                 │          │
   ┌────────────────▼─┐  ┌───────────▼─────────┐  ┌────▼────┐  ┌──▼───────────┐
   │ Engagement Auth  │  │ Track-to-Weapon     │  │ Domain  │  │ Deconfliction│
   │ Workflow Gate    │  │ Decision Support    │  │ Ontology│  │ + Swarm      │
   │ (human-in-loop)  │  │ Ranker              │  │ (15 vt.)│  │ Planner      │
   └──────────────────┘  └──────────┬──────────┘  └─────────┘  └──────────────┘
                                    │
                  ┌─────────────────▼──────────────────┐
                  │    World Model + Sensor Fusion     │
                  │    (HMAC-chained, signed sources)  │
                  └────────────────┬───────────────────┘
                                   │
        ┌──────────────────────────▼──────────────────────────┐
        │      ML Inference Layer (lazy-loaded predictors)    │
        │      16 trained models — see /packages/c2_intel/    │
        │      models/ for the full list and metadata         │
        └──────────────────────────┬──────────────────────────┘
                                   │
   ┌───────────────────────────────▼────────────────────────────────┐
   │  Adapter Layer (32 sensor + protocol adapters)                 │
   │  ATAK · MAVLink · ROS2 · NMEA/2K · AIS/AISStream ·             │
   │  J1939 · ISOBUS · Modbus · BACnet · Meshtastic · LoRaWAN ·     │
   │  Starlink · OpenSky · dump1090 · ONVIF · RTSP · Thermal ·      │
   │  Spot · UR · DJI · Tesla · Kraken · Sentinel Hub · Webhook ·   │
   │  WebSocket · MQTT-Relay · Weather · CAP · Serial · Zigbee      │
   └────────────────────────────────────────────────────────────────┘
```

Three operator role views live in `apps/console/`:
- **OPS** — full-screen map, alert queue, entity detail panel, dispatch
- **COMMAND** — situation feed / map / resource status, handoff brief
- **DEV** — entity explorer, adapter registry, message inspector, schema validator, inference dashboard

---

## Trained ML models (real-data, calibrated)

| Model | Task | Real samples | Metric |
|---|---|---|---|
| `fire_danger_classifier` | 5-class wildfire danger | FIRMS VIIRS + OpenMeteo | F1 0.982 |
| `fire_intensity_regressor` | FRP regression | FIRMS VIIRS + OpenMeteo | — |
| `compound_hazard_scorer` | Multi-hazard 0-100 risk | FIRMS + USGS + OpenMeteo, 1,600 samples | MAE 0.020 |
| `aftershock_lstm` ⚡ | P(M3.0+ aftershock in 12h) | USGS ComCat, 54,411 events | AUC 0.81 |
| `storm_severity_classifier` | NOAA 5-class severity | NOAA Storm Events 2022-2024, 215k rows | F1 0.843 |
| `eonet_hazard_classifier` | NASA EONET multiclass | 5,000 events | F1 0.999 |
| `lightning_ignition_classifier` | P(fire ignition / 25km / 72h) | NOAA + 1.08M FIRMS hotspots | AUC 0.667 |
| `aircraft_anomaly_lstm` ⚡ | LSTM autoencoder anomaly score | OpenSky live, 5,601 aircraft | val MSE 0.003 |
| `maritime_sar_classifier` | P(marine SAR-relevant event 100km/24h) | NDBC + NOAA marine, 30k buoy-hours | F1 0.899, AUC 0.891 |
| `drought_severity_classifier` | USDM drought class | USDM 50-state + OpenMeteo, 5,250 weeks | F1 0.687 |
| `heatwave_transformer` ⚡ | Multi-horizon extreme-heat (1d/3d/7d) | OpenMeteo, 30 cities × 2 yr | AUC 0.96 avg |
| `damage_classifier` / `damage_vision` | Building damage 4-class | xBD (Maxar) | — |
| `corrosion_classifier` / `corrosion_vision` | Infrastructure corrosion | CodeBrim | — |
| `flood_classifier` / `flood_risk_regressor` | Flood inundation + risk | FloodNet + GDACS-derived | — |
| `slope_stability_classifier` | Landslide risk | physics + GBIF | — |
| `counter_uas_classifier` | Authorized vs rogue UAS | rule + synthetic + features | — |
| `vehicle_classifier` / `crowd_estimator` / `deforestation_classifier` / `pipeline_anomaly_classifier` | Specialized | mixed real / synthetic | — |
| `wildfire_lstm` | Sequence FRP regression | (synthetic — FIRMS public CSV span insufficient for daily seq) | MAE 0.017 |

⚡ = deep learning model.

All model metadata (`*_meta.json`) is committed alongside the binary in `packages/c2_intel/models/` — features, training samples, data sources, calibration percentiles, and feature importances.

---

## Capability summary

- **32 sensor / protocol adapters** — see `packages/adapters/`
- **15 vertical domain modules** — wildfire, urban_sar, military, maritime, utilities, agriculture, oilgas, construction, wildlife, flood, traffic, forestry, ports, mining, pipeline
- **16 trained ML models** with real-data calibration metadata
- **Engagement authorization workflow** — human-in-the-loop state machine, single mandatory gate, signed decisions, role-checked, TTL-enforced (`packages/c2_intel/engagement_authorization.py`)
- **Track-to-weapon decision support ranker** — surfaces ranked options to operator (`packages/c2_intel/weapon_target_ranker.py`)
- **3D airspace deconfliction engine** — 0.5s tick, ~200-asset capacity (`packages/deconfliction/`)
- **Hungarian + greedy task assignment** — `packages/swarm/swarm_planner.py`
- **OPA-gated actuator commands** — Ed25519-signed policies, every physical command pre-checked
- **Full security stack** — RBAC, classification labels, mTLS, sensor signing, anti-replay, world-model HMAC chaining, MFA
- **Offline-resilient edge agent** — replay buffer, degraded-mode UI, autonomous waypoint execution
- **PACE comms** — Meshtastic + LoRaWAN + Starlink + cellular fallback
- **ATAK / CoT 2-way** — entity publish + waypoint dispatch (MIL-STD-2525 type codes)
- **Sentinel Hub adapter** — on-demand Sentinel-1 SAR / Sentinel-2 optical / Landsat thermal / MODIS imagery + STAC catalog search + zonal statistics

---

## Engagement authorization invariant

The single hard line in the codebase: **no kinetic action is dispatched without a signed human authorization.**

Implementation:
- `EngagementAuthorizationGate.authorize()` is the only API surface that emits `ENGAGEMENT_AUTHORIZED`.
- It requires: completed PID → ROE check → deconfliction check → ranked options surfaced → operator decision = `AUTHORIZE` → `selected_option` references a viable option → option remains ROE-compliant + deconflicted at decision time → operator role meets the engagement-class requirement → cryptographic signature on the decision payload verifies.
- `AUTHORIZED` carries a TTL; expiry without `ENGAGEMENT_COMPLETE` auto-emits `ENGAGEMENT_DENIED` for audit.
- Every state transition is recorded with timestamp + transition + payload, pushed to an HMAC-chained audit log.

Decision support (target tracks, ranked weapon-target pairings, fire-control solutions presented as options, pattern-of-life, BDA) is in scope. Closed-loop autonomous engagement code that bypasses the human authorization step is out of scope.

---

## Markets

**Civilian** — wildfire response, search & rescue, flood / hurricane / tornado response, infrastructure inspection (pipeline, bridge, dam, power grid), commercial UAV fleets, port security, anti-poaching, agriculture monitoring, public health (heat / smoke / drought / air quality).

**Federal / DoD** — Combat Readiness Deployment, Agile Combat Employment, force protection perimeter, base/FOB monitoring, counter-UAS, CASEVAC escort, BDA, mission rehearsal, deployable C2 readiness, ISR, HADR coordination, blue-force tracking, ATAK interop. Currently pursuing the **USAF CANVAS contract for Combat Readiness Deployments**.

---

## Standards & interoperability

- **CoT / ATAK** — MIL-STD-2525B/D type codes, 2-way publish + waypoint
- **MAVLink** — drone autopilot integration
- **NMEA-0183, NMEA-2000** — maritime, NDBC buoys, vessel telemetry
- **AIS** — vessel position + traffic
- **J1939, ISOBUS** — heavy machinery / construction / agriculture CAN bus
- **Modbus, BACnet** — industrial control + building automation
- **OPA / Rego** — policy gating, Ed25519-signed
- **STAC** — SpatioTemporal Asset Catalog (via Sentinel Hub adapter)
- **Link 16 / VMF** — referenced in scope; gateway adapter not yet shipped
- **9-line MEDEVAC, SALUTE, SITREP** — reporting templates in `packages/domains/military.py`

---

## Console

`apps/console/` — Next.js 14, three role views.

- Font: IBM Plex Mono (data/body) + Orbitron (headings/labels)
- Color: `#00FF9C` green, `#FFB300` amber, `#FF3B3B` red, `#4FC3F7` blue, background `#080C0A`
- WebSocket entity stream + REST alerts/missions/geofences

The OPS view is built around the spec invariant: alert → INVESTIGATE button → entity selected → map flies to entity → entity detail slides in → DISPATCH — under 3 seconds operator time.

---

## Deployment

- **Compose** — `infra/docker/docker-compose.yml`
- **Helm** — `infra/helm/heli-os/`
- **Render** — production deployment running; see internal docs
- **mTLS profile** — `docker compose --profile mtls up` for high-security deployments
- **SITL** — `infra/docker/docker-compose.sitl.yml` for software-in-the-loop testing
- **Edge agent** — Dockerfile at `packages/agent/Dockerfile.agent`, deploys to disconnected/degraded environments

---

## Status

- 16 trained ML models, all with real-data metadata
- 32 sensor/protocol adapters
- 15 vertical domain modules
- Engagement authorization workflow + decision support ranker shipped
- OPS / COMMAND / DEV console views shipped
- Offline edge agent + replay buffer shipped
- Helm + Compose deployment paths
- ML inference dashboard in DEV view

---

## Contact

Branca.ai Inc.
Licensing & partnerships: [licensing@branca.ai](mailto:licensing@branca.ai)
General: [https://branca.ai](https://branca.ai)

This software is proprietary. See [LICENSE](LICENSE).
