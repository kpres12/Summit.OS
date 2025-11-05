# PLAINVIEW — The Autonomous Oilfield Platform (on Summit.OS)

Plainview delivers total operational awareness, predictive safety, and autonomous intervention for oil & gas infrastructure — from wellhead to pipeline — built entirely on Summit.OS services.

## Product framing
- Summit.OS Core: Fabric (comms mesh), Fusion (perception/world model), Intelligence (reasoning/AI), Tasking (missions/scheduling), API Gateway (operator/API surface).
- Plainview Edge Agents: robots, drones, fixed masts running Summit Edge Node (ROS2 bridges + local inference) that speak the same Fabric/contract.
- Plainview Cloud: Console + APIs backed by Summit services; air‑gapped or cloud.

## Industry modules mapped to Summit.OS
1) PipelineGuard — leak/spill prevention and rapid isolation
- Fusion: multi-sensor leak detection (optical/thermal/gas/acoustic) → detections → observations_stream.
- Intelligence: leak scoring + root-cause hints → alerts with severity + recommended action.
- Tasking: auto-dispatch verify/isolate missions to drones/rovers; safety envelopes enforced.

2) RigSight — visual/thermal situational awareness
- Fusion vision stack (apps/fusion): ONNX/Torch models via VisionInference; cameras/FLIR; detections published to MQTT/streams.
- World model overlays: heat plumes, methane spikes, person-in-zone.

3) ValveOps — autonomous actuation
- Tasking: mission primitives for valve identification, alignment, torque sequences, verification.
- ROS2 bridge adapters (packages/summit-os-sdk/bridges) drive arms/UGVs; safety interlocks (pressure/torque/temperature) checked in Intelligence before/while executing.
- Digital twin updates (world model + audit log) on every actuation.

4) FlowIQ — analytics + predictive maintenance
- Intelligence: time‑series models for pressure/flow/temperature anomalies; asset health scoring.
- Training: extend scripts/ai_training_pipeline.py to real data sources; export ONNX for edge/cloud scoring.

## Contracts (no new framework)
- REST + MQTT per docs/API.md and packages/contracts/TOPICS.md.
- Plainview uses the same endpoints; adds domain payloads and topics (see contracts addendum below).
- Schemas stay additive; no breaking changes to v1.

## Data flow (Plainview)
Edge sensors/robots → Fabric (MQTT/Redis) → Fusion (detections, world model) → Intelligence (risk, recommend, guard rails) → Tasking (missions) → API/Console.

## Missions (examples)
- Daily Valve Integrity Sweep (two UGVs) → task graph: navigate → detect valve → align → torque test → log curves.
- Leak Verify + Isolate → trigger from detection → dispatch drone + UGV → visual confirm → close nearest valves → report.
- Valve Safety Test → scheduled or on‑demand.

See missions/plainview/*.yaml for concrete templates.

## Topics addendum (domain)
Keep base topics; add:
- plainview/leaks → leak/spill detection events (from Fusion or Edge).
- valves/{asset_id}/command → actuation requests (ValveOps).
- valves/{asset_id}/status → position/torque curves, interlock state.
- pipeline/pressure/{segment_id} → time‑series taps (optional if SCADA bridged).

Details in packages/contracts/TOPICS_PLAINVIEW.md.

## Model strategy
- Vision: ONNX export for leak/smoke/person; Fusion loads via FUSION_MODEL_PATH/MODEL_REGISTRY.
- Time‑series: FlowIQ models (e.g., STL/ARIMA/LSTM as needed) trained via scripts/ai_training_pipeline.py; serialized (ONNX or torchscript) and deployed to edge/cloud.
- Validation gates: deploy if accuracy/PR thresholds met; telemetry-driven drift checks.

## Safety & compliance hooks
- Interlocks enforced in Intelligence before Tasking commits: torque limit, pressure window, temperature.
- Audit log: every recommendation/action → observations_stream + REST audit endpoint (existing alerts/tasks contracts suffice, additive fields only).

## Deployment modes
- Air‑gapped (k3s on site) or cloud + edge; same compose/k8s manifests with Plainview mission templates and model artifacts mounted.

## MVP scope (0–6 months)
- ValveOps: detect/align/torque sequence + logging.
- PipelineGuard: leak detection + verify/isolate mission.
- RigSight: live map with thermal overlays; anomaly alerts.
- FlowIQ: baseline anomaly scoring and asset health.

## What to build in Summit.OS (delta)
- Add domain topics doc and mission templates (this commit).
- Extend ai_training_pipeline.py to real datasets (future PR).
- Provide example adapters (ROS2 arm/pump) via summit-os-sdk (future PR).
