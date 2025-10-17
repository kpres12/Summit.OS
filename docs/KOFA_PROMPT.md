# KOFA Perception & Decision Stack – Execution Prompt (v1)

You are the engineer responsible for delivering KOFA v1 on top of Summit.OS. Follow this exactly. Report blockers explicitly.

Inputs you must gather/produce first
- Data sources: list S3/GDrive paths or local dirs for (a) drone/tower RGB video, (b) thermal, (c) LiDAR (optional), (d) weather, (e) GNSS/IMU, (f) wildfire clips, (g) “after-pass” cleanup footage.
- Label taxonomies:
  - Detection classes: [smoke, flame, human, vehicle, downed_tree, equipment, wildlife, person_saw, person_blower, powerline_pole, insulator]
  - Segmentation classes: [mineral_soil, duff_litter, grass, brush, rock, road, drivable, non_drivable, structure, pole, gate, hydrant]
  - Tracking: Multi-object tracks for detector outputs
  - Line-cleanliness: binary mask “clean” mineral soil after pass
- Policies/SOPs: min fireline width by fuel/wind class, max grade, standoff heat thresholds, no-go polygons, comms-loss behaviors.

System contracts (already added)
- Perception→Planner JSON schema: packages/contracts/jsonschemas/perception.to.planner.schema.json
- Planner→Actuation JSON schema: packages/contracts/jsonschemas/planner.to.actuation.schema.json

Milestone plan (2-week sprints)
1) Sprint 1 – Data & Baselines
- Curate 50–100 short clips (day/night, dusty) and 200–500 images; balance positives/negatives.
- Label with boxes (smoke/flame/person/vehicle/tools), tracks (ByteTrack/DeepSORT IDs), and 100+ segmentation masks across terrain/asset classes.
- Fine-tune YOLOv8n/yolov8s and RT-DETR-tiny at 640px; export ONNX; measure P/R and latency on CPU and CUDA.
- Enable fusion vision path: set FUSION_ENABLE_VISION_AI=true and mount model to /models; publish image_b64 via MQTT images/# and verify /observations fills.

2) Sprint 2 – Tracking + Segmentation + Risk v0
- Add ByteTrack/DeepSORT into fusion pipeline (associate detector boxes → track_id, velocity).
- Run segmentation: SAM (prompted) for quick masks or Mask2Former distilled; run at 5–10 FPS.
- Build XGBoost risk v0 with features: weather, fuels, slope, proximity to assets, detection confidences; wire into Intelligence (INTELLIGENCE_ENABLE_XGB=true).

3) Sprint 3 – Physics + Surrogate + Planner v1
- Implement Rothermel surface spread + FARSITE-like propagation (batch, near-edge).
- Train small ConvNet/Transformer surrogate on local terrain/weather/state → predict spread rate/extent for planner tick <200 ms.
- Planner: cost map = risk + segmentation + spread; sample passes (width/depth) and enforce SOPs; produce actuation JSON per schema.

4) Sprint 4 – LLM Advisory + UX
- Hosted small LLM (4–8k ctx) that calls planning endpoints and summarizes events; ops-facing, read-only.
- Console: add “Explain” button → advisory + telemetry clips.

Latency targets (edge, Jetson Orin NX)
- Detect+track <40 ms/frame (25 FPS); Seg <80 ms (5–10 FPS); Planner tick <200 ms.

Concrete tasks and commands
1) Stand up dev stack
- make dev
- make health and make logs to verify all services

2) Wire perception model into Fusion
- Place ONNX at models/vision.onnx; set env:
  - FUSION_ENABLE_VISION_AI=true
  - MODEL_REGISTRY=/models
  - FUSION_MODEL_PATH=/models/vision.onnx
  - FUSION_CONF_THRESHOLD=0.6
- Publish frames to MQTT: topic images/camera-001 with JSON: {"image_b64": "...", "device_id": "camera-001", "lat": 37.42, "lon": -122.08, "ts_iso": "..."}
- Verify via GET http://localhost:8002/observations and API Gateway /v1/observations.

3) Train detector(s)
- Start from YOLOv8n/s or RT-DETR-tiny. Suggested hyperparams:
  - imgsz: 640, 1280; mosaic off for eval; conf_thres 0.25; iou 0.6
  - epochs 50–100; cosine lr; augment: hsv, flips, small rotations.
- Export: onnx opset=13; validate with onnxruntime. Store at models/vision.onnx.
- Measure latency with simple ORT benchmark on Orin and x86.

4) Tracking
- Use ByteTrack or DeepSORT; input=detector outputs; output adds track_id and velocity_mps. Integrate into Fusion and push to Redis stream observations_stream with ts_iso.

5) Segmentation
- For speed, start with SAM prompts (terrain priors) at 5 FPS; graduate to Mask2Former distilled head. Classes per taxonomy.
- Encode polygons in segments array in Perception→Planner payload.

6) Risk model v0
- Features: weather API or local sensors; fuels/terrain from GIS; detection confidences; proximity to structures/assets.
- Train XGBoost; export to models/risk_model.json; set INTELLIGENCE_ENABLE_XGB=true and INTELLIGENCE_MODEL_PATH.

7) Physics + surrogate
- Implement Rothermel-based spread in a separate module; generate synthetic sims over local terrain/weather grid.
- Train small surrogate; export ONNX to models/fire_surrogate.onnx; expose in Intelligence or near-edge service.

8) Planner v1
- Build cost map = w1*risk + w2*(non_drivable) + w3*(spread projection) + penalties for SOP violations.
- Sample sequences: choose min-cost plan that meets width/depth constraints; output Planner→Actuation JSON.

9) LLM advisory
- Provide read-only summaries and tool use to call planning endpoints; include “explain” text with links to clips.

10) Data/labeling loop
- Stand up lightweight labeler (CVAT/Label Studio). Bootstrapping: run pretrained, then correct; capture hard negatives.
- Each deployment: ingest new clips → active learning → retrain → validate → ship.

Acceptance criteria
- P/R: smoke P>0.85, R>0.8 day; P>0.75, R>0.7 night/dusty.
- Segmentation IoU ≥0.6 terrain classes; clean-line detection accuracy ≥0.8.
- Latencies within targets; planner tick <200 ms; memory <8 GB on Orin NX for full stack.
- End-to-end demo: images → detections+tracks → segments → risk/advisory → plan → actuation JSON.

Artifacts to deliver
- Datasets with labels + README; model weights (detector, seg, surrogate); ONNX exports; benchmark report; confusion matrices; calibration notes; failure case gallery.

Where to plug things in Summit.OS
- Fusion: apps/fusion/vision_inference.py (replace _postprocess), add tracker, emit to observations_stream.
- Intelligence: enable XGBoost, add surrogate scoring endpoint, advisory summaries.
- Contracts: you already have Perception/Planner schemas in packages/contracts/jsonschemas/.
- Planner: apps/tasking (plans and policy constraints), return actuation JSON; API Gateway proxies.

Report format (each sprint)
- What shipped, metrics vs targets, notable failures and proposed fixes, next sprint plan, dataset deltas, model registry versions.
