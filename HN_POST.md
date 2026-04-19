# Show HN: Heli.OS — open source autonomous coordination for drones and disaster response

**Title (under 80 chars):**
> Show HN: Heli.OS – open source autonomous UAV coordination for disaster response

---

## Body

At 14:33 on a Tuesday, a camera on a ridge detects smoke. Here's what happens in the next 90 seconds — with no human touching a keyboard:

1. A sensor sends `{class: "smoke", confidence: 0.91, lat: 34.12, lon: -118.34}`
2. A trained ML model (GradientBoosting, ONNX, <1ms) scores it CRITICAL and picks SURVEY as the mission type
3. The nearest available UAV gets waypoints over MQTT, adjusted for terrain using SRTM elevation data
4. An alert fires. If no operator acknowledges within your configured timeout, it escalates via webhook
5. The operator opens the console and sees a live HLS video feed from the drone already on-site

The operator's job: watch, verify, decide whether to dispatch ground resources. The software handles everything before that decision.

---

This is **Heli.OS** — the open source alternative to Anduril's LatticeOS for civilian operators.

The coordination software that does this well is closed-source, defense-export-controlled, and costs millions per year. It's not available to a county fire department, a maritime SAR team, an NGO running conservation drones, or a startup building inspection UAVs. We built Heli.OS to change that.

**What's in it:**

- Multi-sensor fusion with Kalman EKF tracking across camera, ADS-B, AIS, MAVLink, CoT/ATAK
- Autonomous mission dispatch trained on 108,000 real-world observations (NASA FIRMS fire data, NOAA Storm Events, GBIF wildlife) — no LLM in the critical path
- Self-improving: retrain the mission planner on your own operator decisions with one command
- Live HLS video from any RTSP source
- Mission replay for incident debrief
- SRTM terrain following on all waypoints
- CoT/ATAK bidirectional bridge
- 30-minute hardware integration (one base class, two methods)
- Full observability stack (Prometheus, Grafana, Jaeger)

**Seven mission types:** SURVEY, MONITOR, SEARCH, PERIMETER, ORBIT, DELIVER, INSPECT

**Quick start:**
```
git clone https://github.com/bigmt-ai/heli-os
cd heli-os && cp .env.example .env
cd infra/docker && docker compose up
python scripts/seed_demo.py  # seeds demo data so the map isn't empty
```

The stack runs entirely local — all ML inference is on-device, no external API calls required. Designed to work air-gapped on field hardware.

AGPL v3. The trained models are included under the same license. Commercial license available for proprietary deployments — see COMMERCIAL_LICENSE.md.

---

**What we're looking for:**

If you're running drones for wildfire detection, SAR, maritime patrol, pipeline inspection, or agriculture — we want to talk. Your operational data (anonymized) is what makes the ML models better for everyone. We're offering free pilot deployments in exchange for feedback.

GitHub: https://github.com/bigmt-ai/heli-os

---

## Subreddit posts (same day as HN)

### r/drone
**Title:** I built an open source autonomous coordination platform for UAV fleets — wildfire response, SAR, inspection [Show and Tell]

Tired of watching expensive closed-source platforms get all the serious drone coordination work. Built Heli.OS — free, AGPL v3, runs on docker compose. A camera detects smoke → UAV dispatches automatically in under 2 seconds, no LLM required. Trained ML models included. Would love feedback from anyone running real operations.

GitHub: https://github.com/bigmt-ai/heli-os

### r/ARES (Amateur Radio Emergency Service)
**Title:** Open source autonomous coordination platform for emergency response — CoT/ATAK compatible

Built Heli.OS for civilian emergency response coordination. Bidirectional CoT/ATAK bridge, so entities show up on your TAK devices. Runs air-gapped. Looking for ARES teams who'd want to pilot it. AGPL v3, self-hosted.

### r/searchandrescue
**Title:** Open source platform for autonomous UAV SAR coordination — looking for field testers

Built an open source alternative to the expensive closed-source drone coordination platforms. Includes a trained ML model for SEARCH missions (missing person, distress signal, overdue vessel → grid pattern dispatch). Would love to talk to any SAR teams running drone ops.

---

## Cold email template (county fire departments)

**Subject:** Free pilot — autonomous wildfire detection software for [COUNTY NAME] Fire

Hi [Name],

I'm the founder of BigMT, and we just open-sourced Heli.OS — autonomous coordination software for wildfire response using drone fleets.

When a camera detects smoke, it automatically dispatches the nearest available UAV, adjusts altitude for terrain, and alerts your operator — in under 90 seconds, no human touching a keyboard until the drone is already on-site.

It runs on any hardware, works air-gapped, and costs nothing to self-host. The ML model was trained on NASA FIRMS fire detection data and NOAA Storm Events.

We're offering free pilot deployments to fire departments willing to give us feedback. I'll set it up, train your team, and support the deployment at no cost.

Would a 30-minute call make sense?

[Name]
kyle@branca.ai
