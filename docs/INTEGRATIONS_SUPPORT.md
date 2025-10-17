# INTEGRATIONS_SUPPORT.md

This document tracks supported upstream integrations, licenses, and failover options.

Upstreams (runtime dependencies)
- ROS 2 (Apache-2.0) — optional; used at edge. Fallback: direct MAVLink via Tasking.
- PX4 (BSD-3) — optional; autopilot firmware. Fallback: ArduPilot/Gazebo SITL.
- ArduPilot (GPLv3) — optional, out-of-process. Fallback: PX4.
- OpenCV (Apache-2.0) / PCL (BSD) — optional AI/3D libs. Fallback: disable models or use CPU-only paths.

Summit.OS stance
- Do not fork/ship proprietary versions of these. Integrate via adapters.
- Keep GPL components out-of-process to avoid copyleft.

Version matrix (initial)
- ROS 2: Humble/Iron (edge-only)
- PX4: 1.14.x
- ArduPilot: 4.5.x
- OpenCV: 4.9+
- PCL: 1.13+

Operational guidance
- Pin versions in container images and requirements.txt.
- Conformance tests via SITL in CI (PX4+ArduPilot).
- Use circuit breakers/timeouts in API Gateway and microservices.
- Maintain internal mirrors; hotfix with patch overlays if required.
