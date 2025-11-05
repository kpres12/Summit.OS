# Plainview Topics (Addendum to TOPICS.md)

This document defines domain-specific topics for the Plainview oil & gas stack. It extends packages/contracts/TOPICS.md without breaking existing contracts.

MQTT (publishers → subscribers)
- plainview/leaks
  - Leak/spill detection events (vision/thermal/gas/acoustic)
  - Payload:
    ```json
    {
      "id": "LEAK-2025-0001",
      "ts": 1730784000.123,
      "source": "fusion|edge",
      "asset_id": "SEG-12",
      "location": { "lat": 31.1234, "lon": -102.5678 },
      "class": "METHANE|OIL|WATER|UNKNOWN",
      "confidence": 0.87,
      "evidence": { "thermal": 0.82, "optical": 0.76, "gas_ppm": 1250 },
      "severity": "LOW|MED|HIGH|CRITICAL"
    }
    ```
- valves/{asset_id}/command
  - Actuation commands emitted by Tasking/Operator
  - Payload:
    ```json
    {
      "command": "OPEN|CLOSE|TORQUE_TEST|INDEX|STOP",
      "params": { "target_angle_deg": 90, "torque_limit_nm": 45, "speed": 0.4 },
      "safety": { "pressure_ok": true, "temp_ok": true },
      "mission_id": "M-VALVE-001",
      "request_id": "REQ-123"
    }
    ```
- valves/{asset_id}/status
  - Robot/edge publishes status/telemetry
  - Payload:
    ```json
    {
      "mission_id": "M-VALVE-001",
      "state": "IDLE|ALIGNING|ACTUATING|VERIFYING|DONE|ERROR",
      "angle_deg": 45.2,
      "torque_nm": 38.1,
      "torque_curve": [[0.0,0.0],[0.2,5.1],[0.4,12.3]],
      "interlocks": { "e_stop": false, "over_torque": false, "over_temp": false },
      "notes": "ok"
    }
    ```
- pipeline/pressure/{segment_id}
  - Optional time-series taps (bridge from SCADA)
  - Payload: { "ts": <unix>, "kpa": <float> }

Redis Streams
- operations_stream
  - Actuation lifecycle events (valve ops/leak isolate)
  - Records mirror MQTT payloads with correlation ids for audit

Notes
- QoS 1 for status/pressure feeds; QoS 2 for commands and leaks.
- Retain optional on valves/{asset_id}/status.
