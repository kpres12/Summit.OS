# Summit.OS MQTT and Stream Topics

Canonical topics and streams (v1):

MQTT (publishers → subscribers)
- telemetry/{device_id}
- alerts/{alert_id}
- missions/{mission_id}
- health/{node_id}/heartbeat
- images/{device_id}

Redis Streams
- telemetry_stream: device telemetry records
- alert_stream: alerts
- mission_stream: mission updates
- observations_stream: fused observations (contract: observation.schema.json)
- observations_dlq: invalid observations or processing errors
- intelligence_dlq: errors during advisory generation

Notes
- Observations idempotency key: sha1(class, lat, lon, confidence, ts, source), 10 min TTL
- Consumer groups: fusion, intelligence on observations_stream
- Retention: telemetry(100k), alerts(200k), missions(50k) (approximate xtrim)
