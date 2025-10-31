# Summit.OS Adapter Templates

Quick-start templates for integrating devices via thin adapters that publish normalized contracts to Summit.OS.

## Templates
- MAVLink (drone autopilots): `summit_os/bridges/mavlink_bridge.py`
- ONVIF (IP cameras): `summit_os/bridges/onvif_bridge.py`
- Generic HTTP (webhooks → MQTT): `summit_os/bridges/generic_http_adapter.py`

## Install
```bash
pip install -e ".[mqtt,adapters]"
```

## Run
```bash
# MAVLink (SITL at 127.0.0.1:14550)
python -m summit_os.bridges.mavlink_bridge --device-id drone-001 --udp 127.0.0.1:14550

# ONVIF
python -m summit_os.bridges.onvif_bridge --device-id cam-001 --host 192.168.1.10 --user admin --password pass

# Generic HTTP (map vendor.lat/vendor.lng fields)
python -m summit_os.bridges.generic_http_adapter --device-id sensor-001 --register \
  --map-lat vendor.lat --map-lon vendor.lng --port 9009 --api http://localhost:8000
# Then POST telemetry:
curl -X POST http://localhost:9009/telemetry -H 'content-type: application/json' \
  -d '{"vendor": {"lat": 34.1, "lng": -117.3}, "temp_c": 21.5}'
```

## Contract Topics
- Telemetry: `telemetry/<device_id>`
- Heartbeat: `health/<device_id>/heartbeat`
- Detections (if applicable): `detections/<device_id>`

## Conformance
- Validate JSON payloads against contracts (optional schemas under `packages/contracts`).
- Ensure QoS 0/1 as appropriate; avoid payloads > 5 MB.

Run SDK tests:
```bash
cd packages/summit-os-sdk
pip install -e ".[dev]"
pytest -q
```
