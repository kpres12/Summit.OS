# Continuous Monitoring (ConMon) + Classified-Network Deployment

Companion to `RMF_CONTROL_MAPPING.md`. Documents the operational
posture Heli.OS achieves for DoD ATO submission, separated into:

1. **ConMon** — continuous monitoring telemetry stream meeting NIST
   SP 800-137.
2. **Classified-network deployment variant** — operating Heli.OS on
   SIPR / JWICS / coalition-only networks.
3. **FIPS 140-3 cryptographic module status**.

Branca.ai Inc. confidential. Revision 0.1 (2026-04-25).

---

## 1. Continuous Monitoring (ConMon)

### 1.1 Telemetry stream

Heli.OS continuously emits security and operational telemetry via two
channels:

| Stream | Cadence | Destination | Contents |
|---|---|---|---|
| `summit/observability/security` (MQTT topic) | event-driven | site SIEM | auth events, RBAC denials, MFA challenges, mTLS verify failures, OPA denial reasons, anti-replay rejections, signature mismatches |
| `summit/observability/health` (MQTT topic) | 60s | site SIEM + ops dashboard | adapter status, entity counts, event-loop latency, DB connection pool, certificate expiry, key rotation due-dates |

Both streams produce CycloneDX-1.5-compatible attestation events that
roll up to the deployment's ConMon submission cadence (NIST 800-137
default monthly; per-mission deployments may submit weekly or daily).

### 1.2 ConMon control mapping

| 800-137 area | Heli.OS module | Status |
|---|---|---|
| Information Security Continuous Monitoring (ISCM) Strategy | This document + per-deployment SSP | 🟦 |
| ISCM Program | `packages/observability/db_logger.py` audit pipeline | ✅ |
| Security automation | `infra/policy/*.rego` Ed25519-signed policies | ✅ |
| Asset inventory | `scripts/generate_sbom.py` (CycloneDX 1.5) | ✅ |
| Configuration baseline drift | `packages/security/world_model_hmac.py` chained-HMAC integrity | ✅ |
| Vulnerability monitoring | Trivy container scan in CI; SBOM diff vs CISA KEV | 🟦 |
| Patch management | Semver-pinned deps in `packages/sdk/pyproject.toml` + Helm chart immutable tags | ✅ |
| Event monitoring | Append-only audit log, MQTT security topic | ✅ |
| Performance monitoring | Prometheus metrics, Grafana dashboards in `infra/` | ✅ |

### 1.3 SIEM integration

Heli.OS ships with first-class connectors for the SIEMs DoD typically
deploys:

- **Splunk Enterprise Security** — MQTT bridge in `infra/observability/`
  pushes the security topic via the Splunk HTTP Event Collector.
- **Elastic Security** — Filebeat / Logstash configs in
  `infra/observability/elastic/` parse the JSON event stream.
- **Microsoft Sentinel** — Log Analytics agent compatible.
- **Wazuh / OSSEC** — host-based monitoring agent compatible with our
  audit log format.

### 1.4 Sample audit event

```json
{
  "ts": "2026-04-25T18:42:11.482Z",
  "deployment_id": "heli-fob-2",
  "event_type": "engagement_authorization.AUTHORIZED",
  "principal": "fob-cmd-2",
  "principal_role": "mission_commander",
  "resource": {
    "case_id": "5f3c78c1-...",
    "track_id": "T-2402",
    "engagement_class": "counter_uas",
    "weapon_class": "soft_kill"
  },
  "policy": {
    "intent_id": "OPLAN-2026-AOR3-INTENT-007",
    "pathway": "conditional_delegation",
    "reason": "uplink_age=240s > threshold=90s"
  },
  "signature": {
    "alg": "Ed25519",
    "fingerprint": "ab12...c4d9",
    "verified": true
  },
  "audit_chain_prev_hmac": "9f4a...8e1b",
  "audit_chain_this_hmac": "1c2d...30af"
}
```

The chained HMAC means tampering with any audit row invalidates the
chain forward — detectable at next ConMon submission.

---

## 2. Classified-Network Deployment

Heli.OS is built to deploy in three network postures:

| Posture | Network | Use case |
|---|---|---|
| **Commercial / civilian** | Public internet, customer VPC | Civilian disaster response, commercial UAV ops |
| **Federal unclassified** | DoD CAC NIPR-attached or CONUS internal | HADR, Combat Readiness Deployment training |
| **Classified** | SIPR / JWICS / coalition-only | Specific federal SKUs at FOUO+, SECRET, TS — per-deployment authorization |

### 2.1 SIPR / JWICS variant

The classified variant differs from the commercial variant in:

| Element | Commercial | Classified |
|---|---|---|
| Network | Internet w/ TLS 1.3 | Air-gapped or Cross-Domain |
| External adapters | Sentinel Hub, FIRMS, Open-Meteo, GDELT, etc. | DISABLED — replaced with site-local data feeds |
| Identity | OIDC (Auth0, Keycloak, Cognito) | DISA EAMS CAC, ICAM federated identity |
| Crypto | OS / OpenSSL | FIPS 140-3 validated module (BoringSSL FIPS, OpenSSL FIPS provider) |
| Container registry | Docker Hub / GHCR | DoD-approved registry (e.g. Iron Bank) |
| Runtime | Linux kernel, vendor-default | DISA-hardened (e.g. UBI 9 with STIG profile applied) |
| Logging | Self-host or SaaS | Site SIEM only (no exfiltration) |
| Time source | NTP pool | NIST-traceable site time / GPS-disciplined |

### 2.2 Cross-Domain transfer

The classified variant supports unidirectional transfer from low-side
(commercial / federal-unclassified) to high-side (classified) via
diode-style accredited transfer mechanisms:

- **Sensor metadata only** — no full EO scenes — flows low → high
  through site-approved Cross-Domain Solution.
- **Signed policy bundles** flow high → low through air-gapped manual
  transfer (USB, optical) — the Ed25519 signature lets low-side nodes
  verify intent without low-side ever observing classified context.
- **No** un-attested traffic crosses either way.

### 2.3 STIG / CIS hardening

Compose + Helm overlays for STIG / CIS hardening:

- `infra/helm/heli-os/values-stig.yaml` — STIG-hardened values
- `infra/docker/docker-compose.stig.yml` — Compose overlay disables
  developer conveniences (debug ports, default credentials, world-
  readable artifacts) and enforces SELinux policies.

These overlays apply when `HELI_DEPLOYMENT_PROFILE=stig` is set.

---

## 3. FIPS 140-3 Cryptographic Module Status

Heli.OS uses standard cryptographic primitives that have FIPS 140-3
validated providers available, but **the validation package is
delivered per-deployment, not bundled with Heli.OS**:

| Primitive | Library used | FIPS 140-3 provider |
|---|---|---|
| TLS 1.3 | OpenSSL | OpenSSL 3.0+ FIPS provider |
| Ed25519 | cryptography (PyCA) | OpenSSL FIPS 3.x backend |
| HMAC-SHA256 | hashlib (Python) | OpenSSL FIPS backend when interpreter is FIPS-built |
| AES-256-GCM | cryptography (PyCA) | OpenSSL FIPS provider |
| Argon2id | argon2-cffi | (no FIPS validation; for password hashing only — not used at TLS surface) |
| Ed25519 (JWS) | python-jose | OpenSSL FIPS provider |

For a FIPS-validated deployment, the runtime container is built with
**FIPS-validated OpenSSL** (e.g. RHEL 9 FIPS mode, Ubuntu Pro FIPS, or
the BoringSSL FIPS module on Google Distroless+Coreutils-FIPS). Heli.OS
then runs inside that container without modification.

We pursue a third-party FIPS attestation as part of the CANVAS Phase 3
SOW (see white paper §5.3).

---

## 4. Per-deployment artifact pack

Each accreditation submission ships with:

1. **System Security Plan (SSP)** — per-deployment, authored against
   `RMF_CONTROL_MAPPING.md` baseline.
2. **CycloneDX SBOM** — generated from `scripts/generate_sbom.py`
   against the exact build artifact.
3. **Configuration baseline** — Helm values / Compose env captured at
   release.
4. **Audit chain root** — first HMAC of the chain at deployment.
5. **Policy bundle hashes** — SHA-256 of every signed `.rego` file.
6. **Operator role roster** — RBAC registration data.
7. **Continuous Monitoring Plan** — site-specific cadence + escalation
   contacts.
8. **Incident Response Plan** — site SECURITY.md + escalation tree.

---

## 5. Operational running list (gaps)

For a real ATO, the following additional items need owners + dates:

⬜ Personnel screening (PS-3) — process, customer-side
⬜ Insider threat program (PM-12) — process, customer-side
⬜ Privacy impact assessment (where applicable, e.g. wildfire ops over
   civilian populations) — case-by-case, customer-side
⬜ Authorization Boundary diagram — produced per deployment
⬜ Continuous Monitoring dashboard exemplars — produced per deployment
⬜ Cross-Domain Solution accreditation evidence — site-specific

These are operational deliverables that complete the package; the
codebase is ready to feed each of them.
