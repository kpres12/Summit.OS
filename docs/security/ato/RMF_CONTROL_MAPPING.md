# NIST 800-53 Rev 5 Control Mapping — Heli.OS

This document maps NIST 800-53 Rev 5 controls to concrete implementation
points in the Heli.OS codebase, infrastructure, and SDLC, for use in
DoD ATO / RMF, CMMC L2, FedRAMP Moderate, or equivalent assessments.

Status legend:
  ✅ Implemented and enforceable in code/config
  🟡 Implemented; requires per-deployment configuration
  🟦 Partial — requires evidence package + customer attestation
  ⬜ Not yet implemented

Revision: 0.1 (2026-04-25). Branca.ai Inc. confidential.

---

## AC — Access Control

| Control | Description | Implementation | Status |
|---|---|---|---|
| AC-2 | Account management | OIDC integration; user store at `packages/security/user_store.py`; admin API in `apps/api_gateway/` | ✅ |
| AC-3 | Access enforcement | RBAC at `packages/security/rbac.py`; OPA policy gate at `infra/policy/*.rego` (Ed25519 signed) | ✅ |
| AC-4 | Information flow enforcement | Classification labels at `packages/security/classification.py`; per-org row-level isolation in Postgres; tenant scoping in `packages/multi_tenant/` | ✅ |
| AC-6 | Least privilege | RBAC roles in `packages/security/rbac.py`; engagement role matrix in `packages/c2_intel/engagement_authorization.py` (`_ROLE_MATRIX`) | ✅ |
| AC-7 | Unsuccessful logon attempts | MFA module at `packages/security/mfa.py`; rate limiting in API gateway | 🟡 |
| AC-12 | Session termination | TTL on sessions in OIDC config; engagement-authorization TTL with auto-DENY | ✅ |
| AC-17 | Remote access | mTLS via `packages/security/mtls.py` + `infra/proxy/`; API gateway requires authenticated session | ✅ |
| AC-20 | Use of external information systems | Adapter framework (`packages/adapters/`) explicitly enumerated; signed source list per deployment | ✅ |

## AU — Audit and Accountability

| Control | Description | Implementation | Status |
|---|---|---|---|
| AU-2 | Event logging | Append-only audit logger at `packages/observability/db_logger.py`; engagement-authorization records every transition | ✅ |
| AU-3 | Content of audit records | All audit events include timestamp, principal, transition, payload, source signature | ✅ |
| AU-4 | Audit log storage capacity | Postgres + S3 archival per `infra/`; retention configurable via `AUDIT_RETENTION_DAYS` | 🟡 |
| AU-9 | Protection of audit information | World-model HMAC chaining (`packages/security/world_model_hmac.py`); audit log is append-only with chained hashes | ✅ |
| AU-10 | Non-repudiation | Engagement-authorization decisions are Ed25519-signed; sensor signing at `packages/security/sensor_signing.py`; OPA policies signed at `packages/policy/signer.py` | ✅ |
| AU-12 | Audit record generation | Centralized event emission via `c2_intel/models.py:C2EventType`; all state transitions emit events | ✅ |

## CM — Configuration Management

| Control | Description | Implementation | Status |
|---|---|---|---|
| CM-2 | Baseline configuration | Helm chart at `infra/helm/heli-os/`; Compose at `infra/docker/`; Render config in repo | ✅ |
| CM-7 | Least functionality | Adapter registry only registers built-ins explicitly listed; no runtime adapter loading from untrusted sources | ✅ |
| CM-8 | System component inventory | SBOM generator at `scripts/generate_sbom.py` (CycloneDX 1.5); produces inventory of all 169+ components incl. ML models | ✅ |
| CM-11 | User-installed software | N/A — no end-user installable extensions; tenant org cannot inject server-side code | ✅ |

## CP — Contingency Planning

| Control | Description | Implementation | Status |
|---|---|---|---|
| CP-2 | Contingency plan | Edge agent (`packages/agent/`) operates offline with replay buffer; degraded-mode UI documented in console | ✅ |
| CP-9 | System backup | Postgres replication + S3 archival per `infra/`; entity store CRDT replication via mesh (`packages/mesh/`) | 🟡 |
| CP-10 | System recovery | Replay buffer + re-sync via mesh; documented RTO/RPO per deployment | 🟡 |

## IA — Identification and Authentication

| Control | Description | Implementation | Status |
|---|---|---|---|
| IA-2 | User identification + authentication | OIDC enforcement, MFA via `packages/security/mfa.py` (FIDO2/WebAuthn supported) | ✅ |
| IA-3 | Device identification + authentication | Sensor signing at `packages/security/sensor_signing.py`; per-device certificates issued by `packages/identity/ca.py` | ✅ |
| IA-5 | Authenticator management | Argon2id password hashing in `packages/security/passwords.py` (OWASP 2024 params); MFA enforcement | ✅ |
| IA-7 | Cryptographic module authentication | TLS 1.3 + mTLS; HMAC-SHA256 for world-model integrity; Ed25519 for policy and decision signatures | ✅ |

## IR — Incident Response

| Control | Description | Implementation | Status |
|---|---|---|---|
| IR-4 | Incident handling | Engagement-authorization audit trail provides forensic timeline; LoAC compliance evidence captured | ✅ |
| IR-6 | Incident reporting | `SECURITY.md` defines vulnerability disclosure process; security@branca.ai | ✅ |

## RA — Risk Assessment

| Control | Description | Implementation | Status |
|---|---|---|---|
| RA-3 | Risk assessment | Threat assessment framework at `packages/threat_assessment/base.py` with `ThreatLevel` enum; per-domain ontologies in `packages/c2_intel/ontology.py` | ✅ |
| RA-5 | Vulnerability scanning | Trivy container scans in CI; `dependency-check` against the SBOM; pinned dependency versions | 🟦 |

## SA — System and Services Acquisition

| Control | Description | Implementation | Status |
|---|---|---|---|
| SA-11 | Developer security testing | Pre-commit + CI pipelines in `.github/workflows/`; pytest + ruff + mypy + Trivy + SBOM generation gated | 🟦 |
| SA-15 | Development process, standards, and tools | Tooling pinned (Python 3.13, Node ≥20, Docker 24+); reproducible Helm + Compose | ✅ |
| SR-3 | Supply chain controls | CycloneDX SBOM generated at every build (`scripts/generate_sbom.py`); pinned dependencies; Ed25519-signed OPA policies | ✅ |
| SR-4 | Provenance | SBOM components include git commit + per-component SHA256; ML model provenance includes training data sources + metrics | ✅ |
| SR-11 | Component authenticity | Sensor signing, policy signing, world-model HMAC chaining all verify provenance at runtime | ✅ |

## SC — System and Communications Protection

| Control | Description | Implementation | Status |
|---|---|---|---|
| SC-7 | Boundary protection | API gateway with auth + RBAC enforcement; mTLS proxy profile; CORS allowlist | ✅ |
| SC-8 | Transmission confidentiality + integrity | TLS 1.3 everywhere; mTLS service-to-service; signed sensor messages | ✅ |
| SC-12 | Cryptographic key establishment + management | Per-deployment KMS or HashiCorp Vault integration via `packages/secret_store/`; per-device certs from internal CA | ✅ |
| SC-13 | Cryptographic protection | AES-256-GCM field-level encryption (`packages/security/field_encryption.py`); SHA-256 + Ed25519 for integrity/auth | ✅ |
| SC-23 | Session authenticity | Anti-replay nonces in `packages/security/anti_replay.py`; auth tokens with TTL | ✅ |

## SI — System and Information Integrity

| Control | Description | Implementation | Status |
|---|---|---|---|
| SI-3 | Malicious code protection | Trivy container scans; dependency provenance via SBOM | 🟡 |
| SI-4 | System monitoring | Prometheus metrics + Grafana dashboards in `infra/`; structured logs to `apps/observability/` | 🟡 |
| SI-7 | Software, firmware, integrity | World-model HMAC + Ed25519 policy signatures verified at load; SBOM hashes on every build artifact | ✅ |
| SI-10 | Information input validation | Pydantic schemas at every API boundary (`packages/schemas/`); OPA pre-flight on every actuator command | ✅ |

---

## Key load-bearing controls for engagement workflows

The following controls form the LoAC/3000.09 audit chain for the
engagement-authorization workflow:

1. **AU-3 / AU-10 / SR-11** — every state transition is recorded with
   timestamp, principal, transition, payload, signature.
2. **IA-2 / IA-7 / AC-3** — operator identity is OIDC-authenticated,
   MFA-verified, and role-checked against `_ROLE_MATRIX` before any
   `AUTHORIZE` decision is accepted.
3. **AC-6 / AU-12** — `EngagementAuthorizationGate.authorize()` is the
   only API surface that emits `ENGAGEMENT_AUTHORIZED`; no bypass path.
4. **SI-7 / SI-10** — option viability (ROE, deconfliction, role,
   signature) re-verified at decision time, not just at surface time.
5. **AC-12** — AUTHORIZED carries a TTL; expiry without
   `ENGAGEMENT_COMPLETE` auto-emits `ENGAGEMENT_DENIED` for the audit log.

---

## What's not yet implemented (fix list for a real ATO pursuit)

⬜ FIPS 140-3 cryptographic module attestation (currently using OS-provided crypto via stdlib)
⬜ Continuous Monitoring (ConMon) telemetry stream meeting NIST 800-137
⬜ Insider threat program (PM-12) — process, not code
⬜ Personnel screening (PS-3) — process, not code
⬜ Contingency plan testing record (CP-4) — operational evidence package
⬜ Authorization Boundary diagram for the SSP — produced per deployment
⬜ Privacy impact assessment artifacts (where applicable)
⬜ Classified-network deployment variant + crypto configuration for SIPR/JWICS

---

## Process

This control mapping is regenerated when:
1. A new control surface is added to the codebase (security, audit, policy module changes)
2. The SBOM contents materially change (new dependencies, new ML models)
3. ATO/RMF assessment cycle requires a fresh artifact

Run `python scripts/generate_sbom.py --version <release>` to produce the
matching SBOM artifact for any submission.
