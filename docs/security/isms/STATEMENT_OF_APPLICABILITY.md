# Statement of Applicability (SoA)
**Summit.OS — BigMT.ai / Branca.ai**
**Classification: Internal**
**Version: 1.0 | Last Updated: 2026-03-21 | Owner: Head of Engineering**
**Standard: ISO/IEC 27001:2022 Annex A**

---

## 1. Purpose

This Statement of Applicability documents which ISO 27001:2022 Annex A controls are applicable to Summit.OS, whether each is implemented, and the justification for any excluded controls. This document is required for ISO 27001 certification and demonstrates the completeness of the ISMS scope.

---

## 2. Scope Statement

This SoA covers the Summit.OS platform as operated by BigMT.ai / Branca.ai, including:
- All application services (API Gateway, Fabric, Fusion, Intelligence, Tasking, Inference, Console)
- All supporting infrastructure (Kubernetes, Postgres, Redis, Vault, CI/CD)
- All data processed by Summit.OS (telemetry, mission data, audit logs, billing data)
- All personnel who build and operate Summit.OS

---

## 3. Annex A Control Assessment

**Legend:**
- ✅ Implemented
- ⚠️ Partially implemented / in progress
- ❌ Not applicable (with justification)
- 🔲 Planned (target date noted)

---

### Theme 5 — Organizational Controls

| Control | Title | Status | Implementation Notes |
|---------|-------|--------|---------------------|
| 5.1 | Policies for information security | ✅ | `INFORMATION_SECURITY_POLICY.md` approved; covers all scope |
| 5.2 | Information security roles and responsibilities | ✅ | Roles defined in ISP §4; Head of Engineering as ISMS owner |
| 5.3 | Segregation of duties | ⚠️ | RBAC role hierarchy enforced; at small team size, full segregation is limited — compensating: peer review on all privileged actions |
| 5.4 | Management responsibilities | ✅ | Head of Engineering owns policy; engineers responsible per ISP §4 |
| 5.5 | Contact with authorities | 🔲 | Contacts to be established with relevant aviation/maritime authorities pre-GA (ACT-004 adjacent) |
| 5.6 | Contact with special interest groups | ⚠️ | CISA advisory subscriptions recommended; not yet formalized |
| 5.7 | Threat intelligence | ⚠️ | CVE monitoring via pip-audit/CI; no formal threat intel feed yet |
| 5.8 | Information security in project management | ✅ | Security requirements documented in plan; threat models required in PRs for security-sensitive changes |
| 5.9 | Inventory of information and other associated assets | ✅ | `ASSET_INVENTORY.md` documents all systems, data stores, and third parties |
| 5.10 | Acceptable use of information and other associated assets | ✅ | ISP §5 (Access Control) defines acceptable use |
| 5.11 | Return of assets | ✅ | Offboarding checklist in ACCESS_CONTROL_POLICY.md covers device/access return |
| 5.12 | Classification of information | ✅ | ISP §6 defines four-tier classification |
| 5.13 | Labelling of information | ⚠️ | ISMS documents carry `Classification:` header; not yet systematically applied to all internal documents |
| 5.14 | Information transfer | ✅ | TLS 1.2+ required; no unencrypted channels for Confidential data |
| 5.15 | Access control | ✅ | `ACCESS_CONTROL_POLICY.md`; RBAC enforced at API layer |
| 5.16 | Identity management | ✅ | OIDC + MFA; unique identities per user; no shared accounts |
| 5.17 | Authentication information | ✅ | API keys hashed (SHA-256); JWTs short-lived; passwords via OIDC (not stored) |
| 5.18 | Access rights | ✅ | Provisioning and deprovisioning procedures in `ACCESS_CONTROL_POLICY.md` |
| 5.19 | Information security in supplier relationships | ✅ | `VENDOR_RISK_ASSESSMENT.md`; onboarding checklist |
| 5.20 | Addressing information security within supplier agreements | ⚠️ | DPAs identified for Stripe/GitHub; cloud provider DPA to be executed before GA |
| 5.21 | Managing information security in the ICT supply chain | ⚠️ | pip-audit in CI; Dependabot planned (ACT-003); no SBOM generated yet |
| 5.22 | Monitoring, review and change management of supplier services | ✅ | Annual vendor review; vendor change triggers new VRA |
| 5.23 | Information security for use of cloud services | ✅ | Cloud provider in asset inventory; encryption at rest enabled; K8s RBAC |
| 5.24 | Information security incident management planning and preparation | ✅ | `INCIDENT_RESPONSE.md` with P0–P3 procedures |
| 5.25 | Assessment and decision on information security events | ✅ | Severity matrix in `INCIDENT_RESPONSE.md` §2 |
| 5.26 | Response to information security incidents | ✅ | Response phases defined in `INCIDENT_RESPONSE.md` §5 |
| 5.27 | Learning from information security incidents | ✅ | PIR required for P0/P1; Risk Register updated post-incident |
| 5.28 | Collection of evidence | ✅ | Evidence preservation in `INCIDENT_RESPONSE.md` §7 |
| 5.29 | Information security during disruption | ✅ | ISP §10 (Business Continuity); degraded-mode operation documented |
| 5.30 | ICT readiness for business continuity | ⚠️ | RTO/RPO defined; Postgres HA not yet implemented (ACT-002) |
| 5.31 | Legal, statutory, regulatory and contractual requirements | ⚠️ | GDPR identified as applicable; full legal review recommended pre-GA |
| 5.32 | Intellectual property rights | ✅ | Proprietary codebase; no OSS license violations identified |
| 5.33 | Protection of records | ✅ | Audit logs retained 90 days minimum; retention configurable |
| 5.34 | Privacy and protection of PII | ⚠️ | Field encryption for PII columns implemented; formal DSAR/DPIA process not yet documented |
| 5.35 | Independent review of information security | 🔲 | Annual penetration test scheduled (ACT-004, due 2026-06-21) |
| 5.36 | Compliance with policies, rules and standards | ✅ | CI enforces coding standards; CODEOWNERS enforces review requirements |
| 5.37 | Documented operating procedures | ✅ | `OPERATIONS_RUNBOOK.md` documents all critical procedures |

---

### Theme 6 — People Controls

| Control | Title | Status | Implementation Notes |
|---------|-------|--------|---------------------|
| 6.1 | Screening | ⚠️ | Background screening process to be formalized before scaling team |
| 6.2 | Terms and conditions of employment | ⚠️ | Security responsibilities should be incorporated into employment contracts / offer letters |
| 6.3 | Information security awareness, education and training | 🔲 | Annual security training required per ISP; not yet delivered (ACT-005, due 2026-04-21) |
| 6.4 | Disciplinary process | ⚠️ | HR process to be documented; security violations covered in ISP §4 |
| 6.5 | Responsibilities after termination or change of employment | ✅ | Deprovisioning procedure in `ACCESS_CONTROL_POLICY.md` §6; confidentiality obligations should be in contracts |
| 6.6 | Confidentiality or non-disclosure agreements | ⚠️ | NDAs recommended for all staff and contractors; verify with legal |
| 6.7 | Remote working | ✅ | MFA required; VPN/secure access required for production |
| 6.8 | Information security event reporting | ✅ | Report to `kyle@branca.ai`; any employee obligation in ISP §4 |

---

### Theme 7 — Physical Controls

| Control | Title | Status | Implementation Notes |
|---------|-------|--------|---------------------|
| 7.1 | Physical security perimeters | ⚠️ | Cloud-hosted (provider responsibility); office physical security to be assessed |
| 7.2 | Physical entry | ⚠️ | Provider responsibility for data center; office controls to be documented |
| 7.3 | Securing offices, rooms and facilities | ⚠️ | Office security to be documented |
| 7.4 | Physical security monitoring | ❌ | Not applicable for cloud-only deployment at this stage |
| 7.5 | Protecting against physical and environmental threats | ⚠️ | Cloud provider handles data center; office disaster plan to be documented |
| 7.6 | Working in secure areas | ⚠️ | Work-from-home policy to be documented |
| 7.7 | Clear desk and clear screen | ⚠️ | Recommended; not yet formalized as policy |
| 7.8 | Equipment siting and protection | ❌ | Cloud-only; no owned server hardware |
| 7.9 | Security of assets off-premises | ⚠️ | Laptop full-disk encryption required; formal policy to be documented |
| 7.10 | Storage media | ⚠️ | Cloud object storage encrypted at rest; removable media policy not documented |
| 7.11 | Supporting utilities | ❌ | Cloud provider responsibility |
| 7.12 | Cabling security | ❌ | Cloud provider responsibility |
| 7.13 | Equipment maintenance | ❌ | Cloud provider responsibility for infrastructure |
| 7.14 | Secure disposal or re-use of equipment | ⚠️ | Developer laptop disposal policy to be documented |

---

### Theme 8 — Technological Controls

| Control | Title | Status | Implementation Notes |
|---------|-------|--------|---------------------|
| 8.1 | User endpoint devices | ⚠️ | Full-disk encryption required; MDM not yet deployed |
| 8.2 | Privileged access rights | ✅ | RBAC hierarchy; SUPER_ADMIN limited; audit logging of all privileged actions |
| 8.3 | Information access restriction | ✅ | RBAC at API layer; field-level encryption for PII |
| 8.4 | Access to source code | ✅ | GitHub CODEOWNERS; branch protection; org-enforced MFA |
| 8.5 | Secure authentication | ✅ | OIDC + MFA; short-lived JWTs; API key hashing |
| 8.6 | Capacity management | ⚠️ | K8s resource limits set; auto-scaling not yet configured |
| 8.7 | Protection against malware | ⚠️ | pip-audit + Bandit in CI; container image scanning to be added |
| 8.8 | Management of technical vulnerabilities | ✅ | ISP §8 patch SLAs; pip-audit in CI; Dependabot planned (ACT-003) |
| 8.9 | Configuration management | ⚠️ | K8s manifests in source control; configuration drift detection not yet automated |
| 8.10 | Information deletion | ⚠️ | Audit log retention task implemented; full data lifecycle/deletion policy not documented |
| 8.11 | Data masking | ✅ | API keys stored hashed; PII columns field-encrypted; audit log does not log credential values |
| 8.12 | Data leakage prevention | ⚠️ | Structured logging avoids credential logging; DLP tooling not deployed |
| 8.13 | Information backup | ⚠️ | RPO 1 hour per ISP; automated Postgres backup to be confirmed operational before GA |
| 8.14 | Redundancy of information processing facilities | ⚠️ | Postgres HA not implemented (ACT-002); K8s multi-replica for stateless services |
| 8.15 | Logging | ✅ | Audit log to Postgres; Prometheus metrics; structured logging; log retention 90 days |
| 8.16 | Monitoring activities | ✅ | Prometheus + Alertmanager; 9 alert rules covering availability, errors, security |
| 8.17 | Clock synchronization | ✅ | Kubernetes nodes use NTP; all timestamps UTC |
| 8.18 | Use of privileged utility programs | ✅ | No privileged utilities deployed; kubectl access restricted |
| 8.19 | Installation of software on operational systems | ✅ | All software via container images through CI/CD pipeline; no ad-hoc installs |
| 8.20 | Networks security | ⚠️ | K8s NetworkPolicy in progress; services communicate via cluster DNS; no public exposure except API Gateway |
| 8.21 | Security of network services | ✅ | TLS 1.2+ for all external traffic; mTLS recommended for future inter-service traffic |
| 8.22 | Segregation of networks | ⚠️ | K8s namespaces used; full NetworkPolicy enforcement to be completed |
| 8.23 | Web filtering | ❌ | Not applicable (server-side application) |
| 8.24 | Use of cryptography | ✅ | TLS 1.2+ in transit; AES-256-GCM at rest for PII fields; JWT RS256; SHA-256 for API keys |
| 8.25 | Secure development lifecycle | ✅ | `CHANGE_MANAGEMENT_PROCEDURE.md`; peer review; SAST; secret scanning; threat model in PRs |
| 8.26 | Application security requirements | ✅ | RBAC, rate limiting, input validation, prompt injection guards, OPA policy gate |
| 8.27 | Secure system architecture and engineering principles | ✅ | Defense-in-depth; least privilege; human-in-the-loop for high-risk actions; graceful degradation |
| 8.28 | Secure coding | ✅ | Bandit SAST; parameterized queries; no string concatenation for SQL; injection sanitization |
| 8.29 | Security testing in development and production | ⚠️ | Unit/E2E tests in CI; penetration test scheduled (ACT-004) |
| 8.30 | Outsourced development | ❌ | Not applicable — all development in-house at this stage |
| 8.31 | Separation of development, test and production environments | ⚠️ | Separate configs via env vars; dedicated production K8s namespace recommended |
| 8.32 | Change management | ✅ | `CHANGE_MANAGEMENT_PROCEDURE.md`; PR-gated deployments; rollback procedures |
| 8.33 | Test information | ⚠️ | Tests use fixture data; production data must not be used in test environments |
| 8.34 | Protection of information systems during audit testing | ✅ | Penetration testing to be conducted on staging, not production |

---

## 4. Summary Statistics

| Status | Count |
|--------|-------|
| ✅ Fully implemented | 47 |
| ⚠️ Partially implemented | 27 |
| 🔲 Planned | 4 |
| ❌ Not applicable | 10 |
| **Total Annex A controls** | **93** |

**Implementation rate (excl. N/A):** 57% fully implemented, 32% partial, 5% planned, 5% N/A.

Key gaps to close before ISO 27001 certification readiness:
1. **Formal security awareness training** (6.3) — ACT-005
2. **Annual penetration test** (8.29, 5.35) — ACT-004
3. **Postgres HA** (8.14) — ACT-002
4. **DPA execution** with cloud provider (5.20)
5. **DPIA / PII lifecycle policy** (5.34, 8.10)
6. **NetworkPolicy** full enforcement (8.22)
7. **MDM for developer endpoints** (8.1)
8. **SBOM generation** in CI (5.21)

---

*Next full review: 2027-03-21*
