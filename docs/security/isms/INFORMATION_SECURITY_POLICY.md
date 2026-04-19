# Information Security Policy
**Summit.OS — BigMT.ai / Branca.ai**
**Classification: Internal**
**Version: 1.0 | Effective: 2026-03-21 | Review: Annually**
**Owner: Head of Engineering**

---

## 1. Purpose

This policy establishes the security principles and obligations that govern how Summit.OS is designed, operated, and accessed. It applies to the platform itself, all personnel who build or operate it, and all third parties with access to Summit.OS systems or data.

Summit.OS is an autonomous systems coordination platform used in civilian operational contexts including disaster response, search and rescue, wildfire operations, and commercial UAV fleet management. The platform handles real-time telemetry, mission-critical tasking decisions, and operational data for its customers. Security failures can have direct physical consequences.

---

## 2. Scope

This policy covers:
- All Summit.OS software systems (API Gateway, Fabric, Fusion, Intelligence, Tasking, Console, adapters)
- All infrastructure (cloud, containerized, edge deployments)
- All data processed by Summit.OS (entity telemetry, mission data, operator actions, billing data)
- All personnel (employees, contractors, contributors)
- All customer environments where Summit.OS is deployed

---

## 3. Security Objectives

The organization commits to:

1. **Confidentiality** — Customer operational data, API keys, and billing information are accessible only to authorized parties
2. **Integrity** — Mission commands, entity state, and audit records are tamper-evident and accurate
3. **Availability** — The platform meets agreed uptime SLAs; degraded-mode operation is safe and documented
4. **Accountability** — All privileged actions are attributable to a specific user with a tamper-evident audit trail
5. **Resilience** — Security failures do not cascade into safety failures; the system fails safely

---

## 4. Roles and Responsibilities

| Role | Security Responsibility |
|------|------------------------|
| **Head of Engineering** | Policy owner; approves exceptions; ensures security reviews before release |
| **All Engineers** | Follow secure coding standards; report vulnerabilities; complete annual security training |
| **DevOps / Platform** | Maintain infrastructure security; rotate credentials; monitor alerts |
| **On-call Engineer** | Respond to security incidents per the Incident Response Procedure within defined SLAs |
| **Any Employee** | Report suspected security incidents immediately; never share credentials |

---

## 5. Access Control

- Access to production systems requires MFA
- All API access uses short-lived tokens (OIDC JWT) or scoped API keys — no long-lived passwords
- Role-based access is enforced at the API layer: `VIEWER → OPERATOR → MISSION_COMMANDER → ADMIN → SUPER_ADMIN`
- Principle of least privilege: permissions are granted for the minimum scope required
- Access is reviewed quarterly; terminated employees are deprovisioned within 24 hours
- Production database credentials are managed via HashiCorp Vault or equivalent secrets management; never hardcoded

---

## 6. Data Classification

| Level | Definition | Examples | Handling |
|-------|-----------|----------|----------|
| **Public** | Deliberately made available externally | API documentation, open-source SDK | No restrictions |
| **Internal** | Business operational data | Architecture docs, runbooks, this policy | Employee access only |
| **Confidential** | Customer data, financial data | Telemetry, mission data, org names, API keys | Encrypted at rest and in transit; access logged |
| **Restricted** | High-sensitivity secrets | Private keys, STRIPE_SECRET_KEY, Vault root token | Vault-managed; access strictly limited; never logged |

All Confidential and Restricted data must be encrypted at rest using AES-256-GCM or equivalent. All data in transit uses TLS 1.2 minimum (TLS 1.3 preferred).

---

## 7. Secure Development

- All code changes require peer review; security-sensitive paths (auth, billing, audit) require additional review from a security-designated approver (see `CODEOWNERS`)
- Dependencies are scanned for known CVEs in every CI run (`pip-audit`)
- Static analysis (Bandit) runs on every PR
- Secrets must never be committed to source control; pre-commit hooks and CI checks enforce this
- Production deployments use the K8s `summit-secrets` Secret; never environment variable literals in manifests
- New features that handle Confidential or Restricted data require a brief threat model documented in the PR

---

## 8. Vulnerability Management

- Critical CVEs (CVSS ≥ 9.0): patch within 24 hours
- High CVEs (CVSS 7.0–8.9): patch within 7 days
- Medium CVEs: patch within 30 days
- Low CVEs: next scheduled release
- Penetration testing: annually minimum; after major architectural changes
- Vulnerability disclosures: report to `kyle@branca.ai`; acknowledged within 48 hours

---

## 9. Incident Response

Security incidents are handled per the **Incident Response Procedure** (`INCIDENT_RESPONSE.md`).

Severity definitions:
- **P0 (Critical)**: Active breach, data exfiltration, ransomware, or safety-impacting compromise. Respond immediately; notify Head of Engineering within 30 minutes.
- **P1 (High)**: Suspected unauthorized access, credential leak, or service compromise. Respond within 4 hours.
- **P2 (Medium)**: Anomalous access patterns, failed intrusion attempts, policy violations. Respond within 24 hours.
- **P3 (Low)**: Security misconfigurations, non-exploitable findings. Resolve within 7 days.

---

## 10. Business Continuity

- Recovery Time Objective (RTO): 4 hours for P0 incidents
- Recovery Point Objective (RPO): 1 hour (based on Postgres backup frequency)
- Runbooks for all critical failure modes are documented in `docs/OPERATIONS_RUNBOOK.md`
- Disaster recovery procedures are tested annually

---

## 11. Third-Party and Supplier Security

- All third-party services with access to Confidential data must complete a vendor risk assessment before onboarding
- Current external dependencies with data access: Stripe (billing), cloud infrastructure provider, optional OIDC provider
- Third-party SLAs and security postures are reviewed annually

---

## 12. Monitoring and Audit

- All API actions are logged to the tamper-evident `audit_log` table (see `middleware/audit.py`)
- Audit logs are retained for 90 days minimum (configurable via `AUDIT_RETENTION_DAYS`)
- Prometheus + Alertmanager monitor for auth failure spikes, error rate anomalies, and service outages
- Audit logs are reviewed monthly for anomalies; auth failure spikes trigger automated alerts

---

## 13. Policy Exceptions

Exceptions to this policy require written approval from the Head of Engineering and must be documented in the Risk Register with a compensating control and a review date no more than 90 days out.

---

## 14. Review and Updates

This policy is reviewed annually or after any significant security incident or architectural change. All employees are notified of material changes.

---

*Approved by: [Head of Engineering — signature required before certification]*
*Next review: 2027-03-21*
