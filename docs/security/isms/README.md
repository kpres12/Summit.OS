# Summit.OS ISMS Document Set
**BigMT.ai / Branca.ai**
**Classification: Internal**
**Version: 1.0 | Last Updated: 2026-03-21**

---

This directory contains the Information Security Management System (ISMS) documentation for Summit.OS. These documents support ISO 27001:2022 and SOC2 Type 2 certification readiness.

---

## Document Index

| Document | Purpose | Owner | Review |
|----------|---------|-------|--------|
| [INFORMATION_SECURITY_POLICY.md](./INFORMATION_SECURITY_POLICY.md) | Master security policy — scope, objectives, roles, data classification, key controls | Head of Engineering | Annually |
| [RISK_REGISTER.md](./RISK_REGISTER.md) | Active risk inventory with inherent/residual scoring and treatment plans | Head of Engineering | Semi-annually |
| [INCIDENT_RESPONSE.md](./INCIDENT_RESPONSE.md) | P0–P3 severity definitions, response phases, escalation contacts, PIR template | Head of Engineering | Annually |
| [ACCESS_CONTROL_POLICY.md](./ACCESS_CONTROL_POLICY.md) | Role hierarchy, provisioning/deprovisioning, MFA requirements, access review | Head of Engineering | Annually |
| [CHANGE_MANAGEMENT_PROCEDURE.md](./CHANGE_MANAGEMENT_PROCEDURE.md) | PR process, security review gates, deployment gates, rollback procedures | Head of Engineering | Annually |
| [ASSET_INVENTORY.md](./ASSET_INVENTORY.md) | All software services, data stores, infrastructure, third-party services, and sensitive data flows | Head of Engineering | Quarterly (additions) / Annually (full) |
| [VENDOR_RISK_ASSESSMENT.md](./VENDOR_RISK_ASSESSMENT.md) | Risk assessments for Stripe, cloud provider, OIDC provider, GitHub | Head of Engineering | Annually |
| [STATEMENT_OF_APPLICABILITY.md](./STATEMENT_OF_APPLICABILITY.md) | ISO 27001:2022 Annex A control applicability mapping (required for certification) | Head of Engineering | Annually |

---

## Certification Readiness Summary

| Standard | Status | Key Gaps |
|----------|--------|---------|
| **SOC2 Type 2** | ~60% controls implemented | Security training (CC1.4), pen test (CC4.1), Postgres HA (A1.2), DPA with cloud provider |
| **ISO 27001:2022** | ~57% Annex A fully implemented | Same gaps + DPIA documentation, NetworkPolicy enforcement, MDM, SBOM |

See `STATEMENT_OF_APPLICABILITY.md` for the full control-by-control breakdown.

---

## Open Action Items (from Risk Register)

| ID | Action | Due |
|----|--------|-----|
| ACT-001 | Deploy upstream WAF/CDN (SEC-005) | Before GA |
| ACT-002 | Enable Postgres HA/replication (OPS-002) | Before GA |
| ACT-003 | Enable GitHub Dependabot (SEC-006) | 2026-04-21 |
| ACT-004 | Schedule first annual penetration test | 2026-06-21 |
| ACT-005 | Complete employee security awareness training | 2026-04-21 |

---

*All documents require Head of Engineering signature before use in external certification processes.*
