# Vendor Risk Assessment
**Summit.OS — BigMT.ai / Branca.ai**
**Classification: Internal**
**Version: 1.0 | Last Updated: 2026-03-21 | Owner: Head of Engineering**

---

## 1. Purpose

This document records the risk assessment for each third-party vendor with access to Summit.OS systems or data. New vendors with access to Confidential or Restricted data must complete this assessment before onboarding.

---

## 2. Assessment Criteria

Each vendor is scored on:

| Criterion | Weight | Description |
|-----------|--------|-------------|
| Data sensitivity | High | What classification of data does the vendor access? |
| Compliance certifications | High | SOC2, ISO 27001, PCI DSS, GDPR adequacy? |
| Breach history | High | Known material breaches in last 5 years |
| Data residency | Medium | Where is data stored/processed? |
| Subprocessor transparency | Medium | Does vendor disclose subprocessors? |
| Contractual protections | Medium | DPA in place? Liability clauses? |
| Business criticality | Medium | What breaks if vendor goes offline? |
| Concentration risk | Low | Is Summit.OS over-dependent on this vendor? |

**Risk tiers:**
- **Low**: Vendor access to non-Confidential data only, or strong certifications + minimal data access
- **Medium**: Some Confidential data access; certifications present but not fully applicable
- **High**: Significant Confidential/Restricted data access; certifications absent or not applicable

---

## 3. Completed Assessments

---

### VRA-001 — Stripe (Billing)

| Field | Value |
|-------|-------|
| **Purpose** | Payment processing and subscription management |
| **Data Access** | Billing contact info, org name, subscription plan. Summit.OS does NOT transmit card data — Stripe-hosted checkout |
| **Data Classification** | Confidential (billing data), but no Restricted data |
| **SOC2 Type 2** | Yes |
| **PCI DSS** | Level 1 (highest level) |
| **ISO 27001** | Yes |
| **Known Breaches** | None material in last 5 years |
| **Data Residency** | USA / EU (configurable) |
| **Subprocessor Disclosure** | Yes — publicly available at stripe.com/legal/service-privacy |
| **DPA Available** | Yes — Stripe Data Processing Agreement |
| **Business Criticality** | Medium — billing failures prevent new subscriptions; existing customers unaffected |
| **Risk Tier** | **Low** |
| **Compensating Controls** | Webhook HMAC signature verification; `STRIPE_WEBHOOK_SECRET` in Vault; no card data in our systems; Stripe has independent incident response capability |
| **Last Reviewed** | 2026-03-21 |
| **Next Review** | 2027-03-21 |

---

### VRA-002 — Cloud Infrastructure Provider

| Field | Value |
|-------|-------|
| **Purpose** | Hosting: Kubernetes cluster, managed PostgreSQL, object storage, networking |
| **Data Access** | All platform data at rest (entities, missions, audit logs, credentials) |
| **Data Classification** | Confidential and Restricted |
| **SOC2 Type 2** | Yes (all major cloud providers) |
| **ISO 27001** | Yes |
| **PCI DSS** | Yes (for applicable services) |
| **Known Breaches** | No material breach affecting customer data in last 5 years (verify for specific provider) |
| **Data Residency** | Select region explicitly during provisioning; document chosen region |
| **Subprocessor Disclosure** | Yes — cloud providers publish subprocessor lists |
| **DPA Available** | Yes — standard cloud provider DPA |
| **Business Criticality** | Critical — all services depend on cloud infrastructure |
| **Concentration Risk** | High — single cloud provider for all workloads |
| **Risk Tier** | **Medium** (due to concentration risk and breadth of data access) |
| **Compensating Controls** | Encryption at rest enabled on all storage; K8s RBAC; network policies; backup/restore tested; disaster recovery procedure documented; consider multi-region for HA |
| **Open Action** | Document specific cloud provider and region; review encryption-at-rest configuration for managed Postgres |
| **Last Reviewed** | 2026-03-21 |
| **Next Review** | 2027-03-21 |

---

### VRA-003 — OIDC Provider (Authentication)

| Field | Value |
|-------|-------|
| **Purpose** | Operator authentication and SSO for Summit.OS console |
| **Data Access** | User identities, authentication events |
| **Data Classification** | Confidential |
| **SOC2 Type 2** | Depends on provider — verify before onboarding |
| **ISO 27001** | Depends on provider — verify before onboarding |
| **Known Breaches** | Verify for specific provider |
| **Data Residency** | Verify for specific provider |
| **DPA Available** | Verify for specific provider |
| **Business Criticality** | High — unavailability prevents operator login (fallback: API key auth available) |
| **Risk Tier** | **Medium** (identity provider; auth fallback exists) |
| **Compensating Controls** | API key authentication as fallback; short-lived JWTs (1 hour max); Summit.OS does not store plaintext credentials — only JWT validation |
| **Open Action** | Confirm specific OIDC provider and complete provider-specific fields above before production |
| **Last Reviewed** | 2026-03-21 |
| **Next Review** | 2027-03-21 |

---

### VRA-004 — GitHub

| Field | Value |
|-------|-------|
| **Purpose** | Source code hosting, CI/CD pipeline execution |
| **Data Access** | Source code (proprietary); CI secrets (via GitHub Secrets, encrypted) |
| **Data Classification** | Internal (source code); Restricted (CI secrets) |
| **SOC2 Type 2** | Yes |
| **ISO 27001** | Yes |
| **Known Breaches** | Minor incidents (e.g., 2023 RSA key exposure); promptly resolved; no customer data breach |
| **Data Residency** | USA (GitHub.com); GitHub Enterprise allows self-hosted |
| **Subprocessor Disclosure** | Yes |
| **DPA Available** | Yes |
| **Business Criticality** | High — CI/CD unavailability blocks deployments; codebase is always locally cloned |
| **Risk Tier** | **Low-Medium** |
| **Compensating Controls** | Organization-enforced MFA; CODEOWNERS enforces security review gates; branch protection on `main`; secrets in GitHub Secrets (not code); local clones provide resilience against outage |
| **Last Reviewed** | 2026-03-21 |
| **Next Review** | 2027-03-21 |

---

## 4. Vendor Onboarding Checklist

Before adding a new vendor that accesses Confidential or Restricted data:

- [ ] Complete this assessment form for the vendor
- [ ] Confirm relevant certifications are current (request documentation)
- [ ] Confirm DPA or equivalent contractual data protection is in place
- [ ] Document what data will be shared and why
- [ ] Add vendor to `ASSET_INVENTORY.md` (TP-series asset)
- [ ] Add vendor to Risk Register if inherent risk score ≥ 10
- [ ] Head of Engineering sign-off before provisioning access

---

## 5. Annual Review

All vendor assessments are reviewed annually. A vendor's risk tier may be elevated if:
- A material breach is disclosed
- Certifications lapse
- Data sharing scope increases
- Contractual protections are found insufficient

---

*Next full review: 2027-03-21*
