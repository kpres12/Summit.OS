# Heli.OS Enterprise

Heli.OS Enterprise is the production-hardened, commercially-licensed deployment
of Heli.OS — built for organizations that need multi-org tenancy, enforced
security controls, SLA-backed support, and a licensing model that doesn't require
open-sourcing their products.

---

## What's different in Enterprise

### Multi-org tenancy
The community edition is single-tenant: one organization, one deployment.
Enterprise enables full row-level isolation across multiple organizations on a
single deployment — each org sees only its own entities, missions, assets, alerts,
and audit logs.

Activated by: `ENTERPRISE_MULTI_TENANT=true`

### Enforced security controls
Community edition ships security controls off-by-default (for easy local dev).
Enterprise enforces them — they cannot be bypassed at runtime:

| Control | Community | Enterprise |
|---|---|---|
| OIDC/SSO authentication | Optional | Enforced |
| RBAC (role-based access) | Optional | Enforced |
| Append-only audit logging | Optional | Enforced |
| AES-256-GCM field encryption | Optional | Enforced |
| MFA | Optional | Enforced |
| API key rotation policy | None | 90-day forced rotation |

### Org management API
Enterprise adds a `SUPER_ADMIN` role with CRUD access to the organization
management API — create orgs, provision users, set per-org resource limits,
view cross-org audit logs.

### Priority support + SLA
| | Community | Enterprise |
|---|---|---|
| Support channel | GitHub Issues | Direct email + Slack |
| Response time | Best effort | 4-hour acknowledgment |
| Uptime SLA | None | 99.9% on managed hosting |
| Incident escalation | None | On-call coverage |

---

## Deployment options

**Self-hosted** — You run the stack on your own infrastructure. We provide
the enterprise-licensed binary (or Docker images), deployment runbooks, and
remote support. You own the data, the keys, and the infrastructure.

**BigMT.ai managed** — We host and operate Heli.OS for you on dedicated
infrastructure. Zero ops burden. Your data stays in your region.

---

## SOC2 readiness

Heli.OS Enterprise includes every technical control required for a SOC2 Type II
audit:

- Argon2id password hashing (OWASP 2024 parameters)
- AES-256-GCM application-layer field encryption
- Append-only audit log with S3 archival
- TLS everywhere (Postgres, MQTT, API gateway)
- RBAC with least-privilege role definitions in OPA
- Geoblock at the API gateway layer
- YubiKey-compatible FIDO2/WebAuthn MFA
- HashiCorp Vault integration for secret management
- Trivy container scanning in CI

Heli.OS is **built to pass a SOC2 audit** — the controls are implemented and
documented. Heli.OS itself is not SOC2 certified; your organization pursues
certification for your specific deployment with a licensed CPA firm. Enterprise
customers get the audit evidence package (control mappings, policy docs,
architecture diagrams) to accelerate that process.

---

## Pricing

Enterprise pricing is based on deployment size and support tier.

Starting at: **$1,500 / month** (self-hosted, up to 10 operators)

Managed hosting, OEM/embedded, and government pricing available on request.

We are early-stage and founder-friendly. If you are a county fire department,
maritime SAR organization, or NGO — reach out before assuming you can't afford it.

---

## Contact

**kyle@branca.ai**

Include your organization name, use case, and approximate team size.
We'll respond within 2 business days.
