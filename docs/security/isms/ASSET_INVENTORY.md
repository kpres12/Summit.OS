# Asset Inventory
**Heli.OS — BigMT.ai / Branca.ai**
**Classification: Internal**
**Version: 1.0 | Last Updated: 2026-03-21 | Owner: Head of Engineering**

---

## 1. Purpose

This inventory catalogues all significant information assets — software systems, data stores, infrastructure, and third-party services — that are in scope for Heli.OS security management. Each asset has a designated owner, classification, and applicable controls.

---

## 2. Software Services

| Asset ID | Service | Description | Location | Owner | Data Classification | Key Controls |
|----------|---------|-------------|----------|-------|--------------------|----|
| SW-001 | API Gateway | FastAPI service; primary entry point for all external clients; handles auth, RBAC, rate limiting, billing, audit logging | `apps/api-gateway/` | Head of Engineering | Confidential | RBAC, rate limiting, JWT auth, OPA policy gate, audit logging |
| SW-002 | Fabric | Entity state service (WorldStore); manages device registry, entity telemetry, geofences | `apps/fabric/` | Head of Engineering | Confidential | JWT auth, RBAC, parameterized SQL |
| SW-003 | Fusion | Sensor data ingestion and fusion; processes ADS-B, AIS, and other sensor feeds | `apps/fusion/` | Head of Engineering | Confidential | Input validation, rate limiting |
| SW-004 | Intelligence | AI reasoning engine; LLM-powered context builder and decision layer | `apps/intelligence/` | Head of Engineering | Confidential | Prompt injection guards, OPA policy gate, `BRAIN_MAX_STEPS` limit |
| SW-005 | Tasking | Mission and task orchestration; dispatches commands to physical assets | `apps/tasking/` | Head of Engineering | Confidential | RBAC (MISSION_COMMANDER+), human approval for HIGH/CRITICAL tasks, audit logging |
| SW-006 | Inference | Ollama/LLM inference service; local model execution | `apps/inference/` | Head of Engineering | Internal | Network isolation, token budget limits |
| SW-007 | Console | Next.js operator UI; OPS, COMMAND, and DEV role views | `apps/console/` | Head of Engineering | Internal (operator UI) | Auth, RBAC, TypeScript strict mode, error boundaries |

---

## 3. Data Stores

| Asset ID | Store | Description | Location | Owner | Data Classification | Key Controls |
|----------|-------|-------------|----------|-------|--------------------|----|
| DS-001 | PostgreSQL (primary) | Canonical store for entities, missions, tasks, audit logs, API keys, device registry | Kubernetes pod / managed cloud DB (prod) | Head of Engineering | Confidential | Encrypted at rest (provider-managed), encrypted PII columns (AES-256-GCM), parameterized queries, no direct external access |
| DS-002 | Redis | Hot cache for entity state; session data; rate limiting state | Kubernetes pod | Head of Engineering | Confidential | In-cluster only (no external exposure), data is transient (TTL-based) |
| DS-003 | Audit Log Table | Append-only `audit_log` table in Postgres; records all privileged API actions | Within DS-001 | Head of Engineering | Confidential | Write-once pattern in middleware; only DELETEable via retention task (automated, age-based); access requires ADMIN role |
| DS-004 | Vault / K8s Secrets | Secrets management: API keys, DB passwords, JWT secrets, Stripe keys | HashiCorp Vault (prod) / K8s Secret (dev) | Head of Engineering | Restricted | Never in source control; AppRole for services; MFA for human access; secret versioning for rollback |

---

## 4. Infrastructure

| Asset ID | Asset | Description | Owner | Data Classification | Key Controls |
|----------|-------|-------------|-------|--------------------|----|
| INF-001 | Kubernetes Cluster | Container orchestration; all services deployed as pods | Head of Engineering | Internal | RBAC, NetworkPolicy, resource limits, no privileged containers |
| INF-002 | Container Registry | Docker images for all services | Head of Engineering | Internal | Image scanning before push; no credentials in layers |
| INF-003 | CI/CD Pipeline | GitHub Actions; builds, tests, deploys all services | Head of Engineering | Internal | Secret scanning, pip-audit, Bandit SAST, required reviews before merge to main |
| INF-004 | Prometheus + Alertmanager | Metrics collection and alerting | Head of Engineering | Internal | Internal-only; no external exposure |
| INF-005 | Grafana | Metrics visualization and dashboards | Head of Engineering | Internal | Authentication required; admin password in Vault |
| INF-006 | Jaeger | Distributed trace collection and visualization | Head of Engineering | Internal | Internal-only |

---

## 5. Third-Party Services

| Asset ID | Service | Purpose | Data Shared | Owner | Risk Assessment | Controls |
|----------|---------|---------|-------------|-------|----------------|---------|
| TP-001 | Stripe | Billing and subscription management | Org name, billing contact, plan selection (NO card data — Stripe-hosted) | Head of Engineering | Low (PCI DSS Level 1, SOC2 Type 2) | Webhook HMAC verification; `STRIPE_WEBHOOK_SECRET` in Vault; no card data ever stored |
| TP-002 | Cloud Infrastructure Provider | Hosting (K8s, managed DB, storage) | All platform data at rest | Head of Engineering | Medium | Encryption at rest enabled; access restricted to authorized engineers; MFA required |
| TP-003 | OIDC Provider (optional) | Operator authentication / SSO | User identities, authentication events | Head of Engineering | Low-Medium | Short-lived JWTs (max 1 hour); no long-lived credentials |
| TP-004 | GitHub | Source code hosting, CI/CD | Source code, CI secrets (via GitHub Secrets) | Head of Engineering | Low-Medium | Org-enforced MFA; CODEOWNERS for security-sensitive paths; branch protection on `main` |
| TP-005 | PagerDuty | On-call alerting | Alert payloads (no PII) | Head of Engineering | Low | Webhook integration only; no data persistence |
| TP-006 | Slack (optional) | Team communication and incident alerts | Alert messages (no PII) | Head of Engineering | Low | Webhook integration only; no secrets in messages |

---

## 6. Sensitive Data Flows

| Flow | Source | Destination | Classification | In-Transit Protection | At-Rest Protection |
|------|--------|-------------|---------------|----------------------|-------------------|
| Entity telemetry | External sensors → Fusion → Fabric (DB) | Confidential | TLS 1.2+ | AES-256 (provider) |
| Mission commands | Operator → API Gateway → Tasking → Assets | Confidential | TLS 1.2+ | AES-256 (provider) |
| Operator credentials | Operator → OIDC Provider → API Gateway | Restricted | TLS 1.3 | Not stored (JWT only) |
| API keys | Generated → DB (`api_keys` table) | Restricted | TLS 1.2+ | SHA-256 hash only |
| Billing data | Operator → Stripe → API Gateway (webhook) | Confidential | TLS 1.3 | HMAC-verified; minimal local storage |
| Audit logs | All services → API Gateway → DB | Confidential | TLS 1.2+ | AES-256 (provider) |
| Secrets | Vault → K8s Secret → Service env | Restricted | Encrypted in Vault | K8s Secret (base64; rely on etcd encryption-at-rest) |
| AI context | WorldStore → Intelligence → Ollama | Confidential (operational) | In-cluster (no TLS needed if same-namespace) | Not persisted |

---

## 7. Asset Review

This inventory is reviewed:
- **Quarterly** for additions or removals of third-party services
- **Annually** as part of the full ISMS review
- **On architectural change** that introduces new data stores, external services, or infrastructure components

Changes to this inventory are made via the standard change management process (`CHANGE_MANAGEMENT_PROCEDURE.md`).

---

*Next full review: 2027-03-21*
