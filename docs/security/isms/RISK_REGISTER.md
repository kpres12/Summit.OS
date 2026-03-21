# Risk Register
**Summit.OS — BigMT.ai / Branca.ai**
**Classification: Internal**
**Version: 1.0 | Last Updated: 2026-03-21 | Owner: Head of Engineering**

---

## How to Use This Register

Each risk is scored on two axes before and after controls are applied:
- **Likelihood**: 1 (Rare) → 5 (Almost Certain)
- **Impact**: 1 (Negligible) → 5 (Critical/Safety-impacting)
- **Risk Score** = Likelihood × Impact (1–25)

Risk appetite: scores **≥ 15** require active treatment and executive sign-off. Scores **10–14** require documented mitigations. Scores **< 10** are accepted with monitoring.

---

## Active Risks

### SEC-001 — API Credential Compromise
| Field | Value |
|-------|-------|
| **Description** | A Summit.OS API key or JWT is stolen (phishing, leaked repo, MITM) and used by an unauthorized party to issue commands or exfiltrate operational data |
| **Likelihood (inherent)** | 3 |
| **Impact (inherent)** | 5 |
| **Inherent Score** | 15 |
| **Controls** | API keys hashed (SHA-256) at rest; OIDC short-lived JWTs; MFA on all operator accounts; API key enforcement (`API_KEY_ENFORCE`); rate limiting on all auth endpoints; audit log captures every key usage |
| **Likelihood (residual)** | 2 |
| **Impact (residual)** | 3 |
| **Residual Score** | 6 |
| **Treatment** | Accepted with monitoring |
| **Owner** | Head of Engineering |
| **Review Date** | 2026-09-21 |

---

### SEC-002 — Prompt Injection via Sensor Data
| Field | Value |
|-------|-------|
| **Description** | A malicious actor controls an external data source (spoofed ADS-B/AIS, compromised sensor) and embeds prompt injection payloads in entity names, descriptions, or alert messages that influence the AI reasoning engine to take unauthorized actions |
| **Likelihood (inherent)** | 3 |
| **Impact (inherent)** | 5 |
| **Inherent Score** | 15 |
| **Controls** | `prompt_guard.py` pattern-based sanitization on all externally-sourced strings; structural isolation (XML delimiters) separating data sections from instruction sections in LLM prompts; conversation history sanitization; token budget enforcement limits context size; OPA policy gate on all tool calls before physical actuation |
| **Likelihood (residual)** | 2 |
| **Impact (residual)** | 3 |
| **Residual Score** | 6 |
| **Treatment** | Accepted with monitoring; additional control: penetration test on AI pipeline annually |
| **Owner** | Head of Engineering |
| **Review Date** | 2026-09-21 |

---

### SEC-003 — Unauthorized Mission Command (Escalation)
| Field | Value |
|-------|-------|
| **Description** | An operator with insufficient privileges (e.g. VIEWER) issues a mission command or task that dispatches physical assets |
| **Likelihood (inherent)** | 3 |
| **Impact (inherent)** | 5 |
| **Inherent Score** | 15 |
| **Controls** | RBAC enforced at API layer: `POST /v1/missions` requires `MISSION_COMMANDER`; `POST /v1/tasks` requires `OPERATOR`; high-risk tasks (`HIGH`/`CRITICAL`) require explicit human approval before dispatch; OPA policy gate validates all actions; all commands logged to audit trail |
| **Likelihood (residual)** | 1 |
| **Impact (residual)** | 4 |
| **Residual Score** | 4 |
| **Treatment** | Accepted with monitoring |
| **Owner** | Head of Engineering |
| **Review Date** | 2026-09-21 |

---

### SEC-004 — Database Credential Exposure
| Field | Value |
|-------|-------|
| **Description** | Postgres or Redis credentials are exposed through misconfiguration, leaked logs, or compromised environment |
| **Likelihood (inherent)** | 3 |
| **Impact (inherent)** | 5 |
| **Inherent Score** | 15 |
| **Controls** | Vault-managed secrets via `packages/secrets/client.py`; K8s `summit-secrets` Secret with `valueFrom.secretKeyRef` (never in manifest literals); `infra/k8s/secrets.yaml` in `.gitignore`; structured logging does not log credential values; field encryption on PII columns provides defense-in-depth even if DB is accessed directly |
| **Likelihood (residual)** | 2 |
| **Impact (residual)** | 3 |
| **Residual Score** | 6 |
| **Treatment** | Accepted with monitoring |
| **Owner** | Head of Engineering |
| **Review Date** | 2026-09-21 |

---

### SEC-005 — Denial of Service / Resource Exhaustion
| Field | Value |
|-------|-------|
| **Description** | An attacker floods the API Gateway with requests, exhausting compute or database connections and making the platform unavailable to legitimate operators |
| **Likelihood (inherent)** | 4 |
| **Impact (inherent)** | 4 |
| **Inherent Score** | 16 |
| **Controls** | Per-IP rate limiting (slowapi): 10/min device registration, 30/min task submission, 20/min mission creation, 5/min MFA endpoints; Prometheus `HighErrorRate` and availability alerts; K8s resource limits on all pods prevent one service consuming all node resources |
| **Likelihood (residual)** | 3 |
| **Impact (residual)** | 3 |
| **Residual Score** | 9 |
| **Treatment** | Partially mitigated at app layer; full mitigation requires upstream WAF/CDN (Cloudflare/AWS Shield) — **open action item for production deployment** |
| **Owner** | Head of Engineering |
| **Review Date** | 2026-06-21 |

---

### SEC-006 — Dependency Vulnerability (Supply Chain)
| Field | Value |
|-------|-------|
| **Description** | A known CVE in a Python or Node.js dependency is exploited before it is patched |
| **Likelihood (inherent)** | 3 |
| **Impact (inherent)** | 4 |
| **Inherent Score** | 12 |
| **Controls** | `pip-audit` runs in every CI build across all 6 service requirements files; Bandit SAST on every PR; Dependabot or equivalent to be enabled on GitHub repo; patch SLAs enforced by this policy (Critical: 24h, High: 7d) |
| **Likelihood (residual)** | 2 |
| **Impact (residual)** | 3 |
| **Residual Score** | 6 |
| **Treatment** | Accepted with monitoring |
| **Owner** | Head of Engineering |
| **Review Date** | 2026-09-21 |

---

### SEC-007 — Insider Threat / Disgruntled Employee
| Field | Value |
|-------|-------|
| **Description** | A current or former employee intentionally exfiltrates data, plants backdoors, or sabotages systems |
| **Likelihood (inherent)** | 2 |
| **Impact (inherent)** | 5 |
| **Inherent Score** | 10 |
| **Controls** | RBAC limits blast radius per role; audit log captures all privileged actions attributed to user identity; CODEOWNERS requires peer review for all security-sensitive paths; production access revoked within 24h of termination; MFA prevents credential sharing |
| **Likelihood (residual)** | 1 |
| **Impact (residual)** | 4 |
| **Residual Score** | 4 |
| **Treatment** | Accepted |
| **Owner** | Head of Engineering |
| **Review Date** | 2026-09-21 |

---

### SEC-008 — Data Breach via Third-Party (Stripe)
| Field | Value |
|-------|-------|
| **Description** | Stripe suffers a breach that exposes billing data for Summit.OS customers |
| **Likelihood (inherent)** | 1 |
| **Impact (inherent)** | 3 |
| **Inherent Score** | 3 |
| **Controls** | Summit.OS never stores full card data — Stripe handles all PCI scope; Stripe webhook signatures verified via HMAC; `STRIPE_WEBHOOK_SECRET` managed via Vault; Stripe has SOC2 Type 2 and PCI DSS Level 1 certification |
| **Likelihood (residual)** | 1 |
| **Impact (residual)** | 2 |
| **Residual Score** | 2 |
| **Treatment** | Accepted |
| **Owner** | Head of Engineering |
| **Review Date** | 2027-03-21 |

---

### OPS-001 — AI Reasoning Hallucination / Incorrect Action
| Field | Value |
|-------|-------|
| **Description** | The LLM reasoning engine (Ollama/Llama) produces a hallucinated or incorrect tool call that dispatches physical assets inappropriately |
| **Likelihood (inherent)** | 3 |
| **Impact (inherent)** | 5 |
| **Inherent Score** | 15 |
| **Controls** | OPA policy gate validates all tool calls before execution; `HIGH`/`CRITICAL` risk tasks require human approval; `BRAIN_MAX_STEPS` limits autonomous action chains; `BRAIN_TEMPERATURE=0.2` (low) for deterministic outputs; brain falls back gracefully (sensor-only mode) when Ollama unavailable; all AI decisions logged with full context |
| **Likelihood (residual)** | 2 |
| **Impact (residual)** | 3 |
| **Residual Score** | 6 |
| **Treatment** | Accepted; human-in-the-loop for high-risk decisions is a design invariant |
| **Owner** | Head of Engineering |
| **Review Date** | 2026-09-21 |

---

### OPS-002 — Single Point of Failure — Postgres
| Field | Value |
|-------|-------|
| **Description** | Postgres unavailability takes down all stateful services (Fabric, Tasking, Intelligence, Gateway) simultaneously |
| **Likelihood (inherent)** | 2 |
| **Impact (inherent)** | 5 |
| **Inherent Score** | 10 |
| **Controls** | K8s readiness/liveness probes; Prometheus `PostgresDown` alert fires within 30s; Redis caches hot entity state for read continuity; services degrade gracefully rather than crash on DB connection failure |
| **Likelihood (residual)** | 2 |
| **Impact (residual)** | 4 |
| **Residual Score** | 8 |
| **Treatment** | Mitigated; full mitigation requires Postgres HA (primary/replica) — **open action item for production** |
| **Owner** | Head of Engineering |
| **Review Date** | 2026-06-21 |

---

## Open Action Items

| ID | Action | Owner | Due |
|----|--------|-------|-----|
| ACT-001 | Deploy upstream WAF/CDN for DDoS protection (SEC-005) | Head of Engineering | Before GA |
| ACT-002 | Enable Postgres HA/replication (OPS-002) | Head of Engineering | Before GA |
| ACT-003 | Enable GitHub Dependabot for automated dependency alerts (SEC-006) | Engineering | 2026-04-21 |
| ACT-004 | Schedule first annual penetration test | Head of Engineering | 2026-06-21 |
| ACT-005 | Complete employee security awareness training for all staff | Head of Engineering | 2026-04-21 |

---

*Next full review: 2026-09-21*
