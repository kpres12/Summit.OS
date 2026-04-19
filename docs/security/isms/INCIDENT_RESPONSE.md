# Incident Response Procedure
**Heli.OS — BigMT.ai / Branca.ai**
**Classification: Internal**
**Version: 1.0 | Last Updated: 2026-03-21 | Owner: Head of Engineering**

---

## 1. Purpose

This procedure defines how Heli.OS responds to security incidents — from initial detection through containment, eradication, recovery, and post-incident review. It applies to all personnel who build, operate, or maintain Heli.OS systems.

---

## 2. Severity Definitions

| Severity | Definition | Response SLA | Examples |
|----------|-----------|--------------|---------|
| **P0 (Critical)** | Active breach, data exfiltration, ransomware, or safety-impacting compromise | Immediate; notify Head of Engineering within 30 minutes | Confirmed unauthorized access to production DB; AI brain dispatching assets without authorization; ransomware detected |
| **P1 (High)** | Suspected unauthorized access, credential leak, or service compromise | Respond within 4 hours | Leaked API key; anomalous auth pattern suggesting compromise; service behaving outside expected parameters |
| **P2 (Medium)** | Anomalous access patterns, failed intrusion attempts, policy violations | Respond within 24 hours | Auth failure spike (>100/min); failed SQL injection attempts; misconfigured CORS detected |
| **P3 (Low)** | Security misconfigurations, non-exploitable findings, compliance gaps | Resolve within 7 days | Dependency with medium CVE; audit log gap; missing security header |

---

## 3. Roles

| Role | Responsibility |
|------|---------------|
| **Incident Commander** | Head of Engineering (or delegate). Owns the incident from declaration to close. Makes containment decisions. |
| **On-call Engineer** | First responder. Triages, executes containment steps, pages Incident Commander for P0/P1. |
| **Communications Lead** | Head of Engineering or CEO. Manages customer and external communications. |
| **Security Reviewer** | Any engineer not directly involved in response. Reviews containment steps before execution on production. |

---

## 4. Detection Sources

- **Automated alerts** — Prometheus/Alertmanager: `AuthFailureSpike`, `AccessDeniedSpike`, `ServiceDown`, `HighErrorRate`
- **Audit log review** — Monthly anomaly review of `audit_log` table; automated alerts on spikes
- **pip-audit / Bandit** — CI pipeline CVE and SAST findings
- **External report** — Vulnerability disclosures to `kyle@branca.ai`
- **Customer report** — Operator reports unexpected behavior or suspected compromise
- **Employee observation** — Any employee who suspects a security issue must report immediately

---

## 5. Response Phases

### Phase 1 — Identify & Triage (0–30 min for P0/P1)

1. **On-call engineer** receives alert or report
2. Classify severity using the table in §2
3. For P0/P1: page Incident Commander immediately via PagerDuty
4. Open incident channel: `#incident-YYYY-MM-DD` in Slack
5. Record initial assessment:
   - What systems are affected?
   - Is data being actively exfiltrated?
   - Is any physical asset (UAV, vessel) behaving abnormally?
6. Preserve evidence: capture logs, metric snapshots, affected API responses before any changes

**P0 Escalation contacts:**
- Head of Engineering: [on-call rotation via PagerDuty]
- CEO: [direct message]
- If safety-impacting (physical asset dispatched without authorization): contact asset operators immediately

### Phase 2 — Contain (0–2 hours for P0, 4 hours for P1)

Execute the appropriate containment action based on incident type:

| Incident Type | Containment Action |
|--------------|-------------------|
| Compromised API key | Immediately revoke key via `DELETE /v1/api-keys/{key_id}` (requires ADMIN). Rotate `FABRIC_JWT_SECRET` if JWT compromise suspected. |
| Compromised operator account | Disable account; revoke all active sessions; force MFA re-enrollment |
| Active data exfiltration | Block source IP at load balancer/WAF; restrict egress if network-level exfiltration |
| Unauthorized physical dispatch | Contact asset operators; issue RTB (Return to Base) command via Tasking API; enable emergency hold in OPA policy |
| SQL injection / active exploit | Take affected service to maintenance mode; restore from last known-good state |
| Prompt injection (AI acting anomalously) | Set `BRAIN_MAX_STEPS=0` via config to halt AI autonomy; switch to sensor-only mode |
| Dependency vulnerability (active exploit) | Patch immediately or pull affected service offline; deploy patch as emergency release |
| Ransomware | Isolate affected systems; do not pay; restore from backups |

All containment actions must be **reviewed by at least one other engineer** before execution on production (except RTB/emergency hold which may be executed unilaterally for safety).

### Phase 3 — Eradicate

1. Identify root cause fully before proceeding
2. Remove malicious artifacts (backdoors, malicious code, unauthorized API keys)
3. Patch the vulnerability
4. Verify patch does not introduce new issues
5. Rotate all credentials that may have been exposed (not just confirmed — suspected)
6. For supply chain: audit transitive dependencies; update `pip-audit` baseline

### Phase 4 — Recover

1. Restore from clean backup if data integrity is in question (RPO: 1 hour)
2. Deploy patched version through normal CI/CD pipeline (security review required)
3. Re-enable services incrementally; verify each before enabling the next
4. Monitor for 24 hours post-recovery for signs of re-compromise
5. Confirm with Incident Commander before declaring recovery complete

### Phase 5 — Post-Incident Review

For all P0/P1 incidents, within 5 business days:

1. **Timeline reconstruction** — precise sequence of events, detection gap analysis
2. **Root cause analysis** — what failed technically, procedurally, and organizationally
3. **Impact assessment** — data affected, systems affected, customer impact
4. **Lessons learned** — what would have prevented this; what would have caught it earlier
5. **Action items** — concrete preventive measures with owners and due dates
6. **Risk Register update** — update or add risk entries in `RISK_REGISTER.md`
7. **Customer notification** (if required) — see §6

Written PIR (Post-Incident Report) is required for P0/P1. Template:
```
Incident ID: INC-YYYY-MM-DD-NNN
Severity: P0/P1
Date/Time Detected:
Date/Time Contained:
Date/Time Resolved:

Summary (2-3 sentences):

Timeline:
  [timestamp] — event

Root Cause:

Impact:
  - Data affected:
  - Systems affected:
  - Customer impact:

What Went Well:

What Needs Improvement:

Action Items:
  [ ] Item — Owner — Due date
```

---

## 6. Customer and Regulatory Notification

| Condition | Notification Required | Timeline |
|-----------|----------------------|---------|
| Personal data breach affecting EU residents | Relevant Data Protection Authority | 72 hours from awareness |
| Material breach affecting customer operational data | Affected customers | As soon as reasonably practicable; no more than 72 hours |
| Billing data compromise | Affected customers + Stripe | Immediately |
| Safety incident (physical asset) | Affected customer + relevant aviation/maritime authority | Immediately |
| No customer data affected | No external notification required | — |

All external notifications are drafted by Communications Lead and reviewed by Head of Engineering and legal counsel before sending.

---

## 7. Evidence Preservation

- Do not delete, modify, or roll over logs before evidence is preserved
- Capture: system logs, Prometheus metrics, audit log export, network captures if available
- Store evidence in a designated incident storage location (not on affected systems)
- Retain for minimum 1 year (longer if litigation is possible)

---

## 8. Communication Guidelines

- **Internal**: Use dedicated incident Slack channel. No speculative attribution.
- **Customer-facing**: Factual, no speculation about root cause until confirmed. Lead with impact and resolution status, not technical details.
- **External/Public**: CEO or Head of Engineering only. Legal review required before any public statement about a breach.
- **Do not**: Post raw log data, CVE details, or attack methodology in public channels before patch is deployed.

---

## 9. Testing

- Incident response tabletop exercise: annually
- Runbook validation: test containment actions (key revocation, service isolation) in staging quarterly
- Alert testing: verify Prometheus → Alertmanager → PagerDuty/Slack path is functional monthly

---

*Next full review: 2027-03-21*
