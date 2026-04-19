# Change Management Procedure
**Heli.OS — BigMT.ai / Branca.ai**
**Classification: Internal**
**Version: 1.0 | Last Updated: 2026-03-21 | Owner: Head of Engineering**

---

## 1. Purpose

This procedure governs how changes to Heli.OS software, infrastructure, and configuration are proposed, reviewed, approved, deployed, and rolled back. It ensures that changes are traceable, reviewed for security impact, and reversible.

---

## 2. Scope

Applies to all changes to:
- Application code (all services: api-gateway, fabric, fusion, intelligence, tasking, inference, console)
- Infrastructure (K8s manifests, Helm charts, Terraform, docker-compose)
- Security configuration (RBAC rules, OPA policies, CORS, rate limits)
- CI/CD pipelines (`.github/workflows/`)
- Secrets and credentials (rotation, addition, removal)
- Third-party integrations (Stripe, OIDC provider, cloud services)
- This document set and other ISMS documents

---

## 3. Change Categories

| Category | Definition | Approval Required | Examples |
|----------|-----------|-------------------|---------|
| **Standard** | Routine change with well-understood risk; follows established procedure | Peer review (1 reviewer) | Bug fix, dependency update, UI change, new non-security feature |
| **Significant** | New feature or architectural change with moderate risk | Peer review + Head of Engineering sign-off | New API endpoint, new data store, new external integration |
| **Security** | Change to auth, billing, audit, RBAC, or security controls | Peer review + security-designated approver (CODEOWNERS) | Changes to `middleware/rbac.py`, `routers/billing.py`, `routers/audit.py`, K8s secrets, OPA policies |
| **Emergency** | Urgent fix required for P0/P1 incident | Incident Commander approval; post-hoc review within 24 hours | Patching active exploit, revoking compromised credentials, hotfix for safety issue |

---

## 4. Standard Change Process

### 4.1 Development

1. Create feature branch from `main` using naming convention: `feat/<description>`, `fix/<description>`, `sec/<description>`
2. Write code following Heli.OS secure coding standards:
   - No hardcoded secrets; use Vault / `get_secret()`
   - All external inputs sanitized before use
   - SQL via parameterized queries (asyncpg `$N` or SQLAlchemy ORM); never string concatenation
   - RBAC applied to any new endpoint handling sensitive data
   - Audit logging added for any privileged or state-mutating action
3. Self-review checklist before opening PR:
   - [ ] No secrets in code or tests
   - [ ] No `TODO: fix security` comments left open
   - [ ] New endpoints have RBAC and rate limiting applied
   - [ ] Threat model documented in PR body if change touches Confidential/Restricted data

### 4.2 CI Gate (automated)

All PRs must pass before merge:
- `pip-audit` — no new Critical/High CVEs introduced
- `bandit` — no new SAST findings above LOW severity (HIGH findings block merge)
- `npm run typecheck` — zero TypeScript errors
- `npm test` — all unit tests pass
- E2E tests (stable subset) — must pass
- Secret scanning — no credentials detected in diff

### 4.3 Code Review

- Minimum 1 approved review required for all PRs
- Security-sensitive paths (defined in `.github/CODEOWNERS`) require additional approval from `@bigmt/security`
- Reviewer checklist:
  - [ ] Logic is correct and edge cases handled
  - [ ] No obvious injection vulnerabilities (SQL, command, prompt)
  - [ ] RBAC applied to new endpoints
  - [ ] No new external data included in LLM prompts without sanitization
  - [ ] Error handling does not leak sensitive information (stack traces, DB details)
  - [ ] Audit logging present for state mutations

### 4.4 Deployment

1. Merge to `main` after approval
2. CI/CD pipeline runs full test suite on merge
3. Deployment to staging (automatic on merge to `main`)
4. Smoke test staging:
   - `/health` returns 200 across all services
   - Authentication flow works end-to-end
   - No new errors in logs
5. Deployment to production: manual trigger by Head of Engineering or delegated deployer
6. Post-deployment monitoring: watch Prometheus dashboards for 30 minutes after deploy

---

## 5. Security Change Process

For changes to security-critical paths (auth, billing, audit, RBAC, OPA policies, K8s secrets):

1. PR body must include a brief threat model:
   - What attack vector does this change address or potentially introduce?
   - What is the blast radius if this change is incorrect?
   - How was the change tested?
2. Two reviewers required: peer reviewer + security-designated approver
3. If change modifies an existing security control, document why the existing control was insufficient or needs updating
4. Post-deploy: run specific security validation (e.g., attempt unauthorized access and confirm 403, verify rate limit fires)

---

## 6. Emergency Change Process

For P0/P1 incidents requiring immediate production changes:

1. Incident Commander authorizes change verbally (Slack message in incident channel is sufficient record)
2. On-call engineer executes change with at least one other engineer observing (if possible)
3. Change is documented in the incident channel in real time:
   - What was changed
   - Why
   - Exact commands/files modified
4. Full PR and retrospective review within 24 hours of incident close
5. If the emergency change introduced technical debt or security debt, a follow-up ticket is created immediately

---

## 7. Infrastructure and Configuration Changes

Changes to K8s manifests, Helm charts, or docker-compose:

- Follow the same PR process as code changes
- `kubectl apply --dry-run=client` must pass before merge
- `helm template` must produce valid YAML before merge
- K8s secret values (`infra/k8s/secrets.yaml`) are never committed to source control (`.gitignore` enforced)
- All manifest changes reviewed by `@bigmt/infra`

---

## 8. Rollback Procedure

| Scenario | Rollback Method |
|----------|----------------|
| Bad application deploy | Re-deploy previous Git SHA via CI/CD; takes ~5 minutes |
| K8s manifest change | `kubectl apply -f <previous-manifest>` |
| Database migration (forward-only) | Apply compensating migration; restore from backup if data is corrupted |
| Secret rotation | Re-issue previous secret from Vault version history |
| Emergency rollback | Incident Commander can authorize `kubectl rollout undo deployment/<service>` without PR |

All rollbacks must be documented with: what was rolled back, when, why, and by whom.

---

## 9. Change Log

Major changes to production are recorded in `CHANGELOG.md` at the project root using [Keep a Changelog](https://keepachangelog.com/) format. Security fixes are noted as `Security:` entries. The changelog is updated on every production deployment.

---

## 10. Periodic Review

- This procedure is reviewed annually or after any significant incident that revealed a gap in the process
- CI/CD pipeline health is reviewed quarterly
- Secret rotation schedule is audited annually

---

*Next full review: 2027-03-21*
