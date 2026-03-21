# Access Control Policy
**Summit.OS — BigMT.ai / Branca.ai**
**Classification: Internal**
**Version: 1.0 | Last Updated: 2026-03-21 | Owner: Head of Engineering**

---

## 1. Purpose

This policy governs how access to Summit.OS systems, data, and infrastructure is granted, reviewed, modified, and revoked. It implements the principle of least privilege and ensures that access rights are proportionate to role requirements and business need.

---

## 2. Role Hierarchy

Summit.OS uses a tiered role model enforced at the API layer via RBAC middleware (`apps/api-gateway/middleware/rbac.py`):

| Role | Access Level | Typical Holder |
|------|-------------|----------------|
| `VIEWER` | Read-only: world state, entities, public alerts | Stakeholders, auditors, read-only integrations |
| `OPERATOR` | VIEWER + submit tasks, acknowledge alerts, manage devices | Operations staff, ground crew |
| `MISSION_COMMANDER` | OPERATOR + create/modify missions, manage geofences | Mission leads, supervisors |
| `ADMIN` | MISSION_COMMANDER + manage users, rotate API keys, view audit logs | Engineering leads, operations managers |
| `SUPER_ADMIN` | Full access including platform configuration, role management | Head of Engineering only |

Role inheritance is strict — each role includes all permissions of roles below it.

**UI Role Mapping** (for console login):
- `ops` → `OPERATOR`
- `command` → `MISSION_COMMANDER`
- `dev` → `ADMIN`

---

## 3. Access Request and Provisioning

### 3.1 New Employee / Contractor
1. Hiring manager submits access request to Head of Engineering specifying:
   - Name, role, start date
   - Required system access (Summit.OS role, infrastructure access, GitHub team)
   - Business justification for each access level
2. Head of Engineering approves access levels within 1 business day
3. On-call engineer provisions access:
   - Create user account with minimum required role
   - Enable MFA requirement
   - Add to appropriate GitHub team (access to codebase)
   - Grant infrastructure access (K8s cluster, cloud console) only if role requires it
4. New user receives credentials via secure channel (password manager share, not email)
5. Provisioning is logged as an audit event

### 3.2 Access Elevation
Any request for role elevation requires:
1. Written request from the employee with business justification
2. Manager approval
3. Head of Engineering approval for ADMIN or SUPER_ADMIN elevation
4. Temporary elevations (e.g., for an incident) expire within 48 hours and must be explicitly time-bounded

### 3.3 Production Access
Production database and infrastructure access is restricted to:
- Head of Engineering
- On-call engineer during active incident (time-bounded)
- Any other access requires written approval from Head of Engineering

---

## 4. Authentication Requirements

| System | Authentication Requirement |
|--------|---------------------------|
| Summit.OS platform (API) | Short-lived OIDC JWT (max 1 hour) or scoped API key |
| Summit.OS console | OIDC + MFA |
| Production cloud console | MFA required; SSO preferred |
| Production Kubernetes cluster | kubeconfig with RBAC; no shared credentials |
| GitHub (codebase) | SSO + MFA enforced at organization level |
| Vault (secrets) | AppRole (services) or OIDC (engineers); MFA for human access |

API keys for service-to-service communication:
- Must be scoped to minimum required permissions
- Stored hashed (SHA-256) in database — plaintext never persisted after issuance
- Rotated at minimum annually or immediately on suspected compromise
- Never committed to source control (enforced by pre-commit hooks + CI secret scanning)

---

## 5. Access Review

| Frequency | Scope | Reviewer |
|-----------|-------|---------|
| **Quarterly** | All user accounts and roles | Head of Engineering |
| **On role change** | Affected user's access | Manager + Head of Engineering |
| **On termination** | All access for departing user | On-call engineer (same day) |
| **Annually** | API keys (service accounts, integration keys) | Head of Engineering |
| **Annually** | Third-party vendor access | Head of Engineering |

Quarterly access review checklist:
- [ ] All accounts have business-justified roles
- [ ] No accounts with elevated roles that are no longer needed
- [ ] No former employee accounts still active
- [ ] All API keys in use; inactive keys revoked
- [ ] Production access limited to required personnel

---

## 6. Offboarding and Deprovisioning

On confirmed termination (voluntary or involuntary):

| Action | Timing | Owner |
|--------|--------|-------|
| Revoke Summit.OS account | Within 24 hours | On-call engineer |
| Revoke GitHub organization access | Within 24 hours | Head of Engineering |
| Revoke cloud infrastructure access | Within 24 hours | On-call engineer |
| Rotate any personal API keys | Within 24 hours | On-call engineer |
| Rotate shared credentials the employee had access to | Within 48 hours | Head of Engineering judgment |
| Remove from PagerDuty / on-call rotation | Same day | Head of Engineering |
| Revoke Vault access | Within 24 hours | On-call engineer |

For involuntary terminations (especially security-sensitive departures), all access is revoked immediately before or simultaneously with notification, and the on-call engineer is notified in advance.

---

## 7. Privileged Access

Privileged access (ADMIN, SUPER_ADMIN, production infrastructure) is subject to additional controls:

- **Just-in-time access**: Prefer temporary elevation over permanent ADMIN grants where possible
- **Session recording**: Production shell sessions are logged where tooling supports it
- **Audit log**: All privileged actions captured in `audit_log` table with full attribution
- **Break-glass procedure**: For emergency access when normal authentication is unavailable, a documented break-glass procedure requires post-access review within 24 hours

---

## 8. API Keys and Service Accounts

| Control | Requirement |
|---------|------------|
| Naming | Service accounts named `svc-<service>-<environment>` |
| Scope | Minimum required API permissions |
| Storage | Vault or K8s Secret (`summit-secrets`); never in source code or environment literals in manifests |
| Rotation | Annual rotation minimum; immediate rotation on compromise or personnel change |
| Logging | All API key usage captured in audit log with key_id (not plaintext key) |
| Expiry | API keys issued with expiry where technically feasible |

---

## 9. Network and Infrastructure Access

- Production cluster kubeconfigs are not shared — each authorized engineer has individual credentials
- SSH access to any host requires key-based authentication; password authentication disabled
- VPN or equivalent required for production cluster access from outside trusted networks
- Firewall rules follow least-privilege: only required ports open between services (defined in K8s NetworkPolicy where deployed)

---

## 10. Exceptions

Exceptions to this policy require written approval from the Head of Engineering and must be:
- Documented with business justification
- Time-bounded (maximum 90 days)
- Reviewed at expiry

---

*Next full review: 2027-03-21*
