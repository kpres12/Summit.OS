/**
 * Engagement Authorization API client.
 *
 * Talks to /engagement/* on the API gateway. The AUTHORIZE endpoint
 * requires the operator's Ed25519 signature in the X-Operator-Signature
 * header — this is the load-bearing human-in-the-loop step for any
 * kinetic action.
 */

import { apiFetch } from './api';

// ---------------------------------------------------------------------------
// Types matching apps/api-gateway/routers/engagement.py schemas
// ---------------------------------------------------------------------------

export type EngagementState =
  | 'detected'
  | 'pid_confirmed'
  | 'roe_cleared'
  | 'deconflicted'
  | 'pending_authorization'
  | 'authorized'
  | 'denied'
  | 'held'
  | 'expired'
  | 'complete';

export interface CaseSummary {
  case_id: string;
  state: EngagementState;
  track_id: string;
  entity_id: string;
  classification: string;
  created_at: string;
  n_audit_entries: number;
  auth_ttl?: string | null;
}

export interface WeaponOption {
  option_id: string;
  weapon_asset_id: string;
  weapon_class: string;
  range_m: number;
  time_of_flight_s: number;
  pk_estimate: number;
  roe_compliant: boolean;
  deconfliction_ok: boolean;
  rationale: string;
}

export interface CaseDetail extends CaseSummary {
  pid?: Record<string, unknown> | null;
  roe?: Record<string, unknown> | null;
  deconfliction?: Record<string, unknown> | null;
  options: WeaponOption[];
  decision?: Record<string, unknown> | null;
  audit: Array<{
    ts: string;
    transition: string;
    to_state: string;
    payload: Record<string, unknown>;
  }>;
}

export type OperatorDecisionType = 'AUTHORIZE' | 'DENY' | 'HOLD' | 'REQUEST_HIGHER';

export interface OperatorDecision {
  decision: OperatorDecisionType;
  operator_id: string;
  operator_role: string;
  rationale: string;
  selected_option?: string;
  engagement_class?: string;
}

// ---------------------------------------------------------------------------
// API surface
// ---------------------------------------------------------------------------

export async function listCases(state?: EngagementState): Promise<CaseSummary[]> {
  const q = state ? `?state=${encodeURIComponent(state)}` : '';
  const r = await apiFetch(`/engagement/cases${q}`);
  return r.json();
}

export async function getCase(caseId: string): Promise<CaseDetail> {
  const r = await apiFetch(`/engagement/cases/${caseId}`);
  return r.json();
}

export async function decideCase(
  caseId: string,
  decision: OperatorDecision,
  signatureBase64?: string,
): Promise<CaseDetail> {
  const headers: Record<string, string> = {};
  if (decision.decision === 'AUTHORIZE') {
    if (!signatureBase64) {
      throw new Error(
        'AUTHORIZE requires an Ed25519 signature in the X-Operator-Signature header. ' +
        'Use signAuthorizationDecision() to produce one.',
      );
    }
    headers['X-Operator-Signature'] = signatureBase64;
  }
  const r = await apiFetch(`/engagement/cases/${caseId}/decide`, {
    method: 'POST',
    body: JSON.stringify(decision),
    headers,
  });
  return r.json();
}

export async function markComplete(
  caseId: string,
  bda?: Record<string, unknown>,
): Promise<CaseDetail> {
  const r = await apiFetch(`/engagement/cases/${caseId}/complete`, {
    method: 'POST',
    body: JSON.stringify(bda ?? null),
  });
  return r.json();
}

export async function expireStale(): Promise<{ expired_case_ids: string[]; count: number }> {
  const r = await apiFetch(`/engagement/expire-stale`, { method: 'POST' });
  return r.json();
}

// ---------------------------------------------------------------------------
// Operator signature helper
// ---------------------------------------------------------------------------

/**
 * Compute the canonical decision payload the gate's verifier will check
 * against. Order + key set MUST match what
 * EngagementAuthorizationGate.authorize() canonicalizes server-side.
 *
 * Operators sign this payload locally with their private Ed25519 key
 * (held outside the browser — typically a hardware token or a CAC reader).
 * For development, `signAuthorizationDecisionWithLocalKey()` exists as a
 * stand-in but should NEVER be used in production.
 */
export function buildCanonicalDecisionPayload(
  caseId: string,
  decision: OperatorDecision,
): string {
  const payload = {
    case_id: caseId,
    decision: decision.decision.toLowerCase(),
    operator_id: decision.operator_id,
    operator_role: decision.operator_role,
    selected_option: decision.selected_option ?? null,
  };
  // Keys sorted, no spaces — matches Python json.dumps(sort_keys=True, separators=(",", ":"))
  return JSON.stringify(payload, Object.keys(payload).sort());
}

/**
 * Returns true if the case is awaiting an operator decision.
 */
export function awaitsDecision(c: CaseSummary): boolean {
  return c.state === 'pending_authorization';
}
