/**
 * EngagementQueue — operator surface for the human-in-the-loop kinetic
 * authorization gate.
 *
 * Lists cases in PENDING_AUTHORIZATION state and lets a credentialed
 * operator review the ranked weapon options + AUTHORIZE / DENY / HOLD
 * the engagement.
 *
 * AUTHORIZE requires the operator's Ed25519 signature on the canonical
 * decision payload. The signing key is held outside the browser
 * (hardware token / CAC reader / external signer) — this component
 * surfaces the canonical payload + a paste field for the signature.
 *
 * For deployment without a hardware signer, an operator workstation
 * sidecar can subscribe to a local IPC channel and produce the signature
 * automatically when the operator confirms via biometrics / PIN.
 */
'use client';

import { useEffect, useState } from 'react';
import {
  CaseDetail,
  CaseSummary,
  buildCanonicalDecisionPayload,
  decideCase,
  getCase,
  listCases,
  markComplete,
} from '../../lib/engagement';

interface Props {
  operatorId: string;
  operatorRole: string;
  /**
   * Optional signer hook. If provided, called with the canonical
   * payload and expected to return a base64-url-encoded Ed25519
   * signature. If absent, a paste field is shown so the operator can
   * supply the signature from an external signer.
   */
  signFn?: (canonicalPayload: string) => Promise<string>;
}

export default function EngagementQueue({ operatorId, operatorRole, signFn }: Props) {
  const [cases, setCases] = useState<CaseSummary[]>([]);
  const [selected, setSelected] = useState<CaseDetail | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Refresh the queue every 5s
  useEffect(() => {
    const refresh = async () => {
      try {
        const all = await listCases('pending_authorization');
        setCases(all);
        setError(null);
      } catch (e: unknown) {
        setError(e instanceof Error ? e.message : String(e));
      }
    };
    refresh();
    const id = setInterval(refresh, 5000);
    return () => clearInterval(id);
  }, []);

  const select = async (caseId: string) => {
    setLoading(true);
    try {
      setSelected(await getCase(caseId));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ display: 'flex', height: '100%', color: '#00FF9C', fontFamily: 'IBM Plex Mono, monospace', background: '#080C0A' }}>
      <div style={{ width: 320, borderRight: '1px solid #00FF9C33', overflowY: 'auto' }}>
        <h2 style={{ padding: '12px 16px', fontFamily: 'Orbitron, sans-serif', fontSize: 14, letterSpacing: 1.5, borderBottom: '1px solid #00FF9C33' }}>
          PENDING AUTHORIZATION
        </h2>
        {error && <div style={{ padding: 12, color: '#FF3B3B' }}>{error}</div>}
        {cases.length === 0 && (
          <div style={{ padding: 16, color: '#888' }}>
            No cases awaiting decision.
          </div>
        )}
        {cases.map((c) => (
          <button
            key={c.case_id}
            onClick={() => select(c.case_id)}
            style={{
              display: 'block',
              width: '100%',
              padding: 12,
              textAlign: 'left',
              background: selected?.case_id === c.case_id ? '#00FF9C20' : 'transparent',
              border: 'none',
              borderBottom: '1px solid #00FF9C22',
              color: 'inherit',
              cursor: 'pointer',
              fontFamily: 'inherit',
            }}
          >
            <div style={{ fontWeight: 600, color: '#FFB300' }}>{c.classification}</div>
            <div style={{ fontSize: 11, color: '#888' }}>{c.entity_id}</div>
            <div style={{ fontSize: 10, color: '#666' }}>{new Date(c.created_at).toLocaleTimeString()}</div>
          </button>
        ))}
      </div>
      <div style={{ flex: 1, padding: 24, overflowY: 'auto' }}>
        {selected ? (
          <CaseDetailView
            caseDetail={selected}
            operatorId={operatorId}
            operatorRole={operatorRole}
            signFn={signFn}
            onAfterDecision={async () => setSelected(await getCase(selected.case_id))}
          />
        ) : loading ? (
          <div>Loading...</div>
        ) : (
          <div style={{ color: '#888' }}>Select a case to review.</div>
        )}
      </div>
    </div>
  );
}

interface DetailProps {
  caseDetail: CaseDetail;
  operatorId: string;
  operatorRole: string;
  signFn?: (canonical: string) => Promise<string>;
  onAfterDecision: () => Promise<void>;
}

function CaseDetailView({ caseDetail, operatorId, operatorRole, signFn, onAfterDecision }: DetailProps) {
  const [selectedOptionId, setSelectedOptionId] = useState<string | null>(
    caseDetail.options[0]?.option_id ?? null,
  );
  const [rationale, setRationale] = useState('');
  const [externalSig, setExternalSig] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const selectedOption = caseDetail.options.find((o) => o.option_id === selectedOptionId) ?? null;

  const decide = async (decision: 'AUTHORIZE' | 'DENY' | 'HOLD') => {
    setErr(null);
    setSubmitting(true);
    try {
      const decisionPayload = {
        decision,
        operator_id: operatorId,
        operator_role: operatorRole,
        rationale: rationale.trim() || (decision === 'AUTHORIZE' ? 'cleared hot' : decision.toLowerCase()),
        selected_option: decision === 'AUTHORIZE' ? selectedOptionId ?? undefined : undefined,
        engagement_class: 'counter_uas',  // TODO: derive from track classification
      };
      let signature: string | undefined;
      if (decision === 'AUTHORIZE') {
        const canonical = buildCanonicalDecisionPayload(caseDetail.case_id, decisionPayload);
        if (signFn) {
          signature = await signFn(canonical);
        } else if (externalSig) {
          signature = externalSig;
        } else {
          throw new Error('AUTHORIZE requires a signature. Either provide signFn prop or paste an external signature.');
        }
      }
      await decideCase(caseDetail.case_id, decisionPayload, signature);
      await onAfterDecision();
    } catch (e: unknown) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div>
      <h2 style={{ fontFamily: 'Orbitron, sans-serif', fontSize: 16, marginBottom: 12 }}>
        {caseDetail.classification} — {caseDetail.entity_id}
      </h2>
      <div style={{ fontSize: 11, color: '#888', marginBottom: 16 }}>
        case_id: {caseDetail.case_id} · state: <span style={{ color: '#FFB300' }}>{caseDetail.state}</span>
      </div>

      <Section title="TRACK">
        <KV k="track_id" v={caseDetail.track_id} />
        <KV k="confidence" v={String(caseDetail.classification)} />
      </Section>

      <Section title="PID">
        {caseDetail.pid ? (
          Object.entries(caseDetail.pid).map(([k, v]) => <KV key={k} k={k} v={String(v)} />)
        ) : (
          <em style={{ color: '#666' }}>not yet submitted</em>
        )}
      </Section>

      <Section title="ROE">
        {caseDetail.roe ? (
          Object.entries(caseDetail.roe).map(([k, v]) => <KV key={k} k={k} v={String(v)} />)
        ) : (
          <em style={{ color: '#666' }}>not yet submitted</em>
        )}
      </Section>

      <Section title="DECONFLICTION">
        {caseDetail.deconfliction ? (
          Object.entries(caseDetail.deconfliction).map(([k, v]) => <KV key={k} k={k} v={String(v)} />)
        ) : (
          <em style={{ color: '#666' }}>not yet submitted</em>
        )}
      </Section>

      <Section title={`WEAPON OPTIONS (${caseDetail.options.length})`}>
        {caseDetail.options.length === 0 && <em style={{ color: '#666' }}>no viable options</em>}
        {caseDetail.options.map((o) => (
          <label
            key={o.option_id}
            style={{
              display: 'block',
              padding: 8,
              marginBottom: 4,
              border: selectedOptionId === o.option_id ? '1px solid #00FF9C' : '1px solid #00FF9C33',
              cursor: 'pointer',
            }}
          >
            <input
              type="radio"
              name="opt"
              value={o.option_id}
              checked={selectedOptionId === o.option_id}
              onChange={() => setSelectedOptionId(o.option_id)}
              style={{ marginRight: 8 }}
            />
            <strong>{o.weapon_class}</strong> via {o.weapon_asset_id} · range {o.range_m.toFixed(0)}m · TOF {o.time_of_flight_s.toFixed(1)}s · PK {(o.pk_estimate * 100).toFixed(0)}%
            <div style={{ fontSize: 10, color: '#888', marginTop: 4 }}>{o.rationale}</div>
          </label>
        ))}
      </Section>

      <Section title="DECISION">
        <textarea
          value={rationale}
          onChange={(e) => setRationale(e.target.value)}
          placeholder="Operator rationale (required for audit trail)"
          rows={3}
          style={{
            width: '100%',
            background: '#0a1410',
            color: '#00FF9C',
            border: '1px solid #00FF9C33',
            padding: 8,
            fontFamily: 'inherit',
            marginBottom: 12,
          }}
        />
        {!signFn && (
          <div>
            <div style={{ fontSize: 11, color: '#888', marginBottom: 4 }}>
              Canonical payload to sign (paste back the base64 signature):
            </div>
            <pre
              style={{
                background: '#0a1410',
                padding: 8,
                fontSize: 10,
                overflowX: 'auto',
                marginBottom: 8,
              }}
            >
              {selectedOptionId
                ? buildCanonicalDecisionPayload(caseDetail.case_id, {
                    decision: 'AUTHORIZE',
                    operator_id: operatorId,
                    operator_role: operatorRole,
                    rationale,
                    selected_option: selectedOptionId,
                  })
                : '(select an option above)'}
            </pre>
            <input
              value={externalSig}
              onChange={(e) => setExternalSig(e.target.value)}
              placeholder="Paste base64-url Ed25519 signature here for AUTHORIZE"
              style={{
                width: '100%',
                background: '#0a1410',
                color: '#00FF9C',
                border: '1px solid #00FF9C33',
                padding: 8,
                fontFamily: 'inherit',
                marginBottom: 12,
              }}
            />
          </div>
        )}
        {err && <div style={{ color: '#FF3B3B', marginBottom: 8 }}>{err}</div>}
        <div style={{ display: 'flex', gap: 8 }}>
          <button
            disabled={submitting || !selectedOptionId || (!signFn && !externalSig)}
            onClick={() => decide('AUTHORIZE')}
            style={btn('#00FF9C')}
          >
            AUTHORIZE
          </button>
          <button disabled={submitting} onClick={() => decide('DENY')} style={btn('#FF3B3B')}>
            DENY
          </button>
          <button disabled={submitting} onClick={() => decide('HOLD')} style={btn('#FFB300')}>
            HOLD
          </button>
          {caseDetail.state === 'authorized' && (
            <button
              disabled={submitting}
              onClick={async () => {
                await markComplete(caseDetail.case_id, { effect: 'manual_complete' });
                await onAfterDecision();
              }}
              style={btn('#4FC3F7')}
            >
              MARK COMPLETE
            </button>
          )}
        </div>
      </Section>

      <Section title={`AUDIT (${caseDetail.audit.length} entries)`}>
        <div style={{ fontSize: 10 }}>
          {caseDetail.audit.map((a, i) => (
            <div key={i} style={{ marginBottom: 4, padding: 4, borderLeft: '2px solid #00FF9C44' }}>
              <span style={{ color: '#FFB300' }}>{a.transition}</span> →{' '}
              <span style={{ color: '#888' }}>{a.to_state}</span>{' '}
              <span style={{ color: '#666' }}>@ {new Date(a.ts).toLocaleTimeString()}</span>
            </div>
          ))}
        </div>
      </Section>
    </div>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div style={{ marginBottom: 24 }}>
      <h3 style={{ fontFamily: 'Orbitron, sans-serif', fontSize: 11, letterSpacing: 1.5, color: '#FFB300', marginBottom: 8 }}>
        {title}
      </h3>
      {children}
    </div>
  );
}

function KV({ k, v }: { k: string; v: string }) {
  return (
    <div style={{ display: 'flex', fontSize: 12, marginBottom: 2 }}>
      <span style={{ width: 160, color: '#888' }}>{k}</span>
      <span>{v}</span>
    </div>
  );
}

function btn(color: string): React.CSSProperties {
  return {
    padding: '8px 16px',
    background: 'transparent',
    color,
    border: `1px solid ${color}`,
    fontFamily: 'inherit',
    cursor: 'pointer',
    letterSpacing: 1,
    fontWeight: 600,
  };
}
