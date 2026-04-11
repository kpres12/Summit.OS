'use client';

/**
 * SecuritySettings — account security management panel.
 *
 * Features:
 *   • View enrolled MFA methods (TOTP + WebAuthn keys)
 *   • Remove a WebAuthn key
 *   • Enroll new MFA methods
 *   • View active sessions
 *   • Revoke individual sessions
 *   • Revoke all other sessions
 */

import React, { useState, useEffect, useCallback } from 'react';
import { useAuth } from '@/components/AuthProvider';
import MFASetup from './MFASetup';

interface MFAStatus {
  totp_enabled:    boolean;
  webauthn_count:  number;
  webauthn_keys:   { id: string; name: string; created_at: string; last_used: string | null }[];
  backup_codes_remaining: number;
}

interface Session {
  session_id:  string;
  created_at:  string;
  last_seen:   string;
  ip_address:  string;
  user_agent:  string;
  is_current:  boolean;
}

type Panel = 'overview' | 'add-mfa' | 'sessions';

export default function SecuritySettings() {
  const { user } = useAuth();
  const [panel, setPanel]         = useState<Panel>('overview');
  const [mfaStatus, setMfaStatus] = useState<MFAStatus | null>(null);
  const [sessions, setSessions]   = useState<Session[]>([]);
  const [loadingMFA, setLoadingMFA] = useState(true);
  const [loadingSess, setLoadingSess] = useState(false);
  const [error, setError]         = useState<string | null>(null);
  const [revoking, setRevoking]   = useState<string | null>(null);

  // ── Fetch MFA status ──
  const fetchMFAStatus = useCallback(async () => {
    setLoadingMFA(true);
    setError(null);
    try {
      const res  = await fetch('/api/auth/mfa/status', { credentials: 'same-origin' });
      const data = await res.json() as MFAStatus & { error?: string };
      if (!res.ok) { setError(data.error ?? 'Failed to load MFA status'); return; }
      setMfaStatus(data);
    } catch {
      setError('Network error');
    } finally {
      setLoadingMFA(false);
    }
  }, []);

  // ── Fetch sessions ──
  const fetchSessions = useCallback(async () => {
    setLoadingSess(true);
    try {
      const res  = await fetch('/api/auth/sessions', { credentials: 'same-origin' });
      const data = await res.json() as { sessions?: Session[]; error?: string };
      if (res.ok) setSessions(data.sessions ?? []);
    } catch { /* silent */ }
    finally { setLoadingSess(false); }
  }, []);

  useEffect(() => { fetchMFAStatus(); }, [fetchMFAStatus]);

  useEffect(() => {
    if (panel === 'sessions') fetchSessions();
  }, [panel, fetchSessions]);

  // ── Remove WebAuthn key ──
  const removeKey = async (keyId: string) => {
    if (revoking) return;
    setRevoking(keyId);
    try {
      const res = await fetch('/api/auth/mfa/webauthn/remove', {
        method:      'DELETE',
        credentials: 'same-origin',
        headers:     { 'Content-Type': 'application/json' },
        body:        JSON.stringify({ key_id: keyId }),
      });
      if (res.ok) await fetchMFAStatus();
    } finally {
      setRevoking(null);
    }
  };

  // ── Revoke a session ──
  const revokeSession = async (sessionId: string) => {
    if (revoking) return;
    setRevoking(sessionId);
    try {
      const res = await fetch('/api/auth/sessions/revoke', {
        method:      'POST',
        credentials: 'same-origin',
        headers:     { 'Content-Type': 'application/json' },
        body:        JSON.stringify({ session_id: sessionId }),
      });
      if (res.ok) setSessions((prev) => prev.filter((s) => s.session_id !== sessionId));
    } finally {
      setRevoking(null);
    }
  };

  const revokeAll = async () => {
    if (revoking) return;
    setRevoking('__all__');
    try {
      const res = await fetch('/api/auth/sessions/revoke-all', {
        method:      'POST',
        credentials: 'same-origin',
      });
      if (res.ok) setSessions((prev) => prev.filter((s) => s.is_current));
    } finally {
      setRevoking(null);
    }
  };

  // ── Add MFA panel ──
  if (panel === 'add-mfa') {
    return (
      <Card>
        <CardHeader onBack={() => setPanel('overview')}>ADD SECOND FACTOR</CardHeader>
        <div style={{ padding: '20px 24px' }}>
          <MFASetup
            onComplete={() => { fetchMFAStatus(); setPanel('overview'); }}
            onCancel={() => setPanel('overview')}
          />
        </div>
      </Card>
    );
  }

  // ── Sessions panel ──
  if (panel === 'sessions') {
    return (
      <Card>
        <CardHeader onBack={() => setPanel('overview')}>ACTIVE SESSIONS</CardHeader>
        <div style={{ padding: '0 0 20px' }}>
          {loadingSess
            ? <LoadingRow />
            : sessions.length === 0
              ? <EmptyRow>NO SESSIONS FOUND</EmptyRow>
              : sessions.map((s) => (
                  <SessionRow
                    key={s.session_id}
                    session={s}
                    onRevoke={() => revokeSession(s.session_id)}
                    revoking={revoking === s.session_id}
                  />
                ))
          }
          {sessions.filter((s) => !s.is_current).length > 1 && (
            <div style={{ padding: '12px 24px', borderTop: '1px solid rgba(0,232,150,0.08)' }}>
              <DangerButton
                onClick={revokeAll}
                disabled={revoking === '__all__'}
              >
                {revoking === '__all__' ? 'REVOKING...' : '✕ REVOKE ALL OTHER SESSIONS'}
              </DangerButton>
            </div>
          )}
        </div>
      </Card>
    );
  }

  // ── Overview ──
  return (
    <Card>
      <CardHeader>ACCOUNT SECURITY</CardHeader>

      {/* User info */}
      <div style={{ padding: '16px 24px', borderBottom: '1px solid rgba(0,232,150,0.08)' }}>
        <div style={labelStyle}>AUTHENTICATED AS</div>
        <div style={{
          fontFamily:    'var(--font-ibm-plex-mono), monospace',
          fontSize:      '12px',
          color:         'rgba(200,230,201,0.8)',
          marginTop:     '4px',
        }}>
          {user?.email ?? '—'}
        </div>
        {user?.roles && user.roles.length > 0 && (
          <div style={{ marginTop: '6px', display: 'flex', gap: '6px', flexWrap: 'wrap' }}>
            {user.roles.map((r) => (
              <span key={r} style={{
                padding:       '2px 8px',
                background:    'rgba(0,232,150,0.06)',
                border:        '1px solid rgba(0,232,150,0.2)',
                color:         '#00E896',
                fontFamily:    'var(--font-ibm-plex-mono), monospace',
                fontSize:      '8px',
                letterSpacing: '0.1em',
              }}>
                {r}
              </span>
            ))}
          </div>
        )}
      </div>

      {error && (
        <div style={{ padding: '0 24px', marginTop: '16px' }}>
          <ErrorBanner>{error}</ErrorBanner>
        </div>
      )}

      {/* MFA section */}
      <div style={{ padding: '16px 24px', borderBottom: '1px solid rgba(0,232,150,0.08)' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
          <div style={labelStyle}>SECOND FACTOR (MFA)</div>
          <button
            onClick={() => setPanel('add-mfa')}
            style={{
              padding:       '5px 10px',
              background:    'rgba(0,232,150,0.08)',
              border:        '1px solid rgba(0,232,150,0.25)',
              color:         '#00E896',
              fontFamily:    'var(--font-ibm-plex-mono), monospace',
              fontSize:      '8px',
              letterSpacing: '0.15em',
              cursor:        'pointer',
            }}
          >
            + ADD
          </button>
        </div>

        {loadingMFA
          ? <LoadingRow />
          : (
            <>
              {/* TOTP */}
              <MFARow
                icon="📱"
                label="AUTHENTICATOR APP"
                status={mfaStatus?.totp_enabled ? 'ENROLLED' : 'NOT ENROLLED'}
                ok={mfaStatus?.totp_enabled ?? false}
                sub={mfaStatus?.totp_enabled
                  ? `${mfaStatus.backup_codes_remaining} backup codes remaining`
                  : 'Authy, Google Authenticator, or any TOTP app'}
              />

              {/* WebAuthn keys */}
              {(mfaStatus?.webauthn_keys ?? []).map((key) => (
                <MFARow
                  key={key.id}
                  icon="🔑"
                  label={key.name.toUpperCase()}
                  status="REGISTERED"
                  ok={true}
                  sub={`Last used: ${key.last_used ? new Date(key.last_used).toLocaleDateString() : 'never'}`}
                  onRemove={() => removeKey(key.id)}
                  removing={revoking === key.id}
                />
              ))}

              {mfaStatus?.webauthn_count === 0 && !mfaStatus.totp_enabled && (
                <div style={{
                  padding:    '10px 12px',
                  background: 'rgba(255,179,0,0.06)',
                  border:     '1px solid rgba(255,179,0,0.2)',
                  fontFamily: 'var(--font-ibm-plex-mono), monospace',
                  fontSize:   '9px',
                  color:      '#FFB300',
                }}>
                  ⚠ NO SECOND FACTOR — ACCOUNT AT RISK
                </div>
              )}
            </>
          )
        }
      </div>

      {/* Sessions */}
      <div style={{ padding: '16px 24px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div style={labelStyle}>SESSIONS</div>
          <button
            onClick={() => setPanel('sessions')}
            style={{
              background:    'transparent',
              border:        'none',
              color:         'rgba(200,230,201,0.35)',
              fontFamily:    'var(--font-ibm-plex-mono), monospace',
              fontSize:      '9px',
              cursor:        'pointer',
              textDecoration: 'underline',
            }}
          >
            MANAGE →
          </button>
        </div>
      </div>
    </Card>
  );
}

// ── Sub-components ──────────────────────────────────────────────────────────

function Card({ children }: { children: React.ReactNode }) {
  return (
    <div style={{
      background: '#0D1210',
      border:     '1px solid rgba(0,232,150,0.2)',
      boxShadow:  '0 0 40px rgba(0,232,150,0.05)',
      width:      '420px',
      maxWidth:   '100%',
    }}>
      {children}
    </div>
  );
}

function CardHeader({ children, onBack }: { children: React.ReactNode; onBack?: () => void }) {
  return (
    <div style={{
      padding:      '16px 24px',
      borderBottom: '1px solid rgba(0,232,150,0.12)',
      display:      'flex',
      alignItems:   'center',
      gap:          '12px',
    }}>
      {onBack && (
        <button
          onClick={onBack}
          style={{
            background: 'transparent', border: 'none',
            color:      'rgba(200,230,201,0.4)',
            fontFamily: 'var(--font-ibm-plex-mono), monospace',
            fontSize:   '10px', cursor: 'pointer', padding: 0,
          }}
        >
          ←
        </button>
      )}
      <div style={{
        fontFamily: 'var(--font-ibm-plex-mono), monospace',
        fontSize:      '10px',
        color:         'rgba(0,232,150,0.5)',
        letterSpacing: '0.3em',
      }}>
        {children}
      </div>
    </div>
  );
}

function MFARow({ icon, label, status, ok, sub, onRemove, removing }: {
  icon: string;
  label: string;
  status: string;
  ok: boolean;
  sub?: string;
  onRemove?: () => void;
  removing?: boolean;
}) {
  return (
    <div style={{
      display:       'flex',
      alignItems:    'center',
      gap:           '10px',
      marginBottom:  '8px',
      padding:       '10px 12px',
      background:    '#080C0A',
      border:        '1px solid rgba(0,232,150,0.08)',
    }}>
      <span style={{ fontSize: '14px' }}>{icon}</span>
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{
          fontFamily:    'var(--font-ibm-plex-mono), monospace',
          fontSize:      '10px',
          color:         'rgba(200,230,201,0.7)',
          letterSpacing: '0.1em',
        }}>
          {label}
        </div>
        {sub && (
          <div style={{
            fontFamily: 'var(--font-ibm-plex-mono), monospace',
            fontSize:   '8px',
            color:      'rgba(200,230,201,0.25)',
            marginTop:  '2px',
          }}>
            {sub}
          </div>
        )}
      </div>
      <span style={{
        fontFamily:    'var(--font-ibm-plex-mono), monospace',
        fontSize:      '8px',
        color:         ok ? '#00E896' : 'rgba(200,230,201,0.25)',
        letterSpacing: '0.1em',
        whiteSpace:    'nowrap',
      }}>
        {status}
      </span>
      {onRemove && (
        <button
          onClick={onRemove}
          disabled={removing}
          style={{
            marginLeft:  '8px',
            padding:     '3px 7px',
            background:  'transparent',
            border:      '1px solid rgba(255,59,59,0.3)',
            color:       removing ? 'rgba(255,59,59,0.3)' : '#FF3B3B',
            fontFamily:  'var(--font-ibm-plex-mono), monospace',
            fontSize:    '8px',
            cursor:      removing ? 'not-allowed' : 'pointer',
          }}
        >
          {removing ? '...' : '✕'}
        </button>
      )}
    </div>
  );
}

function SessionRow({ session, onRevoke, revoking }: {
  session: Session;
  onRevoke: () => void;
  revoking: boolean;
}) {
  const ua = session.user_agent;
  const browser = ua.includes('Chrome') ? 'Chrome' : ua.includes('Firefox') ? 'Firefox'
    : ua.includes('Safari') ? 'Safari' : 'Browser';
  const os = ua.includes('Mac') ? 'macOS' : ua.includes('Windows') ? 'Windows'
    : ua.includes('Linux') ? 'Linux' : 'Unknown OS';

  return (
    <div style={{
      display:       'flex',
      alignItems:    'center',
      gap:           '12px',
      padding:       '12px 24px',
      borderBottom:  '1px solid rgba(0,232,150,0.05)',
      background:    session.is_current ? 'rgba(0,232,150,0.02)' : 'transparent',
    }}>
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{
          display:    'flex',
          alignItems: 'center',
          gap:        '6px',
          marginBottom: '3px',
        }}>
          <span style={{
            fontFamily:    'var(--font-ibm-plex-mono), monospace',
            fontSize:      '10px',
            color:         'rgba(200,230,201,0.7)',
          }}>
            {browser} · {os}
          </span>
          {session.is_current && (
            <span style={{
              padding:       '1px 5px',
              background:    'rgba(0,232,150,0.1)',
              border:        '1px solid rgba(0,232,150,0.25)',
              color:         '#00E896',
              fontFamily:    'var(--font-ibm-plex-mono), monospace',
              fontSize:      '7px',
              letterSpacing: '0.1em',
            }}>
              CURRENT
            </span>
          )}
        </div>
        <div style={{
          fontFamily: 'var(--font-ibm-plex-mono), monospace',
          fontSize:   '8px',
          color:      'rgba(200,230,201,0.25)',
        }}>
          {session.ip_address} · last seen {new Date(session.last_seen).toLocaleDateString()}
        </div>
      </div>
      {!session.is_current && (
        <button
          onClick={onRevoke}
          disabled={revoking}
          style={{
            padding:    '5px 10px',
            background: 'transparent',
            border:     '1px solid rgba(255,59,59,0.3)',
            color:      revoking ? 'rgba(255,59,59,0.3)' : '#FF3B3B',
            fontFamily: 'var(--font-ibm-plex-mono), monospace',
            fontSize:   '8px',
            cursor:     revoking ? 'not-allowed' : 'pointer',
          }}
        >
          {revoking ? '...' : 'REVOKE'}
        </button>
      )}
    </div>
  );
}

function LoadingRow() {
  return (
    <div style={{
      padding:    '16px 24px',
      fontFamily: 'var(--font-ibm-plex-mono), monospace',
      fontSize:   '9px',
      color:      'rgba(200,230,201,0.25)',
    }}>
      LOADING...
    </div>
  );
}

function EmptyRow({ children }: { children: React.ReactNode }) {
  return (
    <div style={{
      padding:    '16px 24px',
      fontFamily: 'var(--font-ibm-plex-mono), monospace',
      fontSize:   '9px',
      color:      'rgba(200,230,201,0.25)',
    }}>
      {children}
    </div>
  );
}

function ErrorBanner({ children }: { children: React.ReactNode }) {
  return (
    <div style={{
      padding:    '10px 12px',
      background: 'rgba(255,59,59,0.08)',
      border:     '1px solid rgba(255,59,59,0.3)',
      fontFamily: 'var(--font-ibm-plex-mono), monospace',
      fontSize:   '10px',
      color:      '#FF3B3B',
      marginBottom: '8px',
    }}>
      ✕ {typeof children === 'string' ? children.toUpperCase() : children}
    </div>
  );
}

function DangerButton({ children, onClick, disabled }: {
  children: React.ReactNode;
  onClick: () => void;
  disabled?: boolean;
}) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      style={{
        padding:       '10px 16px',
        background:    'transparent',
        border:        '1px solid rgba(255,59,59,0.3)',
        color:         disabled ? 'rgba(255,59,59,0.3)' : '#FF3B3B',
        fontFamily:    'var(--font-ibm-plex-mono), monospace',
        fontSize:      '9px',
        letterSpacing: '0.15em',
        cursor:        disabled ? 'not-allowed' : 'pointer',
        width:         '100%',
      }}
    >
      {children}
    </button>
  );
}

const labelStyle: React.CSSProperties = {
  fontFamily: 'var(--font-ibm-plex-mono), monospace',
  fontSize:      '9px',
  color:         'rgba(0,232,150,0.4)',
  letterSpacing: '0.25em',
};
