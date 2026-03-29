'use client';

/**
 * MFASetup — enrollment flow for TOTP and WebAuthn.
 *
 * Steps:
 *   TOTP:    1. Fetch secret → show QR + manual key
 *            2. User scans → enters code to confirm
 *            3. Show one-time backup codes
 *
 *   WebAuthn: 1. Fetch registration options
 *              2. navigator.credentials.create()
 *              3. POST attestation to register-complete
 */

import React, { useState, useCallback } from 'react';
import Image from 'next/image';

type SetupType = 'totp' | 'webauthn';
type TOTPStep  = 'qr' | 'verify' | 'backup';
type WAStep    = 'start' | 'done';

interface Props {
  onComplete: () => void;
  onCancel:   () => void;
}

// ── TOTP Enrollment ──────────────────────────────────────────────────────────
function TOTPEnrollment({ onComplete, onCancel }: Props) {
  const [step, setStep]       = useState<TOTPStep>('qr');
  const [qrUri, setQrUri]     = useState<string | null>(null);
  const [secret, setSecret]   = useState<string | null>(null);
  const [code, setCode]       = useState('');
  const [backups, setBackups] = useState<string[]>([]);
  const [error, setError]     = useState<string | null>(null);
  const [busy, setBusy]       = useState(false);
  const [copiedIdx, setCopiedIdx] = useState<number | null>(null);

  const loadQR = useCallback(async () => {
    setBusy(true);
    setError(null);
    try {
      const res  = await fetch('/api/auth/mfa/totp/enroll', {
        method:      'POST',
        credentials: 'same-origin',
      });
      const data = await res.json() as {
        qr_uri?: string; secret?: string; error?: string;
      };
      if (!res.ok) { setError(data.error ?? 'Enrollment failed'); return; }
      setQrUri(data.qr_uri ?? null);
      setSecret(data.secret ?? null);
    } catch {
      setError('Network error');
    } finally {
      setBusy(false);
    }
  }, []);

  // Load QR on first render
  React.useEffect(() => { loadQR(); }, [loadQR]);

  const verifyCode = async (e: React.FormEvent) => {
    e.preventDefault();
    if (busy || code.length !== 6) return;
    setBusy(true);
    setError(null);
    try {
      const res  = await fetch('/api/auth/mfa/totp/verify', {
        method:      'POST',
        credentials: 'same-origin',
        headers:     { 'Content-Type': 'application/json' },
        body:        JSON.stringify({ code, action: 'enroll' }),
      });
      const data = await res.json() as { ok?: boolean; backup_codes?: string[]; error?: string };
      if (!res.ok) { setError(data.error ?? 'Invalid code'); setCode(''); return; }
      setBackups(data.backup_codes ?? []);
      setStep('backup');
    } catch {
      setError('Network error');
    } finally {
      setBusy(false);
    }
  };

  const copyBackup = (code: string, idx: number) => {
    navigator.clipboard.writeText(code);
    setCopiedIdx(idx);
    setTimeout(() => setCopiedIdx(null), 1500);
  };

  const copyAll = () => {
    navigator.clipboard.writeText(backups.join('\n'));
  };

  // ── Step: QR code ──
  if (step === 'qr') return (
    <div>
      <SectionLabel>STEP 1 OF 2 — SCAN QR CODE</SectionLabel>
      <p style={helpStyle}>
        Open Authy, Google Authenticator, or any TOTP app and scan the code below.
      </p>

      {error && <ErrorBanner>{error}</ErrorBanner>}

      <div style={{
        margin:     '16px 0',
        padding:    '16px',
        background: '#fff',
        display:    'inline-block',
        lineHeight: 0,
      }}>
        {busy && !qrUri
          ? <div style={{ width: 160, height: 160, background: '#eee' }} />
          : qrUri
            ? <Image src={qrUri} alt="TOTP QR Code" width={160} height={160} unoptimized />
            : null}
      </div>

      {secret && (
        <div style={{ marginTop: '8px' }}>
          <div style={{ ...helpStyle, marginBottom: '4px' }}>MANUAL ENTRY KEY:</div>
          <code style={{
            display:       'block',
            padding:       '8px 10px',
            background:    '#080C0A',
            border:        '1px solid rgba(0,255,156,0.15)',
            color:         '#00FF9C',
            fontFamily:    'var(--font-ibm-plex-mono), monospace',
            fontSize:      '11px',
            letterSpacing: '0.1em',
            wordBreak:     'break-all',
          }}>
            {secret}
          </code>
        </div>
      )}

      <div style={{ display: 'flex', gap: '8px', marginTop: '20px' }}>
        <GhostButton onClick={onCancel}>CANCEL</GhostButton>
        <PrimaryButton onClick={() => setStep('verify')} disabled={!qrUri}>
          ► NEXT: VERIFY CODE
        </PrimaryButton>
      </div>
    </div>
  );

  // ── Step: Verify code ──
  if (step === 'verify') return (
    <form onSubmit={verifyCode}>
      <SectionLabel>STEP 2 OF 2 — ENTER CODE TO CONFIRM</SectionLabel>
      <p style={helpStyle}>
        Enter the 6-digit code now shown in your authenticator app.
      </p>

      {error && <ErrorBanner>{error}</ErrorBanner>}

      <input
        type="text"
        inputMode="numeric"
        pattern="[0-9]*"
        maxLength={6}
        value={code}
        onChange={(e) => setCode(e.target.value.replace(/\D/g, '').slice(0, 6))}
        placeholder="000000"
        autoFocus
        autoComplete="one-time-code"
        disabled={busy}
        style={{
          marginTop:     '12px',
          width:         '100%',
          padding:       '14px',
          background:    '#080C0A',
          border:        '1px solid rgba(0,255,156,0.25)',
          color:         '#00FF9C',
          fontFamily:    'var(--font-ibm-plex-mono), monospace',
          fontSize:      '24px',
          letterSpacing: '0.5em',
          textAlign:     'center',
          outline:       'none',
          boxSizing:     'border-box',
        }}
      />

      <div style={{ display: 'flex', gap: '8px', marginTop: '16px' }}>
        <GhostButton onClick={() => { setStep('qr'); setCode(''); setError(null); }}>
          ← BACK
        </GhostButton>
        <PrimaryButton type="submit" disabled={busy || code.length !== 6}>
          {busy ? 'VERIFYING...' : '► CONFIRM ENROLLMENT'}
        </PrimaryButton>
      </div>
    </form>
  );

  // ── Step: Backup codes ──
  return (
    <div>
      <SectionLabel>AUTHENTICATOR ENROLLED — SAVE BACKUP CODES</SectionLabel>
      <p style={{ ...helpStyle, color: '#FFB300' }}>
        These one-time codes let you recover access if you lose your device.
        Each code works once. Store them somewhere safe.
      </p>

      <div style={{
        display:               'grid',
        gridTemplateColumns:   '1fr 1fr',
        gap:                   '6px',
        margin:                '12px 0',
      }}>
        {backups.map((b, i) => (
          <button
            key={i}
            onClick={() => copyBackup(b, i)}
            style={{
              padding:    '8px 10px',
              background: copiedIdx === i ? 'rgba(0,255,156,0.08)' : '#080C0A',
              border:     `1px solid ${copiedIdx === i ? 'rgba(0,255,156,0.4)' : 'rgba(0,255,156,0.12)'}`,
              color:      copiedIdx === i ? '#00FF9C' : 'rgba(200,230,201,0.6)',
              fontFamily: 'var(--font-ibm-plex-mono), monospace',
              fontSize:   '11px',
              cursor:     'pointer',
              textAlign:  'left',
              transition: 'all 150ms',
            }}
          >
            {copiedIdx === i ? '✓ COPIED' : b}
          </button>
        ))}
      </div>

      <div style={{ display: 'flex', gap: '8px', marginTop: '4px' }}>
        <GhostButton onClick={copyAll}>COPY ALL</GhostButton>
        <PrimaryButton onClick={onComplete}>► DONE</PrimaryButton>
      </div>
    </div>
  );
}

// ── WebAuthn Enrollment ───────────────────────────────────────────────────────
function WebAuthnEnrollment({ onComplete, onCancel }: Props) {
  const [step, setStep]     = useState<WAStep>('start');
  const [keyName, setKeyName] = useState('');
  const [error, setError]   = useState<string | null>(null);
  const [busy, setBusy]     = useState(false);

  const register = async () => {
    if (busy) return;
    setBusy(true);
    setError(null);
    try {
      // 1. Get registration options
      const beginRes = await fetch('/api/auth/mfa/webauthn/register-begin', {
        method:      'POST',
        credentials: 'same-origin',
        headers:     { 'Content-Type': 'application/json' },
        body:        JSON.stringify({ key_name: keyName || 'Security Key' }),
      });
      if (!beginRes.ok) {
        setError('Could not start registration');
        return;
      }
      const options = await beginRes.json();

      // 2. Decode and invoke browser API
      const publicKey: PublicKeyCredentialCreationOptions = {
        ...options,
        challenge: base64urlToBuffer(options.challenge),
        user: {
          ...options.user,
          id: base64urlToBuffer(options.user.id),
        },
        excludeCredentials: (options.excludeCredentials ?? []).map(
          (c: { id: string; type: string }) => ({
            ...c,
            id: base64urlToBuffer(c.id),
          })
        ),
      };

      const credential = await navigator.credentials.create({ publicKey }) as PublicKeyCredential;
      if (!credential) {
        setError('Registration cancelled');
        return;
      }

      const attestationResponse = credential.response as AuthenticatorAttestationResponse;

      // 3. Send attestation to server
      const completeRes = await fetch('/api/auth/mfa/webauthn/register-complete', {
        method:      'POST',
        credentials: 'same-origin',
        headers:     { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          id:    credential.id,
          type:  credential.type,
          rawId: bufferToBase64url(credential.rawId),
          response: {
            attestationObject: bufferToBase64url(attestationResponse.attestationObject),
            clientDataJSON:    bufferToBase64url(attestationResponse.clientDataJSON),
          },
          key_name: keyName || 'Security Key',
        }),
      });

      const data = await completeRes.json() as { ok?: boolean; error?: string };
      if (!completeRes.ok) {
        setError(data.error ?? 'Registration failed');
        return;
      }
      setStep('done');
    } catch (err) {
      if (err instanceof DOMException && err.name === 'NotAllowedError') {
        setError('Registration timed out or was cancelled');
      } else {
        setError('Hardware key error');
      }
    } finally {
      setBusy(false);
    }
  };

  if (step === 'done') return (
    <div>
      <SectionLabel>HARDWARE KEY REGISTERED</SectionLabel>
      <div style={{
        padding:    '20px',
        background: 'rgba(0,255,156,0.04)',
        border:     '1px solid rgba(0,255,156,0.15)',
        textAlign:  'center',
        margin:     '12px 0',
      }}>
        <div style={{ fontSize: '28px', marginBottom: '8px' }}>🔑</div>
        <div style={{
          fontFamily: 'var(--font-ibm-plex-mono), monospace',
          fontSize:   '10px',
          color:      '#00FF9C',
        }}>
          {keyName || 'SECURITY KEY'} REGISTERED SUCCESSFULLY
        </div>
      </div>
      <PrimaryButton onClick={onComplete}>► DONE</PrimaryButton>
    </div>
  );

  return (
    <div>
      <SectionLabel>REGISTER HARDWARE KEY</SectionLabel>
      <p style={helpStyle}>
        Supports YubiKey, FIDO2 keys, Touch ID, Face ID, and Windows Hello.
      </p>

      {error && <ErrorBanner>{error}</ErrorBanner>}

      <div style={{ marginTop: '12px', marginBottom: '16px' }}>
        <div style={{ ...helpStyle, marginBottom: '4px' }}>KEY NICKNAME (OPTIONAL)</div>
        <input
          type="text"
          value={keyName}
          onChange={(e) => setKeyName(e.target.value)}
          placeholder="YubiKey 5C"
          maxLength={40}
          disabled={busy}
          style={{
            width:      '100%',
            padding:    '10px 12px',
            background: '#080C0A',
            border:     '1px solid rgba(0,255,156,0.2)',
            color:      'rgba(200,230,201,0.8)',
            fontFamily: 'var(--font-ibm-plex-mono), monospace',
            fontSize:   '12px',
            outline:    'none',
            boxSizing:  'border-box',
          }}
        />
      </div>

      <div style={{ display: 'flex', gap: '8px' }}>
        <GhostButton onClick={onCancel} disabled={busy}>CANCEL</GhostButton>
        <PrimaryButton onClick={register} disabled={busy}>
          {busy ? 'TOUCH YOUR KEY...' : '► REGISTER KEY'}
        </PrimaryButton>
      </div>

      <div style={{ ...helpStyle, marginTop: '12px' }}>
        FIDO2 · WEBAUTHN LEVEL 2 · RESIDENT KEY PREFERRED
      </div>
    </div>
  );
}

// ── Main MFASetup component ───────────────────────────────────────────────────
export default function MFASetup({ onComplete, onCancel }: Props) {
  const [type, setType] = useState<SetupType | null>(null);

  if (type === 'totp')    return <TOTPEnrollment    onComplete={onComplete} onCancel={() => setType(null)} />;
  if (type === 'webauthn') return <WebAuthnEnrollment onComplete={onComplete} onCancel={() => setType(null)} />;

  return (
    <div>
      <SectionLabel>ADD SECOND FACTOR</SectionLabel>
      <p style={helpStyle}>
        Choose your preferred MFA method. Hardware keys (FIDO2) provide the strongest protection.
      </p>

      <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', marginTop: '16px' }}>
        <MethodCard
          onClick={() => setType('webauthn')}
          icon="🔑"
          title="HARDWARE SECURITY KEY"
          description="YubiKey, FIDO2 key, Touch ID, Face ID, Windows Hello"
          badge="STRONGEST"
          badgeColor="#00FF9C"
        />
        <MethodCard
          onClick={() => setType('totp')}
          icon="📱"
          title="AUTHENTICATOR APP"
          description="Authy, Google Authenticator, or any TOTP app"
          badge="RECOMMENDED"
          badgeColor="#FFB300"
        />
      </div>

      <div style={{ marginTop: '16px' }}>
        <GhostButton onClick={onCancel}>CANCEL</GhostButton>
      </div>
    </div>
  );
}

// ── Shared sub-components ─────────────────────────────────────────────────────
function MethodCard({ onClick, icon, title, description, badge, badgeColor }: {
  onClick: () => void;
  icon: string;
  title: string;
  description: string;
  badge: string;
  badgeColor: string;
}) {
  const [hovered, setHovered] = React.useState(false);
  return (
    <button
      onClick={onClick}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        display:    'flex',
        alignItems: 'center',
        gap:        '16px',
        padding:    '14px 16px',
        background: hovered ? 'rgba(0,255,156,0.05)' : '#080C0A',
        border:     `1px solid ${hovered ? 'rgba(0,255,156,0.3)' : 'rgba(0,255,156,0.12)'}`,
        cursor:     'pointer',
        textAlign:  'left',
        transition: 'all 150ms',
      }}
    >
      <span style={{ fontSize: '20px' }}>{icon}</span>
      <div style={{ flex: 1 }}>
        <div style={{
          fontFamily:    'var(--font-orbitron), Orbitron, sans-serif',
          fontSize:      '10px',
          color:         'rgba(200,230,201,0.9)',
          letterSpacing: '0.15em',
          marginBottom:  '3px',
        }}>
          {title}
        </div>
        <div style={{
          fontFamily: 'var(--font-ibm-plex-mono), monospace',
          fontSize:   '9px',
          color:      'rgba(200,230,201,0.35)',
        }}>
          {description}
        </div>
      </div>
      <span style={{
        padding:       '3px 7px',
        background:    `${badgeColor}15`,
        border:        `1px solid ${badgeColor}40`,
        color:         badgeColor,
        fontFamily:    'var(--font-ibm-plex-mono), monospace',
        fontSize:      '8px',
        letterSpacing: '0.1em',
        whiteSpace:    'nowrap',
      }}>
        {badge}
      </span>
    </button>
  );
}

function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <div style={{
      fontFamily:    'var(--font-orbitron), Orbitron, sans-serif',
      fontSize:      '10px',
      color:         'rgba(0,255,156,0.6)',
      letterSpacing: '0.25em',
      marginBottom:  '8px',
    }}>
      {children}
    </div>
  );
}

function ErrorBanner({ children }: { children: React.ReactNode }) {
  return (
    <div style={{
      marginTop:  '8px',
      padding:    '10px 12px',
      background: 'rgba(255,59,59,0.08)',
      border:     '1px solid rgba(255,59,59,0.3)',
      fontFamily: 'var(--font-ibm-plex-mono), monospace',
      fontSize:   '10px',
      color:      '#FF3B3B',
    }}>
      ✕ {typeof children === 'string' ? children.toUpperCase() : children}
    </div>
  );
}

function PrimaryButton({ children, onClick, type = 'button', disabled = false }: {
  children: React.ReactNode;
  onClick?: () => void;
  type?: 'button' | 'submit';
  disabled?: boolean;
}) {
  return (
    <button
      type={type}
      onClick={onClick}
      disabled={disabled}
      style={{
        flex:          1,
        padding:       '11px 16px',
        background:    disabled ? 'rgba(0,255,156,0.1)' : '#00FF9C',
        border:        'none',
        color:         disabled ? 'rgba(0,255,156,0.35)' : '#080C0A',
        fontFamily:    'var(--font-orbitron), Orbitron, sans-serif',
        fontSize:      '10px',
        fontWeight:    700,
        letterSpacing: '0.2em',
        cursor:        disabled ? 'not-allowed' : 'pointer',
        transition:    'all 150ms',
      }}
    >
      {children}
    </button>
  );
}

function GhostButton({ children, onClick, disabled = false }: {
  children: React.ReactNode;
  onClick?: () => void;
  disabled?: boolean;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      style={{
        padding:       '11px 16px',
        background:    'transparent',
        border:        '1px solid rgba(0,255,156,0.2)',
        color:         'rgba(200,230,201,0.4)',
        fontFamily:    'var(--font-ibm-plex-mono), monospace',
        fontSize:      '10px',
        letterSpacing: '0.15em',
        cursor:        disabled ? 'not-allowed' : 'pointer',
        transition:    'all 150ms',
      }}
    >
      {children}
    </button>
  );
}

const helpStyle: React.CSSProperties = {
  fontFamily:    'var(--font-ibm-plex-mono), monospace',
  fontSize:      '9px',
  color:         'rgba(200,230,201,0.4)',
  letterSpacing: '0.1em',
  lineHeight:    1.6,
  margin:        0,
};

// ── Binary helpers (same as mfa/page.tsx) ─────────────────────────────────────
function base64urlToBuffer(b64: string): ArrayBuffer {
  const padded = b64.replace(/-/g, '+').replace(/_/g, '/').padEnd(
    b64.length + (4 - b64.length % 4) % 4, '=',
  );
  const binary = atob(padded);
  const buf    = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) buf[i] = binary.charCodeAt(i);
  return buf.buffer;
}

function bufferToBase64url(buf: ArrayBuffer): string {
  const bytes = new Uint8Array(buf);
  let str = '';
  bytes.forEach((b) => { str += String.fromCharCode(b); });
  return btoa(str).replace(/\+/g, '-').replace(/\//g, '_').replace(/=/g, '');
}
