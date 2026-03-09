'use client';

/**
 * MFA Verification Page
 *
 * Handles the MFA gate step after OIDC authentication.
 * Supports TOTP (Authy / Google Authenticator) and WebAuthn (YubiKey / Passkey).
 *
 * Flow: OIDC callback → sets mfa_pending cookies → redirects here
 *       On success: pending cookies → full session cookies → redirect to /
 */

import React, { useState, useEffect, useRef, useCallback } from 'react';
import { useRouter } from 'next/navigation';
import { useAuth } from '@/components/AuthProvider';

type Method = 'totp' | 'webauthn';

export default function MFAPage() {
  const { isLoading, isAuthenticated, mfaPending, refreshAuth } = useAuth();
  const router = useRouter();

  const [method, setMethod]     = useState<Method>('totp');
  const [code, setCode]         = useState('');
  const [error, setError]       = useState<string | null>(null);
  const [busy, setBusy]         = useState(false);
  const [webauthnBusy, setWebauthnBusy] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  // Redirect if already fully authenticated or not in MFA gate
  useEffect(() => {
    if (isLoading) return;
    if (isAuthenticated) { router.replace('/'); return; }
    if (!mfaPending)     { router.replace('/login'); return; }
    inputRef.current?.focus();
  }, [isLoading, isAuthenticated, mfaPending, router]);

  // ── TOTP submission ────────────────────────────────────────────────────────
  const submitTOTP = useCallback(async (e: React.FormEvent) => {
    e.preventDefault();
    if (busy || code.length !== 6) return;
    setBusy(true);
    setError(null);
    try {
      const res  = await fetch('/api/auth/mfa/totp/verify', {
        method:      'POST',
        credentials: 'same-origin',
        headers:     { 'Content-Type': 'application/json' },
        body:        JSON.stringify({ code, action: 'login' }),
      });
      const data = await res.json() as { ok?: boolean; error?: string };
      if (!res.ok) {
        setError(data.error ?? 'Invalid code');
        setCode('');
        inputRef.current?.focus();
        return;
      }
      await refreshAuth();
      router.replace('/');
    } catch {
      setError('Network error — try again');
    } finally {
      setBusy(false);
    }
  }, [busy, code, refreshAuth, router]);

  // ── WebAuthn authentication ────────────────────────────────────────────────
  const startWebAuthn = useCallback(async () => {
    if (webauthnBusy) return;
    setWebauthnBusy(true);
    setError(null);
    try {
      // 1. Get challenge from server
      const beginRes  = await fetch('/api/auth/mfa/webauthn/auth-begin', {
        method:      'POST',
        credentials: 'same-origin',
      });
      if (!beginRes.ok) {
        setError('Could not start hardware key authentication');
        return;
      }
      const options = await beginRes.json();

      // 2. Decode base64url values required by the browser API
      const publicKey: PublicKeyCredentialRequestOptions = {
        ...options,
        challenge:        base64urlToBuffer(options.challenge),
        allowCredentials: (options.allowCredentials ?? []).map((c: { id: string; type: string }) => ({
          ...c,
          id: base64urlToBuffer(c.id),
        })),
      };

      // 3. Prompt user gesture (touch key / biometric)
      const credential = await navigator.credentials.get({ publicKey }) as PublicKeyCredential;
      if (!credential) {
        setError('Authentication cancelled');
        return;
      }

      const assertionResponse = credential.response as AuthenticatorAssertionResponse;

      // 4. Send assertion to server
      const completeRes = await fetch('/api/auth/mfa/webauthn/auth-complete', {
        method:      'POST',
        credentials: 'same-origin',
        headers:     { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          id:   credential.id,
          type: credential.type,
          rawId: bufferToBase64url(credential.rawId),
          response: {
            authenticatorData: bufferToBase64url(assertionResponse.authenticatorData),
            clientDataJSON:    bufferToBase64url(assertionResponse.clientDataJSON),
            signature:         bufferToBase64url(assertionResponse.signature),
            userHandle:        assertionResponse.userHandle
              ? bufferToBase64url(assertionResponse.userHandle) : null,
          },
        }),
      });
      const data = await completeRes.json() as { ok?: boolean; error?: string };
      if (!completeRes.ok) {
        setError(data.error ?? 'Hardware key verification failed');
        return;
      }
      await refreshAuth();
      router.replace('/');
    } catch (err) {
      if (err instanceof DOMException && err.name === 'NotAllowedError') {
        setError('Authentication timed out or was cancelled');
      } else {
        setError('Hardware key error — try TOTP instead');
      }
    } finally {
      setWebauthnBusy(false);
    }
  }, [webauthnBusy, refreshAuth, router]);

  // Auto-submit when 6 digits entered
  useEffect(() => {
    if (code.length === 6 && !busy) {
      const syntheticEvent = { preventDefault: () => {} } as React.FormEvent;
      submitTOTP(syntheticEvent);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [code]);

  if (isLoading) return null;

  return (
    <div
      className="fixed inset-0 flex flex-col items-center justify-center"
      style={{ background: '#080C0A' }}
    >
      {/* Wordmark */}
      <div className="mb-12 text-center">
        <div style={{
          fontFamily:    'var(--font-orbitron), Orbitron, sans-serif',
          fontSize:      '28px',
          fontWeight:    900,
          color:         '#00FF9C',
          letterSpacing: '0.3em',
          textShadow:    '0 0 20px rgba(0,255,156,0.4), 0 0 40px rgba(0,255,156,0.15)',
        }}>
          SUMMIT.OS
        </div>
        <div style={{
          marginTop:     '6px',
          fontSize:      '10px',
          letterSpacing: '0.35em',
          color:         'rgba(200,230,201,0.35)',
          fontFamily:    'var(--font-ibm-plex-mono), monospace',
        }}>
          MULTI-FACTOR AUTHENTICATION
        </div>
      </div>

      {/* MFA card */}
      <div style={{
        width:      '380px',
        background: '#0D1210',
        border:     '1px solid rgba(0,255,156,0.2)',
        boxShadow:  '0 0 40px rgba(0,255,156,0.07)',
      }}>
        {/* Card header */}
        <div style={{
          padding:      '16px 24px',
          borderBottom: '1px solid rgba(0,255,156,0.12)',
          display:      'flex',
          alignItems:   'center',
          gap:          '12px',
        }}>
          <div style={{
            width: '6px', height: '6px', borderRadius: '50%',
            background: '#FFB300',
            boxShadow:  '0 0 4px #FFB300',
            flexShrink: 0,
          }} />
          <div style={{
            fontFamily:    'var(--font-orbitron), Orbitron, sans-serif',
            fontSize:      '10px',
            color:         'rgba(0,255,156,0.5)',
            letterSpacing: '0.3em',
          }}>
            SECOND FACTOR REQUIRED
          </div>
        </div>

        {/* Method switcher */}
        <div style={{
          display:      'grid',
          gridTemplateColumns: '1fr 1fr',
          borderBottom: '1px solid rgba(0,255,156,0.08)',
        }}>
          {([['totp', 'AUTHENTICATOR'], ['webauthn', 'HARDWARE KEY']] as const).map(([m, label]) => (
            <button
              key={m}
              onClick={() => { setMethod(m); setError(null); setCode(''); }}
              style={{
                padding:       '12px',
                background:    method === m ? 'rgba(0,255,156,0.06)' : 'transparent',
                border:        'none',
                borderBottom:  method === m ? '2px solid #00FF9C' : '2px solid transparent',
                color:         method === m ? '#00FF9C' : 'rgba(200,230,201,0.35)',
                fontFamily:    'var(--font-ibm-plex-mono), monospace',
                fontSize:      '9px',
                letterSpacing: '0.2em',
                cursor:        'pointer',
                transition:    'all 150ms',
              }}
            >
              {label}
            </button>
          ))}
        </div>

        {/* Error banner */}
        {error && (
          <div style={{
            margin:     '16px 24px 0',
            padding:    '10px 12px',
            background: 'rgba(255,59,59,0.08)',
            border:     '1px solid rgba(255,59,59,0.3)',
            fontFamily: 'var(--font-ibm-plex-mono), monospace',
            fontSize:   '10px',
            color:      '#FF3B3B',
          }}>
            ✕ {error.toUpperCase()}
          </div>
        )}

        {/* TOTP panel */}
        {method === 'totp' && (
          <form onSubmit={submitTOTP} style={{ padding: '24px' }}>
            <div style={{
              marginBottom:  '8px',
              fontFamily:    'var(--font-ibm-plex-mono), monospace',
              fontSize:      '9px',
              color:         'rgba(200,230,201,0.4)',
              letterSpacing: '0.2em',
            }}>
              6-DIGIT CODE FROM AUTHY OR AUTHENTICATOR APP
            </div>

            {/* Code input — big OTP-style */}
            <input
              ref={inputRef}
              type="text"
              inputMode="numeric"
              pattern="[0-9]*"
              maxLength={6}
              value={code}
              onChange={(e) => setCode(e.target.value.replace(/\D/g, '').slice(0, 6))}
              placeholder="000000"
              autoComplete="one-time-code"
              disabled={busy}
              style={{
                width:         '100%',
                padding:       '16px',
                background:    '#080C0A',
                border:        '1px solid rgba(0,255,156,0.25)',
                color:         '#00FF9C',
                fontFamily:    'var(--font-ibm-plex-mono), monospace',
                fontSize:      '28px',
                letterSpacing: '0.5em',
                textAlign:     'center',
                outline:       'none',
                boxSizing:     'border-box',
                caretColor:    '#00FF9C',
                opacity:       busy ? 0.5 : 1,
              }}
            />

            <button
              type="submit"
              disabled={busy || code.length !== 6}
              style={{
                marginTop:     '16px',
                width:         '100%',
                padding:       '14px',
                background:    busy || code.length !== 6 ? 'rgba(0,255,156,0.15)' : '#00FF9C',
                border:        'none',
                color:         busy || code.length !== 6 ? 'rgba(0,255,156,0.4)' : '#080C0A',
                fontFamily:    'var(--font-orbitron), Orbitron, sans-serif',
                fontSize:      '12px',
                fontWeight:    700,
                letterSpacing: '0.25em',
                cursor:        busy || code.length !== 6 ? 'not-allowed' : 'pointer',
                transition:    'all 150ms',
              }}
            >
              {busy ? 'VERIFYING...' : '► VERIFY CODE'}
            </button>

            <div style={{
              marginTop:  '12px',
              textAlign:  'center',
              fontFamily: 'var(--font-ibm-plex-mono), monospace',
              fontSize:   '9px',
              color:      'rgba(200,230,201,0.2)',
            }}>
              CODE ROTATES EVERY 30 SECONDS
            </div>
          </form>
        )}

        {/* WebAuthn panel */}
        {method === 'webauthn' && (
          <div style={{ padding: '24px' }}>
            <div style={{
              marginBottom:  '20px',
              padding:       '16px',
              background:    'rgba(0,255,156,0.03)',
              border:        '1px solid rgba(0,255,156,0.1)',
              textAlign:     'center',
            }}>
              <div style={{
                fontSize:   '32px',
                marginBottom: '8px',
                opacity:    webauthnBusy ? 1 : 0.5,
                filter:     webauthnBusy ? 'drop-shadow(0 0 8px rgba(0,255,156,0.6))' : 'none',
                transition: 'all 400ms',
              }}>
                🔑
              </div>
              <div style={{
                fontFamily:    'var(--font-ibm-plex-mono), monospace',
                fontSize:      '9px',
                color:         webauthnBusy ? '#00FF9C' : 'rgba(200,230,201,0.35)',
                letterSpacing: '0.15em',
                lineHeight:    1.6,
              }}>
                {webauthnBusy
                  ? 'TOUCH YOUR SECURITY KEY OR USE BIOMETRIC...'
                  : 'INSERT YUBIKEY OR USE TOUCH ID / FACE ID'}
              </div>
            </div>

            <button
              onClick={startWebAuthn}
              disabled={webauthnBusy}
              style={{
                width:         '100%',
                padding:       '14px',
                background:    webauthnBusy ? 'rgba(0,255,156,0.15)' : '#00FF9C',
                border:        'none',
                color:         webauthnBusy ? 'rgba(0,255,156,0.4)' : '#080C0A',
                fontFamily:    'var(--font-orbitron), Orbitron, sans-serif',
                fontSize:      '12px',
                fontWeight:    700,
                letterSpacing: '0.25em',
                cursor:        webauthnBusy ? 'not-allowed' : 'pointer',
                transition:    'all 150ms',
              }}
            >
              {webauthnBusy ? 'AWAITING GESTURE...' : '► USE HARDWARE KEY'}
            </button>

            {!webauthnBusy && (
              <div style={{
                marginTop:  '12px',
                textAlign:  'center',
                fontFamily: 'var(--font-ibm-plex-mono), monospace',
                fontSize:   '9px',
                color:      'rgba(200,230,201,0.2)',
              }}>
                FIDO2 · WEBAUTHN LEVEL 2
              </div>
            )}
          </div>
        )}

        {/* Footer: sign in with different account */}
        <div style={{
          padding:      '12px 24px',
          borderTop:    '1px solid rgba(0,255,156,0.08)',
          textAlign:    'center',
        }}>
          <button
            onClick={() => { window.location.href = '/api/auth/logout'; }}
            style={{
              background:    'transparent',
              border:        'none',
              fontFamily:    'var(--font-ibm-plex-mono), monospace',
              fontSize:      '9px',
              color:         'rgba(200,230,201,0.25)',
              cursor:        'pointer',
              letterSpacing: '0.1em',
              textDecoration: 'underline',
            }}
          >
            cancel and sign out
          </button>
        </div>
      </div>

      <div style={{
        marginTop:     '24px',
        fontFamily:    'var(--font-ibm-plex-mono), monospace',
        fontSize:      '9px',
        color:         'rgba(200,230,201,0.2)',
        letterSpacing: '0.15em',
      }}>
        SUMMIT.OS v0.1.0
      </div>
    </div>
  );
}

// ── WebAuthn binary helpers ──────────────────────────────────────────────────
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
