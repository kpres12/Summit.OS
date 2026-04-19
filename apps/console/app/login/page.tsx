'use client';

import { useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { useAuth } from '@/components/AuthProvider';

export default function LoginPage() {
  const { isAuthenticated, isLoading, login } = useAuth();
  const router = useRouter();

  // Already logged in — bounce to app
  useEffect(() => {
    if (!isLoading && isAuthenticated) router.replace('/');
  }, [isAuthenticated, isLoading, router]);

  // Read error from query string (set by /api/auth/callback on failure)
  const errorMsg = typeof window !== 'undefined'
    ? new URLSearchParams(window.location.search).get('error')
    : null;

  if (isLoading) return null;

  return (
    <div
      className="fixed inset-0 flex flex-col items-center justify-center"
      style={{ background: '#080C0A' }}
    >
      {/* Wordmark */}
      <div className="mb-12 text-center">
        <div
          style={{
            fontFamily:  'var(--font-orbitron), Orbitron, sans-serif',
            fontSize:    '28px',
            fontWeight:  900,
            color:       '#00FF9C',
            letterSpacing: '0.3em',
            textShadow:  '0 0 20px rgba(0,255,156,0.4), 0 0 40px rgba(0,255,156,0.15)',
          }}
        >
          HELI.OS
        </div>
        <div
          style={{
            marginTop:   '6px',
            fontSize:    '10px',
            letterSpacing: '0.35em',
            color:       'rgba(200,230,201,0.35)',
            fontFamily:  'var(--font-ibm-plex-mono), monospace',
          }}
        >
          AUTONOMOUS SYSTEMS COORDINATION PLATFORM
        </div>
      </div>

      {/* Login card */}
      <div
        style={{
          width:      '380px',
          background: '#0D1210',
          border:     '1px solid rgba(0,255,156,0.2)',
          boxShadow:  '0 0 40px rgba(0,255,156,0.07)',
        }}
      >
        {/* Card header */}
        <div
          style={{
            padding:      '16px 24px',
            borderBottom: '1px solid rgba(0,255,156,0.12)',
          }}
        >
          <div style={{ fontFamily: 'var(--font-orbitron), Orbitron, sans-serif', fontSize: '10px', color: 'rgba(0,255,156,0.5)', letterSpacing: '0.3em' }}>
            SYSTEM ACCESS
          </div>
        </div>

        {/* Status indicators */}
        <div style={{ padding: '20px 24px', borderBottom: '1px solid rgba(0,255,156,0.08)' }}>
          {[
            { label: 'FABRIC',     ok: true  },
            { label: 'OIDC READY', ok: true  },
            { label: 'MFA',        ok: true  },
            { label: 'CREDENTIALS', ok: false, text: 'AWAITING' },
          ].map((item) => (
            <div key={item.label} className="flex items-center gap-3 mb-2 last:mb-0">
              <div style={{
                width: '6px', height: '6px', borderRadius: '50%',
                background: item.ok ? '#00FF9C' : '#FFB300',
                boxShadow:  item.ok ? '0 0 4px #00FF9C' : '0 0 4px #FFB300',
                flexShrink: 0,
              }} />
              <span style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', fontSize: '10px', color: 'rgba(200,230,201,0.5)' }}>
                {item.label}
              </span>
              <span style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', fontSize: '10px', color: item.ok ? '#00FF9C' : '#FFB300', marginLeft: 'auto' }}>
                {item.text ?? 'NOMINAL'}
              </span>
            </div>
          ))}
        </div>

        {/* Error */}
        {errorMsg && (
          <div style={{
            margin: '16px 24px 0',
            padding: '10px 12px',
            background: 'rgba(255,59,59,0.08)',
            border: '1px solid rgba(255,59,59,0.3)',
            fontFamily: 'var(--font-ibm-plex-mono), monospace',
            fontSize: '10px',
            color: '#FF3B3B',
          }}>
            AUTH ERROR: {decodeURIComponent(errorMsg).toUpperCase()}
          </div>
        )}

        {/* Authenticate button */}
        <div style={{ padding: '24px' }}>
          <button
            onClick={login}
            style={{
              width:         '100%',
              padding:       '14px',
              background:    '#00FF9C',
              border:        'none',
              color:         '#080C0A',
              fontFamily:    'var(--font-orbitron), Orbitron, sans-serif',
              fontSize:      '12px',
              fontWeight:    700,
              letterSpacing: '0.25em',
              cursor:        'pointer',
              transition:    'background 150ms',
            }}
            onMouseEnter={(e) => ((e.currentTarget as HTMLButtonElement).style.background = '#00CC74')}
            onMouseLeave={(e) => ((e.currentTarget as HTMLButtonElement).style.background = '#00FF9C')}
          >
            ► AUTHENTICATE
          </button>

          <div style={{
            marginTop:  '16px',
            textAlign:  'center',
            fontFamily: 'var(--font-ibm-plex-mono), monospace',
            fontSize:   '9px',
            color:      'rgba(200,230,201,0.25)',
            lineHeight: 1.6,
          }}>
            OIDC · PKCE · MFA ENFORCED
          </div>
        </div>
      </div>

      <div style={{
        marginTop:   '24px',
        fontFamily:  'var(--font-ibm-plex-mono), monospace',
        fontSize:    '9px',
        color:       'rgba(200,230,201,0.2)',
        letterSpacing: '0.15em',
      }}>
        HELI.OS v0.1.0
      </div>
    </div>
  );
}
