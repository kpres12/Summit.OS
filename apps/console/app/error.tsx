'use client';

import { useEffect } from 'react';

export default function Error({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    console.error('[Summit.OS] Route error:', error);
  }, [error]);

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        height: '100vh',
        background: '#080C0A',
        gap: '16px',
        fontFamily: 'var(--font-ibm-plex-mono), monospace',
      }}
    >
      <span
        style={{
          fontFamily: 'var(--font-orbitron), Orbitron, sans-serif',
          color: '#FF3B3B',
          fontSize: '11px',
          letterSpacing: '0.2em',
          fontWeight: 700,
        }}
      >
        SYSTEM ERROR
      </span>
      <span style={{ color: 'rgba(255,59,59,0.7)', fontSize: '10px', maxWidth: '420px', textAlign: 'center' }}>
        {error.message || 'An unexpected error occurred.'}
      </span>
      {error.digest && (
        <span style={{ color: 'rgba(255,255,255,0.25)', fontSize: '9px' }}>
          REF: {error.digest}
        </span>
      )}
      <button
        onClick={reset}
        style={{
          marginTop: '8px',
          padding: '8px 24px',
          background: 'transparent',
          border: '1px solid rgba(255,59,59,0.4)',
          color: '#FF3B3B',
          fontFamily: 'var(--font-ibm-plex-mono), monospace',
          fontSize: '10px',
          letterSpacing: '0.15em',
          cursor: 'pointer',
        }}
        onMouseEnter={(e) => { (e.currentTarget.style.background = 'rgba(255,59,59,0.08)'); }}
        onMouseLeave={(e) => { (e.currentTarget.style.background = 'transparent'); }}
      >
        RETRY
      </button>
    </div>
  );
}
