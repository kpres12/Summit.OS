'use client';

import { useEffect } from 'react';

export default function GlobalError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    console.error('[Summit.OS] Global error:', error);
  }, [error]);

  return (
    <html lang="en">
      <body style={{ margin: 0 }}>
        <div
          style={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            height: '100vh',
            background: '#080C0A',
            gap: '16px',
            fontFamily: 'monospace',
          }}
        >
          <span
            style={{
              color: '#FF3B3B',
              fontSize: '11px',
              letterSpacing: '0.2em',
              fontWeight: 700,
              textTransform: 'uppercase',
            }}
          >
            Critical System Error
          </span>
          <span style={{ color: 'rgba(255,59,59,0.7)', fontSize: '10px', maxWidth: '420px', textAlign: 'center' }}>
            {error.message || 'Summit.OS encountered an unrecoverable error.'}
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
              fontFamily: 'monospace',
              fontSize: '10px',
              letterSpacing: '0.15em',
              cursor: 'pointer',
            }}
          >
            RELOAD
          </button>
        </div>
      </body>
    </html>
  );
}
