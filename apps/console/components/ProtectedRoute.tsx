'use client';

/**
 * ProtectedRoute — auth is ALWAYS enforced, no env-var opt-out.
 *
 * Routing logic:
 *   loading           → show authenticating screen (no flash)
 *   mfaPending        → redirect to /mfa
 *   not authenticated → redirect to /login
 *   authenticated     → render children
 */

import { useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { useAuth } from './AuthProvider';

export default function ProtectedRoute({ children }: { children: React.ReactNode }) {
  const { isAuthenticated, isLoading, mfaPending } = useAuth();
  const router = useRouter();

  useEffect(() => {
    if (isLoading) return;
    if (mfaPending) { router.replace('/mfa');   return; }
    if (!isAuthenticated) { router.replace('/login'); }
  }, [isAuthenticated, isLoading, mfaPending, router]);

  if (isLoading) {
    return (
      <div
        className="fixed inset-0 flex flex-col items-center justify-center gap-4"
        style={{ background: '#080C0A' }}
      >
        <div
          style={{
            fontFamily:    'var(--font-orbitron), Orbitron, sans-serif',
            color:         '#00FF9C',
            fontSize:      '11px',
            letterSpacing: '0.4em',
            textShadow:    '0 0 8px rgba(0,255,156,0.5)',
          }}
        >
          AUTHENTICATING
        </div>
        {/* Animated progress bar */}
        <div style={{ width: '120px', height: '1px', background: 'rgba(0,255,156,0.15)', position: 'relative', overflow: 'hidden' }}>
          <div
            style={{
              position:        'absolute',
              inset:           0,
              background:      'linear-gradient(90deg, transparent 0%, #00FF9C 50%, transparent 100%)',
              animation:       'auth-scan 1.4s ease-in-out infinite',
              backgroundSize:  '200% 100%',
            }}
          />
        </div>
        <style>{`
          @keyframes auth-scan {
            0%   { background-position: -100% 0; }
            100% { background-position:  200% 0; }
          }
        `}</style>
      </div>
    );
  }

  if (!isAuthenticated || mfaPending) return null;

  return <>{children}</>;
}
