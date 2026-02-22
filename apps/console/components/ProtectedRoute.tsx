'use client';

import { useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { useAuth } from './AuthProvider';

/**
 * Opt-in route guard.
 *
 * When NEXT_PUBLIC_AUTH_ENABLED is "true", unauthenticated users are
 * redirected to /login.  When the flag is absent or "false" the children
 * render unconditionally (local-dev mode).
 */
const AUTH_ENABLED = process.env.NEXT_PUBLIC_AUTH_ENABLED === 'true';

export default function ProtectedRoute({ children }: { children: React.ReactNode }) {
  const { isAuthenticated, isLoading } = useAuth();
  const router = useRouter();

  useEffect(() => {
    if (AUTH_ENABLED && !isLoading && !isAuthenticated) {
      router.replace('/login');
    }
  }, [isAuthenticated, isLoading, router]);

  if (AUTH_ENABLED && isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-[#0A0A0A]">
        <div className="text-[#00FF91] text-xl font-mono animate-pulse">
          AUTHENTICATING...
        </div>
      </div>
    );
  }

  if (AUTH_ENABLED && !isAuthenticated) {
    return null; // will redirect
  }

  return <>{children}</>;
}
