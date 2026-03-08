'use client';

/**
 * AuthProvider — tokens live exclusively in httpOnly cookies.
 *
 * Client-side code NEVER sees a JWT. The auth state is bootstrapped by
 * fetching /api/auth/me on mount, which reads the httpOnly id_token
 * server-side and returns only the sanitized user object.
 *
 * Login  → redirects to /api/auth/login  (OIDC + PKCE, server-side)
 * Logout → POST /api/auth/logout (cookie clear) then OIDC SLO redirect
 */

import React, {
  createContext, useContext, useState, useEffect, useCallback,
} from 'react';

export interface User {
  id:     string;
  email:  string;
  name:   string;
  org_id: string;
  roles:  string[];
}

interface AuthContextType {
  user:            User | null;
  isLoading:       boolean;
  isAuthenticated: boolean;
  mfaPending:      boolean;
  login:           () => void;
  logout:          () => Promise<void>;
  refreshAuth:     () => Promise<void>;
  // Legacy compat — no longer returns a real token (tokens are httpOnly)
  getAccessToken:  () => null;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user,      setUser]      = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [mfaPending, setMfaPending] = useState(false);

  const refreshAuth = useCallback(async () => {
    try {
      const res = await fetch('/api/auth/me', { credentials: 'same-origin' });
      if (res.ok) {
        const data = await res.json() as { user: User | null; mfaPending?: boolean };
        setUser(data.user);
        setMfaPending(!!data.mfaPending);
      } else {
        setUser(null);
        setMfaPending(false);
      }
    } catch {
      setUser(null);
      setMfaPending(false);
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => { refreshAuth(); }, [refreshAuth]);

  const login = () => {
    // Initiates OIDC + PKCE server-side — browser is redirected to IdP
    window.location.href = '/api/auth/login';
  };

  const logout = async () => {
    try {
      const res  = await fetch('/api/auth/logout', { method: 'POST', credentials: 'same-origin' });
      const data = await res.json() as { ok: boolean; logoutUrl?: string };
      setUser(null);
      setMfaPending(false);
      // Complete OIDC single-logout so the provider also clears its session
      window.location.href = data.logoutUrl ?? '/login';
    } catch {
      window.location.href = '/login';
    }
  };

  return (
    <AuthContext.Provider value={{
      user,
      isLoading,
      isAuthenticated: !!user && !mfaPending,
      mfaPending,
      login,
      logout,
      refreshAuth,
      getAccessToken: () => null,
    }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth(): AuthContextType {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error('useAuth must be used within AuthProvider');
  return ctx;
}
