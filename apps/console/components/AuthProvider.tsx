'use client';

import React, { createContext, useContext, useState, useEffect } from 'react';
import {
  User,
  authConfig,
  generatePKCE,
  buildAuthorizationUrl,
  extractUser,
  buildLogoutUrl,
} from '../lib/auth';

interface AuthContextType {
  user: User | null;
  isLoading: boolean;
  isAuthenticated: boolean;
  login: () => Promise<void>;
  logout: () => void;
  getAccessToken: () => string | null;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    // Check for existing session
    const idToken = localStorage.getItem('id_token');
    if (idToken) {
      const userData = extractUser(idToken);
      setUser(userData);
    }
    setIsLoading(false);
  }, []);

  const login = async () => {
    try {
      const pkce = await generatePKCE();
      
      // Store code verifier for callback
      sessionStorage.setItem('oidc_code_verifier', pkce.codeVerifier);
      
      // Build authorization URL and redirect
      const authUrl = buildAuthorizationUrl(authConfig, {
        codeChallenge: pkce.codeChallenge,
        codeChallengeMethod: pkce.codeChallengeMethod,
      });
      
      window.location.href = authUrl;
    } catch (error) {
      console.error('Login failed:', error);
    }
  };

  const logout = () => {
    const idToken = localStorage.getItem('id_token');
    
    // Clear local storage
    localStorage.removeItem('access_token');
    localStorage.removeItem('id_token');
    localStorage.removeItem('refresh_token');
    sessionStorage.clear();
    
    setUser(null);
    
    // Redirect to OIDC logout
    const logoutUrl = buildLogoutUrl(authConfig, idToken || undefined);
    window.location.href = logoutUrl;
  };

  const getAccessToken = () => {
    return localStorage.getItem('access_token');
  };

  return (
    <AuthContext.Provider
      value={{
        user,
        isLoading,
        isAuthenticated: !!user,
        login,
        logout,
        getAccessToken,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}
