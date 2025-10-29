'use client';

import { useEffect, useState } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import { authConfig, exchangeCodeForTokens, extractUser } from '../../../lib/auth';

export default function AuthCallbackPage() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const handleCallback = async () => {
      try {
        const code = searchParams.get('code');
        const state = searchParams.get('state');
        const errorParam = searchParams.get('error');

        if (errorParam) {
          setError(`Authentication error: ${errorParam}`);
          return;
        }

        if (!code) {
          setError('No authorization code received');
          return;
        }

        // Validate state
        const storedState = sessionStorage.getItem('oidc_state');
        if (state !== storedState) {
          setError('Invalid state parameter');
          return;
        }

        // Get code verifier
        const codeVerifier = sessionStorage.getItem('oidc_code_verifier');
        if (!codeVerifier) {
          setError('Missing code verifier');
          return;
        }

        // Exchange code for tokens
        const tokens = await exchangeCodeForTokens(code, codeVerifier, authConfig);

        // Store tokens
        localStorage.setItem('access_token', tokens.access_token);
        localStorage.setItem('id_token', tokens.id_token);
        if (tokens.refresh_token) {
          localStorage.setItem('refresh_token', tokens.refresh_token);
        }

        // Extract user info
        const user = extractUser(tokens.id_token);
        if (!user) {
          setError('Failed to extract user information from token');
          return;
        }

        // Clean up session storage
        sessionStorage.removeItem('oidc_state');
        sessionStorage.removeItem('oidc_nonce');
        sessionStorage.removeItem('oidc_code_verifier');

        // Redirect to home
        router.push('/');
      } catch (err) {
        console.error('Auth callback error:', err);
        setError(err instanceof Error ? err.message : 'Authentication failed');
      }
    };

    handleCallback();
  }, [searchParams, router]);

  if (error) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-[#0A0A0A]">
        <div className="text-center space-y-4">
          <div className="text-[#FF3333] text-xl font-mono">Authentication Failed</div>
          <div className="text-[#00FF91] text-sm font-mono">{error}</div>
          <button
            onClick={() => router.push('/')}
            className="mt-4 px-6 py-2 bg-[#00FF91] text-[#0A0A0A] font-mono hover:bg-[#00CC74] transition-colors"
          >
            Return Home
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-[#0A0A0A]">
      <div className="text-center space-y-4">
        <div className="text-[#00FF91] text-xl font-mono animate-pulse">
          Authenticating...
        </div>
        <div className="w-16 h-16 border-4 border-[#00FF91] border-t-transparent rounded-full animate-spin mx-auto" />
      </div>
    </div>
  );
}
