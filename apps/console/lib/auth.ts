// OIDC Authentication for Heli.OS Console

export interface User {
  id: string;
  email: string;
  name: string;
  org_id: string;
  roles: string[];
}

export interface AuthConfig {
  issuer: string;
  clientId: string;
  clientSecret?: string;
  redirectUri: string;
  scope: string;
}

// Default OIDC configuration from environment
export const authConfig: AuthConfig = {
  issuer: process.env.NEXT_PUBLIC_OIDC_ISSUER || 'https://auth.heli-os.local',
  clientId: process.env.NEXT_PUBLIC_OIDC_CLIENT_ID || 'summit-console',
  clientSecret: process.env.OIDC_CLIENT_SECRET,
  redirectUri: process.env.NEXT_PUBLIC_OIDC_REDIRECT_URI || 'http://localhost:3000/auth/callback',
  scope: 'openid profile email org_id roles',
};

// Generate PKCE code challenge
export async function generatePKCE() {
  const codeVerifier = generateRandomString(128);
  const codeChallenge = await generateCodeChallenge(codeVerifier);
  
  return {
    codeVerifier,
    codeChallenge,
    codeChallengeMethod: 'S256',
  };
}

function generateRandomString(length: number): string {
  const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~';
  let result = '';
  const randomValues = new Uint8Array(length);
  crypto.getRandomValues(randomValues);
  
  for (let i = 0; i < length; i++) {
    result += chars[randomValues[i] % chars.length];
  }
  
  return result;
}

async function generateCodeChallenge(codeVerifier: string): Promise<string> {
  const encoder = new TextEncoder();
  const data = encoder.encode(codeVerifier);
  const hash = await crypto.subtle.digest('SHA-256', data);
  
  return base64URLEncode(hash);
}

function base64URLEncode(buffer: ArrayBuffer): string {
  const bytes = new Uint8Array(buffer);
  let binary = '';
  for (let i = 0; i < bytes.length; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  
  return btoa(binary)
    .replace(/\+/g, '-')
    .replace(/\//g, '_')
    .replace(/=/g, '');
}

// Build authorization URL
export function buildAuthorizationUrl(config: AuthConfig, pkce: { codeChallenge: string; codeChallengeMethod: string }): string {
  const state = generateRandomString(32);
  const nonce = generateRandomString(32);
  
  // Store state and nonce in sessionStorage for validation
  if (typeof window !== 'undefined') {
    sessionStorage.setItem('oidc_state', state);
    sessionStorage.setItem('oidc_nonce', nonce);
  }
  
  const params = new URLSearchParams({
    response_type: 'code',
    client_id: config.clientId,
    redirect_uri: config.redirectUri,
    scope: config.scope,
    state,
    nonce,
    code_challenge: pkce.codeChallenge,
    code_challenge_method: pkce.codeChallengeMethod,
  });
  
  return `${config.issuer}/protocol/openid-connect/auth?${params.toString()}`;
}

// Exchange authorization code for tokens
export async function exchangeCodeForTokens(
  code: string,
  codeVerifier: string,
  config: AuthConfig
): Promise<{
  access_token: string;
  id_token: string;
  refresh_token?: string;
  expires_in: number;
}> {
  const response = await fetch(`${config.issuer}/protocol/openid-connect/token`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/x-www-form-urlencoded',
    },
    body: new URLSearchParams({
      grant_type: 'authorization_code',
      code,
      redirect_uri: config.redirectUri,
      client_id: config.clientId,
      code_verifier: codeVerifier,
      ...(config.clientSecret && { client_secret: config.clientSecret }),
    }),
  });
  
  if (!response.ok) {
    throw new Error(`Token exchange failed: ${response.statusText}`);
  }
  
  return response.json();
}

// Parse JWT token (without verification - for client-side only)
export function parseJwt(token: string): Record<string, unknown> | null {
  try {
    const base64Url = token.split('.')[1];
    const base64 = base64Url.replace(/-/g, '+').replace(/_/g, '/');
    const jsonPayload = decodeURIComponent(
      atob(base64)
        .split('')
        .map((c) => '%' + ('00' + c.charCodeAt(0).toString(16)).slice(-2))
        .join('')
    );
    
    return JSON.parse(jsonPayload);
  } catch (error) {
    console.error('Failed to parse JWT:', error);
    return null;
  }
}

// Extract user from ID token
export function extractUser(idToken: string): User | null {
  const payload = parseJwt(idToken);
  if (!payload) return null;
  
  return {
    id: String(payload.sub ?? ''),
    email: String(payload.email ?? ''),
    name: String(payload.name ?? payload.preferred_username ?? ''),
    org_id: String(payload.org_id ?? 'default'),
    roles: (payload.roles ?? (payload.realm_access as Record<string, unknown>)?.roles ?? []) as string[],
  };
}

// Logout URL
export function buildLogoutUrl(config: AuthConfig, idToken?: string): string {
  const params = new URLSearchParams({
    client_id: config.clientId,
    post_logout_redirect_uri: window.location.origin,
    ...(idToken && { id_token_hint: idToken }),
  });
  
  return `${config.issuer}/protocol/openid-connect/logout?${params.toString()}`;
}
