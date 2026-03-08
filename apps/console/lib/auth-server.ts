/**
 * Summit.OS — Server-side auth utilities (Node.js only, never imported by client components)
 *
 * Handles: PKCE generation, OIDC token exchange, JWT decoding, cookie config.
 * Tokens are NEVER exposed to client-side JavaScript. All token handling goes
 * through Next.js API routes that set/read httpOnly cookies.
 */

import crypto from 'crypto';

// ─── Config ──────────────────────────────────────────────────────────────────

export const AUTH_CONFIG = {
  issuer:       process.env.OIDC_ISSUER       ?? process.env.NEXT_PUBLIC_OIDC_ISSUER       ?? 'https://auth.summit-os.local',
  clientId:     process.env.OIDC_CLIENT_ID    ?? process.env.NEXT_PUBLIC_OIDC_CLIENT_ID    ?? 'summit-console',
  clientSecret: process.env.OIDC_CLIENT_SECRET,
  // Redirect URI must point to the API route, not the page
  redirectUri:  process.env.OIDC_REDIRECT_URI ??
    `${process.env.NEXT_PUBLIC_APP_URL ?? 'http://localhost:3000'}/api/auth/callback`,
  scope:        'openid profile email',
  apiUrl:       process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000',
};

// ─── Cookie names & options ───────────────────────────────────────────────────

export const COOKIE_NAMES = {
  ACCESS_TOKEN:  'summit_at',
  ID_TOKEN:      'summit_it',
  REFRESH_TOKEN: 'summit_rt',
  // MFA gate — tokens held here until second factor verified
  MFA_PENDING_AT: 'summit_mfa_at',
  MFA_PENDING_IT: 'summit_mfa_it',
  MFA_PENDING_RT: 'summit_mfa_rt',
  // Short-lived PKCE / state for the OIDC redirect
  PKCE_VERIFIER: 'summit_pkce_v',
  OIDC_STATE:    'summit_oidc_s',
  OIDC_NONCE:    'summit_oidc_n',
};

const IS_PROD = process.env.NODE_ENV === 'production';

// Used for PKCE / state cookies (short-lived, survive the redirect)
export const TRANSIENT_COOKIE_OPTS = {
  httpOnly: true,
  secure:   IS_PROD,
  sameSite: 'lax' as const, // 'lax' required for cross-site redirects
  path:     '/',
  maxAge:   300, // 5 minutes
};

// Used for session cookies
export const SESSION_COOKIE_OPTS = {
  httpOnly: true,
  secure:   IS_PROD,
  sameSite: 'strict' as const,
  path:     '/',
};

// ─── PKCE helpers ─────────────────────────────────────────────────────────────

export function generateCodeVerifier(): string {
  // 96 random bytes → 128 base64url characters
  return crypto.randomBytes(96).toString('base64url');
}

export async function generateCodeChallenge(verifier: string): Promise<string> {
  const hash = crypto.createHash('sha256').update(verifier).digest();
  return hash.toString('base64url');
}

export function generateNonce(): string {
  return crypto.randomBytes(32).toString('base64url');
}

// ─── Authorization URL ────────────────────────────────────────────────────────

export function buildAuthorizationUrl(params: {
  state:         string;
  codeChallenge: string;
  nonce:         string;
}): string {
  const url = new URL(`${AUTH_CONFIG.issuer}/protocol/openid-connect/auth`);
  url.searchParams.set('response_type',          'code');
  url.searchParams.set('client_id',              AUTH_CONFIG.clientId);
  url.searchParams.set('redirect_uri',           AUTH_CONFIG.redirectUri);
  url.searchParams.set('scope',                  AUTH_CONFIG.scope);
  url.searchParams.set('state',                  params.state);
  url.searchParams.set('nonce',                  params.nonce);
  url.searchParams.set('code_challenge',         params.codeChallenge);
  url.searchParams.set('code_challenge_method',  'S256');
  return url.toString();
}

// ─── Token exchange ───────────────────────────────────────────────────────────

export interface TokenSet {
  access_token:   string;
  id_token:       string;
  refresh_token?: string;
  expires_in:     number;
}

export async function exchangeCode(code: string, codeVerifier: string): Promise<TokenSet> {
  const body = new URLSearchParams({
    grant_type:    'authorization_code',
    code,
    redirect_uri:  AUTH_CONFIG.redirectUri,
    client_id:     AUTH_CONFIG.clientId,
    code_verifier: codeVerifier,
    ...(AUTH_CONFIG.clientSecret && { client_secret: AUTH_CONFIG.clientSecret }),
  });

  const res = await fetch(`${AUTH_CONFIG.issuer}/protocol/openid-connect/token`, {
    method:  'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body,
  });

  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    throw new Error(`Token exchange failed (${res.status}): ${text}`);
  }

  return res.json() as Promise<TokenSet>;
}

export async function refreshAccessToken(refreshToken: string): Promise<TokenSet> {
  const body = new URLSearchParams({
    grant_type:    'refresh_token',
    refresh_token: refreshToken,
    client_id:     AUTH_CONFIG.clientId,
    ...(AUTH_CONFIG.clientSecret && { client_secret: AUTH_CONFIG.clientSecret }),
  });

  const res = await fetch(`${AUTH_CONFIG.issuer}/protocol/openid-connect/token`, {
    method:  'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body,
  });

  if (!res.ok) throw new Error(`Token refresh failed (${res.status})`);
  return res.json() as Promise<TokenSet>;
}

// ─── JWT utilities (no signature verification — that's the backend's job) ─────

export function decodeJwtPayload(token: string): Record<string, unknown> | null {
  try {
    const [, payload] = token.split('.');
    return JSON.parse(Buffer.from(payload, 'base64url').toString('utf-8'));
  } catch {
    return null;
  }
}

export function isTokenExpired(token: string): boolean {
  const payload = decodeJwtPayload(token);
  if (!payload?.exp) return true;
  // 30-second clock skew buffer
  return Date.now() / 1000 > (payload.exp as number) - 30;
}

export interface SessionUser {
  id:     string;
  email:  string;
  name:   string;
  org_id: string;
  roles:  string[];
}

export function extractUser(idToken: string): SessionUser | null {
  const p = decodeJwtPayload(idToken);
  if (!p?.sub) return null;
  return {
    id:     String(p.sub),
    email:  String(p.email ?? ''),
    name:   String(p.name ?? p.preferred_username ?? ''),
    org_id: String(p.org_id ?? 'default'),
    roles:  ((p.roles ?? (p.realm_access as Record<string, unknown>)?.roles ?? []) as string[]),
  };
}

// ─── Cookie helpers ───────────────────────────────────────────────────────────

import type { NextResponse } from 'next/server';

export function setSessionCookies(response: NextResponse, tokens: TokenSet): void {
  const opts = SESSION_COOKIE_OPTS;
  response.cookies.set(COOKIE_NAMES.ACCESS_TOKEN,  tokens.access_token, { ...opts, maxAge: tokens.expires_in });
  response.cookies.set(COOKIE_NAMES.ID_TOKEN,      tokens.id_token,     { ...opts, maxAge: tokens.expires_in });
  if (tokens.refresh_token) {
    response.cookies.set(COOKIE_NAMES.REFRESH_TOKEN, tokens.refresh_token, { ...opts, maxAge: 60 * 60 * 24 * 7 });
  }
}

export function setMfaPendingCookies(response: NextResponse, tokens: TokenSet): void {
  const opts = { ...TRANSIENT_COOKIE_OPTS, maxAge: 600, sameSite: 'lax' as const }; // 10 min to complete MFA
  response.cookies.set(COOKIE_NAMES.MFA_PENDING_AT, tokens.access_token, opts);
  response.cookies.set(COOKIE_NAMES.MFA_PENDING_IT, tokens.id_token, opts);
  if (tokens.refresh_token) {
    response.cookies.set(COOKIE_NAMES.MFA_PENDING_RT, tokens.refresh_token, opts);
  }
}

export function clearAllAuthCookies(response: NextResponse): void {
  for (const name of Object.values(COOKIE_NAMES)) {
    response.cookies.delete(name);
  }
}

export function clearPkceCookies(response: NextResponse): void {
  response.cookies.delete(COOKIE_NAMES.PKCE_VERIFIER);
  response.cookies.delete(COOKIE_NAMES.OIDC_STATE);
  response.cookies.delete(COOKIE_NAMES.OIDC_NONCE);
}
