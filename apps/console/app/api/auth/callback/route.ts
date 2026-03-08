/**
 * GET /api/auth/callback
 *
 * Handles the OIDC authorization code callback entirely server-side.
 * Validates state, exchanges code for tokens, checks MFA status,
 * sets httpOnly session cookies, and redirects to app or /mfa.
 *
 * This route must be set as the OIDC redirect_uri. The browser never
 * sees any token — they go straight into httpOnly cookies.
 */

import { type NextRequest, NextResponse } from 'next/server';
import {
  AUTH_CONFIG,
  COOKIE_NAMES,
  exchangeCode,
  extractUser,
  setSessionCookies,
  setMfaPendingCookies,
  clearPkceCookies,
} from '@/lib/auth-server';

export async function GET(request: NextRequest): Promise<NextResponse> {
  const { searchParams, origin } = request.nextUrl;

  const code      = searchParams.get('code');
  const state     = searchParams.get('state');
  const errorCode = searchParams.get('error');
  const errorDesc = searchParams.get('error_description');

  const loginUrl = `${origin}/login`;

  // ── OIDC provider sent an error ───────────────────────────────────────────
  if (errorCode) {
    const msg = errorDesc ? `${errorCode}: ${errorDesc}` : errorCode;
    return NextResponse.redirect(`${loginUrl}?error=${encodeURIComponent(msg)}`);
  }

  if (!code) {
    return NextResponse.redirect(`${loginUrl}?error=no_code`);
  }

  // ── CSRF: validate state matches what we stored ───────────────────────────
  const storedState    = request.cookies.get(COOKIE_NAMES.OIDC_STATE)?.value;
  const codeVerifier   = request.cookies.get(COOKIE_NAMES.PKCE_VERIFIER)?.value;

  if (!storedState || !codeVerifier) {
    return NextResponse.redirect(`${loginUrl}?error=session_expired`);
  }

  if (!state || !crypto.timingSafeEqual(Buffer.from(state), Buffer.from(storedState))) {
    return NextResponse.redirect(`${loginUrl}?error=invalid_state`);
  }

  // ── Exchange code for tokens (server-to-server, no browser) ──────────────
  let tokens;
  try {
    tokens = await exchangeCode(code, codeVerifier);
  } catch (err) {
    console.error('[auth/callback] token exchange failed:', err);
    return NextResponse.redirect(`${loginUrl}?error=token_exchange_failed`);
  }

  const user = extractUser(tokens.id_token);
  if (!user) {
    return NextResponse.redirect(`${loginUrl}?error=invalid_id_token`);
  }

  // ── Check if this user has MFA enrolled ───────────────────────────────────
  let mfaRequired = false;
  try {
    const mfaRes = await fetch(`${AUTH_CONFIG.apiUrl}/auth/mfa/status`, {
      headers: { Authorization: `Bearer ${tokens.access_token}` },
      signal: AbortSignal.timeout(3000),
    });
    if (mfaRes.ok) {
      const mfaStatus = await mfaRes.json() as { mfa_enabled: boolean; mfa_method: string };
      mfaRequired = mfaStatus.mfa_enabled && mfaStatus.mfa_method !== 'none';
    }
  } catch {
    // MFA service unreachable — log and proceed without MFA gate
    // In hardened deployments, flip this to fail-closed
    console.warn('[auth/callback] MFA status check unreachable — proceeding without MFA gate');
  }

  // ── Route based on MFA requirement ───────────────────────────────────────
  if (mfaRequired) {
    const response = NextResponse.redirect(`${origin}/mfa`);
    setMfaPendingCookies(response, tokens);
    clearPkceCookies(response);
    return response;
  }

  const response = NextResponse.redirect(`${origin}/`);
  setSessionCookies(response, tokens);
  clearPkceCookies(response);
  return response;
}
