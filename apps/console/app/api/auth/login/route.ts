/**
 * GET /api/auth/login
 *
 * Initiates OIDC authorization code + PKCE flow entirely server-side.
 * Generates a fresh code verifier, challenge, state, and nonce.
 * Stores verifier/state/nonce in short-lived httpOnly cookies (5 min),
 * then redirects the browser to the OIDC provider.
 *
 * The browser never sees the code verifier.
 */

import { NextResponse } from 'next/server';
import {
  generateCodeVerifier,
  generateCodeChallenge,
  generateNonce,
  buildAuthorizationUrl,
  COOKIE_NAMES,
  TRANSIENT_COOKIE_OPTS,
} from '@/lib/auth-server';

export async function GET(): Promise<NextResponse> {
  const codeVerifier = generateCodeVerifier();
  const codeChallenge = await generateCodeChallenge(codeVerifier);
  const state = generateNonce();
  const nonce = generateNonce();

  const authUrl = buildAuthorizationUrl({ state, codeChallenge, nonce });

  const response = NextResponse.redirect(authUrl);

  response.cookies.set(COOKIE_NAMES.PKCE_VERIFIER, codeVerifier, TRANSIENT_COOKIE_OPTS);
  response.cookies.set(COOKIE_NAMES.OIDC_STATE,    state,        TRANSIENT_COOKIE_OPTS);
  response.cookies.set(COOKIE_NAMES.OIDC_NONCE,    nonce,        TRANSIENT_COOKIE_OPTS);

  return response;
}
