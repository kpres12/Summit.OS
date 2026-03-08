/**
 * POST /api/auth/mfa/webauthn/auth-complete
 *
 * Completes WebAuthn authentication. On success, promotes the pending
 * session to a full session (same as TOTP login verification).
 */

import { type NextRequest, NextResponse } from 'next/server';
import {
  AUTH_CONFIG,
  COOKIE_NAMES,
  SESSION_COOKIE_OPTS,
  extractUser,
} from '@/lib/auth-server';

export async function POST(request: NextRequest): Promise<NextResponse> {
  const accessToken = request.cookies.get(COOKIE_NAMES.MFA_PENDING_AT)?.value;
  if (!accessToken) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
  }

  const credential = await request.json();

  const backendRes = await fetch(`${AUTH_CONFIG.apiUrl}/auth/mfa/webauthn/authenticate/complete`, {
    method:  'POST',
    headers: {
      Authorization:  `Bearer ${accessToken}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(credential),
    signal: AbortSignal.timeout(10000),
  });

  const data = await backendRes.json();
  if (!backendRes.ok) {
    return NextResponse.json(data, { status: backendRes.status });
  }

  const idToken      = request.cookies.get(COOKIE_NAMES.MFA_PENDING_IT)?.value;
  const refreshToken = request.cookies.get(COOKIE_NAMES.MFA_PENDING_RT)?.value;

  if (!idToken) {
    return NextResponse.json({ error: 'MFA session expired' }, { status: 401 });
  }

  const user = extractUser(idToken);
  if (!user) {
    return NextResponse.json({ error: 'Invalid session' }, { status: 401 });
  }

  const response = NextResponse.json({ ok: true, user });

  response.cookies.set(COOKIE_NAMES.ACCESS_TOKEN,  accessToken,  { ...SESSION_COOKIE_OPTS, maxAge: 3600 });
  response.cookies.set(COOKIE_NAMES.ID_TOKEN,       idToken,      { ...SESSION_COOKIE_OPTS, maxAge: 3600 });
  if (refreshToken) {
    response.cookies.set(COOKIE_NAMES.REFRESH_TOKEN, refreshToken, { ...SESSION_COOKIE_OPTS, maxAge: 604800 });
  }

  response.cookies.delete(COOKIE_NAMES.MFA_PENDING_AT);
  response.cookies.delete(COOKIE_NAMES.MFA_PENDING_IT);
  response.cookies.delete(COOKIE_NAMES.MFA_PENDING_RT);

  return response;
}
