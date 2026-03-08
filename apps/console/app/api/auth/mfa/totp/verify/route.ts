/**
 * POST /api/auth/mfa/totp/verify
 *
 * Two purposes, selected by the `action` field:
 *
 *   action: "enroll"  — Verify the first TOTP code after scanning the QR.
 *                       Activates TOTP and returns one-time backup codes.
 *                       Uses the active session access token.
 *
 *   action: "login"   — Verify TOTP during the MFA gate step.
 *                       Uses the mfa_pending access token. On success,
 *                       promotes pending cookies to full session cookies.
 */

import { type NextRequest, NextResponse } from 'next/server';
import {
  AUTH_CONFIG,
  COOKIE_NAMES,
  SESSION_COOKIE_OPTS,
  extractUser,
} from '@/lib/auth-server';

export async function POST(request: NextRequest): Promise<NextResponse> {
  const body = await request.json() as { token?: string; action?: string };
  const { token, action } = body;

  if (!token || typeof token !== 'string') {
    return NextResponse.json({ error: 'Missing token' }, { status: 400 });
  }

  const isLogin = action === 'login';

  const accessToken = isLogin
    ? request.cookies.get(COOKIE_NAMES.MFA_PENDING_AT)?.value
    : request.cookies.get(COOKIE_NAMES.ACCESS_TOKEN)?.value;

  if (!accessToken) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
  }

  const endpoint = isLogin
    ? `${AUTH_CONFIG.apiUrl}/auth/mfa/totp/validate`
    : `${AUTH_CONFIG.apiUrl}/auth/mfa/totp/enroll/verify`;

  const backendRes = await fetch(endpoint, {
    method: 'POST',
    headers: {
      Authorization:  `Bearer ${accessToken}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ token }),
    signal: AbortSignal.timeout(5000),
  });

  const data = await backendRes.json();

  if (!backendRes.ok) {
    return NextResponse.json(data, { status: backendRes.status });
  }

  // Login verification succeeded — promote pending session to full session
  if (isLogin) {
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

    response.cookies.set(COOKIE_NAMES.ACCESS_TOKEN, accessToken, { ...SESSION_COOKIE_OPTS, maxAge: 3600 });
    response.cookies.set(COOKIE_NAMES.ID_TOKEN, idToken, { ...SESSION_COOKIE_OPTS, maxAge: 3600 });
    if (refreshToken) {
      response.cookies.set(COOKIE_NAMES.REFRESH_TOKEN, refreshToken, { ...SESSION_COOKIE_OPTS, maxAge: 604800 });
    }

    // Clear MFA pending cookies
    response.cookies.delete(COOKIE_NAMES.MFA_PENDING_AT);
    response.cookies.delete(COOKIE_NAMES.MFA_PENDING_IT);
    response.cookies.delete(COOKIE_NAMES.MFA_PENDING_RT);

    return response;
  }

  return NextResponse.json(data);
}
