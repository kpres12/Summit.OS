/**
 * POST /api/auth/mfa/webauthn/auth-begin
 *
 * Begins WebAuthn authentication challenge during the MFA gate step.
 * Uses the mfa_pending access token (not a full session token).
 */

import { type NextRequest, NextResponse } from 'next/server';
import { AUTH_CONFIG, COOKIE_NAMES } from '@/lib/auth-server';

export async function POST(request: NextRequest): Promise<NextResponse> {
  // During login, we use the pending token; during re-auth, the active token
  const accessToken =
    request.cookies.get(COOKIE_NAMES.MFA_PENDING_AT)?.value ??
    request.cookies.get(COOKIE_NAMES.ACCESS_TOKEN)?.value;

  if (!accessToken) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
  }

  const res = await fetch(`${AUTH_CONFIG.apiUrl}/auth/mfa/webauthn/authenticate/begin`, {
    method:  'POST',
    headers: {
      Authorization:  `Bearer ${accessToken}`,
      'Content-Type': 'application/json',
    },
    signal: AbortSignal.timeout(5000),
  });

  const data = await res.json();
  return NextResponse.json(data, { status: res.status });
}
