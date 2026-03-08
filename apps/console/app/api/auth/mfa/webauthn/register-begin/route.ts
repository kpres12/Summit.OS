/**
 * POST /api/auth/mfa/webauthn/register-begin
 *
 * Begins WebAuthn credential registration. Returns PublicKeyCredentialCreationOptions.
 * The browser passes these to navigator.credentials.create().
 */

import { type NextRequest, NextResponse } from 'next/server';
import { AUTH_CONFIG, COOKIE_NAMES } from '@/lib/auth-server';

export async function POST(request: NextRequest): Promise<NextResponse> {
  const accessToken = request.cookies.get(COOKIE_NAMES.ACCESS_TOKEN)?.value;
  if (!accessToken) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
  }

  const body = await request.json().catch(() => ({}));

  const res = await fetch(`${AUTH_CONFIG.apiUrl}/auth/mfa/webauthn/register/begin`, {
    method:  'POST',
    headers: {
      Authorization:  `Bearer ${accessToken}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(5000),
  });

  const data = await res.json();
  return NextResponse.json(data, { status: res.status });
}
