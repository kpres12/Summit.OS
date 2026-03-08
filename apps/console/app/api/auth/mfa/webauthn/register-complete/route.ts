/**
 * POST /api/auth/mfa/webauthn/register-complete
 *
 * Completes WebAuthn registration. Accepts the PublicKeyCredential from
 * navigator.credentials.create() and forwards it to the backend for
 * cryptographic verification and credential storage.
 */

import { type NextRequest, NextResponse } from 'next/server';
import { AUTH_CONFIG, COOKIE_NAMES } from '@/lib/auth-server';

export async function POST(request: NextRequest): Promise<NextResponse> {
  const accessToken = request.cookies.get(COOKIE_NAMES.ACCESS_TOKEN)?.value;
  if (!accessToken) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
  }

  const credential = await request.json();

  const res = await fetch(`${AUTH_CONFIG.apiUrl}/auth/mfa/webauthn/register/complete`, {
    method:  'POST',
    headers: {
      Authorization:  `Bearer ${accessToken}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(credential),
    signal: AbortSignal.timeout(10000),
  });

  const data = await res.json();
  return NextResponse.json(data, { status: res.status });
}
