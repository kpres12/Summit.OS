/**
 * POST /api/auth/mfa/totp/enroll
 *
 * Begins TOTP enrollment. Returns a QR code data URI and the raw secret
 * (for manual entry). The secret is held server-side by the MFA service
 * and only activated after the user verifies the first code.
 */

import { type NextRequest, NextResponse } from 'next/server';
import { AUTH_CONFIG, COOKIE_NAMES } from '@/lib/auth-server';

export async function POST(request: NextRequest): Promise<NextResponse> {
  const accessToken = request.cookies.get(COOKIE_NAMES.ACCESS_TOKEN)?.value;
  if (!accessToken) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
  }

  const res = await fetch(`${AUTH_CONFIG.apiUrl}/auth/mfa/totp/enroll/begin`, {
    method: 'POST',
    headers: { Authorization: `Bearer ${accessToken}` },
    signal: AbortSignal.timeout(5000),
  });

  const data = await res.json();
  return NextResponse.json(data, { status: res.status });
}
