/**
 * POST /api/auth/sessions/revoke-all
 *
 * Revokes all sessions except the current one.
 */

import { type NextRequest, NextResponse } from 'next/server';
import { AUTH_CONFIG, COOKIE_NAMES } from '@/lib/auth-server';

export async function POST(request: NextRequest): Promise<NextResponse> {
  const accessToken = request.cookies.get(COOKIE_NAMES.ACCESS_TOKEN)?.value;
  if (!accessToken) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
  }

  const res = await fetch(`${AUTH_CONFIG.apiUrl}/auth/sessions/revoke-all`, {
    method:  'POST',
    headers: { Authorization: `Bearer ${accessToken}` },
    signal:  AbortSignal.timeout(5000),
  });

  const data = await res.json();
  return NextResponse.json(data, { status: res.status });
}
