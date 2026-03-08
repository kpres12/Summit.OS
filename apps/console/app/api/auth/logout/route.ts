/**
 * POST /api/auth/logout
 *
 * Clears all session cookies and returns the OIDC provider logout URL.
 * The client is responsible for redirecting to logoutUrl to complete
 * the single-logout (SLO) flow with the OIDC provider.
 */

import { type NextRequest, NextResponse } from 'next/server';
import {
  AUTH_CONFIG,
  COOKIE_NAMES,
  clearAllAuthCookies,
} from '@/lib/auth-server';

export async function POST(request: NextRequest): Promise<NextResponse> {
  const idToken = request.cookies.get(COOKIE_NAMES.ID_TOKEN)?.value;

  // Build OIDC logout URL so the provider also clears its session
  const logoutParams = new URLSearchParams({
    client_id: AUTH_CONFIG.clientId,
    post_logout_redirect_uri: process.env.NEXT_PUBLIC_APP_URL ?? 'http://localhost:3000',
    ...(idToken && { id_token_hint: idToken }),
  });
  const logoutUrl = `${AUTH_CONFIG.issuer}/protocol/openid-connect/logout?${logoutParams}`;

  const response = NextResponse.json({ ok: true, logoutUrl });
  clearAllAuthCookies(response);
  return response;
}
