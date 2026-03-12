/**
 * GET /api/auth/me
 *
 * Returns the current user's identity from the httpOnly id_token cookie.
 * This is the ONLY way client-side code learns who is logged in.
 * The actual token never leaves the server.
 *
 * Also handles silent token refresh: if the access_token is expired but
 * a refresh_token is present, a new access_token is silently obtained.
 */

import { type NextRequest, NextResponse } from 'next/server';
import {
  COOKIE_NAMES,
  SESSION_COOKIE_OPTS,
  extractUser,
  isTokenExpired,
  refreshAccessToken,
} from '@/lib/auth-server';

export async function GET(request: NextRequest): Promise<NextResponse> {
  // Dev bypass — skip auth when NEXT_PUBLIC_DEV_BYPASS_AUTH=true
  if (process.env.NEXT_PUBLIC_DEV_BYPASS_AUTH === 'true') {
    return NextResponse.json({
      user: {
        id:     'dev-user',
        email:  'dev@summit.local',
        name:   'Dev Operator',
        org_id: 'dev',
        roles:  ['ADMIN'],
      },
      mfaPending: false,
    });
  }

  const idToken      = request.cookies.get(COOKIE_NAMES.ID_TOKEN)?.value;
  const accessToken  = request.cookies.get(COOKIE_NAMES.ACCESS_TOKEN)?.value;
  const refreshToken = request.cookies.get(COOKIE_NAMES.REFRESH_TOKEN)?.value;
  const mfaPending   = !!request.cookies.get(COOKIE_NAMES.MFA_PENDING_AT)?.value;

  // No session at all
  if (!idToken) {
    return NextResponse.json({ user: null }, { status: 401 });
  }

  // Access token expired — try silent refresh
  if (accessToken && isTokenExpired(accessToken) && refreshToken) {
    try {
      const refreshed = await refreshAccessToken(refreshToken);
      const user = extractUser(refreshed.id_token);
      if (!user) {
        return NextResponse.json({ user: null }, { status: 401 });
      }

      const response = NextResponse.json({ user, mfaPending: false });
      response.cookies.set(COOKIE_NAMES.ACCESS_TOKEN, refreshed.access_token, {
        ...SESSION_COOKIE_OPTS,
        maxAge: refreshed.expires_in,
      });
      response.cookies.set(COOKIE_NAMES.ID_TOKEN, refreshed.id_token, {
        ...SESSION_COOKIE_OPTS,
        maxAge: refreshed.expires_in,
      });
      if (refreshed.refresh_token) {
        response.cookies.set(COOKIE_NAMES.REFRESH_TOKEN, refreshed.refresh_token, {
          ...SESSION_COOKIE_OPTS,
          maxAge: 60 * 60 * 24 * 7,
        });
      }
      return response;
    } catch {
      // Refresh failed — clear session
      const response = NextResponse.json({ user: null, reason: 'session_expired' }, { status: 401 });
      for (const name of Object.values(COOKIE_NAMES)) response.cookies.delete(name);
      return response;
    }
  }

  // ID token expired
  if (isTokenExpired(idToken)) {
    return NextResponse.json({ user: null, reason: 'session_expired' }, { status: 401 });
  }

  const user = extractUser(idToken);
  if (!user) {
    return NextResponse.json({ user: null }, { status: 401 });
  }

  return NextResponse.json({ user, mfaPending });
}
