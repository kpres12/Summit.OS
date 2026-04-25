/**
 * Satellite imagery search — two sources:
 *
 * FREE  Sentinel-2 L2A via Element84 Earth Search STAC (no key required)
 *       10 m resolution, 5-day revisit, global coverage
 *
 * PAID  BlackSky via UP42 (requires UP42_API_KEY + UP42_PROJECT_ID env vars)
 *       35–83 cm resolution, on-demand tasking, <1-hr response
 */

import { NextRequest, NextResponse } from 'next/server';

const UP42_API_KEY    = process.env.UP42_API_KEY;
const UP42_PROJECT_ID = process.env.UP42_PROJECT_ID;
const STAC_URL        = 'https://earth-search.aws.element84.com/v1/search';

// ── Types ─────────────────────────────────────────────────────────────────────

export interface ImageryScene {
  id: string;
  provider: 'sentinel-2' | 'blacksky';
  acquired_iso: string;
  cloud_pct: number;
  resolution_m: number;
  bbox: [number, number, number, number]; // [w, s, e, n]
  thumbnail_url: string | null;
  preview_url: string | null;
  status: 'archive' | 'tasked' | 'delivered';
}

export interface ImageryResponse {
  scenes: ImageryScene[];
  providers: { sentinel2: boolean; blacksky: boolean };
  fetchedAt: string;
}

// ── Cache — keyed by bbox string, 10 min TTL ──────────────────────────────────

const _cache = new Map<string, { data: ImageryResponse; expiresAt: number }>();

// ── Sentinel-2 via Element84 Earth Search STAC (free) ────────────────────────

interface StacItem {
  id: string;
  bbox: [number, number, number, number];
  properties: { datetime: string; 'eo:cloud_cover': number };
  assets: {
    thumbnail?: { href: string };
    overview?: { href: string };
    visual?:   { href: string };
  };
}

async function searchSentinel2(
  bbox: [number, number, number, number],
  days = 30,
): Promise<ImageryScene[]> {
  const end   = new Date();
  const start = new Date(end.getTime() - days * 86_400_000);
  const body  = {
    collections: ['sentinel-2-l2a'],
    bbox,
    datetime: `${start.toISOString().slice(0, 10)}T00:00:00Z/${end.toISOString().slice(0, 10)}T23:59:59Z`,
    query:    { 'eo:cloud_cover': { lt: 40 } },
    limit:    12,
    sortby:   [{ field: 'datetime', direction: 'desc' }],
  };

  const res = await fetch(STAC_URL, {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify(body),
    signal:  AbortSignal.timeout(8_000),
  });
  if (!res.ok) return [];

  const json = await res.json() as { features: StacItem[] };
  return (json.features ?? []).map(f => ({
    id:             f.id,
    provider:       'sentinel-2',
    acquired_iso:   f.properties.datetime,
    cloud_pct:      Math.round(f.properties['eo:cloud_cover'] ?? 0),
    resolution_m:   10,
    bbox:           f.bbox,
    thumbnail_url:  f.assets.thumbnail?.href ?? f.assets.overview?.href ?? null,
    preview_url:    f.assets.visual?.href ?? null,
    status:         'archive',
  }));
}

// ── BlackSky via UP42 (requires UP42_API_KEY + UP42_PROJECT_ID) ───────────────

async function getUp42Token(): Promise<string | null> {
  if (!UP42_API_KEY || !UP42_PROJECT_ID) return null;
  try {
    const res = await fetch('https://api.up42.com/oauth/token', {
      method:  'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body:    new URLSearchParams({
        grant_type:    'client_credentials',
        client_id:     UP42_PROJECT_ID,
        client_secret: UP42_API_KEY,
      }),
      signal: AbortSignal.timeout(5_000),
    });
    if (!res.ok) return null;
    const data = await res.json() as { access_token?: string };
    return data.access_token ?? null;
  } catch {
    return null;
  }
}

interface Up42Feature {
  id: string;
  properties: {
    acquisitionDate: string;
    resolution?: number;
    cloudCoverage?: number;
  };
  bbox: [number, number, number, number];
  assets?: { thumbnail?: { href: string } };
}

async function searchBlackSky(
  bbox: [number, number, number, number],
): Promise<ImageryScene[]> {
  const token = await getUp42Token();
  if (!token) return [];

  try {
    const res = await fetch(
      'https://api.up42.com/v2/assets/stac/search',
      {
        method:  'POST',
        headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' },
        body:    JSON.stringify({
          collections: ['blacksky-mono'],
          bbox,
          limit: 8,
        }),
        signal: AbortSignal.timeout(8_000),
      },
    );
    if (!res.ok) return [];
    const json = await res.json() as { features: Up42Feature[] };
    return (json.features ?? []).map(f => ({
      id:             f.id,
      provider:       'blacksky',
      acquired_iso:   f.properties.acquisitionDate,
      cloud_pct:      Math.round(f.properties.cloudCoverage ?? 0),
      resolution_m:   f.properties.resolution ?? 0.83,
      bbox:           f.bbox,
      thumbnail_url:  f.assets?.thumbnail?.href ?? null,
      preview_url:    null,
      status:         'archive',
    }));
  } catch {
    return [];
  }
}

// ── Route ─────────────────────────────────────────────────────────────────────

export async function POST(req: NextRequest) {
  let bbox: [number, number, number, number];
  let days: number;

  try {
    const body = await req.json() as { bbox?: unknown; days?: unknown };
    const b = body.bbox as number[];
    if (!Array.isArray(b) || b.length !== 4) throw new Error('bbox required');
    bbox = b as [number, number, number, number];
    days = typeof body.days === 'number' ? Math.min(body.days, 90) : 30;
  } catch {
    return NextResponse.json({ error: 'bbox: [w,s,e,n] required' }, { status: 400 });
  }

  const cacheKey = `${bbox.join(',')}|${days}`;
  const cached = _cache.get(cacheKey);
  if (cached && Date.now() < cached.expiresAt) {
    return NextResponse.json(cached.data);
  }

  const [s2, bs] = await Promise.allSettled([
    searchSentinel2(bbox, days),
    searchBlackSky(bbox),
  ]);

  const scenes: ImageryScene[] = [
    ...(s2.status === 'fulfilled' ? s2.value : []),
    ...(bs.status === 'fulfilled' ? bs.value : []),
  ].sort((a, b) => new Date(b.acquired_iso).getTime() - new Date(a.acquired_iso).getTime());

  const response: ImageryResponse = {
    scenes,
    providers: {
      sentinel2: s2.status === 'fulfilled' && s2.value.length > 0,
      blacksky:  !!UP42_API_KEY,
    },
    fetchedAt: new Date().toISOString(),
  };

  _cache.set(cacheKey, { data: response, expiresAt: Date.now() + 10 * 60_000 });
  return NextResponse.json(response);
}
