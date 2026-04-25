/**
 * Satellite imagery search — free sources + optional paid
 *
 * FREE (no key, no account):
 *   Sentinel-2 L2A  — 10 m optical,  5-day revisit,   global
 *   Sentinel-1 GRD  — 10 m SAR,      6–12 day revisit, all-weather/night
 *   Landsat 8/9 C2  — 30 m optical,  16-day revisit,  global archive
 *   NAIP            — 0.6–1 m aerial, annual,          US only
 *   All via Element84 Earth Search STAC (earth-search.aws.element84.com)
 *
 * PAID (requires UP42_API_KEY + UP42_PROJECT_ID):
 *   BlackSky Gen-2/3 — 35–83 cm, on-demand tasking, <1-hr response
 */

import { NextRequest, NextResponse } from 'next/server';

const UP42_API_KEY    = process.env.UP42_API_KEY;
const UP42_PROJECT_ID = process.env.UP42_PROJECT_ID;
const STAC_URL        = 'https://earth-search.aws.element84.com/v1/search';

// ── Types ─────────────────────────────────────────────────────────────────────

export type ImageryProvider = 'sentinel-2' | 'sentinel-1' | 'landsat' | 'naip' | 'blacksky';

export interface ImageryScene {
  id: string;
  provider: ImageryProvider;
  acquired_iso: string;
  cloud_pct: number;
  resolution_m: number;
  bbox: [number, number, number, number]; // [w, s, e, n]
  thumbnail_url: string | null;
  preview_url: string | null;
  status: 'archive' | 'tasked' | 'delivered';
  /** SAR polarisation band (Sentinel-1 only) */
  polarisation?: string;
}

export interface ImageryResponse {
  scenes: ImageryScene[];
  providers: Record<ImageryProvider, boolean>;
  fetchedAt: string;
}

// ── Cache — keyed by bbox + days, 10-min TTL ─────────────────────────────────

const _cache = new Map<string, { data: ImageryResponse; expiresAt: number }>();

// ── Element84 STAC helper ─────────────────────────────────────────────────────

interface StacItem {
  id: string;
  bbox: [number, number, number, number];
  properties: Record<string, unknown>;
  assets: Record<string, { href: string } | undefined>;
}

async function stacSearch(
  collections: string[],
  bbox: [number, number, number, number],
  days: number,
  extra: Record<string, unknown> = {},
): Promise<StacItem[]> {
  const end   = new Date();
  const start = new Date(end.getTime() - days * 86_400_000);
  const body  = {
    collections,
    bbox,
    datetime: `${start.toISOString().slice(0, 10)}T00:00:00Z/${end.toISOString().slice(0, 10)}T23:59:59Z`,
    limit: 10,
    sortby: [{ field: 'datetime', direction: 'desc' }],
    ...extra,
  };
  try {
    const res = await fetch(STAC_URL, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(body),
      signal:  AbortSignal.timeout(8_000),
    });
    if (!res.ok) return [];
    const json = await res.json() as { features?: StacItem[] };
    return json.features ?? [];
  } catch {
    return [];
  }
}

// ── Per-provider parsers ──────────────────────────────────────────────────────

async function searchSentinel2(bbox: [number, number, number, number], days: number): Promise<ImageryScene[]> {
  const items = await stacSearch(
    ['sentinel-2-l2a'], bbox, days,
    { query: { 'eo:cloud_cover': { lt: 40 } } },
  );
  return items.map(f => ({
    id:            f.id,
    provider:      'sentinel-2',
    acquired_iso:  String(f.properties.datetime ?? ''),
    cloud_pct:     Math.round(Number(f.properties['eo:cloud_cover'] ?? 0)),
    resolution_m:  10,
    bbox:          f.bbox,
    thumbnail_url: f.assets.thumbnail?.href ?? f.assets.overview?.href ?? null,
    preview_url:   f.assets.visual?.href ?? null,
    status:        'archive',
  }));
}

async function searchSentinel1(bbox: [number, number, number, number], days: number): Promise<ImageryScene[]> {
  const items = await stacSearch(['sentinel-1-grd'], bbox, days);
  return items.map(f => ({
    id:            f.id,
    provider:      'sentinel-1',
    acquired_iso:  String(f.properties.datetime ?? ''),
    cloud_pct:     0, // SAR is all-weather
    resolution_m:  10,
    bbox:          f.bbox,
    thumbnail_url: f.assets.thumbnail?.href ?? null,
    preview_url:   f.assets.vv?.href ?? f.assets.vh?.href ?? null,
    status:        'archive',
    polarisation:  String(f.properties['sar:polarizations'] ?? 'VV+VH'),
  }));
}

async function searchLandsat(bbox: [number, number, number, number], days: number): Promise<ImageryScene[]> {
  const items = await stacSearch(
    ['landsat-c2-l2'], bbox, days,
    { query: { 'eo:cloud_cover': { lt: 50 } } },
  );
  return items.map(f => ({
    id:            f.id,
    provider:      'landsat',
    acquired_iso:  String(f.properties.datetime ?? ''),
    cloud_pct:     Math.round(Number(f.properties['eo:cloud_cover'] ?? 0)),
    resolution_m:  30,
    bbox:          f.bbox,
    thumbnail_url: f.assets.thumbnail?.href ?? null,
    preview_url:   f.assets.red?.href ?? null,
    status:        'archive',
  }));
}

async function searchNaip(bbox: [number, number, number, number], days: number): Promise<ImageryScene[]> {
  // NAIP is annual — use a longer window
  const items = await stacSearch(['naip'], bbox, Math.max(days, 400));
  return items.map(f => ({
    id:            f.id,
    provider:      'naip',
    acquired_iso:  String(f.properties.datetime ?? ''),
    cloud_pct:     0,
    resolution_m:  0.6,
    bbox:          f.bbox,
    thumbnail_url: f.assets.thumbnail?.href ?? null,
    preview_url:   f.assets.image?.href ?? null,
    status:        'archive',
  }));
}

// ── BlackSky via UP42 ─────────────────────────────────────────────────────────

async function getUp42Token(): Promise<string | null> {
  if (!UP42_API_KEY || !UP42_PROJECT_ID) return null;
  try {
    const res = await fetch('https://api.up42.com/oauth/token', {
      method:  'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body:    new URLSearchParams({
        grant_type: 'client_credentials',
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

async function searchBlackSky(bbox: [number, number, number, number]): Promise<ImageryScene[]> {
  const token = await getUp42Token();
  if (!token) return [];
  try {
    const res = await fetch('https://api.up42.com/v2/assets/stac/search', {
      method:  'POST',
      headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' },
      body:    JSON.stringify({ collections: ['blacksky-mono'], bbox, limit: 8 }),
      signal:  AbortSignal.timeout(8_000),
    });
    if (!res.ok) return [];
    const json = await res.json() as {
      features: { id: string; bbox: [number,number,number,number];
        properties: { acquisitionDate: string; resolution?: number; cloudCoverage?: number };
        assets?: { thumbnail?: { href: string } } }[]
    };
    return (json.features ?? []).map(f => ({
      id:            f.id,
      provider:      'blacksky',
      acquired_iso:  f.properties.acquisitionDate,
      cloud_pct:     Math.round(f.properties.cloudCoverage ?? 0),
      resolution_m:  f.properties.resolution ?? 0.83,
      bbox:          f.bbox,
      thumbnail_url: f.assets?.thumbnail?.href ?? null,
      preview_url:   null,
      status:        'archive',
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
  const hit = _cache.get(cacheKey);
  if (hit && Date.now() < hit.expiresAt) return NextResponse.json(hit.data);

  const [s2, s1, ls, naip, bs] = await Promise.allSettled([
    searchSentinel2(bbox, days),
    searchSentinel1(bbox, days),
    searchLandsat(bbox, days),
    searchNaip(bbox, days),
    searchBlackSky(bbox),
  ]);

  const scenes: ImageryScene[] = [
    ...(s2.status   === 'fulfilled' ? s2.value   : []),
    ...(s1.status   === 'fulfilled' ? s1.value   : []),
    ...(ls.status   === 'fulfilled' ? ls.value   : []),
    ...(naip.status === 'fulfilled' ? naip.value : []),
    ...(bs.status   === 'fulfilled' ? bs.value   : []),
  ].sort((a, b) => new Date(b.acquired_iso).getTime() - new Date(a.acquired_iso).getTime());

  const ok = (r: PromiseSettledResult<ImageryScene[]>) =>
    r.status === 'fulfilled' && r.value.length > 0;

  const response: ImageryResponse = {
    scenes,
    providers: {
      'sentinel-2': ok(s2),
      'sentinel-1': ok(s1),
      'landsat':    ok(ls),
      'naip':       ok(naip),
      'blacksky':   !!UP42_API_KEY,
    },
    fetchedAt: new Date().toISOString(),
  };

  _cache.set(cacheKey, { data: response, expiresAt: Date.now() + 10 * 60_000 });
  return NextResponse.json(response);
}
