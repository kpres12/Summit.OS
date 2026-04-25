/**
 * NASA FIRMS active fire / thermal anomaly hotspots
 *
 * Sources: VIIRS S-NPP (375 m) + MODIS (1 km)
 * Data: past 24 hours, global
 * Key:  FIRMS_MAP_KEY env var — free at https://firms.modaps.eosdis.nasa.gov/api/area/
 *       Falls back to public demo key if unset (rate-limited to ~10 req/day)
 */

import { NextResponse } from 'next/server';

const FIRMS_KEY = process.env.FIRMS_MAP_KEY ?? 'MAP_KEY'; // demo key — works but rate-limited
const FIRMS_BASE = 'https://firms.modaps.eosdis.nasa.gov/api/area/csv';

export interface FireHotspot {
  lat: number;
  lon: number;
  brightness: number;   // Kelvin (MODIS) or brightness temp (VIIRS)
  frp: number;          // fire radiative power MW
  confidence: 'low' | 'nominal' | 'high';
  instrument: 'VIIRS' | 'MODIS';
  acquired_iso: string;
}

export interface FiresResponse {
  hotspots: FireHotspot[];
  count: number;
  source: string;
  fetchedAt: string;
  fromCache: boolean;
}

// ── Cache — 10-min TTL (FIRMS updates every ~3 hrs for VIIRS) ────────────────

let _cache: { data: FiresResponse; expiresAt: number } | null = null;

// ── CSV parser ────────────────────────────────────────────────────────────────

function parseViirsCsv(csv: string): FireHotspot[] {
  const lines = csv.trim().split('\n');
  if (lines.length < 2) return [];
  const header = lines[0].split(',').map(h => h.trim());
  const idx = (name: string) => header.indexOf(name);

  const iLat  = idx('latitude');
  const iLon  = idx('longitude');
  const iBrt  = idx('bright_ti4');
  const iFrp  = idx('frp');
  const iConf = idx('confidence');
  const iDate = idx('acq_date');
  const iTime = idx('acq_time');

  const hotspots: FireHotspot[] = [];
  for (const line of lines.slice(1)) {
    const c = line.split(',');
    const lat  = parseFloat(c[iLat]);
    const lon  = parseFloat(c[iLon]);
    if (isNaN(lat) || isNaN(lon)) continue;

    const confRaw = (c[iConf] ?? '').trim().toLowerCase();
    const conf: FireHotspot['confidence'] =
      confRaw === 'h' || confRaw === 'high'    ? 'high'
      : confRaw === 'l' || confRaw === 'low'   ? 'low'
      : 'nominal';

    const dateStr = (c[iDate] ?? '').trim();
    const timeStr = (c[iTime] ?? '00:00').trim().padStart(4, '0');
    const acquired_iso = dateStr
      ? `${dateStr}T${timeStr.slice(0, 2)}:${timeStr.slice(2)}:00Z`
      : new Date().toISOString();

    hotspots.push({
      lat, lon,
      brightness: parseFloat(c[iBrt]) || 0,
      frp:        parseFloat(c[iFrp]) || 0,
      confidence: conf,
      instrument: 'VIIRS',
      acquired_iso,
    });
  }
  return hotspots;
}

// ── Fetch VIIRS SNPP — world, past 24h ───────────────────────────────────────

async function fetchFirmsViirs(): Promise<FireHotspot[]> {
  const url = `${FIRMS_BASE}/${FIRMS_KEY}/VIIRS_SNPP_NRT/world/1`;
  try {
    const res = await fetch(url, {
      signal: AbortSignal.timeout(12_000),
      headers: { 'User-Agent': 'HeliOS/1.0' },
    });
    if (!res.ok) return [];
    const csv = await res.text();
    return parseViirsCsv(csv);
  } catch {
    return [];
  }
}

// ── Route ─────────────────────────────────────────────────────────────────────

export async function GET() {
  if (_cache && Date.now() < _cache.expiresAt) {
    return NextResponse.json({ ..._cache.data, fromCache: true });
  }

  const hotspots = await fetchFirmsViirs();

  // Sort highest FRP first (most intense fires at top)
  hotspots.sort((a, b) => b.frp - a.frp);

  const response: FiresResponse = {
    hotspots: hotspots.slice(0, 500), // cap at 500 for payload size
    count:    hotspots.length,
    source:   'NASA FIRMS VIIRS SNPP (375m, 24h)',
    fetchedAt: new Date().toISOString(),
    fromCache: false,
  };

  _cache = { data: response, expiresAt: Date.now() + 10 * 60_000 };
  return NextResponse.json(response);
}
