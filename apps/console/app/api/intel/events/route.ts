import { NextResponse } from 'next/server';

// ── Types ─────────────────────────────────────────────────────────────────────

export interface IntelEvent {
  id: string;
  type: 'earthquake' | 'flood' | 'cyclone' | 'volcano' | 'fire' | 'drought';
  title: string;
  severity: 'GREEN' | 'ORANGE' | 'RED';
  lat: number;
  lon: number;
  country: string;
  ts_iso: string;
  source: string;
  url?: string;
}

export interface IntelEventsResponse {
  events: IntelEvent[];
  fetchedAt: string;
  fromCache: boolean;
}

// ── Cache (5-min TTL — events don't change faster than that) ──────────────────

let _cache: { data: IntelEventsResponse; expiresAt: number } | null = null;
const CACHE_TTL_MS = 5 * 60 * 1000;

// ── USGS: M2.5+ earthquakes past 24h ─────────────────────────────────────────

async function fetchEarthquakes(): Promise<IntelEvent[]> {
  const res = await fetch(
    'https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/2.5_day.geojson',
    { next: { revalidate: 300 } },
  );
  if (!res.ok) return [];
  const json = await res.json() as {
    features: {
      id: string;
      properties: { title: string; mag: number; time: number; url: string; place: string };
      geometry: { coordinates: [number, number, number] };
    }[];
  };

  return json.features.slice(0, 20).map(f => {
    const mag = f.properties.mag;
    return {
      id:       `usgs-${f.id}`,
      type:     'earthquake',
      title:    f.properties.title,
      severity: mag >= 6.0 ? 'RED' : mag >= 4.5 ? 'ORANGE' : 'GREEN',
      lat:      f.geometry.coordinates[1],
      lon:      f.geometry.coordinates[0],
      country:  f.properties.place,
      ts_iso:   new Date(f.properties.time).toISOString(),
      source:   'USGS',
      url:      f.properties.url,
    } satisfies IntelEvent;
  });
}

// ── GDACS: global disaster alerts ─────────────────────────────────────────────

interface GdacsItem {
  title?: string | { _text: string };
  'gdacs:alertlevel'?: string | { _text: string };
  'gdacs:latitude'?: string | { _text: string };
  'gdacs:longitude'?: string | { _text: string };
  'gdacs:country'?: string | { _text: string };
  'gdacs:eventtype'?: string | { _text: string };
  pubDate?: string | { _text: string };
  link?: string | { _text: string };
  guid?: string | { _text: string };
}

function gdacsText(v: string | { _text: string } | undefined): string {
  if (!v) return '';
  return typeof v === 'string' ? v : v._text;
}

async function fetchGdacs(): Promise<IntelEvent[]> {
  try {
    const res = await fetch('https://www.gdacs.org/xml/rss.xml', {
      next: { revalidate: 300 },
      headers: { 'User-Agent': 'HeliOS/1.0' },
    });
    if (!res.ok) return [];
    const text = await res.text();

    // Minimal XML → JSON using regex (no xml2js in edge runtime)
    const items: IntelEvent[] = [];
    const itemBlocks = text.match(/<item>([\s\S]*?)<\/item>/g) ?? [];
    for (const block of itemBlocks.slice(0, 15)) {
      const get = (tag: string) => {
        const m = block.match(new RegExp(`<${tag}[^>]*>([^<]*)<\/${tag}>`));
        return m ? m[1].trim() : '';
      };
      const getCdata = (tag: string) => {
        const m = block.match(new RegExp(`<${tag}[^>]*><!\\[CDATA\\[([^\\]]*)]\\]><\/${tag}>`));
        return m ? m[1].trim() : get(tag);
      };

      const alertLevel = get('gdacs:alertlevel').toUpperCase();
      const eventType  = get('gdacs:eventtype').toLowerCase();
      const lat        = parseFloat(get('gdacs:latitude'))  || 0;
      const lon        = parseFloat(get('gdacs:longitude')) || 0;
      const country    = get('gdacs:country');
      const title      = getCdata('title');
      const pubDate    = get('pubDate');
      const link       = getCdata('link');
      const guid       = get('guid');

      if (!title || !lat) continue;

      const typeMap: Record<string, IntelEvent['type']> = {
        eq: 'earthquake', fl: 'flood', tc: 'cyclone',
        vo: 'volcano',    wf: 'fire',   dr: 'drought',
      };

      items.push({
        id:       `gdacs-${guid || title.slice(0, 16)}`,
        type:     typeMap[eventType] ?? 'flood',
        title,
        severity: alertLevel === 'RED' ? 'RED' : alertLevel === 'ORANGE' ? 'ORANGE' : 'GREEN',
        lat,
        lon,
        country,
        ts_iso:   pubDate ? new Date(pubDate).toISOString() : new Date().toISOString(),
        source:   'GDACS',
        url:      link || undefined,
      });
    }
    return items;
  } catch {
    return [];
  }
}

// ── Route handler ─────────────────────────────────────────────────────────────

export async function GET() {
  if (_cache && Date.now() < _cache.expiresAt) {
    return NextResponse.json({ ..._cache.data, fromCache: true });
  }

  const [quakes, gdacs] = await Promise.allSettled([fetchEarthquakes(), fetchGdacs()]);

  const events: IntelEvent[] = [
    ...(quakes.status === 'fulfilled' ? quakes.value : []),
    ...(gdacs.status  === 'fulfilled' ? gdacs.value  : []),
  ].sort((a, b) => new Date(b.ts_iso).getTime() - new Date(a.ts_iso).getTime());

  const response: IntelEventsResponse = {
    events,
    fetchedAt: new Date().toISOString(),
    fromCache: false,
  };

  _cache = { data: response, expiresAt: Date.now() + CACHE_TTL_MS };
  return NextResponse.json(response);
}
