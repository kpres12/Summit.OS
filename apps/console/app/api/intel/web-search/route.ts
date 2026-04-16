import { NextRequest, NextResponse } from 'next/server';

const SERPER_API_KEY = process.env.SERPER_API_KEY;
const FIRECRAWL_API_KEY = process.env.FIRECRAWL_API_KEY;

// ─── Simple in-memory cache (per-process, TTL 10 min) ─────────────────────────

interface CacheEntry {
  data: WebSearchResponse;
  expiresAt: number;
}

const cache = new Map<string, CacheEntry>();
const CACHE_TTL_MS = 10 * 60 * 1000;

function getCached(key: string): WebSearchResponse | null {
  const entry = cache.get(key);
  if (!entry) return null;
  if (Date.now() > entry.expiresAt) {
    cache.delete(key);
    return null;
  }
  return entry.data;
}

function setCached(key: string, data: WebSearchResponse) {
  cache.set(key, { data, expiresAt: Date.now() + CACHE_TTL_MS });
}

// ─── Types ────────────────────────────────────────────────────────────────────

export interface OsintResult {
  title: string;
  url: string;
  snippet: string;
  source: string;
  content?: string;         // scraped markdown from Firecrawl (top results only)
  crawledAt: string;        // ISO timestamp
}

export interface WebSearchResponse {
  query: string;
  results: OsintResult[];
  cachedAt: string;
  fromCache: boolean;
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

function sourceDomain(url: string): string {
  try {
    return new URL(url).hostname.replace(/^www\./, '');
  } catch {
    return url;
  }
}

async function serperSearch(query: string): Promise<{ title: string; link: string; snippet: string }[]> {
  const res = await fetch('https://google.serper.dev/search', {
    method: 'POST',
    headers: {
      'X-API-KEY': SERPER_API_KEY!,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ q: query, num: 8 }),
  });
  if (!res.ok) throw new Error(`Serper error: ${res.status}`);
  const data = await res.json() as { organic?: { title: string; link: string; snippet: string }[] };
  return data.organic ?? [];
}

async function firecrawlScrape(url: string): Promise<string | null> {
  try {
    const res = await fetch('https://api.firecrawl.dev/v1/scrape', {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${FIRECRAWL_API_KEY}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        url,
        formats: ['markdown'],
        onlyMainContent: true,
      }),
    });
    if (!res.ok) return null;
    const data = await res.json() as { success?: boolean; data?: { markdown?: string } };
    return data?.data?.markdown?.slice(0, 2000) ?? null;   // cap at 2 000 chars
  } catch {
    return null;
  }
}

// ─── Route handler ────────────────────────────────────────────────────────────

export async function POST(req: NextRequest) {
  if (!SERPER_API_KEY || !FIRECRAWL_API_KEY) {
    return NextResponse.json({ error: 'OSINT keys not configured' }, { status: 503 });
  }

  let query: string;
  let scrapeTop: number;

  try {
    const body = await req.json() as { query?: string; scrapeTop?: number };
    query = (body.query ?? '').trim();
    scrapeTop = Math.min(body.scrapeTop ?? 2, 3);   // max 3 scrapes per request
  } catch {
    return NextResponse.json({ error: 'Invalid JSON body' }, { status: 400 });
  }

  if (!query) {
    return NextResponse.json({ error: 'query is required' }, { status: 400 });
  }

  const cacheKey = `${query}|${scrapeTop}`;
  const cached = getCached(cacheKey);
  if (cached) {
    return NextResponse.json({ ...cached, fromCache: true });
  }

  try {
    const organic = await serperSearch(query);

    // Scrape top N results in parallel
    const scrapeTargets = organic.slice(0, scrapeTop);
    const scraped = await Promise.all(scrapeTargets.map((r) => firecrawlScrape(r.link)));

    const results: OsintResult[] = organic.map((r, i) => ({
      title: r.title,
      url: r.link,
      snippet: r.snippet,
      source: sourceDomain(r.link),
      content: scraped[i] ?? undefined,
      crawledAt: new Date().toISOString(),
    }));

    const response: WebSearchResponse = {
      query,
      results,
      cachedAt: new Date().toISOString(),
      fromCache: false,
    };

    setCached(cacheKey, response);
    return NextResponse.json(response);
  } catch (err) {
    const msg = err instanceof Error ? err.message : 'Unknown error';
    return NextResponse.json({ error: msg }, { status: 502 });
  }
}
