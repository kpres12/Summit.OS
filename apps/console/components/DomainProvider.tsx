'use client';

import React, { createContext, useContext, useState, useEffect, useCallback } from 'react';
import type { DomainConfig } from '@/lib/domains/types';

/**
 * DomainProvider — runtime domain theming for Summit.OS.
 *
 * Loads domain configs from /domains/*.json at runtime.
 * To add a new domain:
 *   1. Create a JSON file in public/domains/ matching the DomainConfig schema
 *   2. Add an entry to public/domains/index.json
 *   3. That's it — no source code changes needed
 *
 * This is what makes Summit.OS a platform, not a product.
 */

const STORAGE_KEY = 'summit_domain';

// ── Minimal fallback (used before JSON loads) ───────────────────────────────

const FALLBACK: DomainConfig = {
  id: 'default',
  name: 'Summit.OS',
  description: 'General-purpose operations platform',
  palette: {
    accent: '#00FF9C',
    accentDim: '#00CC74',
    accentDark: '#00AA5E',
    warning: '#FFB300',
    critical: '#FF3B3B',
    nominal: '#4AEDC4',
    active: '#4FC3F7',
    backgroundTint: '#080C0A',
    panelBg: '#0D1210',
    border: 'rgba(0,255,156,0.15)',
    scanline: 'rgba(0,255,156,0.05)',
  },
  entityLabels: {},
  assetTypes: [],
  mapLayers: [],
  commandExamples: [],
  alertTypes: [],
  missionTemplates: [],
  terminology: {
    mission: 'Mission',
    asset: 'Asset',
    alert: 'Alert',
    entity: 'Entity',
    operatorView: 'Operator',
    supervisorView: 'Supervisor',
  },
};

// ── Context ─────────────────────────────────────────────────────────────────

interface DomainIndexEntry {
  id: string;
  name: string;
  file: string;
}

interface DomainContextValue {
  /** The active DomainConfig. */
  config: DomainConfig;
  /** Switch domain at runtime. Persists to localStorage. */
  setDomain: (id: string) => void;
  /** All registered domains (for the picker — id + name only until loaded). */
  domains: DomainConfig[];
  /** True while the domain index is loading. */
  loading: boolean;
}

const DomainContext = createContext<DomainContextValue>({
  config: FALLBACK,
  setDomain: () => {},
  domains: [FALLBACK],
  loading: true,
});

export function useDomain() {
  return useContext(DomainContext);
}

// ── CSS variable injection ──────────────────────────────────────────────────

function applyPalette(palette: DomainConfig['palette']) {
  const root = document.documentElement;
  root.style.setProperty('--accent', palette.accent);
  root.style.setProperty('--accent-dim', palette.accentDim);
  root.style.setProperty('--accent-dark', palette.accentDark);
  root.style.setProperty('--warning', palette.warning);
  root.style.setProperty('--critical', palette.critical);
  root.style.setProperty('--nominal', palette.nominal);
  root.style.setProperty('--color-active', palette.active);
  root.style.setProperty('--background', palette.backgroundTint);
  root.style.setProperty('--background-panel', palette.panelBg);
  root.style.setProperty('--border', palette.border);
  root.style.setProperty('--scanline', palette.scanline);

  // Derived opacity variants
  root.style.setProperty('--accent-5', `${palette.accent}0D`);
  root.style.setProperty('--accent-10', `${palette.accent}1A`);
  root.style.setProperty('--accent-15', `${palette.accent}26`);
  root.style.setProperty('--accent-30', `${palette.accent}4D`);
  root.style.setProperty('--accent-50', `${palette.accent}80`);
}

// ── Fetchers ────────────────────────────────────────────────────────────────

async function fetchIndex(): Promise<DomainIndexEntry[]> {
  try {
    const r = await fetch('/domains/index.json');
    if (!r.ok) return [];
    return await r.json();
  } catch {
    return [];
  }
}

async function fetchDomain(file: string): Promise<DomainConfig | null> {
  try {
    const r = await fetch(`/domains/${file}`);
    if (!r.ok) return null;
    return await r.json();
  } catch {
    return null;
  }
}

// ── Provider ────────────────────────────────────────────────────────────────

export default function DomainProvider({ children }: { children: React.ReactNode }) {
  const [config, setConfig] = useState<DomainConfig>(FALLBACK);
  const [domains, setDomains] = useState<DomainConfig[]>([FALLBACK]);
  const [cache, setCache] = useState<Record<string, DomainConfig>>({ default: FALLBACK });
  const [loading, setLoading] = useState(true);

  // 1. Load the domain index + the user's stored domain on mount
  useEffect(() => {
    (async () => {
      const entries = await fetchIndex();

      // Load all domains in parallel (they're tiny JSON files)
      const loaded: DomainConfig[] = [];
      const newCache: Record<string, DomainConfig> = {};
      await Promise.all(
        entries.map(async (entry) => {
          const d = await fetchDomain(entry.file);
          if (d) {
            loaded.push(d);
            newCache[d.id] = d;
          }
        }),
      );

      // Sort to match index order
      const ordered = entries
        .map((e) => newCache[e.id])
        .filter(Boolean) as DomainConfig[];

      setDomains(ordered.length > 0 ? ordered : [FALLBACK]);
      setCache(newCache);

      // Resolve the user's stored preference
      const stored = localStorage.getItem(STORAGE_KEY);
      const resolved = (stored && newCache[stored]) || ordered[0] || FALLBACK;
      setConfig(resolved);
      applyPalette(resolved.palette);
      setLoading(false);
    })();
  }, []);

  const setDomain = useCallback(
    (id: string) => {
      const next = cache[id] || FALLBACK;
      setConfig(next);
      localStorage.setItem(STORAGE_KEY, next.id);
      applyPalette(next.palette);
    },
    [cache],
  );

  return (
    <DomainContext.Provider value={{ config, setDomain, domains, loading }}>
      {children}
    </DomainContext.Provider>
  );
}
