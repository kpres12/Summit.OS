'use client';

import React, { useState, useEffect, useCallback, useRef } from 'react';
import PanelHeader from '@/components/ui/PanelHeader';
import SectionHeader from '@/components/ui/SectionHeader';
import { apiFetch } from '@/lib/api';
import { publishOsint } from '@/lib/osintBus';
import { useEntityStream } from '@/hooks/useEntityStream';
import type { IntelEvent } from '@/app/api/intel/events/route';
import type { ImageryScene, ImageryResponse } from '@/app/api/intel/imagery/route';

// ─── OSINT / Web Search types ─────────────────────────────────────────────────

interface OsintResult {
  title: string;
  url: string;
  snippet: string;
  source: string;
  content?: string;
  crawledAt: string;
}

interface WebSearchResponse {
  query: string;
  results: OsintResult[];
  cachedAt: string;
  fromCache: boolean;
}

interface SatelliteData {
  entity_id: string;
  name: string;
  sat_id: string;
  sat_type: string;
  position: { lat: number; lon: number; alt: number };
  speed_kms: number;
  heading_deg: number;
  period_min: number;
}

interface JamZone {
  id: string;
  name: string;
  lat: number;
  lon: number;
  radius_km: number;
  intensity: number;
  source: string;
}

interface Vessel {
  id: string;
  name: string;
  type: string;
  lat: number;
  lon: number;
  heading: number;
  speed_kts: number;
  status?: string;
}

interface NoFlyZone {
  id: string;
  name: string;
  severity: string;
  source: string;
  active: boolean;
}

const SAT_TYPE_COLOR: Record<string, string> = {
  station: 'var(--accent)',
  sar: '#4FC3F7',
  reconnaissance: 'var(--warning)',
  optical: '#B39DDB',
  comms: 'var(--text-dim)',
};

const JAM_COLOR = (intensity: number) =>
  intensity > 0.8 ? 'var(--critical)' : intensity > 0.5 ? 'var(--warning)' : 'var(--text-dim)';

const _VESSEL_ICON: Record<string, string> = {
  tanker: '⛽', cargo: '📦', container: '🔲',
};

const EVENT_TYPE_LABEL: Record<IntelEvent['type'], string> = {
  earthquake: 'EQ',  flood: 'FL', cyclone: 'TC',
  volcano: 'VO', fire: 'WF', drought: 'DR',
};
const EVENT_SEVERITY_COLOR: Record<IntelEvent['severity'], string> = {
  RED: 'var(--critical)', ORANGE: 'var(--warning)', GREEN: 'var(--accent)',
};

// Synthetic RF signal profile per sensor entity classification
function rfProfile(classification: string): { freq: string; bw: string; waveform: string } {
  const c = classification.toLowerCase();
  if (c.includes('radar'))   return { freq: '9.3–9.5 GHz', bw: '1.0 MHz', waveform: 'LFM chirp' };
  if (c.includes('rf'))      return { freq: '0.5–3.0 GHz', bw: '500 kHz', waveform: 'CW / burst' };
  if (c.includes('ads-b') || c.includes('adsb'))
                             return { freq: '1090 MHz',    bw: '1.0 MHz', waveform: 'PPM squitter' };
  if (c.includes('ais'))     return { freq: '161–162 MHz', bw: '25 kHz',  waveform: 'GMSK TDMA' };
  if (c.includes('lidar'))   return { freq: '905 nm (IR)',  bw: '—',      waveform: 'ToF pulsed' };
  return                            { freq: '2.4 GHz ISM', bw: '20 MHz',  waveform: 'FHSS' };
}

export default function OpsIntel() {
  const { entityList } = useEntityStream();
  const sensorEntities = entityList.filter(e => e.domain === 'sensor');

  const [satellites, setSatellites] = useState<SatelliteData[]>([]);
  const [jamZones, setJamZones] = useState<JamZone[]>([]);
  const [vessels, setVessels] = useState<Vessel[]>([]);
  const [noFlyZones, setNoFlyZones] = useState<NoFlyZone[]>([]);
  const [events, setEvents] = useState<IntelEvent[]>([]);
  const [eventsLoading, setEventsLoading] = useState(false);
  const [imagery, setImagery] = useState<ImageryResponse | null>(null);
  const [imageryLoading, setImageryLoading] = useState(false);
  const [activeTab, setActiveTab] = useState<'sat' | 'jam' | 'mar' | 'nfz' | 'web' | 'sigint' | 'events' | 'img'>('sat');

  // ── Web / OSINT state ──────────────────────────────────────────────────────
  const [webQuery, setWebQuery] = useState('');
  const [webResults, setWebResults] = useState<WebSearchResponse | null>(null);
  const [webLoading, setWebLoading] = useState(false);
  const [webError, setWebError] = useState<string | null>(null);
  const [expandedResult, setExpandedResult] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const [loading, setLoading] = useState(true);

  const fetchAll = useCallback(async () => {
    try {
      const [satRes, jamRes, marRes, nfzRes] = await Promise.all([
        apiFetch('/v1/satellites').then(r => r.json()).catch(() => ({ satellites: [] })),
        apiFetch('/v1/gpsjam').then(r => r.json()).catch(() => ({ zones: [] })),
        apiFetch('/v1/maritime').then(r => r.json()).catch(() => ({ vessels: [] })),
        apiFetch('/v1/noflyzones').then(r => r.json()).catch(() => ({ zones: [] })),
      ]);
      setSatellites(satRes.satellites || []);
      setJamZones(jamRes.zones || []);
      setVessels(marRes.vessels || []);
      setNoFlyZones(nfzRes.zones || []);
    } catch {
      // non-fatal
    } finally {
      setLoading(false);
    }
  }, []);

  const fetchEvents = useCallback(async () => {
    setEventsLoading(true);
    try {
      const res = await fetch('/api/intel/events');
      if (res.ok) {
        const data = await res.json() as { events: IntelEvent[] };
        setEvents(data.events ?? []);
      }
    } catch {
      // non-fatal
    } finally {
      setEventsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchAll();
    const interval = setInterval(fetchAll, 15000);
    return () => clearInterval(interval);
  }, [fetchAll]);

  useEffect(() => {
    fetchEvents();
    const interval = setInterval(fetchEvents, 5 * 60 * 1000); // 5-min cadence
    return () => clearInterval(interval);
  }, [fetchEvents]);

  const runWebSearch = useCallback(async (q: string) => {
    const query = q.trim();
    if (!query) return;
    setWebLoading(true);
    setWebError(null);
    setExpandedResult(null);
    try {
      const res = await fetch('/api/intel/web-search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, scrapeTop: 2 }),
      });
      if (!res.ok) {
        const err = await res.json() as { error?: string };
        throw new Error(err.error ?? `HTTP ${res.status}`);
      }
      const data = await res.json() as WebSearchResponse;
      setWebResults(data);
      if (data.results.length > 0) {
        publishOsint({
          query,
          topSnippet: data.results[0].snippet,
          source: data.results[0].source,
          resultCount: data.results.length,
        });
      }
    } catch (err) {
      setWebError(err instanceof Error ? err.message : 'Search failed');
    } finally {
      setWebLoading(false);
    }
  }, []);

  const fetchImagery = useCallback(async () => {
    // Default AOR: ±1° around San Francisco (matches simulated entities)
    setImageryLoading(true);
    try {
      const res = await fetch('/api/intel/imagery', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ bbox: [-123.5, 36.8, -121.4, 38.9], days: 30 }),
      });
      if (res.ok) setImagery(await res.json() as ImageryResponse);
    } catch {
      // non-fatal
    } finally {
      setImageryLoading(false);
    }
  }, []);

  useEffect(() => {
    if (activeTab === 'img' && !imagery && !imageryLoading) fetchImagery();
  }, [activeTab, imagery, imageryLoading, fetchImagery]);

  const tabs = [
    { id: 'sat'    as const, label: 'SAT',    count: satellites.length },
    { id: 'jam'    as const, label: 'GPS',    count: jamZones.filter(z => z.intensity > 0.5).length },
    { id: 'mar'    as const, label: 'MAR',    count: vessels.length },
    { id: 'nfz'    as const, label: 'NFZ',    count: noFlyZones.filter(z => z.active).length },
    { id: 'sigint' as const, label: 'RF',     count: sensorEntities.length },
    { id: 'events' as const, label: 'EVT',    count: events.filter(e => e.severity !== 'GREEN').length },
    { id: 'img'    as const, label: 'IMG',    count: imagery?.scenes.length ?? 0 },
    { id: 'web'    as const, label: 'WEB',    count: webResults?.results.length ?? 0 },
  ];

  return (
    <div className="flex flex-col h-full panel-scanline">
      <PanelHeader title="INTEL" count={satellites.length + jamZones.length} />

      {/* Source tabs */}
      <div
        className="flex-none flex"
        style={{ borderBottom: '1px solid var(--border)' }}
      >
        {tabs.map(tab => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className="flex-1 py-2 text-[10px] tracking-widest transition-colors"
            style={{
              fontFamily: 'var(--font-ibm-plex-mono), monospace',
              color: activeTab === tab.id ? 'var(--accent)' : 'var(--text-muted)',
              background: activeTab === tab.id ? 'var(--accent-5)' : 'transparent',
              border: 'none',
              borderBottom: activeTab === tab.id ? '2px solid var(--accent)' : '2px solid transparent',
              cursor: 'pointer',
            }}
          >
            {tab.label}
            {tab.count > 0 && (
              <span style={{ marginLeft: 4, color: activeTab === tab.id ? 'var(--accent-50)' : 'var(--text-muted)' }}>
                {tab.count}
              </span>
            )}
          </button>
        ))}
      </div>

      <div className="flex-1 overflow-y-auto">
        {loading && (
          <div className="flex items-center justify-center h-20">
            <span style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: 'var(--text-muted)', fontSize: 10 }}>
              FETCHING...
            </span>
          </div>
        )}

        {/* Satellites */}
        {activeTab === 'sat' && !loading && (
          <>
            <SectionHeader title="ORBITAL ASSETS" />
            <div className="px-3 pb-2 text-[9px]" style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: 'var(--text-muted)', lineHeight: 1.6 }}>
              Live positions computed from public orbital elements. Includes commercial, civil, and foreign reconnaissance satellites.
            </div>
            {satellites.map(sat => (
              <div
                key={sat.entity_id}
                className="px-3 py-2 flex flex-col gap-0.5"
                style={{ borderBottom: '1px solid var(--accent-5)' }}
              >
                <div className="flex items-center justify-between">
                  <span className="text-[11px] font-bold" style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: SAT_TYPE_COLOR[sat.sat_type] || 'var(--text-dim)' }}>
                    {sat.name}
                  </span>
                  <span className="text-[9px] px-1" style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: 'var(--text-muted)', border: '1px solid var(--accent-10)' }}>
                    {sat.sat_type.toUpperCase()}
                  </span>
                </div>
                <div className="flex gap-3 text-[9px]" style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: 'var(--text-muted)' }}>
                  <span>{sat.position.lat.toFixed(1)}°, {sat.position.lon.toFixed(1)}°</span>
                  <span>{(sat.position.alt / 1000).toFixed(0)} km ALT</span>
                  <span>{sat.period_min.toFixed(0)} min ORB</span>
                </div>
              </div>
            ))}
          </>
        )}

        {/* GPS Jamming */}
        {activeTab === 'jam' && !loading && (
          <>
            <SectionHeader title="GPS INTERFERENCE" />
            <div className="px-3 pb-2 text-[9px]" style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: 'var(--text-muted)', lineHeight: 1.6 }}>
              Jamming inferred from commercial aircraft GPS degradation signals. Source: ELINT aggregation.
            </div>
            {jamZones.map(zone => (
              <div
                key={zone.id}
                className="px-3 py-2.5 flex flex-col gap-1"
                style={{ borderBottom: '1px solid var(--accent-5)' }}
              >
                <div className="flex items-center justify-between">
                  <span className="text-[11px] font-bold" style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: JAM_COLOR(zone.intensity) }}>
                    {zone.name}
                  </span>
                  <span className="text-[10px] font-bold" style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: JAM_COLOR(zone.intensity) }}>
                    {Math.round(zone.intensity * 100)}%
                  </span>
                </div>
                <div className="h-1" style={{ background: 'var(--accent-5)', borderRadius: 1 }}>
                  <div style={{ width: `${zone.intensity * 100}%`, height: '100%', background: JAM_COLOR(zone.intensity), borderRadius: 1 }} />
                </div>
                <div className="flex gap-3 text-[9px]" style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: 'var(--text-muted)' }}>
                  <span>{zone.radius_km} km radius</span>
                  <span>{zone.source}</span>
                </div>
              </div>
            ))}
          </>
        )}

        {/* Maritime */}
        {activeTab === 'mar' && !loading && (
          <>
            <SectionHeader title="MARITIME TRAFFIC" />
            <div className="px-3 pb-2 text-[9px]" style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: 'var(--text-muted)', lineHeight: 1.6 }}>
              AIS vessel tracking — Persian Gulf, Red Sea, Strait of Hormuz.
            </div>
            {vessels.map(v => (
              <div
                key={v.id}
                className="px-3 py-2 flex flex-col gap-0.5"
                style={{ borderBottom: '1px solid var(--accent-5)' }}
              >
                <div className="flex items-center justify-between">
                  <span className="text-[11px] font-bold" style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: v.status === 'anchored' ? 'var(--text-dim)' : '#4FC3F7' }}>
                    {v.name}
                  </span>
                  {v.status === 'anchored' ? (
                    <span className="text-[9px] px-1" style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: 'var(--warning)', border: '1px solid color-mix(in srgb, var(--warning) 30%, transparent)' }}>ANCHORED</span>
                  ) : (
                    <span className="text-[9px]" style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: 'var(--text-muted)' }}>{v.speed_kts.toFixed(1)} kts</span>
                  )}
                </div>
                <div className="flex gap-3 text-[9px]" style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: 'var(--text-muted)' }}>
                  <span>{v.type.toUpperCase()}</span>
                  <span>{v.lat.toFixed(2)}°, {v.lon.toFixed(2)}°</span>
                  <span>HDG {v.heading.toFixed(0)}°</span>
                </div>
              </div>
            ))}
          </>
        )}

        {/* No-fly zones */}
        {activeTab === 'nfz' && !loading && (
          <>
            <SectionHeader title="ACTIVE AIRSPACE CLOSURES" />
            <div className="px-3 pb-2 text-[9px]" style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: 'var(--text-muted)', lineHeight: 1.6 }}>
              NOTAM-sourced airspace restrictions and conflict zone closures.
            </div>
            {noFlyZones.map(z => (
              <div
                key={z.id}
                className="px-3 py-2.5 flex flex-col gap-1"
                style={{ borderBottom: '1px solid var(--accent-5)', opacity: z.active ? 1 : 0.4 }}
              >
                <div className="flex items-center justify-between">
                  <span className="text-[11px] font-bold" style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: z.severity === 'CRITICAL' ? 'var(--critical)' : 'var(--warning)' }}>
                    {z.name}
                  </span>
                  <span className="text-[9px] px-1" style={{
                    fontFamily: 'var(--font-ibm-plex-mono), monospace',
                    color: z.severity === 'CRITICAL' ? 'var(--critical)' : 'var(--warning)',
                    border: `1px solid ${z.severity === 'CRITICAL' ? 'color-mix(in srgb, var(--critical) 30%, transparent)' : 'color-mix(in srgb, var(--warning) 30%, transparent)'}`,
                  }}>
                    {z.severity}
                  </span>
                </div>
                <div className="text-[9px]" style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: 'var(--text-muted)' }}>
                  {z.source} · {z.active ? 'ACTIVE' : 'INACTIVE'}
                </div>
              </div>
            ))}
          </>
        )}
        {/* SIGINT — RF sensor profiles */}
        {activeTab === 'sigint' && (
          <>
            <SectionHeader title="RF / SIGINT SENSORS" />
            <div className="px-3 pb-2 text-[9px]" style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: 'var(--text-muted)', lineHeight: 1.6 }}>
              Active sensor nodes reporting to fusion engine. Frequency / waveform derived from classification.
            </div>
            {sensorEntities.length === 0 && (
              <div className="flex items-center justify-center h-16">
                <span style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: 'var(--text-muted)', fontSize: 10 }}>
                  NO SENSOR ENTITIES IN STREAM
                </span>
              </div>
            )}
            {sensorEntities.map(e => {
              const rf = rfProfile(e.classification);
              const snr = Math.round(60 + (e.confidence ?? 0.9) * 30);
              const status = e.track_state === 'confirmed' ? 'ACTIVE' : e.track_state === 'coasting' ? 'COAST' : 'INIT';
              return (
                <div key={e.entity_id} className="px-3 py-2.5 flex flex-col gap-1.5"
                  style={{ borderBottom: '1px solid var(--accent-5)' }}>
                  <div className="flex items-center justify-between">
                    <span className="text-[11px] font-bold"
                      style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: 'var(--accent)' }}>
                      {e.callsign ?? e.entity_id}
                    </span>
                    <span className="text-[9px] px-1"
                      style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace',
                        color: status === 'ACTIVE' ? 'var(--accent)' : 'var(--warning)',
                        border: '1px solid var(--accent-15)' }}>
                      {status}
                    </span>
                  </div>
                  <div className="text-[9px]"
                    style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: 'var(--text-dim)' }}>
                    {e.classification}
                  </div>
                  <div className="grid gap-x-3 gap-y-0.5 text-[9px]"
                    style={{ gridTemplateColumns: '1fr 1fr', fontFamily: 'var(--font-ibm-plex-mono), monospace', color: 'var(--text-muted)' }}>
                    <span>FREQ  <span style={{ color: 'var(--accent)' }}>{rf.freq}</span></span>
                    <span>BW    <span style={{ color: 'var(--accent)' }}>{rf.bw}</span></span>
                    <span>MODE  <span style={{ color: 'var(--accent)' }}>{rf.waveform}</span></span>
                    <span>SNR   <span style={{ color: 'var(--accent)' }}>{snr} dB</span></span>
                    <span>LAT   <span style={{ color: 'var(--text-dim)' }}>{e.position.lat.toFixed(4)}°</span></span>
                    <span>LON   <span style={{ color: 'var(--text-dim)' }}>{e.position.lon.toFixed(4)}°</span></span>
                  </div>
                  {/* SNR bar */}
                  <div style={{ height: 3, background: 'var(--accent-5)', marginTop: 2 }}>
                    <div style={{ width: `${Math.min(100, snr)}%`, height: '100%',
                      background: snr > 80 ? 'var(--accent)' : snr > 60 ? 'var(--warning)' : 'var(--critical)' }} />
                  </div>
                </div>
              );
            })}
          </>
        )}

        {/* EVENTS — GDACS + USGS real-world feed */}
        {activeTab === 'events' && (
          <>
            <SectionHeader title="GLOBAL EVENTS FEED" />
            <div className="px-3 pb-2 text-[9px]" style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: 'var(--text-muted)', lineHeight: 1.6 }}>
              Live disaster alerts — GDACS + USGS. Updates every 5 min. Click row to copy coordinates.
            </div>
            {eventsLoading && events.length === 0 && (
              <div className="flex items-center justify-center h-16">
                <span style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: 'var(--text-muted)', fontSize: 10 }}>
                  FETCHING GDACS + USGS...
                </span>
              </div>
            )}
            {events.map(ev => (
              <div key={ev.id}
                className="px-3 py-2.5 flex flex-col gap-0.5 cursor-pointer"
                style={{ borderBottom: '1px solid var(--accent-5)',
                  borderLeft: `3px solid ${EVENT_SEVERITY_COLOR[ev.severity]}` }}
                onClick={() => navigator.clipboard?.writeText(`${ev.lat},${ev.lon}`).catch(() => {})}
                title="Click to copy coordinates">
                <div className="flex items-center gap-2">
                  <span className="text-[9px] px-1 font-bold"
                    style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace',
                      color: EVENT_SEVERITY_COLOR[ev.severity],
                      border: `1px solid color-mix(in srgb, ${EVENT_SEVERITY_COLOR[ev.severity]} 30%, transparent)` }}>
                    {EVENT_TYPE_LABEL[ev.type]}
                  </span>
                  <span className="text-[9px] px-1"
                    style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: 'var(--text-muted)',
                      border: '1px solid var(--accent-10)' }}>
                    {ev.source}
                  </span>
                  <span className="text-[9px] ml-auto"
                    style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: 'var(--text-muted)' }}>
                    {new Date(ev.ts_iso).toUTCString().slice(5, 22)}
                  </span>
                </div>
                <span className="text-[10px] leading-snug"
                  style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: 'var(--text-dim)' }}>
                  {ev.title}
                </span>
                <span className="text-[9px]"
                  style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: 'var(--text-muted)' }}>
                  {ev.lat.toFixed(2)}°, {ev.lon.toFixed(2)}° · {ev.country}
                </span>
              </div>
            ))}
            {!eventsLoading && events.length === 0 && (
              <div className="flex items-center justify-center h-16">
                <span style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: 'var(--text-muted)', fontSize: 10 }}>
                  NO EVENTS — CHECK NETWORK
                </span>
              </div>
            )}
          </>
        )}

        {/* Imagery — Sentinel-2 (free) + BlackSky (paid via UP42) */}
        {activeTab === 'img' && (
          <>
            <SectionHeader title="SATELLITE IMAGERY" />
            <div className="px-3 pb-2 text-[9px]" style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: 'var(--text-muted)', lineHeight: 1.6 }}>
              Sentinel-2 (10 m, free) · BlackSky (35 cm, requires UP42 key). Past 30 days, &lt;40% cloud cover.
            </div>

            {/* Provider badges */}
            <div className="flex gap-2 px-3 pb-3">
              <span className="text-[9px] px-1.5 py-0.5" style={{
                fontFamily: 'var(--font-ibm-plex-mono), monospace',
                color: 'var(--accent)',
                border: '1px solid var(--accent-20)',
                background: 'var(--accent-5)',
              }}>S2 LIVE</span>
              <span className="text-[9px] px-1.5 py-0.5" style={{
                fontFamily: 'var(--font-ibm-plex-mono), monospace',
                color: imagery?.providers.blacksky ? 'var(--accent)' : 'var(--text-muted)',
                border: `1px solid ${imagery?.providers.blacksky ? 'var(--accent-20)' : 'var(--border)'}`,
                background: imagery?.providers.blacksky ? 'var(--accent-5)' : 'transparent',
              }}>{imagery?.providers.blacksky ? 'BLACKSKY LIVE' : 'BLACKSKY — SET UP42_API_KEY'}</span>
            </div>

            {imageryLoading && (
              <div className="flex items-center justify-center h-16">
                <span style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: 'var(--text-muted)', fontSize: 10 }}>
                  QUERYING ELEMENT84 STAC...
                </span>
              </div>
            )}

            {!imageryLoading && imagery?.scenes.length === 0 && (
              <div className="flex items-center justify-center h-16">
                <span style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: 'var(--text-muted)', fontSize: 10 }}>
                  NO SCENES IN WINDOW
                </span>
              </div>
            )}

            {imagery?.scenes.map(scene => (
              <div key={scene.id} className="px-3 py-2.5 flex gap-3"
                style={{ borderBottom: '1px solid var(--accent-5)' }}>
                {/* Thumbnail */}
                <div style={{ flexShrink: 0, width: 52, height: 52, background: 'var(--accent-5)',
                  border: '1px solid var(--accent-10)', overflow: 'hidden' }}>
                  {scene.thumbnail_url
                    ? <img src={scene.thumbnail_url} alt="scene" style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
                    : <div style={{ width: '100%', height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                        <span style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', fontSize: 8, color: 'var(--text-muted)' }}>NO IMG</span>
                      </div>
                  }
                </div>

                {/* Metadata */}
                <div className="flex flex-col gap-0.5 flex-1 min-w-0">
                  <div className="flex items-center gap-1.5">
                    <span className="text-[9px] px-1" style={{
                      fontFamily: 'var(--font-ibm-plex-mono), monospace',
                      color: scene.provider === 'blacksky' ? 'var(--warning)' : '#4FC3F7',
                      border: `1px solid ${scene.provider === 'blacksky' ? 'color-mix(in srgb, var(--warning) 30%, transparent)' : 'rgba(79,195,247,0.25)'}`,
                    }}>
                      {scene.provider === 'blacksky' ? 'BLACKSKY' : 'S2'}
                    </span>
                    <span className="text-[9px]" style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: 'var(--accent)' }}>
                      {scene.resolution_m < 1 ? `${(scene.resolution_m * 100).toFixed(0)} cm` : `${scene.resolution_m} m`}
                    </span>
                    <span className="text-[9px] ml-auto" style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: 'var(--text-muted)' }}>
                      ☁ {scene.cloud_pct}%
                    </span>
                  </div>
                  <span className="text-[9px]" style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: 'var(--text-dim)' }}>
                    {new Date(scene.acquired_iso).toUTCString().slice(5, 22)}
                  </span>
                  <span className="text-[9px] truncate" style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: 'var(--text-muted)' }}>
                    {scene.bbox[0].toFixed(1)}°W {scene.bbox[1].toFixed(1)}°S → {scene.bbox[2].toFixed(1)}°E {scene.bbox[3].toFixed(1)}°N
                  </span>
                  {scene.preview_url && (
                    <a href={scene.preview_url} target="_blank" rel="noreferrer"
                      className="text-[9px] self-start"
                      style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: 'var(--accent)', textDecoration: 'none' }}
                      onClick={e => e.stopPropagation()}>
                      [ VIEW FULL ]
                    </a>
                  )}
                </div>
              </div>
            ))}

            {/* Re-fetch button */}
            {!imageryLoading && (
              <div className="px-3 pt-2 pb-3">
                <button onClick={fetchImagery} className="text-[9px] tracking-widest" style={{
                  fontFamily: 'var(--font-ibm-plex-mono), monospace',
                  color: 'var(--text-muted)',
                  background: 'transparent',
                  border: '1px solid var(--border)',
                  padding: '3px 8px',
                  cursor: 'pointer',
                  width: '100%',
                }}>
                  REFRESH IMAGERY
                </button>
              </div>
            )}
          </>
        )}

        {/* Web / OSINT */}
        {activeTab === 'web' && (
          <>
            <SectionHeader title="WEB INTELLIGENCE" />
            <div className="px-3 pb-2 text-[9px]" style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: 'var(--text-muted)', lineHeight: 1.6 }}>
              Open-source web search via Serper + Firecrawl. Results tagged [OSINT]. Cache TTL: 10 min.
            </div>

            {/* Search input */}
            <div
              className="flex gap-1 px-3 pb-3"
              style={{ borderBottom: '1px solid var(--accent-5)' }}
            >
              <input
                ref={inputRef}
                value={webQuery}
                onChange={(e) => setWebQuery(e.target.value)}
                onKeyDown={(e) => { if (e.key === 'Enter') runWebSearch(webQuery); }}
                placeholder="wildfire perimeter CA-2024..."
                className="flex-1 px-2 py-1 text-[10px] outline-none"
                style={{
                  fontFamily: 'var(--font-ibm-plex-mono), monospace',
                  background: 'var(--accent-5)',
                  border: '1px solid var(--accent-10)',
                  color: 'var(--text-dim)',
                  caretColor: 'var(--accent)',
                }}
              />
              <button
                onClick={() => runWebSearch(webQuery)}
                disabled={webLoading || !webQuery.trim()}
                className="px-3 py-1 text-[10px] font-bold tracking-widest transition-colors"
                style={{
                  fontFamily: 'var(--font-ibm-plex-mono), monospace',
                  background: webLoading ? 'transparent' : 'var(--accent)',
                  color: webLoading ? 'var(--text-muted)' : '#080C0A',
                  border: webLoading ? '1px solid var(--accent-10)' : 'none',
                  cursor: webLoading || !webQuery.trim() ? 'not-allowed' : 'pointer',
                  opacity: !webQuery.trim() ? 0.4 : 1,
                }}
              >
                {webLoading ? '...' : 'GO'}
              </button>
            </div>

            {/* Error */}
            {webError && (
              <div className="mx-3 my-2 px-2 py-1 text-[9px]" style={{
                fontFamily: 'var(--font-ibm-plex-mono), monospace',
                color: 'var(--critical)',
                border: '1px solid color-mix(in srgb, var(--critical) 25%, transparent)',
                background: 'color-mix(in srgb, var(--critical) 5%, transparent)',
              }}>
                ERR: {webError}
              </div>
            )}

            {/* Loading state */}
            {webLoading && (
              <div className="flex items-center justify-center h-20">
                <span style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: 'var(--text-muted)', fontSize: 10 }}>
                  QUERYING SERPER + FIRECRAWL...
                </span>
              </div>
            )}

            {/* Results */}
            {!webLoading && webResults && (
              <>
                <div className="px-3 pt-2 pb-1 flex items-center justify-between">
                  <span className="text-[9px]" style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: 'var(--text-muted)' }}>
                    {webResults.results.length} RESULTS
                  </span>
                  <span className="text-[9px]" style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: 'var(--text-muted)' }}>
                    {webResults.fromCache ? '[CACHED]' : '[LIVE]'}
                  </span>
                </div>
                {webResults.results.map((r) => {
                  const isExpanded = expandedResult === r.url;
                  return (
                    <div
                      key={r.url}
                      className="px-3 py-2 flex flex-col gap-1 cursor-pointer"
                      style={{ borderBottom: '1px solid var(--accent-5)', borderLeft: '3px solid var(--accent-10)' }}
                      onClick={() => setExpandedResult(isExpanded ? null : r.url)}
                    >
                      {/* Source badge + title */}
                      <div className="flex items-start gap-2">
                        <span
                          className="flex-none text-[8px] px-1 py-0.5 mt-0.5"
                          style={{
                            fontFamily: 'var(--font-ibm-plex-mono), monospace',
                            color: 'var(--accent)',
                            border: '1px solid var(--accent-20)',
                            background: 'var(--accent-5)',
                            whiteSpace: 'nowrap',
                          }}
                        >
                          OSINT
                        </span>
                        <span
                          className="text-[10px] leading-tight font-bold"
                          style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: 'var(--text-dim)' }}
                        >
                          {r.title}
                        </span>
                      </div>

                      {/* Source domain */}
                      <span className="text-[9px]" style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: 'var(--accent)', opacity: 0.6 }}>
                        {r.source}
                      </span>

                      {/* Snippet */}
                      <span className="text-[9px] leading-relaxed" style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: 'var(--text-muted)' }}>
                        {r.snippet}
                      </span>

                      {/* Expanded scraped content */}
                      {isExpanded && r.content && (
                        <div
                          className="mt-1 px-2 py-2 text-[9px] leading-relaxed whitespace-pre-wrap"
                          style={{
                            fontFamily: 'var(--font-ibm-plex-mono), monospace',
                            color: 'var(--text-muted)',
                            background: 'var(--accent-5)',
                            border: '1px solid var(--accent-10)',
                            maxHeight: '200px',
                            overflowY: 'auto',
                          }}
                        >
                          {r.content}
                        </div>
                      )}
                      {isExpanded && !r.content && (
                        <div className="mt-1 text-[9px]" style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: 'var(--text-muted)' }}>
                          No scraped content for this result.
                        </div>
                      )}

                      {/* Toggle indicator */}
                      <span className="text-[8px] self-end" style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: 'var(--text-muted)' }}>
                        {isExpanded ? '[ COLLAPSE ]' : r.content ? '[ EXPAND ]' : ''}
                      </span>
                    </div>
                  );
                })}
              </>
            )}

            {/* Empty state */}
            {!webLoading && !webResults && !webError && (
              <div className="flex flex-col items-center justify-center gap-2 py-8 px-4">
                <span className="text-[9px] text-center" style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: 'var(--text-muted)', lineHeight: 1.8 }}>
                  EXAMPLE QUERIES{'\n'}
                  wildfire perimeter CA-LNU-2024{'\n'}
                  FAA TFR Sacramento County{'\n'}
                  hurricane track NHC advisory{'\n'}
                  missing persons Sierra Nevada
                </span>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
