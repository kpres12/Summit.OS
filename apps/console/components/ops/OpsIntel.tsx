'use client';

import React, { useState, useEffect, useCallback } from 'react';
import PanelHeader from '@/components/ui/PanelHeader';
import SectionHeader from '@/components/ui/SectionHeader';
import { apiFetch } from '@/lib/api';

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

const VESSEL_ICON: Record<string, string> = {
  tanker: '⛽', cargo: '📦', container: '🔲',
};

export default function OpsIntel() {
  const [satellites, setSatellites] = useState<SatelliteData[]>([]);
  const [jamZones, setJamZones] = useState<JamZone[]>([]);
  const [vessels, setVessels] = useState<Vessel[]>([]);
  const [noFlyZones, setNoFlyZones] = useState<NoFlyZone[]>([]);
  const [activeTab, setActiveTab] = useState<'sat' | 'jam' | 'mar' | 'nfz'>('sat');
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

  useEffect(() => {
    fetchAll();
    const interval = setInterval(fetchAll, 15000);
    return () => clearInterval(interval);
  }, [fetchAll]);

  const tabs = [
    { id: 'sat' as const, label: 'SAT', count: satellites.length },
    { id: 'jam' as const, label: 'GPS', count: jamZones.filter(z => z.intensity > 0.5).length },
    { id: 'mar' as const, label: 'MAR', count: vessels.length },
    { id: 'nfz' as const, label: 'NFZ', count: noFlyZones.filter(z => z.active).length },
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
      </div>
    </div>
  );
}
