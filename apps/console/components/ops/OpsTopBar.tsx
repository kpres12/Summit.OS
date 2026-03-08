'use client';

import React, { useState, useEffect } from 'react';
import { useEntityStream } from '@/hooks/useEntityStream';

interface OpsTopBarProps {
  onSwitchRole: () => void;
}

function utcString(d: Date): string {
  const y = d.getUTCFullYear();
  const mo = String(d.getUTCMonth() + 1).padStart(2, '0');
  const day = String(d.getUTCDate()).padStart(2, '0');
  const h = String(d.getUTCHours()).padStart(2, '0');
  const m = String(d.getUTCMinutes()).padStart(2, '0');
  const s = String(d.getUTCSeconds()).padStart(2, '0');
  return `${y}-${mo}-${day} // ${h}:${m}:${s}Z`;
}

export default function OpsTopBar({ onSwitchRole }: OpsTopBarProps) {
  const { connected } = useEntityStream();
  const [now, setNow] = useState<Date>(new Date());

  useEffect(() => {
    const t = setInterval(() => setNow(new Date()), 1000);
    return () => clearInterval(t);
  }, []);

  const statusPills = [
    { label: 'FABRIC', ok: true },
    { label: 'INFERENCE', ok: true },
    { label: 'MESH', ok: true },
  ];

  return (
    <div
      className="flex-none flex items-center px-4 relative"
      style={{
        height: '40px',
        background: '#0D1210',
        borderBottom: '1px solid rgba(0,255,156,0.15)',
      }}
    >
      {/* Left */}
      <div className="flex items-center gap-3 z-10">
        <span
          className="text-sm font-bold tracking-widest"
          style={{ fontFamily: 'var(--font-orbitron), Orbitron, sans-serif', color: '#00FF9C' }}
        >
          SUMMIT.OS
        </span>
        <span style={{ color: 'rgba(0,255,156,0.3)' }}>|</span>
        <span
          className="text-xs italic"
          style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: 'rgba(200,230,201,0.35)' }}
        >
          NO ACTIVE MISSION
        </span>
      </div>

      {/* Center — absolute */}
      <div
        className="absolute left-1/2 -translate-x-1/2 text-xs"
        style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: 'rgba(200,230,201,0.45)' }}
      >
        {utcString(now)}
      </div>

      {/* Right */}
      <div className="ml-auto flex items-center gap-3 z-10">
        {/* Status pills */}
        {statusPills.map((p) => (
          <div key={p.label} className="flex items-center gap-1.5">
            <div
              className="w-1.5 h-1.5 rounded-full"
              style={{ background: p.ok ? '#00FF9C' : '#FF3B3B' }}
            />
            <span
              className="text-[10px] tracking-widest"
              style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: 'rgba(200,230,201,0.45)' }}
            >
              {p.label}
            </span>
          </div>
        ))}

        {/* WS indicator */}
        <div className="flex items-center gap-1.5">
          <div
            className="w-1.5 h-1.5 rounded-full"
            style={{
              background: connected ? '#00FF9C' : '#FF3B3B',
              animation: connected ? 'none' : 'blink 1s infinite',
            }}
          />
          <span
            className="text-[10px] tracking-widest"
            style={{
              fontFamily: 'var(--font-ibm-plex-mono), monospace',
              color: connected ? '#00FF9C' : '#FF3B3B',
            }}
          >
            {connected ? 'WS LIVE' : 'WS DOWN'}
          </span>
        </div>

        {/* Role switch button */}
        <button
          onClick={onSwitchRole}
          className="text-[10px] tracking-widest px-2 py-0.5 transition-colors"
          style={{
            fontFamily: 'var(--font-ibm-plex-mono), monospace',
            color: 'rgba(200,230,201,0.45)',
            border: '1px solid rgba(0,255,156,0.15)',
            background: 'transparent',
          }}
          onMouseEnter={(e) => {
            (e.currentTarget as HTMLButtonElement).style.color = '#00FF9C';
            (e.currentTarget as HTMLButtonElement).style.borderColor = 'rgba(0,255,156,0.5)';
          }}
          onMouseLeave={(e) => {
            (e.currentTarget as HTMLButtonElement).style.color = 'rgba(200,230,201,0.45)';
            (e.currentTarget as HTMLButtonElement).style.borderColor = 'rgba(0,255,156,0.15)';
          }}
        >
          ⊕ ROLE
        </button>
      </div>
    </div>
  );
}
