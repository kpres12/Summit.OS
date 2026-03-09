'use client';

import React, { useState, useEffect, useRef } from 'react';
import { useEntityStream } from '@/hooks/useEntityStream';
import { useAuth } from '@/components/AuthProvider';
import { highestRole, roleLabel } from '@/lib/rbac';
import SecuritySettings from '@/components/auth/SecuritySettings';

interface OpsTopBarProps {
  onSwitchRole: () => void;
}

function utcString(d: Date): string {
  const y   = d.getUTCFullYear();
  const mo  = String(d.getUTCMonth() + 1).padStart(2, '0');
  const day = String(d.getUTCDate()).padStart(2, '0');
  const h   = String(d.getUTCHours()).padStart(2, '0');
  const m   = String(d.getUTCMinutes()).padStart(2, '0');
  const s   = String(d.getUTCSeconds()).padStart(2, '0');
  return `${y}-${mo}-${day} // ${h}:${m}:${s}Z`;
}

export default function OpsTopBar({ onSwitchRole }: OpsTopBarProps) {
  const { connected }             = useEntityStream();
  const { user, logout }          = useAuth();
  const [now, setNow]             = useState<Date>(new Date());
  const [menuOpen, setMenuOpen]   = useState(false);
  const [showSecurity, setShowSecurity] = useState(false);
  const menuRef                   = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const t = setInterval(() => setNow(new Date()), 1000);
    return () => clearInterval(t);
  }, []);

  // Close menu on outside click
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        setMenuOpen(false);
      }
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, []);

  const topRole = highestRole(user?.roles ?? []);

  const statusPills = [
    { label: 'FABRIC',    ok: true },
    { label: 'INFERENCE', ok: true },
    { label: 'MESH',      ok: true },
  ];

  return (
    <>
      <div
        className="flex-none flex items-center px-4 relative"
        style={{
          height:       '40px',
          background:   '#0D1210',
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
                animation:  connected ? 'none' : 'blink 1s infinite',
              }}
            />
            <span
              className="text-[10px] tracking-widest"
              style={{
                fontFamily: 'var(--font-ibm-plex-mono), monospace',
                color:      connected ? '#00FF9C' : '#FF3B3B',
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
              color:      'rgba(200,230,201,0.45)',
              border:     '1px solid rgba(0,255,156,0.15)',
              background: 'transparent',
            }}
            onMouseEnter={(e) => {
              (e.currentTarget as HTMLButtonElement).style.color       = '#00FF9C';
              (e.currentTarget as HTMLButtonElement).style.borderColor = 'rgba(0,255,156,0.5)';
            }}
            onMouseLeave={(e) => {
              (e.currentTarget as HTMLButtonElement).style.color       = 'rgba(200,230,201,0.45)';
              (e.currentTarget as HTMLButtonElement).style.borderColor = 'rgba(0,255,156,0.15)';
            }}
          >
            ⊕ ROLE
          </button>

          {/* User menu */}
          <div ref={menuRef} style={{ position: 'relative' }}>
            <button
              onClick={() => setMenuOpen((o) => !o)}
              style={{
                display:       'flex',
                alignItems:    'center',
                gap:           '6px',
                padding:       '3px 8px',
                background:    menuOpen ? 'rgba(0,255,156,0.08)' : 'transparent',
                border:        `1px solid ${menuOpen ? 'rgba(0,255,156,0.4)' : 'rgba(0,255,156,0.15)'}`,
                cursor:        'pointer',
                transition:    'all 150ms',
              }}
            >
              <span style={{
                width:        '18px',
                height:       '18px',
                borderRadius: '50%',
                background:   'rgba(0,255,156,0.15)',
                border:       '1px solid rgba(0,255,156,0.3)',
                display:      'flex',
                alignItems:   'center',
                justifyContent: 'center',
                fontSize:     '8px',
                color:        '#00FF9C',
                fontFamily:   'var(--font-orbitron), Orbitron, sans-serif',
                flexShrink:   0,
              }}>
                {user?.name?.[0]?.toUpperCase() ?? '?'}
              </span>
              {topRole && (
                <span style={{
                  fontFamily:    'var(--font-ibm-plex-mono), monospace',
                  fontSize:      '9px',
                  color:         'rgba(200,230,201,0.5)',
                  letterSpacing: '0.1em',
                }}>
                  {topRole}
                </span>
              )}
              <span style={{ color: 'rgba(0,255,156,0.4)', fontSize: '8px' }}>▾</span>
            </button>

            {/* Dropdown */}
            {menuOpen && (
              <div style={{
                position:   'absolute',
                top:        'calc(100% + 4px)',
                right:      0,
                width:      '220px',
                background: '#0D1210',
                border:     '1px solid rgba(0,255,156,0.2)',
                boxShadow:  '0 8px 32px rgba(0,0,0,0.6)',
                zIndex:     100,
              }}>
                {/* User info */}
                <div style={{
                  padding:      '12px 14px',
                  borderBottom: '1px solid rgba(0,255,156,0.08)',
                }}>
                  <div style={{
                    fontFamily: 'var(--font-ibm-plex-mono), monospace',
                    fontSize:   '10px',
                    color:      'rgba(200,230,201,0.7)',
                    marginBottom: '2px',
                    overflow:   'hidden',
                    textOverflow: 'ellipsis',
                    whiteSpace: 'nowrap',
                  }}>
                    {user?.email ?? '—'}
                  </div>
                  {topRole && (
                    <div style={{
                      fontFamily:    'var(--font-ibm-plex-mono), monospace',
                      fontSize:      '8px',
                      color:         '#00FF9C',
                      letterSpacing: '0.1em',
                    }}>
                      {roleLabel(topRole)}
                    </div>
                  )}
                </div>

                {/* Menu items */}
                <MenuItem
                  icon="🔒"
                  label="SECURITY SETTINGS"
                  onClick={() => { setMenuOpen(false); setShowSecurity(true); }}
                />
                <div style={{ borderTop: '1px solid rgba(0,255,156,0.06)', margin: '4px 0' }} />
                <MenuItem
                  icon="⏻"
                  label="SIGN OUT"
                  danger
                  onClick={() => { setMenuOpen(false); logout(); }}
                />
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Security settings modal */}
      {showSecurity && (
        <div
          className="fixed inset-0 flex items-center justify-center"
          style={{ background: 'rgba(0,0,0,0.7)', zIndex: 200 }}
          onClick={(e) => { if (e.target === e.currentTarget) setShowSecurity(false); }}
        >
          <div style={{ position: 'relative' }}>
            <button
              onClick={() => setShowSecurity(false)}
              style={{
                position:   'absolute',
                top:        '-32px',
                right:      0,
                background: 'transparent',
                border:     'none',
                color:      'rgba(200,230,201,0.4)',
                fontFamily: 'var(--font-ibm-plex-mono), monospace',
                fontSize:   '10px',
                cursor:     'pointer',
                letterSpacing: '0.1em',
              }}
            >
              ✕ CLOSE
            </button>
            <SecuritySettings />
          </div>
        </div>
      )}
    </>
  );
}

function MenuItem({ icon, label, onClick, danger = false }: {
  icon: string;
  label: string;
  onClick: () => void;
  danger?: boolean;
}) {
  const [hovered, setHovered] = useState(false);
  return (
    <button
      onClick={onClick}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        width:      '100%',
        display:    'flex',
        alignItems: 'center',
        gap:        '10px',
        padding:    '9px 14px',
        background: hovered ? (danger ? 'rgba(255,59,59,0.06)' : 'rgba(0,255,156,0.04)') : 'transparent',
        border:     'none',
        color:      danger
          ? (hovered ? '#FF3B3B' : 'rgba(255,59,59,0.6)')
          : (hovered ? '#00FF9C' : 'rgba(200,230,201,0.5)'),
        fontFamily:    'var(--font-ibm-plex-mono), monospace',
        fontSize:      '9px',
        letterSpacing: '0.15em',
        cursor:        'pointer',
        textAlign:     'left',
        transition:    'all 120ms',
      }}
    >
      <span style={{ fontSize: '11px' }}>{icon}</span>
      {label}
    </button>
  );
}
