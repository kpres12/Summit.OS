'use client';

import React, { useState, useEffect, useRef } from 'react';
import { useEntityStream } from '@/hooks/useEntityStream';
import { useAuth } from '@/components/AuthProvider';
import { useDomain } from '@/components/DomainProvider';
import { highestRole, roleLabel } from '@/lib/rbac';
import StatusDot from '@/components/ui/StatusDot';
import SecuritySettings from '@/components/auth/SecuritySettings';

interface OpsTopBarProps {
  onSwitchRole: () => void;
  /** Optional mission name displayed in the header */
  missionName?: string;
  /** Number of active assets */
  assetCount?: number;
  /** Number of active alerts */
  alertCount?: number;
  /** Link quality 0–1; shown as colored bar (red <0.3, amber 0.3–0.7, green >0.7) */
  linkQuality?: number;
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

export default function OpsTopBar({ onSwitchRole, missionName, assetCount, alertCount, linkQuality }: OpsTopBarProps) {
  const { connected }             = useEntityStream();
  const { user, logout }          = useAuth();
  const { config, setDomain, domains } = useDomain();
  const [now, setNow]             = useState<Date>(new Date());
  const [menuOpen, setMenuOpen]   = useState(false);
  const [showSecurity, setShowSecurity] = useState(false);
  const [showDomainPicker, setShowDomainPicker] = useState(false);
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
      <header
        role="banner"
        className="flex-none flex items-center px-4 relative crt-band"
        style={{
          height:       '40px',
          background:   'var(--background-panel)',
          borderBottom: '1px solid var(--border)',
        }}
      >
        {/* Left */}
        <div className="flex items-center gap-3 z-10">
          <span
            className="text-sm font-bold tracking-widest"
            style={{ fontFamily: 'var(--font-orbitron), Orbitron, sans-serif', color: 'var(--accent)' }}
          >
            SUMMIT.OS
          </span>
          <span style={{ color: 'var(--accent-30)' }}>|</span>
          <button
            onClick={() => setShowDomainPicker(true)}
            className="summit-btn text-xs italic"
            aria-label="Change domain"
            style={{
              fontFamily: 'var(--font-ibm-plex-mono), monospace',
              color: 'var(--text-dim)',
              background: 'none',
              border: 'none',
            }}
          >
            {config.name.toUpperCase()}
          </button>

          {/* Optional mission context pills */}
          {missionName && (
            <>
              <span style={{ color: 'var(--accent-30)' }}>|</span>
              <span
                className="text-[10px] tracking-widest"
                style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: 'var(--text-dim)' }}
              >
                {missionName.toUpperCase()}
              </span>
            </>
          )}
          {assetCount !== undefined && (
            <span
              className="text-[10px] tracking-widest"
              style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: 'var(--accent)' }}
              title="Active assets"
            >
              {assetCount} ASSETS
            </span>
          )}
          {alertCount !== undefined && (
            <span
              className="text-[10px] tracking-widest"
              style={{
                fontFamily: 'var(--font-ibm-plex-mono), monospace',
                color: alertCount > 0 ? 'var(--critical)' : 'var(--text-dim)',
              }}
              title="Active alerts"
            >
              {alertCount} ALERTS
            </span>
          )}
          {linkQuality !== undefined && (
            <div
              className="flex items-center gap-1"
              title={`Link quality: ${Math.round(linkQuality * 100)}%`}
            >
              <span
                className="text-[9px] tracking-widest"
                style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: 'var(--text-muted)' }}
              >
                LINK
              </span>
              <div
                style={{
                  width: '32px',
                  height: '6px',
                  background: 'var(--border)',
                  position: 'relative',
                  overflow: 'hidden',
                }}
              >
                <div
                  style={{
                    position: 'absolute',
                    left: 0,
                    top: 0,
                    bottom: 0,
                    width: `${Math.round(Math.max(0, Math.min(1, linkQuality)) * 100)}%`,
                    background: linkQuality < 0.3
                      ? 'var(--critical)'
                      : linkQuality < 0.7
                        ? 'var(--warning)'
                        : 'var(--accent)',
                    transition: 'width 500ms ease',
                  }}
                />
              </div>
            </div>
          )}
        </div>

        {/* Center — absolute */}
        <div
          className="absolute left-1/2 -translate-x-1/2 text-xs"
          style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: 'var(--text-dim)' }}
        >
          {utcString(now)}
        </div>

        {/* Right */}
        <div className="ml-auto flex items-center gap-3 z-10">
          {/* Status pills */}
          {statusPills.map((p) => (
            <div key={p.label} className="flex items-center gap-1.5">
              <StatusDot variant={p.ok ? 'accent' : 'critical'} glow={p.ok} />
              <span
                className="text-[10px] tracking-widest"
                style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: 'var(--text-dim)' }}
              >
                {p.label}
              </span>
            </div>
          ))}

          {/* WS indicator */}
          <div className="flex items-center gap-1.5">
            <StatusDot
              variant={connected ? 'accent' : 'critical'}
              glow={connected}
            />
            <span
              className="text-[10px] tracking-widest"
              style={{
                fontFamily: 'var(--font-ibm-plex-mono), monospace',
                color:      connected ? 'var(--accent)' : 'var(--critical)',
              }}
            >
              {connected ? 'WS LIVE' : 'WS DOWN'}
            </span>
          </div>

          {/* Role switch button */}
          <button
            onClick={onSwitchRole}
            aria-label="Switch view"
            className="summit-btn text-[10px] tracking-widest px-2 py-0.5"
            style={{
              fontFamily: 'var(--font-ibm-plex-mono), monospace',
              color:      'var(--text-dim)',
              border:     '1px solid var(--border)',
              background: 'transparent',
            }}
          >
            SWITCH VIEW
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
                background:    menuOpen ? 'var(--accent-5)' : 'transparent',
                border:        `1px solid ${menuOpen ? 'var(--accent-30)' : 'var(--border)'}`,
                cursor:        'pointer',
                transition:    'all 150ms',
              }}
            >
              <span style={{
                width:        '18px',
                height:       '18px',
                borderRadius: '50%',
                background:   'var(--accent-15)',
                border:       '1px solid var(--accent-30)',
                display:      'flex',
                alignItems:   'center',
                justifyContent: 'center',
                fontSize:     '8px',
                color:        'var(--accent)',
                fontFamily:   'var(--font-ibm-plex-mono), monospace',
                flexShrink:   0,
              }}>
                {user?.name?.[0]?.toUpperCase() ?? '?'}
              </span>
              {topRole && (
                <span style={{
                  fontFamily:    'var(--font-ibm-plex-mono), monospace',
                  fontSize:      '9px',
                  color:         'var(--text-dim)',
                  letterSpacing: '0.1em',
                }}>
                  {topRole}
                </span>
              )}
              <span style={{ color: 'var(--accent-30)', fontSize: '8px' }}>▾</span>
            </button>

            {/* Dropdown */}
            {menuOpen && (
              <div style={{
                position:   'absolute',
                top:        'calc(100% + 4px)',
                right:      0,
                width:      '220px',
                background: 'var(--background-panel)',
                border:     '1px solid var(--accent-15)',
                boxShadow:  '0 8px 32px rgba(0,0,0,0.6)',
                zIndex:     100,
              }}>
                {/* User info */}
                <div style={{
                  padding:      '12px 14px',
                  borderBottom: '1px solid var(--accent-5)',
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
                      color:         'var(--accent)',
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
                <div style={{ borderTop: '1px solid var(--accent-5)', margin: '4px 0' }} />
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
      </header>

      {/* Domain picker modal */}
      {showDomainPicker && (
        <div
          className="fixed inset-0 flex items-center justify-center"
          style={{ background: 'rgba(0,0,0,0.7)', zIndex: 200 }}
          onClick={(e) => { if (e.target === e.currentTarget) setShowDomainPicker(false); }}
        >
          <div style={{
            background: 'var(--background-panel)',
            border: '1px solid var(--border)',
            padding: '24px',
            width: '400px',
            maxWidth: '90vw',
          }}>
            <div
              className="text-[10px] font-bold tracking-[0.2em] mb-4"
              style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: 'var(--text-dim)' }}
            >
              SELECT DOMAIN
            </div>
            <div className="flex flex-col gap-2">
              {domains.map((d) => (
                <button
                  key={d.id}
                  onClick={() => { setDomain(d.id); setShowDomainPicker(false); }}
                  className="summit-btn text-left px-3 py-3"
                  style={{
                    background: config.id === d.id ? 'var(--accent-5)' : 'transparent',
                    border: `1px solid ${config.id === d.id ? 'var(--accent-50)' : 'var(--border)'}`,
                    color: config.id === d.id ? 'var(--accent)' : 'var(--text-dim)',
                  }}
                >
                  <div style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', fontSize: 11, fontWeight: 700, letterSpacing: '0.12em' }}>
                    {d.name.toUpperCase()}
                  </div>
                  <div style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', fontSize: 9, marginTop: 2, color: 'var(--text-muted)' }}>
                    {d.description}
                  </div>
                </button>
              ))}
            </div>
            <button
              onClick={() => setShowDomainPicker(false)}
              className="summit-btn w-full mt-4 text-[10px] tracking-widest py-2"
              style={{ color: 'var(--text-dim)', border: '1px solid var(--border)', background: 'transparent' }}
            >
              CANCEL
            </button>
          </div>
        </div>
      )}

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
  return (
    <button
      onClick={onClick}
      className={danger ? 'menu-item-danger' : 'menu-item'}
      style={{
        width:      '100%',
        display:    'flex',
        alignItems: 'center',
        gap:        '10px',
        padding:    '9px 14px',
        border:     'none',
        fontFamily:    'var(--font-ibm-plex-mono), monospace',
        fontSize:      '9px',
        letterSpacing: '0.15em',
        cursor:        'pointer',
        textAlign:     'left',
        background:    'transparent',
      }}
    >
      <span style={{ fontSize: '11px' }}>{icon}</span>
      {label}
    </button>
  );
}
