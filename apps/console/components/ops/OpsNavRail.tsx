'use client';

import React, { useRef, useCallback } from 'react';
import Link from 'next/link';
import { useTier } from '@/hooks/useTier';

type PanelId = 'alerts' | 'entities' | 'missions' | 'layers' | 'hardware' | 'system' | 'mission-builder' | 'intel' | 'log' | 'tasks' | 'reports';

// "entities" panel id kept for compatibility — label shown as "Assets"
const NAV_ITEMS: { id: PanelId; icon: string; label: string }[] = [
  { id: 'alerts',   icon: '⚠',  label: 'Alerts' },
  { id: 'entities', icon: '◉',  label: 'Assets' },
  { id: 'tasks',    icon: '▶',  label: 'Active Tasks' },
  { id: 'missions', icon: '⬡',  label: 'Missions' },
  { id: 'intel',    icon: '⊛',  label: 'Intel' },
  { id: 'reports',  icon: '⊟',  label: 'Reports' },
  { id: 'layers',   icon: '◫',  label: 'Layers' },
  { id: 'hardware', icon: '⊕',  label: 'Hardware' },
  { id: 'log',      icon: '◎',  label: 'Action Log' },
  { id: 'system',   icon: '⚙',  label: 'System' },
];

const PLAN_MISSION_ITEM = { id: 'mission-builder' as PanelId, icon: '✛', label: 'Plan Mission' };

interface OpsNavRailProps {
  activePanel: string | null;
  onSelect: (panel: string | null) => void;
}

const TIER_BADGE: Record<string, string> = { free: 'FREE', pro: 'PRO', org: 'ORG', enterprise: 'ENT' };

export default function OpsNavRail({ activePanel, onSelect }: OpsNavRailProps) {
  const listRef = useRef<HTMLDivElement>(null);
  const { tier } = useTier();

  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    const buttons = listRef.current?.querySelectorAll<HTMLButtonElement>('button[data-nav]');
    if (!buttons?.length) return;
    const idx = Array.from(buttons).findIndex((b) => b === document.activeElement);
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      buttons[(idx + 1) % buttons.length].focus();
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      buttons[(idx - 1 + buttons.length) % buttons.length].focus();
    }
  }, []);

  return (
    <nav
      role="navigation"
      aria-label="Operations panels"
      className="flex-none flex flex-col"
      style={{
        width: '48px',
        background: 'var(--background-panel)',
        borderRight: '1px solid var(--border)',
      }}
    >
      <div ref={listRef} className="flex flex-col pt-2 flex-1" onKeyDown={handleKeyDown}>
        {NAV_ITEMS.map((item) => {
          const isActive = activePanel === item.id;
          return (
            /* Tooltip wrapper — label appears to the right on hover */
            <div key={item.id} className="relative group">
              <button
                data-nav
                onClick={() => onSelect(isActive ? null : item.id)}
                aria-label={item.label}
                aria-pressed={isActive}
                className="summit-btn flex items-center justify-center"
                style={{
                  width: '48px',
                  height: '40px',
                  background: isActive ? 'var(--accent-5)' : 'transparent',
                  color: isActive ? 'var(--accent)' : 'var(--text-dim)',
                  borderLeft: isActive ? '2px solid var(--accent)' : '2px solid transparent',
                }}
              >
                <span className="text-base leading-none" aria-hidden="true">{item.icon}</span>
              </button>
              {/* Tooltip — visible on hover, positioned to the right of the rail */}
              <div
                className="pointer-events-none absolute left-full top-1/2 ml-2 px-2 py-1 opacity-0 group-hover:opacity-100 transition-opacity z-50 whitespace-nowrap"
                style={{
                  transform: 'translateY(-50%)',
                  background: 'var(--background-panel)',
                  border: '1px solid var(--border)',
                  fontFamily: 'var(--font-ibm-plex-mono), monospace',
                  fontSize: '10px',
                  letterSpacing: '0.1em',
                  color: 'var(--accent)',
                  pointerEvents: 'none',
                }}
                aria-hidden="true"
              >
                {item.label.toUpperCase()}
              </div>
            </div>
          );
        })}
      </div>

      {/* Plan Mission — primary action, pinned above tier badge */}
      <div className="flex-none px-1.5 pb-2 relative group">
        <button
          data-nav
          onClick={() => onSelect(activePanel === PLAN_MISSION_ITEM.id ? null : PLAN_MISSION_ITEM.id)}
          aria-label={PLAN_MISSION_ITEM.label}
          aria-pressed={activePanel === PLAN_MISSION_ITEM.id}
          className="summit-btn flex items-center justify-center w-full"
          style={{
            height: '36px',
            background: activePanel === PLAN_MISSION_ITEM.id ? 'var(--accent-10)' : 'var(--accent-5)',
            border: `1px solid ${activePanel === PLAN_MISSION_ITEM.id ? 'var(--accent-50)' : 'var(--accent-15)'}`,
            color: activePanel === PLAN_MISSION_ITEM.id ? 'var(--accent)' : 'var(--accent-50)',
          }}
        >
          <span className="text-base leading-none" aria-hidden="true">{PLAN_MISSION_ITEM.icon}</span>
        </button>
        <div
          className="pointer-events-none absolute left-full top-1/2 ml-2 px-2 py-1 opacity-0 group-hover:opacity-100 transition-opacity z-50 whitespace-nowrap"
          style={{
            transform: 'translateY(-50%)',
            background: 'var(--background-panel)',
            border: '1px solid var(--accent-30)',
            fontFamily: 'var(--font-ibm-plex-mono), monospace',
            fontSize: '10px',
            letterSpacing: '0.1em',
            color: 'var(--accent)',
          }}
          aria-hidden="true"
        >
          PLAN MISSION
        </div>
      </div>

      {/* Billing / plan link */}
      <Link
        href="/billing"
        aria-label={`Plan: ${TIER_BADGE[tier] ?? 'FREE'} — upgrade`}
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          height: '32px',
          margin: '0 4px 4px',
          borderRadius: '3px',
          background: tier === 'free' ? 'var(--accent-5)' : 'transparent',
          border: tier === 'free' ? '1px solid var(--accent-15)' : '1px solid transparent',
          color: tier === 'free' ? 'var(--accent-50)' : 'var(--text-dim)',
          fontSize: '8px',
          fontFamily: 'var(--font-ibm-plex-mono), monospace',
          letterSpacing: '0.12em',
          textDecoration: 'none',
          transition: 'color 0.15s',
        }}
      >
        {TIER_BADGE[tier] ?? 'FREE'}
      </Link>

      <div
        className="flex items-center justify-center pb-3"
        aria-hidden="true"
        style={{
          fontFamily: 'var(--font-ibm-plex-mono), monospace',
          color: 'var(--text-muted)',
          fontSize: '8px',
          letterSpacing: '0.15em',
          writingMode: 'vertical-rl',
          transform: 'rotate(180deg)',
        }}
      >
        OPS
      </div>
    </nav>
  );
}
