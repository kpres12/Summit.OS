'use client';

import React, { useRef, useCallback } from 'react';

type PanelId = 'alerts' | 'entities' | 'missions' | 'layers' | 'hardware' | 'system' | 'mission-builder';

const NAV_ITEMS: { id: PanelId; icon: string; label: string }[] = [
  { id: 'alerts', icon: '⚠', label: 'Alerts' },
  { id: 'entities', icon: '◉', label: 'Entities' },
  { id: 'missions', icon: '⬡', label: 'Missions' },
  { id: 'layers', icon: '◫', label: 'Layers' },
  { id: 'hardware', icon: '⊕', label: 'Hardware' },
  { id: 'system', icon: '⚙', label: 'System' },
];

const PLAN_MISSION_ITEM = { id: 'mission-builder' as PanelId, icon: '✛', label: 'Plan Mission' };

interface OpsNavRailProps {
  activePanel: string | null;
  onSelect: (panel: string | null) => void;
}

export default function OpsNavRail({ activePanel, onSelect }: OpsNavRailProps) {
  const listRef = useRef<HTMLDivElement>(null);

  // Arrow-key navigation within the nav rail
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
      {/* Nav buttons */}
      <div ref={listRef} className="flex flex-col pt-2 flex-1" onKeyDown={handleKeyDown}>
        {NAV_ITEMS.map((item) => {
          const isActive = activePanel === item.id;
          return (
            <button
              key={item.id}
              data-nav
              onClick={() => onSelect(isActive ? null : item.id)}
              aria-label={item.label}
              aria-pressed={isActive}
              className="summit-btn relative flex items-center justify-center"
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
          );
        })}
      </div>

      {/* Plan Mission — primary action button, pinned above OPS label */}
      <div className="flex-none px-1.5 pb-2">
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
      </div>

      {/* Bottom label */}
      <div
        className="flex items-center justify-center pb-3"
        aria-hidden="true"
        style={{
          fontFamily: 'var(--font-orbitron), Orbitron, sans-serif',
          color: 'var(--accent-30)',
          fontSize: '8px',
          letterSpacing: '0.1em',
          writingMode: 'vertical-rl',
          transform: 'rotate(180deg)',
        }}
      >
        OPS
      </div>
    </nav>
  );
}
