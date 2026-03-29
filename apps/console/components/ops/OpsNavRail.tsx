'use client';

import React from 'react';

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
  return (
    <div
      className="flex-none flex flex-col"
      style={{
        width: '48px',
        background: '#0D1210',
        borderRight: '1px solid rgba(0,255,156,0.15)',
      }}
    >
      {/* Nav buttons */}
      <div className="flex flex-col pt-2 flex-1">
        {NAV_ITEMS.map((item) => {
          const isActive = activePanel === item.id;
          return (
            <button
              key={item.id}
              onClick={() => onSelect(isActive ? null : item.id)}
              title={item.label}
              className="relative flex items-center justify-center transition-colors"
              style={{
                width: '48px',
                height: '40px',
                background: isActive ? 'rgba(0,255,156,0.08)' : 'transparent',
                color: isActive ? '#00FF9C' : 'rgba(200,230,201,0.45)',
                borderLeft: isActive ? '2px solid #00FF9C' : '2px solid transparent',
                outline: 'none',
                cursor: 'pointer',
              }}
              onMouseEnter={(e) => {
                if (!isActive) (e.currentTarget as HTMLButtonElement).style.color = '#00FF9C';
              }}
              onMouseLeave={(e) => {
                if (!isActive) (e.currentTarget as HTMLButtonElement).style.color = 'rgba(200,230,201,0.45)';
              }}
            >
              <span className="text-base leading-none">{item.icon}</span>
            </button>
          );
        })}
      </div>

      {/* Plan Mission — primary action button, pinned above OPS label */}
      <div className="flex-none px-1.5 pb-2">
        <button
          onClick={() => onSelect(activePanel === PLAN_MISSION_ITEM.id ? null : PLAN_MISSION_ITEM.id)}
          title={PLAN_MISSION_ITEM.label}
          className="flex items-center justify-center w-full transition-colors"
          style={{
            height: '36px',
            background: activePanel === PLAN_MISSION_ITEM.id ? 'rgba(0,255,156,0.12)' : 'rgba(0,255,156,0.06)',
            border: `1px solid ${activePanel === PLAN_MISSION_ITEM.id ? 'rgba(0,255,156,0.5)' : 'rgba(0,255,156,0.2)'}`,
            color: activePanel === PLAN_MISSION_ITEM.id ? '#00FF9C' : 'rgba(0,255,156,0.6)',
            cursor: 'pointer',
          }}
          onMouseEnter={(e) => { (e.currentTarget as HTMLButtonElement).style.color = '#00FF9C'; }}
          onMouseLeave={(e) => {
            if (activePanel !== PLAN_MISSION_ITEM.id)
              (e.currentTarget as HTMLButtonElement).style.color = 'rgba(0,255,156,0.6)';
          }}
        >
          <span className="text-base leading-none">{PLAN_MISSION_ITEM.icon}</span>
        </button>
      </div>

      {/* Bottom label */}
      <div
        className="flex items-center justify-center pb-3"
        style={{
          fontFamily: 'var(--font-orbitron), Orbitron, sans-serif',
          color: 'rgba(0,255,156,0.3)',
          fontSize: '8px',
          letterSpacing: '0.1em',
          writingMode: 'vertical-rl',
          transform: 'rotate(180deg)',
        }}
      >
        OPS
      </div>
    </div>
  );
}
