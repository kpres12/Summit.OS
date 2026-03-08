'use client';

import React from 'react';
import { Role } from '@/hooks/useRole';

interface RoleCard {
  role: Role;
  label: string;
  subtitle: string;
  description: string;
  icon: string;
}

const ROLES: RoleCard[] = [
  {
    role: 'ops',
    label: 'OPS',
    subtitle: 'Field Operator',
    description: 'Full-screen map. Asset monitoring. Mission dispatch.',
    icon: '◉',
  },
  {
    role: 'command',
    label: 'COMMAND',
    subtitle: 'Mission Commander',
    description: 'Situation overview. Resource management. Approval authority.',
    icon: '⬡',
  },
  {
    role: 'dev',
    label: 'DEV',
    subtitle: 'Integration Developer',
    description: 'Entity explorer. Adapter registry. Message inspector.',
    icon: '⚙',
  },
];

interface RolePickerProps {
  onSelect: (role: Role) => void;
}

export default function RolePicker({ onSelect }: RolePickerProps) {
  return (
    <div
      className="fixed inset-0 flex flex-col items-center justify-center"
      style={{ background: '#080C0A' }}
    >
      {/* Wordmark */}
      <div className="mb-2 text-center">
        <h1
          className="text-3xl font-bold tracking-widest"
          style={{
            fontFamily: 'var(--font-orbitron), Orbitron, sans-serif',
            color: '#00FF9C',
            textShadow: '0 0 20px rgba(0,255,156,0.4), 0 0 40px rgba(0,255,156,0.2)',
          }}
        >
          SUMMIT.OS
        </h1>
        <div
          className="mt-1 text-xs tracking-[0.3em]"
          style={{ color: 'rgba(200,230,201,0.45)', fontFamily: 'var(--font-ibm-plex-mono), monospace' }}
        >
          SELECT OPERATOR MODE
        </div>
      </div>

      {/* Separator */}
      <div
        className="w-64 my-8"
        style={{ height: '1px', background: 'rgba(0,255,156,0.15)' }}
      />

      {/* Role cards */}
      <div className="flex gap-6">
        {ROLES.map((r) => (
          <RoleCardButton key={r.role} card={r} onSelect={onSelect} />
        ))}
      </div>

      {/* Footer */}
      <div
        className="mt-12 text-[10px] tracking-widest"
        style={{ color: 'rgba(200,230,201,0.25)', fontFamily: 'var(--font-ibm-plex-mono), monospace' }}
      >
        AUTONOMOUS SYSTEMS COORDINATION PLATFORM
      </div>
    </div>
  );
}

function RoleCardButton({ card, onSelect }: { card: RoleCard; onSelect: (r: Role) => void }) {
  const [hovered, setHovered] = React.useState(false);

  return (
    <button
      onClick={() => onSelect(card.role)}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      className="flex flex-col items-center text-left transition-all duration-150 cursor-pointer"
      style={{
        width: '200px',
        padding: '24px 20px',
        background: hovered ? 'rgba(0,255,156,0.05)' : '#0D1210',
        border: `1px solid ${hovered ? 'rgba(0,255,156,0.6)' : 'rgba(0,255,156,0.15)'}`,
        boxShadow: hovered ? '0 0 20px rgba(0,255,156,0.15), inset 0 0 20px rgba(0,255,156,0.03)' : 'none',
        outline: 'none',
      }}
    >
      {/* Icon */}
      <div
        className="text-3xl mb-4"
        style={{ color: hovered ? '#00FF9C' : 'rgba(200,230,201,0.3)' }}
      >
        {card.icon}
      </div>

      {/* Role label */}
      <div
        className="text-lg font-bold tracking-widest mb-1"
        style={{
          fontFamily: 'var(--font-orbitron), Orbitron, sans-serif',
          color: hovered ? '#00FF9C' : 'rgba(200,230,201,0.7)',
        }}
      >
        {card.label}
      </div>

      {/* Subtitle */}
      <div
        className="text-[11px] tracking-wider mb-4"
        style={{
          fontFamily: 'var(--font-orbitron), Orbitron, sans-serif',
          color: hovered ? 'rgba(0,255,156,0.7)' : 'rgba(200,230,201,0.35)',
        }}
      >
        {card.subtitle}
      </div>

      {/* Separator */}
      <div
        className="w-full mb-4"
        style={{ height: '1px', background: hovered ? 'rgba(0,255,156,0.3)' : 'rgba(0,255,156,0.1)' }}
      />

      {/* Description */}
      <div
        className="text-[11px] leading-relaxed text-center"
        style={{
          fontFamily: 'var(--font-ibm-plex-mono), monospace',
          color: 'rgba(200,230,201,0.45)',
        }}
      >
        {card.description}
      </div>
    </button>
  );
}
