'use client';

import React from 'react';
import { Role } from '@/hooks/useRole';
import { useAuth } from '@/components/AuthProvider';
import { allowedViews, highestRole, roleLabel } from '@/lib/rbac';

interface RoleCard {
  role: Role;
  label: string;
  subtitle: string;
  description: string;
  icon: string;
}

const ALL_ROLES: RoleCard[] = [
  {
    role: 'ops',
    label: 'OPS',
    subtitle: 'Field Coordinator',
    description: 'Full-screen map. Live entity tracking. Task assignment.',
    icon: '◉',
  },
  {
    role: 'command',
    label: 'COMMAND',
    subtitle: 'Operations Lead',
    description: 'Operations overview. Resource management. Mission approvals.',
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
  const { user } = useAuth();
  const roles     = user?.roles ?? [];
  const allowed   = allowedViews(roles);
  const topRole   = highestRole(roles);

  // Filter cards to only those the user has access to
  const visibleCards = ALL_ROLES.filter((r) => allowed.includes(r.role));

  // If only one view is allowed, auto-select it
  React.useEffect(() => {
    if (visibleCards.length === 1) onSelect(visibleCards[0].role);
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div
      className="fixed inset-0 flex flex-col items-center justify-center"
      style={{ background: 'var(--background)' }}
    >
      {/* Wordmark */}
      <div className="mb-2 text-center">
        <h1
          className="text-3xl font-bold tracking-widest"
          style={{
            fontFamily: 'var(--font-orbitron), Orbitron, sans-serif',
            color: 'var(--accent)',
            textShadow: '0 0 20px var(--accent-30), 0 0 40px var(--accent-15)',
          }}
        >
          SUMMIT.OS
        </h1>
        <div
          className="mt-1 text-xs tracking-[0.3em]"
          style={{ color: 'var(--text-dim)', fontFamily: 'var(--font-ibm-plex-mono), monospace' }}
        >
          SELECT OPERATOR MODE
        </div>
      </div>

      {/* Separator */}
      <div
        className="w-64 my-8"
        style={{ height: '1px', background: 'var(--accent-15)' }}
      />

      {/* Role cards — only what this user can access */}
      {visibleCards.length > 0 ? (
        <div className="flex gap-6">
          {visibleCards.map((r) => (
            <RoleCardButton key={r.role} card={r} onSelect={onSelect} />
          ))}
        </div>
      ) : (
        <div style={{
          padding:    '24px 32px',
          background: 'color-mix(in srgb, var(--critical) 6%, transparent)',
          border:     '1px solid color-mix(in srgb, var(--critical) 20%, transparent)',
          fontFamily: 'var(--font-ibm-plex-mono), monospace',
          fontSize:   '11px',
          color:      'var(--critical)',
          textAlign:  'center',
        }}>
          ACCESS DENIED — NO ROLES ASSIGNED<br />
          <span style={{ color: 'var(--text-muted)', fontSize: '9px', marginTop: '6px', display: 'block' }}>
            Contact your administrator to be assigned a role.
          </span>
        </div>
      )}

      {/* User identity + role badge */}
      {user && (
        <div style={{
          marginTop:  '32px',
          display:    'flex',
          alignItems: 'center',
          gap:        '10px',
          fontFamily: 'var(--font-ibm-plex-mono), monospace',
          fontSize:   '9px',
          color:      'var(--text-muted)',
        }}>
          <span>{user.email}</span>
          {topRole && (
            <span style={{
              padding:       '2px 8px',
              background:    'var(--accent-5)',
              border:        '1px solid var(--accent-15)',
              color:         'var(--accent)',
              letterSpacing: '0.1em',
            }}>
              {roleLabel(topRole)}
            </span>
          )}
        </div>
      )}

      {/* Footer */}
      <div
        className="mt-8 text-[10px] tracking-widest"
        style={{ color: 'var(--text-muted)', fontFamily: 'var(--font-ibm-plex-mono), monospace' }}
      >
        AUTONOMOUS SYSTEMS COORDINATION PLATFORM
      </div>
    </div>
  );
}

function RoleCardButton({ card, onSelect }: { card: RoleCard; onSelect: (r: Role) => void }) {
  return (
    <button
      onClick={() => onSelect(card.role)}
      className="role-card flex flex-col items-center text-left cursor-pointer"
      style={{ width: '200px', padding: '24px 20px', outline: 'none' }}
    >
      {/* Icon */}
      <div
        className="role-card-icon text-3xl mb-4"
        style={{ color: 'var(--text-muted)' }}
      >
        {card.icon}
      </div>

      {/* Role label */}
      <div
        className="role-card-label text-lg font-bold tracking-widest mb-1"
        style={{
          fontFamily: 'var(--font-orbitron), Orbitron, sans-serif',
          color: 'var(--text-dim)',
        }}
      >
        {card.label}
      </div>

      {/* Subtitle */}
      <div
        className="role-card-subtitle text-[11px] tracking-wider mb-4"
        style={{
          fontFamily: 'var(--font-orbitron), Orbitron, sans-serif',
          color: 'var(--text-muted)',
        }}
      >
        {card.subtitle}
      </div>

      {/* Separator */}
      <div
        className="role-card-divider w-full mb-4"
        style={{ height: '1px', background: 'var(--accent-10)' }}
      />

      {/* Description */}
      <div
        className="text-[11px] leading-relaxed text-center"
        style={{
          fontFamily: 'var(--font-ibm-plex-mono), monospace',
          color: 'var(--text-dim)',
        }}
      >
        {card.description}
      </div>
    </button>
  );
}
