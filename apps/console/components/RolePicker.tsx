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
  /** The role that was active before opening the picker. Used to show current marker + enable back. */
  currentRole?: Role | null;
  /** If provided, shows a back button that returns to the previous view without switching. */
  onBack?: () => void;
}

export default function RolePicker({ onSelect, currentRole, onBack }: RolePickerProps) {
  const { user } = useAuth();
  const roles     = user?.roles ?? [];
  const allowed   = allowedViews(roles);
  const topRole   = highestRole(roles);

  const visibleCards = ALL_ROLES.filter((r) => allowed.includes(r.role));

  React.useEffect(() => {
    if (visibleCards.length === 1) onSelect(visibleCards[0].role);
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const isSwitching = !!onBack;

  return (
    <div
      className="fixed inset-0 flex flex-col items-center justify-center"
      style={{ background: 'var(--background)' }}
    >
      {/* Back button — only shown when switching from an active view */}
      {onBack && (
        <button
          onClick={onBack}
          className="summit-btn absolute text-[10px] tracking-widest px-3 py-1.5"
          style={{
            top: '16px',
            left: '16px',
            fontFamily: 'var(--font-ibm-plex-mono), monospace',
            color: 'var(--text-dim)',
            background: 'transparent',
            border: '1px solid var(--border)',
          }}
        >
          ← BACK
        </button>
      )}

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
          {isSwitching ? 'SWITCH VIEW' : 'SELECT OPERATOR MODE'}
        </div>
      </div>

      {/* Separator */}
      <div
        className="w-64 my-8"
        style={{ height: '1px', background: 'var(--accent-15)' }}
      />

      {/* Role cards */}
      {visibleCards.length > 0 ? (
        <div className="flex gap-6">
          {visibleCards.map((r) => (
            <RoleCardButton
              key={r.role}
              card={r}
              isCurrent={r.role === currentRole}
              onSelect={onSelect}
            />
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

      {/* User identity */}
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

function RoleCardButton({
  card,
  isCurrent,
  onSelect,
}: {
  card: RoleCard;
  isCurrent: boolean;
  onSelect: (r: Role) => void;
}) {
  return (
    <button
      onClick={() => onSelect(card.role)}
      aria-label={`${card.label} — ${card.subtitle}: ${card.description}${isCurrent ? ' (active)' : ''}`}
      aria-pressed={isCurrent}
      className="role-card flex flex-col items-center text-left cursor-pointer relative"
      style={{
        width: '200px',
        padding: '24px 20px',
        borderColor: isCurrent ? 'var(--accent-50)' : undefined,
        background: isCurrent ? 'var(--accent-5)' : undefined,
      }}
    >
      {/* Current view indicator */}
      {isCurrent && (
        <div
          className="absolute top-2 right-2 text-[8px] tracking-widest px-1.5 py-0.5"
          style={{
            fontFamily: 'var(--font-ibm-plex-mono), monospace',
            color: 'var(--accent)',
            border: '1px solid var(--accent-30)',
            background: 'var(--accent-5)',
          }}
        >
          ACTIVE
        </div>
      )}

      {/* Icon */}
      <div
        className="role-card-icon text-3xl mb-4"
        style={{ color: isCurrent ? 'var(--accent)' : 'var(--text-muted)' }}
      >
        {card.icon}
      </div>

      {/* Role label */}
      <div
        className="role-card-label text-lg font-bold tracking-widest mb-1"
        style={{
          fontFamily: 'var(--font-orbitron), Orbitron, sans-serif',
          color: isCurrent ? 'var(--accent)' : 'var(--text-dim)',
        }}
      >
        {card.label}
      </div>

      {/* Subtitle */}
      <div
        className="role-card-subtitle text-[11px] tracking-wider mb-4"
        style={{
          fontFamily: 'var(--font-orbitron), Orbitron, sans-serif',
          color: isCurrent ? 'var(--accent-50)' : 'var(--text-muted)',
        }}
      >
        {card.subtitle}
      </div>

      {/* Separator */}
      <div
        className="role-card-divider w-full mb-4"
        style={{ height: '1px', background: isCurrent ? 'var(--accent-30)' : 'var(--accent-10)' }}
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
