'use client';

import React from 'react';

interface ActionButtonProps {
  label: string;
  onClick: () => void;
  /** Color variant. Default = accent (green). */
  variant?: 'accent' | 'critical' | 'warning' | 'dim' | 'primary';
  disabled?: boolean;
  /** Fill the button solid (like ASSIGN TASK). Default = outline/ghost. */
  fill?: boolean;
  className?: string;
  'aria-label'?: string;
}

const VARIANT_CSS: Record<string, { color: string; border: string; hoverBg: string }> = {
  accent:  { color: 'var(--accent)',   border: 'var(--accent-30)',                    hoverBg: 'var(--accent-5)' },
  critical:{ color: 'var(--critical)', border: 'rgba(255,59,59,0.4)',                 hoverBg: 'rgba(255,59,59,0.08)' },
  warning: { color: 'var(--warning)',  border: 'rgba(255,179,0,0.4)',                 hoverBg: 'rgba(255,179,0,0.08)' },
  dim:     { color: 'var(--text-dim)', border: 'var(--border)',                       hoverBg: 'rgba(200,230,201,0.05)' },
  primary: { color: 'var(--background)', border: 'transparent',                       hoverBg: 'var(--accent-dim)' },
};

export default function ActionButton({
  label,
  onClick,
  variant = 'accent',
  disabled = false,
  fill = false,
  className = '',
  ...rest
}: ActionButtonProps) {
  const v = VARIANT_CSS[variant] || VARIANT_CSS.accent;
  const isPrimary = variant === 'primary' || fill;

  return (
    <button
      onClick={onClick}
      disabled={disabled}
      aria-label={rest['aria-label'] || label}
      className={`summit-btn text-[10px] tracking-widest ${className}`}
      style={{
        fontFamily: isPrimary ? 'var(--font-orbitron), Orbitron, sans-serif' : 'var(--font-ibm-plex-mono), monospace',
        fontWeight: isPrimary ? 700 : 400,
        letterSpacing: isPrimary ? '0.2em' : '0.15em',
        color: isPrimary ? 'var(--background)' : v.color,
        background: isPrimary ? 'var(--accent)' : 'transparent',
        border: isPrimary ? 'none' : `1px solid ${v.border}`,
        cursor: disabled ? 'default' : 'pointer',
        opacity: disabled ? 0.5 : 1,
        width: '100%',
        padding: isPrimary ? '10px 0' : '6px 0',
      }}
    >
      {label}
    </button>
  );
}
