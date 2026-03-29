'use client';

import React from 'react';

interface ProgressBarProps {
  /** 0–100. */
  value: number;
  color?: string;
  label?: string;
  showValue?: boolean;
}

export default function ProgressBar({ value, color = 'var(--accent)', label, showValue = true }: ProgressBarProps) {
  const pct = Math.max(0, Math.min(100, value));
  return (
    <div className="mb-1">
      {(label || showValue) && (
        <div className="flex items-center justify-between mb-1">
          {label && (
            <span
              className="text-[10px]"
              style={{ color: 'var(--text-dim)', fontFamily: 'var(--font-ibm-plex-mono), monospace' }}
            >
              {label}
            </span>
          )}
          {showValue && (
            <span
              className="text-[11px] font-bold"
              style={{ color, fontFamily: 'var(--font-ibm-plex-mono), monospace' }}
            >
              {Math.round(pct)}%
            </span>
          )}
        </div>
      )}
      <div
        className="h-1.5 w-full rounded-full overflow-hidden"
        style={{ background: 'var(--accent-10)' }}
        role="progressbar"
        aria-valuenow={pct}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-label={label || 'Progress'}
      >
        <div
          className="h-full rounded-full transition-all duration-300"
          style={{ width: `${pct}%`, background: color }}
        />
      </div>
    </div>
  );
}
