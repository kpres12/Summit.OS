'use client';

import React from 'react';

interface DataRowProps {
  label: string;
  value: string;
  valueColor?: string;
}

export default function DataRow({ label, value, valueColor }: DataRowProps) {
  return (
    <div className="flex items-baseline justify-between py-0.5">
      <span
        className="text-[10px]"
        style={{ color: 'var(--text-dim)', fontFamily: 'var(--font-ibm-plex-mono), monospace' }}
      >
        {label}
      </span>
      <span
        className="text-[11px] font-bold"
        style={{ color: valueColor || 'var(--accent)', fontFamily: 'var(--font-ibm-plex-mono), monospace' }}
      >
        {value}
      </span>
    </div>
  );
}
