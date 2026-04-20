'use client';

import React from 'react';

interface DataRowProps {
  label: string;
  value: string;
  valueColor?: string;
}

export default function DataRow({ label, value, valueColor }: DataRowProps) {
  return (
    <div
      className="flex items-baseline py-2"
      style={{ borderBottom: '1px solid var(--accent-5)', gap: '12px' }}
    >
      <span
        style={{
          color: 'var(--text-dim)',
          fontFamily: 'var(--font-ibm-plex-mono), monospace',
          fontSize: '10px',
          letterSpacing: '0.08em',
          flexShrink: 0,
          minWidth: '72px',
        }}
      >
        {label}
      </span>
      <span
        title={value}
        style={{
          color: valueColor || 'rgba(200,230,201,0.9)',
          fontFamily: 'var(--font-ibm-plex-mono), monospace',
          fontSize: '11px',
          fontWeight: 500,
          textAlign: 'right',
          flex: 1,
          overflow: 'hidden',
          textOverflow: 'ellipsis',
          whiteSpace: 'nowrap',
          letterSpacing: '0.04em',
        }}
      >
        {value}
      </span>
    </div>
  );
}
