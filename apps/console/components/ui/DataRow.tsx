'use client';

import React from 'react';

interface DataRowProps {
  label: string;
  value: string;
  valueColor?: string;
}

export default function DataRow({ label, value, valueColor }: DataRowProps) {
  return (
    <div className="flex items-center justify-between py-1.5" style={{ borderBottom: '1px solid var(--accent-5)' }}>
      <span
        className="text-[11px]"
        style={{ color: 'var(--text-dim)', fontFamily: 'var(--font-ibm-plex-mono), monospace', flexShrink: 0 }}
      >
        {label}
      </span>
      <span
        className="text-[12px] font-medium text-right"
        style={{
          color: valueColor || 'rgba(200,230,201,0.85)',
          fontFamily: 'var(--font-ibm-plex-mono), monospace',
          marginLeft: '12px',
        }}
      >
        {value}
      </span>
    </div>
  );
}
