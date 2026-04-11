'use client';

import React from 'react';

interface SectionHeaderProps {
  title: string;
}

export default function SectionHeader({ title }: SectionHeaderProps) {
  return (
    <div
      className="text-[9px] font-bold tracking-[0.18em] uppercase pt-3 pb-1"
      style={{
        fontFamily: 'var(--font-ibm-plex-mono), monospace',
        color: 'var(--text-muted)',
        borderBottom: '1px solid var(--accent-10)',
        marginBottom: '4px',
      }}
    >
      {title}
    </div>
  );
}
