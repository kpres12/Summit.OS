'use client';

import React from 'react';

interface SectionHeaderProps {
  title: string;
}

export default function SectionHeader({ title }: SectionHeaderProps) {
  return (
    <div
      className="text-[10px] font-bold tracking-[0.18em] uppercase"
      style={{
        fontFamily: 'var(--font-ibm-plex-mono), monospace',
        color: 'var(--text-dim)',
        borderBottom: '1px solid var(--accent-10)',
        paddingTop: '16px',
        paddingBottom: '6px',
        marginBottom: '2px',
      }}
    >
      {title}
    </div>
  );
}
