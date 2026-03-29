'use client';

import React from 'react';

interface EmptyStateProps {
  message: string;
  hint?: string;
}

export default function EmptyState({ message, hint }: EmptyStateProps) {
  return (
    <div className="flex flex-col items-center justify-center h-full gap-2 px-4">
      <span
        className="text-[10px] tracking-widest text-center"
        style={{ color: 'var(--text-dim)', fontFamily: 'var(--font-ibm-plex-mono), monospace' }}
      >
        {message}
      </span>
      {hint && (
        <span
          className="text-[9px] text-center leading-relaxed"
          style={{ color: 'var(--text-muted)', fontFamily: 'var(--font-ibm-plex-mono), monospace' }}
        >
          {hint}
        </span>
      )}
    </div>
  );
}
