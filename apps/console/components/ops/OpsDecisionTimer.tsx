'use client';

/**
 * Decision timer — shows elapsed time since an alert was raised.
 * Turns amber at 30s, red at 60s, pulses at 90s.
 * Used in the alert queue to indicate urgency.
 */

import React, { useEffect, useState } from 'react';

interface OpsDecisionTimerProps {
  startIso: string;
}

function elapsedSeconds(startIso: string): number {
  try {
    const start = new Date(startIso).getTime();
    return Math.max(0, Math.floor((Date.now() - start) / 1000));
  } catch {
    return 0;
  }
}

function formatMmSs(totalSeconds: number): string {
  const mm = Math.floor(totalSeconds / 60);
  const ss = totalSeconds % 60;
  return `${String(mm).padStart(2, '0')}:${String(ss).padStart(2, '0')}`;
}

export default function OpsDecisionTimer({ startIso }: OpsDecisionTimerProps): JSX.Element {
  const [elapsed, setElapsed] = useState(() => elapsedSeconds(startIso));

  useEffect(() => {
    const interval = setInterval(() => {
      setElapsed(elapsedSeconds(startIso));
    }, 1000);
    return () => clearInterval(interval);
  }, [startIso]);

  let color: string;
  if (elapsed >= 60) {
    color = 'var(--critical)';
  } else if (elapsed >= 30) {
    color = 'var(--warning)';
  } else {
    color = 'var(--accent)';
  }

  const shouldPulse = elapsed >= 90;

  return (
    <span
      className={shouldPulse ? 'alert-pulse' : undefined}
      style={{
        fontFamily: 'var(--font-ibm-plex-mono), monospace',
        fontSize: '10px',
        color,
        letterSpacing: '0.05em',
        flexShrink: 0,
      }}
      aria-label={`Alert age: ${formatMmSs(elapsed)}`}
    >
      {formatMmSs(elapsed)}
    </span>
  );
}
