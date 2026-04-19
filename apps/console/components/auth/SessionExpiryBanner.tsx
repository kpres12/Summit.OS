'use client';

import React, { useState, useEffect } from 'react';
import { useAuth } from '@/components/AuthProvider';

export default function SessionExpiryBanner() {
  const { sessionExp, refreshAuth } = useAuth();
  const [secsLeft, setSecsLeft]     = useState<number | null>(null);
  const [renewing, setRenewing]     = useState(false);

  useEffect(() => {
    if (!sessionExp) return;
    const update = () => setSecsLeft(Math.floor(sessionExp - Date.now() / 1000));
    update();
    const t = setInterval(update, 1000);
    return () => clearInterval(t);
  }, [sessionExp]);

  if (!secsLeft || secsLeft > 300) return null;

  const mins      = Math.floor(Math.max(0, secsLeft) / 60);
  const secs      = Math.max(0, secsLeft) % 60;
  const isUrgent  = secsLeft < 60;
  const color     = isUrgent ? 'var(--critical)' : 'var(--warning)';

  const handleRenew = async () => {
    setRenewing(true);
    try {
      await fetch('/api/auth/refresh', { method: 'POST', credentials: 'same-origin' });
      await refreshAuth();
    } finally {
      setRenewing(false);
    }
  };

  return (
    <div
      role="alert"
      aria-live="assertive"
      style={{
        position:        'fixed',
        top:             '40px',
        left:            '50%',
        transform:       'translateX(-50%)',
        zIndex:          300,
        display:         'flex',
        alignItems:      'center',
        gap:             '12px',
        padding:         '6px 16px',
        background:      'var(--background-panel)',
        border:          `1px solid ${color}`,
        borderTop:       `2px solid ${color}`,
        boxShadow:       '0 4px 16px rgba(0,0,0,0.5)',
        fontFamily:      'var(--font-ibm-plex-mono), monospace',
      }}
    >
      <span style={{ fontSize: '10px', color, letterSpacing: '0.1em' }}>
        SESSION EXPIRES {mins > 0 ? `${mins}m ${secs}s` : `${secs}s`}
      </span>
      <button
        onClick={handleRenew}
        disabled={renewing}
        aria-label="Renew session"
        style={{
          fontFamily:    'var(--font-ibm-plex-mono), monospace',
          fontSize:      '9px',
          letterSpacing: '0.12em',
          padding:       '3px 10px',
          background:    color,
          color:         'var(--background)',
          border:        'none',
          cursor:        renewing ? 'default' : 'pointer',
          opacity:       renewing ? 0.7 : 1,
        }}
      >
        {renewing ? 'RENEWING...' : 'RENEW'}
      </button>
    </div>
  );
}
