'use client';

import React, { useEffect, useState } from 'react';

export default function TopOverlay() {
  const [time, setTime] = useState<Date | null>(null);

  useEffect(() => {
    setTime(new Date());
    const interval = setInterval(() => setTime(new Date()), 1000);
    return () => clearInterval(interval);
  }, []);

  const formatTime = (date: Date) => {
    return date.toISOString().replace('T', ' ').slice(0, 19) + ' UTC';
  };

  return (
    <div className="h-12 bg-zinc-950 border-b border-zinc-800 flex items-center justify-between px-5 relative z-10">
      {/* Left — Identity + Time */}
      <div className="flex items-center gap-5">
        <div className="text-white text-sm font-semibold tracking-wide">
          SUMMIT.OS
        </div>
        <div className="h-5 w-px bg-zinc-700" />
        <div className="text-xs text-zinc-400 font-mono" suppressHydrationWarning>
          {time ? formatTime(time) : '—'}
        </div>
      </div>

      {/* Right — Status Indicators */}
      <div className="flex items-center gap-5">
        <StatusIcon label="FABRIC" status="OK" color="green" />
        <StatusIcon label="AI" status="ACTIVE" color="green" />
        <StatusIcon label="LINK" status="NOMINAL" color="green" />
        <StatusIcon label="MESH" status="3 PEERS" color="green" />
      </div>
    </div>
  );
}

interface StatusIconProps {
  label: string;
  status: string;
  color: 'green' | 'amber' | 'red';
}

function StatusIcon({ label, status, color }: StatusIconProps) {
  const dotColor = {
    green: 'bg-emerald-500',
    amber: 'bg-amber-500',
    red: 'bg-red-500',
  };

  return (
    <div className="flex items-center gap-1.5">
      <div className={`w-1.5 h-1.5 rounded-full ${dotColor[color]}`} />
      <div className="text-[10px] text-zinc-500 uppercase tracking-wider">{label}</div>
      <div className="text-[10px] text-zinc-300 font-mono">{status}</div>
    </div>
  );
}
