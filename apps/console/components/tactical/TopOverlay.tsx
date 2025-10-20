'use client';

import React, { useEffect, useState } from 'react';

export default function TopOverlay() {
  // Avoid SSR/client mismatch by initializing on client after mount
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
    <div className="h-16 bg-[#0F0F0F] border-b-2 border-[#00FF91]/20 flex items-center justify-between px-6 relative z-10">
      {/* Left Section - Mission Info */}
      <div className="flex items-center gap-6">
        <div className="text-[#00FF91] text-xl font-semibold tracking-wider text-glow">
          SUMMIT.OS
        </div>
        <div className="h-8 w-px bg-[#00FF91]/30" />
        <div className="flex flex-col">
          <div className="text-[10px] text-[#006644] uppercase tracking-wider">Mission Time</div>
          <div className="text-xs text-[#00CC74] font-mono" suppressHydrationWarning>
            {time ? formatTime(time) : 'SYNC...'}
          </div>
        </div>
        <div className="flex flex-col">
          <div className="text-[10px] text-[#006644] uppercase tracking-wider">Grid Ref</div>
          <div className="text-xs text-[#00CC74] font-mono">34.0522°N 118.2437°W</div>
        </div>
      </div>

      {/* Center Section - BigMT.ai branding */}
      <div className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2">
        <div className="text-[#006644] text-xs tracking-[0.3em] uppercase font-light">
          BigMT.ai Tactical Operations
        </div>
      </div>

      {/* Right Section - Status Indicators */}
      <div className="flex items-center gap-4">
        {/* Weather */}
        <StatusIcon label="WX" status="CLEAR" color="green" />
        
        {/* AI State */}
        <StatusIcon label="AI" status="ACTIVE" color="green" blink />
        
        {/* Comms Link */}
        <StatusIcon label="LINK" status="NOMINAL" color="green" />
        
        {/* Encryption */}
        <StatusIcon label="ENC" status="AES-256" color="green" />
      </div>
    </div>
  );
}

interface StatusIconProps {
  label: string;
  status: string;
  color: 'green' | 'amber' | 'red';
  blink?: boolean;
}

function StatusIcon({ label, status, color, blink }: StatusIconProps) {
  const colorMap = {
    green: '#00FF91',
    amber: '#FF9933',
    red: '#FF3333',
  };

  return (
    <div className="flex flex-col items-center">
      <div className={`w-2 h-2 rounded-full mb-1 ${blink ? 'animate-pulse' : ''}`} 
           style={{ 
             backgroundColor: colorMap[color],
             boxShadow: `0 0 4px ${colorMap[color]}, 0 0 8px ${colorMap[color]}`
           }} />
      <div className="text-[9px] text-[#006644] uppercase tracking-wider">{label}</div>
      <div className="text-[10px] text-[#00CC74] font-mono">{status}</div>
    </div>
  );
}
