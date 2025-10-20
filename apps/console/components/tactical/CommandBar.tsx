'use client';

import React, { useState, useEffect, useRef } from 'react';

export default function CommandBar() {
  const [command, setCommand] = useState('');
  const [history, setHistory] = useState<string[]>([]);
  const [historyIndex, setHistoryIndex] = useState(-1);
  const inputRef = useRef<HTMLInputElement>(null);

  // Mock system stats
  const [cpuUsage, setCpuUsage] = useState(34);
  const [netThroughput, setNetThroughput] = useState(156);
  const [energyDraw, setEnergyDraw] = useState(2.4);

  useEffect(() => {
    // Simulate fluctuating stats
    const interval = setInterval(() => {
      setCpuUsage(prev => Math.max(10, Math.min(95, prev + (Math.random() - 0.5) * 10)));
      setNetThroughput(prev => Math.max(50, Math.min(500, prev + (Math.random() - 0.5) * 30)));
      setEnergyDraw(prev => Math.max(1.0, Math.min(5.0, prev + (Math.random() - 0.5) * 0.3)));
    }, 2000);

    return () => clearInterval(interval);
  }, []);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (command.trim()) {
      setHistory(prev => [...prev, command]);
      // Here you would handle the command execution
      console.log('Command:', command);
      setCommand('');
      setHistoryIndex(-1);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'ArrowUp') {
      e.preventDefault();
      if (history.length > 0) {
        const newIndex = historyIndex === -1 ? history.length - 1 : Math.max(0, historyIndex - 1);
        setHistoryIndex(newIndex);
        setCommand(history[newIndex]);
      }
    } else if (e.key === 'ArrowDown') {
      e.preventDefault();
      if (historyIndex !== -1) {
        const newIndex = Math.min(history.length - 1, historyIndex + 1);
        if (newIndex === history.length - 1 && historyIndex === newIndex) {
          setHistoryIndex(-1);
          setCommand('');
        } else {
          setHistoryIndex(newIndex);
          setCommand(history[newIndex]);
        }
      }
    }
  };

  return (
    <div className="h-20 bg-[#0F0F0F] border-t-2 border-[#00FF91]/20 flex items-center gap-6 px-6 relative z-10">
      {/* System Stats */}
      <div className="flex items-center gap-6 min-w-[400px]">
        <StatBar label="CPU" value={cpuUsage} unit="%" max={100} />
        <StatBar label="NET" value={netThroughput} unit="KB/s" max={500} />
        <StatBar label="PWR" value={energyDraw} unit="kW" max={5} decimals={1} />
      </div>

      {/* Vertical Divider */}
      <div className="h-12 w-px bg-[#00FF91]/30" />

      {/* Command Prompt */}
      <form onSubmit={handleSubmit} className="flex-1 flex items-center gap-3">
        <div className="text-[#00FF91] text-xl font-bold text-glow">{'>'}</div>
        <input
          ref={inputRef}
          type="text"
          value={command}
          onChange={(e) => setCommand(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="ENTER COMMAND..."
          className="flex-1 bg-transparent border-none outline-none text-[#00CC74] font-mono text-sm placeholder-[#006644] caret-[#00FF91]"
          autoFocus
        />
        <div className="text-[10px] text-[#006644] font-mono tracking-wider">
          [CTRL+C TO ABORT]
        </div>
      </form>
    </div>
  );
}

interface StatBarProps {
  label: string;
  value: number;
  unit: string;
  max: number;
  decimals?: number;
}

function StatBar({ label, value, unit, max, decimals = 0 }: StatBarProps) {
  const percentage = (value / max) * 100;
  const displayValue = decimals > 0 ? value.toFixed(decimals) : Math.round(value);
  
  // Color based on usage
  const getColor = () => {
    if (percentage > 80) return '#FF3333';
    if (percentage > 60) return '#FF9933';
    return '#00FF91';
  };

  const color = getColor();

  return (
    <div className="flex flex-col gap-1 min-w-[100px]">
      <div className="flex items-center justify-between">
        <div className="text-[10px] text-[#006644] uppercase tracking-wider">{label}</div>
        <div className="text-xs font-mono" style={{ color }}>
          {displayValue} {unit}
        </div>
      </div>
      <div className="h-1.5 bg-[#0A0A0A] border border-[#00FF91]/20 relative overflow-hidden">
        <div 
          className="h-full transition-all duration-500"
          style={{ 
            width: `${percentage}%`,
            backgroundColor: color,
            boxShadow: `0 0 8px ${color}`
          }}
        />
      </div>
    </div>
  );
}
