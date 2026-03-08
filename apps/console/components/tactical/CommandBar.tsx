'use client';

import React, { useState, useEffect, useRef } from 'react';

const EXAMPLE_COMMANDS = [
  'patrol sector 4',
  'return all assets',
  'survey grid alpha',
  'status drone-01',
  'deploy sensor array bravo',
];

export default function CommandBar() {
  const [command, setCommand] = useState('');
  const [history, setHistory] = useState<string[]>([]);
  const [historyIndex, setHistoryIndex] = useState(-1);
  const inputRef = useRef<HTMLInputElement>(null);

  const [placeholderIdx, setPlaceholderIdx] = useState(0);
  const [cpuUsage, setCpuUsage] = useState(34);
  const [netThroughput, setNetThroughput] = useState(156);

  useEffect(() => {
    const interval = setInterval(() => {
      setCpuUsage(prev => Math.max(10, Math.min(95, prev + (Math.random() - 0.5) * 10)));
      setNetThroughput(prev => Math.max(50, Math.min(500, prev + (Math.random() - 0.5) * 30)));
    }, 3000);
    return () => clearInterval(interval);
  }, []);

  // Rotate placeholder examples
  useEffect(() => {
    const interval = setInterval(() => {
      setPlaceholderIdx(prev => (prev + 1) % EXAMPLE_COMMANDS.length);
    }, 4000);
    return () => clearInterval(interval);
  }, []);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (command.trim()) {
      setHistory(prev => [...prev, command]);
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
    <div className="h-10 bg-[#0A0A0A] border-t-2 border-[#00FF91]/20 flex items-center gap-4 px-4 relative z-10">
      {/* Compact stats */}
      <div className="flex items-center gap-4 text-[10px] font-mono text-[#006644]">
        <span>CPU <span className={cpuUsage > 80 ? 'text-[#FF3333]' : cpuUsage > 60 ? 'text-[#FF9933]' : 'text-[#00CC74]'}>{Math.round(cpuUsage)}%</span></span>
        <span>NET <span className="text-[#00CC74]">{Math.round(netThroughput)} KB/s</span></span>
      </div>

      <div className="h-4 w-px bg-[#00FF91]/20" />

      {/* Command input */}
      <form onSubmit={handleSubmit} className="flex-1 flex items-center gap-2">
        <span className="text-[#00FF91] text-sm font-mono">$</span>
        <input
          ref={inputRef}
          type="text"
          value={command}
          onChange={(e) => setCommand(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={EXAMPLE_COMMANDS[placeholderIdx]}
          className="flex-1 bg-transparent border-none outline-none text-[#00CC74] font-mono text-xs placeholder-[#004422] caret-[#00FF91]"
          autoFocus
        />
      </form>
    </div>
  );
}
