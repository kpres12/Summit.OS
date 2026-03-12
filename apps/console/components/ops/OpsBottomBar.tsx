'use client';

import React, { useState, useEffect, useRef, useCallback } from 'react';
import { useEntityStream, EntityData } from '@/hooks/useEntityStream';
import { connectWebSocket } from '@/lib/api';

const API = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

const EXAMPLE_COMMANDS = [
  'patrol sector 4',
  'return all assets',
  'survey grid alpha',
  'status drone-01',
  'deploy sensor array bravo',
];

function domainTag(domain: string): string {
  switch (domain) {
    case 'aerial': return 'UAV';
    case 'ground': return 'GND';
    case 'maritime': return 'MAR';
    case 'fixed': return 'FIX';
    case 'sensor': return 'SEN';
    default: return '???';
  }
}

function entityTypeColor(type: string): string {
  switch (type) {
    case 'active': return '#00FF9C';
    case 'alert': return '#FF3B3B';
    case 'neutral': return 'rgba(200,230,201,0.45)';
    default: return '#FFB300';
  }
}

interface DetectionChip {
  id: string;
  domain: string;
  callsign: string;
  entityType: string;
  confidence: number;
  ts: number;
}

interface OpsBottomBarProps {
  onInvestigateEntity?: (callsign: string) => void;
}

export default function OpsBottomBar({ onInvestigateEntity }: OpsBottomBarProps) {
  const { entityCount, entityList } = useEntityStream();
  const [chips, setChips] = useState<DetectionChip[]>([]);
  const [command, setCommand] = useState('');
  const [cmdFeedback, setCmdFeedback] = useState<string | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const [history, setHistory] = useState<string[]>([]);
  const [historyIndex, setHistoryIndex] = useState(-1);
  const [placeholderIdx, setPlaceholderIdx] = useState(0);
  const [uptimeSecs, setUptimeSecs] = useState(0);
  const feedRef = useRef<HTMLDivElement>(null);
  const mountTime = useRef(Date.now());

  // Uptime counter
  useEffect(() => {
    const t = setInterval(() => {
      setUptimeSecs(Math.floor((Date.now() - mountTime.current) / 1000));
    }, 1000);
    return () => clearInterval(t);
  }, []);

  // Rotate placeholder
  useEffect(() => {
    const t = setInterval(() => {
      setPlaceholderIdx((p) => (p + 1) % EXAMPLE_COMMANDS.length);
    }, 4000);
    return () => clearInterval(t);
  }, []);

  // WS feed for detection chips
  const handleMsg = useCallback((data: unknown) => {
    const msg = data as { type?: string; data?: EntityData };
    if (msg.type === 'entity_update' && msg.data) {
      const e = msg.data;
      const chip: DetectionChip = {
        id: `${e.entity_id}-${Date.now()}`,
        domain: e.domain,
        callsign: e.callsign || e.entity_id.slice(0, 8),
        entityType: e.entity_type,
        confidence: e.confidence,
        ts: Date.now(),
      };
      setChips((prev) => [...prev, chip].slice(-20));
    }
  }, []);

  useEffect(() => {
    const ws = connectWebSocket(handleMsg);
    return () => { ws?.close(); };
  }, [handleMsg]);

  // Clean up polling on unmount
  useEffect(() => () => { if (pollRef.current) clearInterval(pollRef.current); }, []);

  // Auto-scroll feed to right
  useEffect(() => {
    if (feedRef.current) {
      feedRef.current.scrollLeft = feedRef.current.scrollWidth;
    }
  }, [chips]);

  const formatUptime = (secs: number): string => {
    const h = Math.floor(secs / 3600);
    const m = Math.floor((secs % 3600) / 60);
    const s = secs % 60;
    return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
  };

  const activeEntities = entityList.filter((e) => e.speed_mps > 0.5);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const cmd = command.trim();
    if (!cmd) return;
    setHistory((prev) => [...prev, cmd]);
    setCommand('');
    setHistoryIndex(-1);
    setCmdFeedback('SENDING...');
    try {
      const r = await fetch(`${API}/agents`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mission_objective: cmd }),
      });
      const data = await r.json();
      if (!data.ok) {
        setCmdFeedback(`ERR: ${data.error || 'failed'}`);
        setTimeout(() => setCmdFeedback(null), 3000);
        return;
      }
      const agentId = data.agent_id;
      setCmdFeedback(`AGENT ${agentId} RUNNING`);
      // Poll agent status until terminal state
      if (pollRef.current) clearInterval(pollRef.current);
      let polls = 0;
      pollRef.current = setInterval(async () => {
        polls++;
        try {
          const sr = await fetch(`${API}/agents`);
          const sd = await sr.json();
          const agent = (sd.agents || []).find((a: { agent_id: string; status: string }) => a.agent_id === agentId);
          if (agent) {
            const s = agent.status as string;
            setCmdFeedback(`AGENT ${agentId} ${s}`);
            if (['COMPLETED', 'FAILED', 'CANCELLED'].includes(s) || polls >= 6) {
              clearInterval(pollRef.current!);
              pollRef.current = null;
              setTimeout(() => setCmdFeedback(null), 2000);
            }
          } else if (polls >= 6) {
            clearInterval(pollRef.current!);
            pollRef.current = null;
            setTimeout(() => setCmdFeedback(null), 2000);
          }
        } catch {
          clearInterval(pollRef.current!);
          pollRef.current = null;
          setTimeout(() => setCmdFeedback(null), 2000);
        }
      }, 4000);
    } catch {
      setCmdFeedback('SERVER UNREACHABLE');
      setTimeout(() => setCmdFeedback(null), 3000);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'ArrowUp') {
      e.preventDefault();
      if (history.length > 0) {
        const newIdx = historyIndex === -1 ? history.length - 1 : Math.max(0, historyIndex - 1);
        setHistoryIndex(newIdx);
        setCommand(history[newIdx]);
      }
    } else if (e.key === 'ArrowDown') {
      e.preventDefault();
      if (historyIndex !== -1) {
        const newIdx = Math.min(history.length - 1, historyIndex + 1);
        if (newIdx === history.length - 1 && historyIndex === newIdx) {
          setHistoryIndex(-1);
          setCommand('');
        } else {
          setHistoryIndex(newIdx);
          setCommand(history[newIdx]);
        }
      }
    }
  };

  return (
    <div
      className="flex-none flex items-stretch"
      style={{
        height: '52px',
        background: '#0D1210',
        borderTop: '1px solid rgba(0,255,156,0.15)',
      }}
    >
      {/* LEFT: Detection feed */}
      <div
        className="flex-none flex items-center overflow-hidden"
        style={{ width: '35%', borderRight: '1px solid rgba(0,255,156,0.1)' }}
      >
        <div
          ref={feedRef}
          className="flex items-center gap-1.5 px-3 overflow-x-auto"
          style={{ scrollbarWidth: 'none', msOverflowStyle: 'none' }}
        >
          {chips.length === 0 && (
            <span
              className="text-[10px] whitespace-nowrap"
              style={{ color: 'rgba(200,230,201,0.25)', fontFamily: 'var(--font-ibm-plex-mono), monospace' }}
            >
              DETECTION FEED OFFLINE
            </span>
          )}
          {chips.map((chip) => {
            const color = entityTypeColor(chip.entityType);
            return (
              <div
                key={chip.id}
                onClick={() => onInvestigateEntity?.(chip.callsign)}
                className="flex-none text-[9px] px-1.5 py-0.5 whitespace-nowrap transition-colors"
                style={{
                  fontFamily: 'var(--font-ibm-plex-mono), monospace',
                  color,
                  border: `1px solid ${color}40`,
                  background: `${color}10`,
                  cursor: onInvestigateEntity ? 'pointer' : 'default',
                }}
                onMouseEnter={(e) => {
                  if (onInvestigateEntity) (e.currentTarget as HTMLDivElement).style.background = `${color}25`;
                }}
                onMouseLeave={(e) => {
                  (e.currentTarget as HTMLDivElement).style.background = `${color}10`;
                }}
              >
                [{domainTag(chip.domain)}] {chip.callsign} {Math.round(chip.confidence * 100)}%
              </div>
            );
          })}
        </div>
      </div>

      {/* CENTER: Command input */}
      <div className="flex-1 flex items-center px-3" style={{ borderRight: '1px solid rgba(0,255,156,0.1)' }}>
        <form onSubmit={handleSubmit} className="flex items-center gap-2 w-full">
          <span
            className="text-sm font-bold"
            style={{ color: '#00FF9C', fontFamily: 'var(--font-ibm-plex-mono), monospace' }}
          >
            &gt;
          </span>
          {cmdFeedback ? (
            <span
              className="flex-1 text-xs"
              style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: cmdFeedback.startsWith('ERR') || cmdFeedback === 'SERVER UNREACHABLE' ? '#FF3B3B' : '#00FF9C' }}
            >
              {cmdFeedback}
            </span>
          ) : (
            <input
              type="text"
              value={command}
              onChange={(e) => setCommand(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={EXAMPLE_COMMANDS[placeholderIdx]}
              className="flex-1 text-xs outline-none"
              style={{
                fontFamily: 'var(--font-ibm-plex-mono), monospace',
                color: '#00FF9C',
                background: 'transparent',
                border: 'none',
                caretColor: '#00FF9C',
              }}
            />
          )}
        </form>
      </div>

      {/* RIGHT: Stats */}
      <div
        className="flex-none flex items-center px-4 gap-4"
        style={{ width: '25%' }}
      >
        <div className="flex flex-col gap-0.5">
          <span
            className="text-[10px]"
            style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: 'rgba(200,230,201,0.45)' }}
          >
            {activeEntities.length} ACTIVE
          </span>
          <span
            className="text-[10px]"
            style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: 'rgba(200,230,201,0.45)' }}
          >
            {entityCount} ENTITIES
          </span>
          <span
            className="text-[10px]"
            style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: 'rgba(200,230,201,0.35)' }}
          >
            UP {formatUptime(uptimeSecs)}
          </span>
        </div>
      </div>
    </div>
  );
}
