'use client';

import React, { useState, useEffect, useCallback, useRef } from 'react';
import { useEntityStream } from '@/hooks/useEntityStream';
import { connectWebSocket } from '@/lib/api';
import ErrorBoundary from '@/components/ErrorBoundary';

interface DevLayoutProps {
  onSwitchRole: () => void;
}

type DevView = 'entity-explorer' | 'adapter-registry' | 'message-inspector' | 'schema-validator' | 'inference-dashboard';

const NAV_ITEMS: { id: DevView; label: string }[] = [
  { id: 'entity-explorer', label: 'Entity Explorer' },
  { id: 'adapter-registry', label: 'Adapter Registry' },
  { id: 'message-inspector', label: 'Message Inspector' },
  { id: 'schema-validator', label: 'Schema Validator' },
  { id: 'inference-dashboard', label: 'Inference Dashboard' },
];

const VIEW_LABELS: Record<DevView, string> = {
  'entity-explorer': 'ENTITY EXPLORER',
  'adapter-registry': 'ADAPTER REGISTRY',
  'message-inspector': 'MESSAGE INSPECTOR',
  'schema-validator': 'SCHEMA VALIDATOR',
  'inference-dashboard': 'INFERENCE DASHBOARD',
};

// ─── Entity Explorer ─────────────────────────────────────────

function EntityExplorer() {
  const { entityList } = useEntityStream();
  const [search, setSearch] = useState('');
  const [typeFilter, setTypeFilter] = useState<string>('ALL');
  const [statusFilter, setStatusFilter] = useState<string>('ALL');
  const [expandedId, setExpandedId] = useState<string | null>(null);

  const types = ['ALL', 'active', 'alert', 'neutral', 'unknown'];
  const statuses = ['ALL', 'aerial', 'ground', 'maritime', 'fixed', 'sensor'];

  const filtered = entityList.filter((e) => {
    const q = search.toLowerCase();
    const matchSearch = !q ||
      e.entity_id.toLowerCase().includes(q) ||
      (e.callsign || '').toLowerCase().includes(q) ||
      e.classification.toLowerCase().includes(q);
    const matchType = typeFilter === 'ALL' || e.entity_type === typeFilter;
    const matchStatus = statusFilter === 'ALL' || e.domain === statusFilter;
    return matchSearch && matchType && matchStatus;
  });

  function entityTypeColor(type: string): string {
    switch (type) {
      case 'active': return '#00E896';
      case 'alert': return '#FF3B3B';
      case 'neutral': return 'rgba(200,230,201,0.45)';
      default: return '#FFB300';
    }
  }

  return (
    <div className="flex flex-col h-full">
      {/* Controls */}
      <div
        className="flex-none px-4 py-3 flex items-center gap-3"
        style={{ borderBottom: '1px solid rgba(0,232,150,0.15)' }}
      >
        <input
          type="text"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Search entities..."
          className="text-xs px-3 py-1.5 outline-none flex-1"
          style={{
            fontFamily: 'var(--font-ibm-plex-mono), monospace',
            background: '#111916',
            border: '1px solid rgba(0,232,150,0.2)',
            color: '#00E896',
          }}
        />
        <select
          value={typeFilter}
          onChange={(e) => setTypeFilter(e.target.value)}
          className="text-[10px] px-2 py-1.5 outline-none"
          style={{
            fontFamily: 'var(--font-ibm-plex-mono), monospace',
            background: '#111916',
            border: '1px solid rgba(0,232,150,0.2)',
            color: 'rgba(200,230,201,0.7)',
          }}
        >
          {types.map((t) => <option key={t} value={t}>{t === 'ALL' ? 'All Types' : t}</option>)}
        </select>
        <select
          value={statusFilter}
          onChange={(e) => setStatusFilter(e.target.value)}
          className="text-[10px] px-2 py-1.5 outline-none"
          style={{
            fontFamily: 'var(--font-ibm-plex-mono), monospace',
            background: '#111916',
            border: '1px solid rgba(0,232,150,0.2)',
            color: 'rgba(200,230,201,0.7)',
          }}
        >
          {statuses.map((s) => <option key={s} value={s}>{s === 'ALL' ? 'All Domains' : s}</option>)}
        </select>
        <button
          onClick={() => console.log('Create entity — not implemented')}
          className="text-[10px] px-3 py-1.5 tracking-widest transition-colors flex-none"
          style={{
            fontFamily: 'var(--font-ibm-plex-mono), monospace',
            color: '#4FC3F7',
            border: '1px solid rgba(79,195,247,0.4)',
            background: 'transparent',
            cursor: 'not-allowed',
            opacity: 0.6,
          }}
        >
          + CREATE ENTITY
        </button>
      </div>

      {/* Table */}
      <div className="flex-1 overflow-y-auto">
        {/* Table header */}
        <div
          className="flex items-center px-4 py-2 text-[9px] tracking-widest sticky top-0"
          style={{
            background: '#0D1210',
            borderBottom: '1px solid rgba(0,232,150,0.15)',
            fontFamily: 'var(--font-ibm-plex-mono), monospace',
            color: 'rgba(0,232,150,0.5)',
          }}
        >
          <span style={{ width: '180px' }}>ID</span>
          <span style={{ width: '100px' }}>TYPE</span>
          <span style={{ width: '90px' }}>DOMAIN</span>
          <span style={{ width: '90px' }}>STATUS</span>
          <span style={{ width: '130px' }}>LAST UPDATED</span>
          <span className="flex-1">COORDINATES</span>
        </div>

        {filtered.length === 0 && (
          <div className="flex flex-col items-center justify-center h-48 gap-3">
            <div
              className="text-[10px] tracking-widest"
              style={{ color: 'rgba(200,230,201,0.35)', fontFamily: 'var(--font-ibm-plex-mono), monospace' }}
            >
              NO ENTITIES
            </div>
            <div
              className="text-[9px] text-center leading-relaxed"
              style={{ color: 'rgba(200,230,201,0.2)', fontFamily: 'var(--font-ibm-plex-mono), monospace' }}
            >
              Connect an adapter using the Heli.OS SDK<br />pip install heli-os-sdk
            </div>
          </div>
        )}

        {filtered.map((e) => {
          const isExpanded = expandedId === e.entity_id;
          const color = entityTypeColor(e.entity_type);
          const diff = Math.floor((Date.now() / 1000) - e.last_seen);
          const age = diff < 60 ? `${diff}s ago` : diff < 3600 ? `${Math.floor(diff / 60)}m ago` : `${Math.floor(diff / 3600)}h ago`;

          return (
            <div key={e.entity_id}>
              <div
                className="flex items-center px-4 py-2 cursor-pointer transition-colors"
                style={{
                  borderBottom: '1px solid rgba(0,232,150,0.05)',
                  background: isExpanded ? 'rgba(0,232,150,0.04)' : 'transparent',
                }}
                onClick={() => setExpandedId(isExpanded ? null : e.entity_id)}
                onMouseEnter={(el) => { if (!isExpanded) (el.currentTarget as HTMLDivElement).style.background = 'rgba(0,232,150,0.03)'; }}
                onMouseLeave={(el) => { if (!isExpanded) (el.currentTarget as HTMLDivElement).style.background = 'transparent'; }}
              >
                <span
                  className="text-[10px] font-bold"
                  style={{ width: '180px', fontFamily: 'var(--font-ibm-plex-mono), monospace', color, flexShrink: 0 }}
                >
                  {e.entity_id.slice(0, 16)}
                </span>
                <span
                  className="text-[10px]"
                  style={{ width: '100px', fontFamily: 'var(--font-ibm-plex-mono), monospace', color, flexShrink: 0 }}
                >
                  {e.entity_type}
                </span>
                <span
                  className="text-[10px]"
                  style={{ width: '90px', fontFamily: 'var(--font-ibm-plex-mono), monospace', color: 'rgba(200,230,201,0.6)', flexShrink: 0 }}
                >
                  {e.domain}
                </span>
                <span
                  className="text-[10px]"
                  style={{ width: '90px', fontFamily: 'var(--font-ibm-plex-mono), monospace', color: 'rgba(200,230,201,0.6)', flexShrink: 0 }}
                >
                  {e.track_state || 'active'}
                </span>
                <span
                  className="text-[10px]"
                  style={{ width: '130px', fontFamily: 'var(--font-ibm-plex-mono), monospace', color: 'rgba(200,230,201,0.4)', flexShrink: 0 }}
                >
                  {age}
                </span>
                <span
                  className="text-[10px] flex-1"
                  style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: 'rgba(200,230,201,0.4)' }}
                >
                  {e.position.lat.toFixed(4)}, {e.position.lon.toFixed(4)}
                </span>
              </div>

              {/* Expanded row */}
              {isExpanded && (
                <div
                  className="px-6 py-4"
                  style={{
                    background: '#111916',
                    borderBottom: '1px solid rgba(0,232,150,0.1)',
                    borderLeft: '3px solid #00E896',
                  }}
                >
                  <div className="grid grid-cols-2 gap-6">
                    <div>
                      <div
                        className="text-[9px] tracking-widest mb-2"
                        style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: 'rgba(0,232,150,0.5)' }}
                      >
                        ENTITY DATA
                      </div>
                      <pre
                        className="text-[10px] leading-relaxed overflow-x-auto"
                        style={{
                          fontFamily: 'var(--font-ibm-plex-mono), monospace',
                          color: 'rgba(200,230,201,0.7)',
                          background: '#0D1210',
                          padding: '8px',
                          border: '1px solid rgba(0,232,150,0.1)',
                        }}
                      >
                        {JSON.stringify(e, null, 2)}
                      </pre>
                    </div>
                    <div>
                      <div
                        className="text-[9px] tracking-widest mb-2"
                        style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: 'rgba(0,232,150,0.5)' }}
                      >
                        POSITION
                      </div>
                      <div
                        className="p-3"
                        style={{ background: '#0D1210', border: '1px solid rgba(0,232,150,0.1)' }}
                      >
                        {[
                          ['LAT', e.position.lat.toFixed(6)],
                          ['LON', e.position.lon.toFixed(6)],
                          ['ALT', `${e.position.alt.toFixed(0)} m`],
                          ['HDG', `${e.position.heading_deg.toFixed(1)}°`],
                          ['SPD', `${e.speed_mps.toFixed(1)} m/s`],
                        ].map(([label, val]) => (
                          <div key={label} className="flex justify-between py-0.5">
                            <span className="text-[10px]" style={{ color: 'rgba(200,230,201,0.45)', fontFamily: 'var(--font-ibm-plex-mono), monospace' }}>{label}</span>
                            <span className="text-[10px] font-bold" style={{ color: '#00E896', fontFamily: 'var(--font-ibm-plex-mono), monospace' }}>{val}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ─── Adapter Registry ─────────────────────────────────────────

const MOCK_ADAPTERS = [
  { name: 'Summit SDK (Python)', version: '1.2.0', status: 'online', entityCount: 0, msgRate: 0, errorRate: 0, conformance: 100 },
  { name: 'MAVLINK Bridge', version: '0.8.3', status: 'offline', entityCount: 0, msgRate: 0, errorRate: 0, conformance: 87 },
  { name: 'ADS-B Receiver', version: '2.1.0', status: 'online', entityCount: 0, msgRate: 0, errorRate: 0, conformance: 95 },
];

function AdapterRegistry() {
  return (
    <div className="flex flex-col h-full overflow-y-auto p-4">
      <div className="grid gap-4" style={{ gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))' }}>
        {MOCK_ADAPTERS.map((adapter) => (
          <div
            key={adapter.name}
            className="flex flex-col"
            style={{
              background: '#111916',
              border: '1px solid rgba(0,232,150,0.15)',
              padding: '16px',
            }}
          >
            {/* Header */}
            <div className="flex items-start justify-between mb-3">
              <div>
                <div
                  className="text-xs font-bold"
                  style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: '#00E896' }}
                >
                  {adapter.name}
                </div>
                <div
                  className="text-[10px]"
                  style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: 'rgba(200,230,201,0.4)' }}
                >
                  v{adapter.version}
                </div>
              </div>
              <div className="flex items-center gap-2">
                <span
                  className="text-[9px] px-1.5 py-0.5"
                  style={{
                    fontFamily: 'var(--font-ibm-plex-mono), monospace',
                    color: 'rgba(200,230,201,0.4)',
                    border: '1px solid rgba(200,230,201,0.15)',
                    background: 'rgba(200,230,201,0.05)',
                  }}
                >
                  [SIMULATED]
                </span>
                <div
                  className="w-2 h-2 rounded-full"
                  style={{ background: adapter.status === 'online' ? '#00E896' : '#FF3B3B' }}
                />
              </div>
            </div>

            {/* Stats */}
            <div className="flex gap-4 mb-3">
              {[
                { label: 'ENTITIES', value: String(adapter.entityCount) },
                { label: 'MSG/S', value: String(adapter.msgRate) },
                { label: 'ERR%', value: `${adapter.errorRate}%` },
              ].map((s) => (
                <div key={s.label} className="flex flex-col">
                  <span className="text-[9px]" style={{ color: 'rgba(200,230,201,0.4)', fontFamily: 'var(--font-ibm-plex-mono), monospace' }}>{s.label}</span>
                  <span className="text-xs font-bold" style={{ color: '#00E896', fontFamily: 'var(--font-ibm-plex-mono), monospace' }}>{s.value}</span>
                </div>
              ))}
            </div>

            {/* Conformance */}
            <div className="mb-3">
              <div className="flex justify-between mb-1">
                <span className="text-[9px]" style={{ color: 'rgba(200,230,201,0.4)', fontFamily: 'var(--font-ibm-plex-mono), monospace' }}>CONFORMANCE</span>
                <span className="text-[10px] font-bold" style={{ color: adapter.conformance >= 90 ? '#00E896' : '#FFB300', fontFamily: 'var(--font-ibm-plex-mono), monospace' }}>{adapter.conformance}%</span>
              </div>
              <div className="h-1 w-full rounded-full overflow-hidden" style={{ background: 'rgba(0,232,150,0.1)' }}>
                <div className="h-full" style={{ width: `${adapter.conformance}%`, background: adapter.conformance >= 90 ? '#00E896' : '#FFB300' }} />
              </div>
            </div>

            {/* Sparkline placeholder */}
            <div
              className="mb-3 flex items-end gap-px"
              style={{ height: '24px' }}
            >
              {Array.from({ length: 20 }).map((_, i) => (
                <div
                  key={i}
                  style={{
                    flex: 1,
                    height: '2px',
                    background: 'rgba(0,232,150,0.2)',
                  }}
                />
              ))}
            </div>

            {/* Buttons */}
            <div className="flex gap-2">
              <button
                onClick={() => console.log('View logs:', adapter.name)}
                className="flex-1 text-[10px] py-1.5 tracking-wider transition-colors"
                style={{
                  fontFamily: 'var(--font-ibm-plex-mono), monospace',
                  color: 'rgba(200,230,201,0.6)',
                  border: '1px solid rgba(0,232,150,0.2)',
                  background: 'transparent',
                  cursor: 'pointer',
                }}
              >
                VIEW LOGS
              </button>
              <button
                onClick={() => console.log('Run conformance:', adapter.name)}
                className="flex-1 text-[10px] py-1.5 tracking-wider transition-colors"
                style={{
                  fontFamily: 'var(--font-ibm-plex-mono), monospace',
                  color: '#00E896',
                  border: '1px solid rgba(0,232,150,0.3)',
                  background: 'transparent',
                  cursor: 'pointer',
                }}
              >
                RUN CONFORMANCE
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ─── Message Inspector ────────────────────────────────────────

interface WsMessage {
  id: string;
  timestamp: number;
  topic: string;
  direction: 'IN';
  size: number;
  valid: boolean;
  payload: unknown;
}

function MessageInspector() {
  const [messages, setMessages] = useState<WsMessage[]>([]);
  const [paused, setPaused] = useState(false);
  const [topicFilter, setTopicFilter] = useState('');
  const [selectedMsg, setSelectedMsg] = useState<WsMessage | null>(null);
  const pausedRef = useRef(paused);
  pausedRef.current = paused;
  const containerRef = useRef<HTMLDivElement>(null);

  const handleMsg = useCallback((data: unknown) => {
    if (pausedRef.current) return;
    const raw = JSON.stringify(data);
    const msg = data as { type?: string };
    const wm: WsMessage = {
      id: `${Date.now()}-${Math.random()}`,
      timestamp: Date.now(),
      topic: msg.type || 'unknown',
      direction: 'IN',
      size: raw.length,
      valid: true,
      payload: data,
    };
    setMessages((prev) => [wm, ...prev].slice(0, 500));
  }, []);

  useEffect(() => {
    const ws = connectWebSocket(handleMsg);
    return () => { ws?.close(); };
  }, [handleMsg]);

  const filtered = messages.filter((m) =>
    !topicFilter || m.topic.toLowerCase().includes(topicFilter.toLowerCase())
  );

  useEffect(() => {
    if (!paused && containerRef.current) {
      containerRef.current.scrollTop = 0;
    }
  }, [messages, paused]);

  return (
    <div className="flex flex-col h-full">
      {/* Controls */}
      <div
        className="flex-none flex items-center gap-3 px-4 py-2"
        style={{ borderBottom: '1px solid rgba(0,232,150,0.15)' }}
      >
        <button
          onClick={() => setPaused((p) => !p)}
          className="text-[10px] px-3 py-1.5 tracking-widest transition-colors"
          style={{
            fontFamily: 'var(--font-ibm-plex-mono), monospace',
            color: paused ? '#FFB300' : '#00E896',
            border: `1px solid ${paused ? 'rgba(255,179,0,0.4)' : 'rgba(0,232,150,0.3)'}`,
            background: 'transparent',
            cursor: 'pointer',
          }}
        >
          {paused ? 'RESUME' : 'PAUSE'}
        </button>
        <input
          type="text"
          value={topicFilter}
          onChange={(e) => setTopicFilter(e.target.value)}
          placeholder="Filter by topic..."
          className="text-xs px-3 py-1.5 outline-none"
          style={{
            fontFamily: 'var(--font-ibm-plex-mono), monospace',
            background: '#111916',
            border: '1px solid rgba(0,232,150,0.2)',
            color: '#00E896',
            width: '200px',
          }}
        />
        <span
          className="text-[10px] ml-auto"
          style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: 'rgba(200,230,201,0.4)' }}
        >
          {filtered.length} messages
        </span>
      </div>

      {/* Table */}
      <div className="flex-1 overflow-hidden flex flex-col">
        {/* Table header */}
        <div
          className="flex items-center px-4 py-1.5 text-[9px] tracking-widest flex-none"
          style={{
            background: '#0D1210',
            borderBottom: '1px solid rgba(0,232,150,0.15)',
            fontFamily: 'var(--font-ibm-plex-mono), monospace',
            color: 'rgba(0,232,150,0.5)',
          }}
        >
          <span style={{ width: '90px' }}>TIMESTAMP</span>
          <span style={{ width: '120px' }}>TOPIC</span>
          <span style={{ width: '60px' }}>DIR</span>
          <span style={{ width: '80px' }}>SIZE</span>
          <span>VALIDATION</span>
        </div>

        <div ref={containerRef} className="flex-1 overflow-y-auto">
          {filtered.length === 0 && (
            <div
              className="flex items-center justify-center h-20 text-[10px]"
              style={{ color: 'rgba(200,230,201,0.25)', fontFamily: 'var(--font-ibm-plex-mono), monospace' }}
            >
              {paused ? 'PAUSED' : 'LISTENING...'}
            </div>
          )}
          {filtered.map((m) => {
            const isSelected = selectedMsg?.id === m.id;
            const d = new Date(m.timestamp);
            const timeStr = `${String(d.getHours()).padStart(2,'0')}:${String(d.getMinutes()).padStart(2,'0')}:${String(d.getSeconds()).padStart(2,'0')}.${String(d.getMilliseconds()).padStart(3,'0')}`;
            return (
              <React.Fragment key={m.id}>
                <div
                  onClick={() => setSelectedMsg(isSelected ? null : m)}
                  className="flex items-center px-4 py-1.5 cursor-pointer transition-colors"
                  style={{
                    borderBottom: '1px solid rgba(0,232,150,0.04)',
                    background: isSelected ? 'rgba(0,232,150,0.06)' : 'transparent',
                  }}
                  onMouseEnter={(el) => { if (!isSelected) (el.currentTarget as HTMLDivElement).style.background = 'rgba(0,232,150,0.03)'; }}
                  onMouseLeave={(el) => { if (!isSelected) (el.currentTarget as HTMLDivElement).style.background = 'transparent'; }}
                >
                  <span className="text-[10px]" style={{ width: '90px', fontFamily: 'var(--font-ibm-plex-mono), monospace', color: 'rgba(200,230,201,0.4)', flexShrink: 0 }}>{timeStr}</span>
                  <span className="text-[10px] font-bold" style={{ width: '120px', fontFamily: 'var(--font-ibm-plex-mono), monospace', color: '#00E896', flexShrink: 0 }}>{m.topic}</span>
                  <span className="text-[10px]" style={{ width: '60px', fontFamily: 'var(--font-ibm-plex-mono), monospace', color: 'rgba(200,230,201,0.4)', flexShrink: 0 }}>{m.direction}</span>
                  <span className="text-[10px]" style={{ width: '80px', fontFamily: 'var(--font-ibm-plex-mono), monospace', color: 'rgba(200,230,201,0.4)', flexShrink: 0 }}>{m.size}B</span>
                  <span className="text-[10px]" style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: m.valid ? '#00E896' : '#FF3B3B' }}>
                    {m.valid ? 'PASS' : 'FAIL'}
                  </span>
                </div>
                {isSelected && (
                  <div
                    className="px-4 py-3"
                    style={{
                      background: '#111916',
                      borderBottom: '1px solid rgba(0,232,150,0.1)',
                      borderLeft: '3px solid #00E896',
                    }}
                  >
                    <pre
                      className="text-[10px] leading-relaxed overflow-x-auto"
                      style={{
                        fontFamily: 'var(--font-ibm-plex-mono), monospace',
                        color: 'rgba(200,230,201,0.7)',
                        maxHeight: '200px',
                      }}
                    >
                      {JSON.stringify(m.payload, null, 2)}
                    </pre>
                  </div>
                )}
              </React.Fragment>
            );
          })}
        </div>
      </div>
    </div>
  );
}

// ─── Schema Validator ─────────────────────────────────────────

function SchemaValidator() {
  const [input, setInput] = useState('');
  const [result, setResult] = useState<{ valid: boolean; error?: string; fields?: string[] } | null>(null);
  const [toast, setToast] = useState<string | null>(null);

  const validate = () => {
    try {
      const parsed = JSON.parse(input);
      const fields = Object.keys(parsed);
      setResult({ valid: true, fields });
    } catch (err: unknown) {
      const error = err instanceof Error ? err.message : String(err);
      setResult({ valid: false, error });
    }
  };

  const publish = () => {
    console.log('Publish to topic:', input);
    setToast('PUBLISHED');
    setTimeout(() => setToast(null), 2000);
  };

  return (
    <div className="flex flex-col h-full p-4">
      <div className="flex gap-4 flex-1 overflow-hidden">
        {/* Left: input */}
        <div className="flex-1 flex flex-col">
          <div
            className="text-[9px] tracking-widest mb-2"
            style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: 'rgba(0,232,150,0.5)' }}
          >
            JSON INPUT
          </div>
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder={'{\n  "entity_id": "...",\n  "entity_type": "friendly"\n}'}
            className="flex-1 text-xs p-3 outline-none resize-none"
            style={{
              fontFamily: 'var(--font-ibm-plex-mono), monospace',
              background: '#111916',
              border: '1px solid rgba(0,232,150,0.2)',
              color: '#00E896',
            }}
          />
          <div className="flex gap-2 mt-3">
            <button
              onClick={validate}
              className="flex-1 text-[10px] py-2 tracking-widest transition-colors"
              style={{
                fontFamily: 'var(--font-ibm-plex-mono), monospace',
                color: '#00E896',
                border: '1px solid rgba(0,232,150,0.4)',
                background: 'transparent',
                cursor: 'pointer',
              }}
            >
              VALIDATE
            </button>
            <button
              onClick={publish}
              disabled={!result?.valid}
              className="flex-1 text-[10px] py-2 tracking-widest transition-colors"
              style={{
                fontFamily: 'var(--font-ibm-plex-mono), monospace',
                color: result?.valid ? '#4FC3F7' : 'rgba(200,230,201,0.3)',
                border: `1px solid ${result?.valid ? 'rgba(79,195,247,0.4)' : 'rgba(200,230,201,0.1)'}`,
                background: 'transparent',
                cursor: result?.valid ? 'pointer' : 'not-allowed',
              }}
            >
              PUBLISH TO TOPIC
            </button>
          </div>
        </div>

        {/* Right: results */}
        <div className="flex-1 flex flex-col">
          <div
            className="text-[9px] tracking-widest mb-2"
            style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: 'rgba(0,232,150,0.5)' }}
          >
            VALIDATION RESULTS
          </div>
          <div
            className="flex-1 p-3"
            style={{
              background: '#111916',
              border: `1px solid ${result ? (result.valid ? 'rgba(0,232,150,0.3)' : 'rgba(255,59,59,0.3)') : 'rgba(0,232,150,0.1)'}`,
            }}
          >
            {!result && (
              <div
                className="text-[10px]"
                style={{ color: 'rgba(200,230,201,0.3)', fontFamily: 'var(--font-ibm-plex-mono), monospace' }}
              >
                Enter JSON and click VALIDATE
              </div>
            )}
            {result && !result.valid && (
              <div>
                <div
                  className="text-xs font-bold mb-2"
                  style={{ color: '#FF3B3B', fontFamily: 'var(--font-ibm-plex-mono), monospace' }}
                >
                  FAIL — INVALID JSON
                </div>
                <div
                  className="text-[10px]"
                  style={{ color: '#FF3B3B', fontFamily: 'var(--font-ibm-plex-mono), monospace' }}
                >
                  {result.error}
                </div>
              </div>
            )}
            {result?.valid && (
              <div>
                <div
                  className="text-xs font-bold mb-3"
                  style={{ color: '#00E896', fontFamily: 'var(--font-ibm-plex-mono), monospace' }}
                >
                  PASS — VALID JSON
                </div>
                <div
                  className="text-[9px] tracking-widest mb-2"
                  style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: 'rgba(0,232,150,0.5)' }}
                >
                  FIELDS ({result.fields?.length})
                </div>
                {result.fields?.map((f) => (
                  <div
                    key={f}
                    className="text-[10px] py-0.5"
                    style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: 'rgba(200,230,201,0.6)' }}
                  >
                    · {f}
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Toast */}
      {toast && (
        <div
          className="fixed bottom-6 right-6 px-4 py-2 text-xs tracking-widest"
          style={{
            fontFamily: 'var(--font-ibm-plex-mono), monospace',
            color: '#00E896',
            background: '#0D1210',
            border: '1px solid rgba(0,232,150,0.5)',
            boxShadow: '0 0 20px rgba(0,232,150,0.15)',
            zIndex: 100,
          }}
        >
          {toast}
        </div>
      )}
    </div>
  );
}

// ─── Inference Dashboard ─────────────────────────────────────

const FALLBACK_MODELS = [
  { name: 'Object Detection', version: 'YOLOv8-S', status: 'online', latency: 12, accuracy: 94.2 },
  { name: 'Classification', version: 'ResNet-50', status: 'online', latency: 8, accuracy: 91.7 },
  { name: 'Segmentation', version: 'SAM-L', status: 'offline', latency: 45, accuracy: 88.5 },
];

const INFERENCE_URL = process.env.NEXT_PUBLIC_API_URL
  ? process.env.NEXT_PUBLIC_API_URL.replace(':8000', ':8006')
  : 'http://localhost:8006';

interface InferenceModel {
  name: string;
  version: string;
  status: string;
  latency: number;
  accuracy: number;
}

function InferenceDashboard() {
  const [models, setModels] = useState<InferenceModel[]>(FALLBACK_MODELS);
  const [liveData, setLiveData] = useState(false);
  const [reqCount, setReqCount] = useState(0);

  useEffect(() => {
    fetch(`${INFERENCE_URL}/health`)
      .then((r) => r.json())
      .then((d) => {
        if (d.status === 'ok') {
          setLiveData(true);
          return fetch(`${INFERENCE_URL}/models`);
        }
        return null;
      })
      .then((r) => r ? r.json() : null)
      .then((d) => {
        if (d?.models) {
          setModels(d.models.map((m: { name?: string; model_id?: string; path?: string; status?: string }) => ({
            name: m.name || m.model_id || 'Unknown',
            version: m.path?.split('/').pop()?.replace('.onnx', '') || 'unknown',
            status: m.status || 'online',
            latency: 0,
            accuracy: 0,
          })));
        }
      })
      .catch((e: Error) => console.warn('[DevLayout] inference models fetch failed:', e.message));

    // Poll request count from health endpoint
    const t = setInterval(() => {
      fetch(`${INFERENCE_URL}/health`)
        .then((r) => r.json())
        .then((d) => { if (d.total_requests !== undefined) setReqCount(d.total_requests); })
        .catch((e: Error) => console.warn('[DevLayout] inference health poll failed:', e.message));
    }, 5000);
    return () => clearInterval(t);
  }, []);

  return (
    <div className="flex flex-col h-full overflow-y-auto p-4 gap-6">
      {/* Model Status */}
      <div>
        <div className="flex items-center gap-3 mb-3">
          <span
            className="text-xs font-bold tracking-widest"
            style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: '#00E896' }}
          >
            MODEL STATUS
          </span>
          <span
            className="text-[9px] px-1.5 py-0.5"
            style={{
              fontFamily: 'var(--font-ibm-plex-mono), monospace',
              color: liveData ? '#00E896' : 'rgba(200,230,201,0.4)',
              border: `1px solid ${liveData ? 'rgba(0,232,150,0.3)' : 'rgba(200,230,201,0.15)'}`,
            }}
          >
            {liveData ? '[LIVE]' : '[SIMULATED]'}
          </span>
          {reqCount > 0 && (
            <span className="text-[9px]" style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: 'rgba(200,230,201,0.4)' }}>
              {reqCount} reqs
            </span>
          )}
        </div>
        <div style={{ border: '1px solid rgba(0,232,150,0.15)' }}>
          <div
            className="flex items-center px-4 py-2 text-[9px] tracking-widest"
            style={{
              background: '#0D1210',
              borderBottom: '1px solid rgba(0,232,150,0.15)',
              fontFamily: 'var(--font-ibm-plex-mono), monospace',
              color: 'rgba(0,232,150,0.5)',
            }}
          >
            <span style={{ flex: 2 }}>MODEL</span>
            <span style={{ flex: 1 }}>VERSION</span>
            <span style={{ flex: 1 }}>STATUS</span>
            <span style={{ flex: 1 }}>LATENCY</span>
            <span style={{ flex: 1 }}>ACCURACY</span>
          </div>
          {models.map((m) => (
            <div
              key={m.name}
              className="flex items-center px-4 py-2.5"
              style={{ borderBottom: '1px solid rgba(0,232,150,0.05)', background: '#111916' }}
            >
              <span className="text-[11px] font-bold" style={{ flex: 2, fontFamily: 'var(--font-ibm-plex-mono), monospace', color: '#00E896' }}>{m.name}</span>
              <span className="text-[10px]" style={{ flex: 1, fontFamily: 'var(--font-ibm-plex-mono), monospace', color: 'rgba(200,230,201,0.5)' }}>{m.version}</span>
              <span className="text-[10px]" style={{ flex: 1, fontFamily: 'var(--font-ibm-plex-mono), monospace', color: m.status === 'online' ? '#00E896' : '#FF3B3B' }}>{m.status.toUpperCase()}</span>
              <span className="text-[10px]" style={{ flex: 1, fontFamily: 'var(--font-ibm-plex-mono), monospace', color: 'rgba(200,230,201,0.5)' }}>{m.latency}ms</span>
              <span className="text-[10px]" style={{ flex: 1, fontFamily: 'var(--font-ibm-plex-mono), monospace', color: '#00E896' }}>{m.accuracy}%</span>
            </div>
          ))}
        </div>
      </div>

      {/* Recent Detections */}
      <div>
        <div className="flex items-center gap-3 mb-3">
          <span
            className="text-xs font-bold tracking-widest"
            style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: '#00E896' }}
          >
            RECENT DETECTIONS
          </span>
          <span
            className="text-[9px] px-1.5 py-0.5"
            style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: 'rgba(200,230,201,0.4)', border: '1px solid rgba(200,230,201,0.15)' }}
          >
            [SIMULATED]
          </span>
        </div>
        <div className="grid gap-3" style={{ gridTemplateColumns: 'repeat(3, 1fr)' }}>
          {Array.from({ length: 6 }).map((_, i) => (
            <div
              key={i}
              className="p-3"
              style={{
                background: '#111916',
                border: '1px solid rgba(0,232,150,0.15)',
              }}
            >
              <div className="flex items-center justify-between mb-2">
                <span className="text-[9px] tracking-wider" style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: 'rgba(0,232,150,0.5)' }}>DETECTION {i + 1}</span>
                <span className="text-[9px]" style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: '#00E896' }}>
                  {(85 + Math.round(Math.random() * 14))}%
                </span>
              </div>
              <div className="text-xs font-bold mb-1" style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: 'rgba(200,230,201,0.7)' }}>
                {['UAV', 'VEHICLE', 'PERSON', 'VESSEL', 'AIRCRAFT', 'SENSOR'][i]}
              </div>
              <div className="text-[9px]" style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: 'rgba(200,230,201,0.35)' }}>
                {(i * 7 + 3)}s ago · YOLOv8-S
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Performance */}
      <div>
        <div className="flex items-center gap-3 mb-3">
          <span
            className="text-xs font-bold tracking-widest"
            style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: '#00E896' }}
          >
            PERFORMANCE
          </span>
          <span
            className="text-[9px] px-1.5 py-0.5"
            style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: 'rgba(200,230,201,0.4)', border: '1px solid rgba(200,230,201,0.15)' }}
          >
            [SIMULATED]
          </span>
        </div>
        <div
          className="p-4 text-[11px] text-center"
          style={{
            background: '#111916',
            border: '1px solid rgba(0,232,150,0.1)',
            fontFamily: 'var(--font-ibm-plex-mono), monospace',
            color: 'rgba(200,230,201,0.35)',
          }}
        >
          [SIMULATED] Live charts would appear here when inference service is connected
        </div>
      </div>
    </div>
  );
}

// ─── Dev Layout ───────────────────────────────────────────────

export default function DevLayout({ onSwitchRole }: DevLayoutProps) {
  const [activeView, setActiveView] = useState<DevView>('entity-explorer');

  const renderView = () => {
    switch (activeView) {
      case 'entity-explorer': return <EntityExplorer />;
      case 'adapter-registry': return <AdapterRegistry />;
      case 'message-inspector': return <MessageInspector />;
      case 'schema-validator': return <SchemaValidator />;
      case 'inference-dashboard': return <InferenceDashboard />;
    }
  };

  return (
    <ErrorBoundary>
    <div className="fixed inset-0 flex flex-row" style={{ background: '#080C0A' }}>
      {/* Sidebar */}
      <div
        className="flex-none flex flex-col"
        style={{
          width: '200px',
          background: '#0D1210',
          borderRight: '1px solid rgba(0,232,150,0.15)',
        }}
      >
        {/* Top wordmark */}
        <div
          className="px-4 py-3"
          style={{ borderBottom: '1px solid rgba(0,232,150,0.15)' }}
        >
          <div
            className="text-xs font-bold tracking-widest"
            style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: 'rgba(0,232,150,0.5)' }}
          >
            HELI.OS DEV
          </div>
        </div>

        {/* Separator */}
        <div style={{ height: '1px', background: 'rgba(0,232,150,0.08)' }} />

        {/* Nav items */}
        <div className="flex-1 py-2">
          {NAV_ITEMS.map((item) => {
            const isActive = activeView === item.id;
            return (
              <button
                key={item.id}
                onClick={() => setActiveView(item.id)}
                className="w-full text-left px-4 py-2.5 text-xs transition-colors"
                style={{
                  fontFamily: 'var(--font-ibm-plex-mono), monospace',
                  color: isActive ? '#00E896' : 'rgba(200,230,201,0.5)',
                  background: isActive ? 'rgba(0,232,150,0.05)' : 'transparent',
                  borderTop: 'none',
                  borderRight: 'none',
                  borderBottom: 'none',
                  borderLeft: isActive ? '2px solid #00E896' : '2px solid transparent',
                  cursor: 'pointer',
                  display: 'block',
                }}
              >
                {item.label}
              </button>
            );
          })}
        </div>

        {/* Bottom: switch role */}
        <div
          className="flex-none p-3"
          style={{ borderTop: '1px solid rgba(0,232,150,0.15)' }}
        >
          <button
            onClick={onSwitchRole}
            className="w-full text-[10px] py-2 tracking-widest transition-colors"
            style={{
              fontFamily: 'var(--font-ibm-plex-mono), monospace',
              color: 'rgba(200,230,201,0.45)',
              border: '1px solid rgba(0,232,150,0.15)',
              background: 'transparent',
              cursor: 'pointer',
            }}
            onMouseEnter={(e) => {
              (e.currentTarget as HTMLButtonElement).style.color = '#00E896';
              (e.currentTarget as HTMLButtonElement).style.borderColor = 'rgba(0,232,150,0.4)';
            }}
            onMouseLeave={(e) => {
              (e.currentTarget as HTMLButtonElement).style.color = 'rgba(200,230,201,0.45)';
              (e.currentTarget as HTMLButtonElement).style.borderColor = 'rgba(0,232,150,0.15)';
            }}
          >
            ← SWITCH ROLE
          </button>
        </div>
      </div>

      {/* Content area */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Header strip */}
        <div
          className="flex-none flex items-center px-4"
          style={{ height: '40px', background: '#0D1210', borderBottom: '1px solid rgba(0,232,150,0.15)' }}
        >
          <span
            className="text-sm font-bold tracking-widest"
            style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: '#00E896' }}
          >
            {VIEW_LABELS[activeView]}
          </span>
        </div>

        {/* View content */}
        <div className="flex-1 overflow-hidden">
          {renderView()}
        </div>
      </div>
    </div>
    </ErrorBoundary>
  );
}
