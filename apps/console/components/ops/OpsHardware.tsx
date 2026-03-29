'use client';

import React, { useState, useEffect, useCallback } from 'react';
import { useEntityStream } from '@/hooks/useEntityStream';

const API = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

// ── Types ──────────────────────────────────────────────────────────────────

interface AdapterStatus {
  id: string;
  name: string;
  protocol: string;
  status: 'online' | 'offline' | 'error';
  entity_count: number;
  last_seen: number;
  connection: string;
}

const PROTOCOLS = [
  { id: 'mavlink',  label: 'MAVLink (Drone)',          placeholder: 'udp:192.168.1.100:14550' },
  { id: 'modbus',   label: 'Modbus/TCP (PLC / Sensor)', placeholder: '192.168.1.10:502' },
  { id: 'opcua',    label: 'OPC-UA (Industrial)',       placeholder: 'opc.tcp://192.168.1.20:4840' },
  { id: 'rtsp',     label: 'RTSP Camera',               placeholder: 'rtsp://192.168.1.50/stream' },
  { id: 'opensky',  label: 'ADS-B / OpenSky',           placeholder: '(no config needed)' },
  { id: 'celestrak',label: 'Satellite / CelesTrak',     placeholder: '(no config needed)' },
];

// ── Sub-components ─────────────────────────────────────────────────────────

function StatusDot({ status }: { status: AdapterStatus['status'] }) {
  const color = status === 'online' ? '#00FF9C' : status === 'error' ? '#FF3B3B' : 'rgba(200,230,201,0.3)';
  return (
    <div
      style={{
        width: 6, height: 6, borderRadius: '50%',
        background: color,
        boxShadow: status === 'online' ? `0 0 6px ${color}` : 'none',
        flexShrink: 0,
      }}
    />
  );
}

function AdapterCard({ adapter }: { adapter: AdapterStatus }) {
  const age = Math.floor((Date.now() / 1000) - adapter.last_seen);
  const ageStr = age < 60 ? `${age}s ago` : `${Math.floor(age / 60)}m ago`;

  return (
    <div
      style={{
        padding: '10px 12px',
        background: adapter.status === 'online' ? 'rgba(0,255,156,0.03)' : 'rgba(0,0,0,0.2)',
        borderLeft: `2px solid ${adapter.status === 'online' ? 'rgba(0,255,156,0.4)' : 'rgba(200,230,201,0.1)'}`,
        marginBottom: 6,
      }}
    >
      <div className="flex items-center gap-2 mb-1">
        <StatusDot status={adapter.status} />
        <span style={{
          fontFamily: 'var(--font-orbitron), Orbitron, sans-serif',
          color: adapter.status === 'online' ? '#00FF9C' : 'rgba(200,230,201,0.4)',
          fontSize: 10,
          fontWeight: 700,
          letterSpacing: '0.1em',
        }}>
          {adapter.name.toUpperCase()}
        </span>
        <span style={{
          marginLeft: 'auto',
          fontFamily: 'var(--font-ibm-plex-mono), monospace',
          color: 'rgba(200,230,201,0.3)',
          fontSize: 9,
        }}>
          {adapter.status === 'online' ? `${adapter.entity_count} entities` : 'offline'}
        </span>
      </div>
      <div style={{
        fontFamily: 'var(--font-ibm-plex-mono), monospace',
        color: 'rgba(200,230,201,0.3)',
        fontSize: 9,
      }}>
        {adapter.protocol.toUpperCase()} · {adapter.connection} · {ageStr}
      </div>
    </div>
  );
}

// ── Main component ─────────────────────────────────────────────────────────

export default function OpsHardware() {
  const { entityList } = useEntityStream();
  const [adapters, setAdapters] = useState<AdapterStatus[]>([]);
  const [protocol, setProtocol] = useState(PROTOCOLS[0].id);
  const [connection, setConnection] = useState('');
  const [deviceName, setDeviceName] = useState('');
  const [adding, setAdding] = useState(false);
  const [feedback, setFeedback] = useState<{ ok: boolean; msg: string } | null>(null);
  const [showForm, setShowForm] = useState(false);

  // Derive live adapter list from entities + poll /adapters endpoint
  const refreshAdapters = useCallback(async () => {
    try {
      const r = await fetch(`${API}/adapters`);
      if (r.ok) {
        const data = await r.json();
        setAdapters(data.adapters || []);
        return;
      }
    } catch { /* fall through to entity-derived list */ }

    // Fallback: derive from entity sources
    const bySource: Record<string, AdapterStatus> = {};
    for (const e of entityList) {
      const src = e.source_sensors?.[0] || 'unknown';
      if (!bySource[src]) {
        bySource[src] = {
          id: src,
          name: src.replace(/-/g, ' '),
          protocol: src.includes('mavlink') ? 'mavlink'
            : src.includes('adsb') ? 'adsb'
            : src.includes('modbus') ? 'modbus'
            : 'unknown',
          status: 'online',
          entity_count: 0,
          last_seen: e.last_seen,
          connection: src,
        };
      }
      bySource[src].entity_count++;
      if (e.last_seen > bySource[src].last_seen) bySource[src].last_seen = e.last_seen;
    }
    setAdapters(Object.values(bySource));
  }, [entityList]);

  useEffect(() => {
    refreshAdapters();
    const t = setInterval(refreshAdapters, 5000);
    return () => clearInterval(t);
  }, [refreshAdapters]);

  const selectedProtocol = PROTOCOLS.find(p => p.id === protocol) || PROTOCOLS[0];

  const handleAdd = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!connection.trim() && !['opensky', 'celestrak'].includes(protocol)) return;
    setAdding(true);
    setFeedback(null);
    try {
      const r = await fetch(`${API}/adapters`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          protocol,
          connection: connection.trim() || selectedProtocol.placeholder,
          name: deviceName.trim() || `${protocol}-${Date.now()}`,
        }),
      });
      const data = await r.json();
      if (data.ok) {
        setFeedback({ ok: true, msg: `${(deviceName || protocol).toUpperCase()} connected` });
        setConnection('');
        setDeviceName('');
        setShowForm(false);
        setTimeout(refreshAdapters, 1000);
      } else {
        setFeedback({ ok: false, msg: data.error || 'Connection failed' });
      }
    } catch {
      setFeedback({ ok: false, msg: 'Could not reach server' });
    } finally {
      setAdding(false);
    }
  };

  const onlineCount = adapters.filter(a => a.status === 'online').length;

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div
        className="flex-none px-3 py-2 flex items-center justify-between"
        style={{ borderBottom: '1px solid rgba(0,255,156,0.15)' }}
      >
        <div className="flex items-center gap-2">
          <span style={{
            fontFamily: 'var(--font-orbitron), Orbitron, sans-serif',
            color: '#00FF9C', fontSize: 11, fontWeight: 700, letterSpacing: '0.1em',
          }}>
            HARDWARE
          </span>
          <span style={{
            fontFamily: 'var(--font-ibm-plex-mono), monospace',
            color: 'rgba(200,230,201,0.4)', fontSize: 9,
          }}>
            {onlineCount} online
          </span>
        </div>
        <button
          onClick={() => { setShowForm(f => !f); setFeedback(null); }}
          style={{
            fontFamily: 'var(--font-ibm-plex-mono), monospace',
            fontSize: 9, color: showForm ? '#FF3B3B' : '#00FF9C',
            background: 'none', border: `1px solid ${showForm ? 'rgba(255,59,59,0.3)' : 'rgba(0,255,156,0.3)'}`,
            padding: '3px 8px', cursor: 'pointer', letterSpacing: '0.1em',
          }}
        >
          {showForm ? 'CANCEL' : '+ ADD'}
        </button>
      </div>

      <div className="flex-1 overflow-y-auto">
        {/* Add equipment form */}
        {showForm && (
          <form
            onSubmit={handleAdd}
            style={{
              padding: '12px',
              borderBottom: '1px solid rgba(0,255,156,0.15)',
              background: 'rgba(0,255,156,0.02)',
            }}
          >
            {/* Protocol selector */}
            <div style={{ marginBottom: 8 }}>
              <div style={{
                fontFamily: 'var(--font-ibm-plex-mono), monospace',
                color: 'rgba(200,230,201,0.45)', fontSize: 9,
                letterSpacing: '0.1em', marginBottom: 4,
              }}>
                HARDWARE TYPE
              </div>
              <div className="flex flex-col gap-1">
                {PROTOCOLS.map(p => (
                  <button
                    key={p.id}
                    type="button"
                    onClick={() => { setProtocol(p.id); setConnection(''); }}
                    style={{
                      textAlign: 'left', padding: '5px 8px',
                      background: protocol === p.id ? 'rgba(0,255,156,0.08)' : 'transparent',
                      border: `1px solid ${protocol === p.id ? 'rgba(0,255,156,0.4)' : 'rgba(0,255,156,0.1)'}`,
                      color: protocol === p.id ? '#00FF9C' : 'rgba(200,230,201,0.5)',
                      fontFamily: 'var(--font-ibm-plex-mono), monospace',
                      fontSize: 10, cursor: 'pointer',
                    }}
                  >
                    {p.label}
                  </button>
                ))}
              </div>
            </div>

            {/* Device name */}
            <div style={{ marginBottom: 8 }}>
              <div style={{
                fontFamily: 'var(--font-ibm-plex-mono), monospace',
                color: 'rgba(200,230,201,0.45)', fontSize: 9,
                letterSpacing: '0.1em', marginBottom: 4,
              }}>
                NAME (optional)
              </div>
              <input
                type="text"
                value={deviceName}
                onChange={e => setDeviceName(e.target.value)}
                placeholder={`e.g. Field Drone Alpha`}
                style={{
                  width: '100%', background: 'rgba(0,0,0,0.3)',
                  border: '1px solid rgba(0,255,156,0.2)',
                  color: '#00FF9C', padding: '5px 8px',
                  fontFamily: 'var(--font-ibm-plex-mono), monospace', fontSize: 10,
                  outline: 'none',
                }}
              />
            </div>

            {/* Connection string */}
            {!['opensky', 'celestrak'].includes(protocol) && (
              <div style={{ marginBottom: 10 }}>
                <div style={{
                  fontFamily: 'var(--font-ibm-plex-mono), monospace',
                  color: 'rgba(200,230,201,0.45)', fontSize: 9,
                  letterSpacing: '0.1em', marginBottom: 4,
                }}>
                  CONNECTION
                </div>
                <input
                  type="text"
                  value={connection}
                  onChange={e => setConnection(e.target.value)}
                  placeholder={selectedProtocol.placeholder}
                  style={{
                    width: '100%', background: 'rgba(0,0,0,0.3)',
                    border: '1px solid rgba(0,255,156,0.2)',
                    color: '#00FF9C', padding: '5px 8px',
                    fontFamily: 'var(--font-ibm-plex-mono), monospace', fontSize: 10,
                    outline: 'none',
                  }}
                />
              </div>
            )}

            <button
              type="submit"
              disabled={adding}
              style={{
                width: '100%', padding: '7px',
                background: adding ? 'rgba(0,255,156,0.3)' : '#00FF9C',
                color: '#080C0A', border: 'none', cursor: adding ? 'wait' : 'pointer',
                fontFamily: 'var(--font-orbitron), Orbitron, sans-serif',
                fontSize: 10, fontWeight: 700, letterSpacing: '0.2em',
              }}
            >
              {adding ? 'CONNECTING...' : 'CONNECT'}
            </button>

            {feedback && (
              <div style={{
                marginTop: 8, padding: '5px 8px',
                fontFamily: 'var(--font-ibm-plex-mono), monospace', fontSize: 10,
                color: feedback.ok ? '#00FF9C' : '#FF3B3B',
                border: `1px solid ${feedback.ok ? 'rgba(0,255,156,0.2)' : 'rgba(255,59,59,0.2)'}`,
              }}>
                {feedback.ok ? '✓ ' : '✗ '}{feedback.msg}
              </div>
            )}
          </form>
        )}

        {/* Connected hardware list */}
        <div style={{ padding: '10px 12px' }}>
          {adapters.length === 0 ? (
            <div style={{
              fontFamily: 'var(--font-ibm-plex-mono), monospace',
              color: 'rgba(200,230,201,0.25)', fontSize: 10,
              textAlign: 'center', padding: '24px 0',
            }}>
              No hardware connected.<br />
              <span style={{ fontSize: 9 }}>Press + ADD to connect a device.</span>
            </div>
          ) : (
            adapters.map(a => <AdapterCard key={a.id} adapter={a} />)
          )}
        </div>
      </div>
    </div>
  );
}
