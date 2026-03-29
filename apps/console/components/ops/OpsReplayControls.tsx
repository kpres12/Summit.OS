'use client';

import React from 'react';
import { UseReplayResult } from '@/hooks/useReplay';

interface OpsReplayControlsProps {
  replay: UseReplayResult;
  onClose: () => void;
}

export default function OpsReplayControls({ replay, onClose }: OpsReplayControlsProps) {
  const { timeline, snapshot, index, isPlaying, isLoading, error, play, pause, seek, step } = replay;

  const mono: React.CSSProperties = {
    fontFamily: 'var(--font-ibm-plex-mono), monospace',
  };
  const orbitron: React.CSSProperties = {
    fontFamily: 'var(--font-orbitron), Orbitron, sans-serif',
  };

  if (isLoading) {
    return (
      <div style={{ ...mono, color: 'rgba(200,230,201,0.45)', fontSize: 10, padding: 8 }}>
        LOADING REPLAY…
      </div>
    );
  }

  if (error) {
    return (
      <div style={{ ...mono, color: '#FF3B3B', fontSize: 10, padding: 8 }}>
        REPLAY ERROR: {error}
      </div>
    );
  }

  if (!timeline || timeline.count === 0) {
    return (
      <div style={{ ...mono, color: 'rgba(200,230,201,0.35)', fontSize: 10, padding: 8 }}>
        NO REPLAY DATA
      </div>
    );
  }

  const currentTs = snapshot?.ts_iso
    ? new Date(snapshot.ts_iso).toLocaleTimeString()
    : '--:--:--';

  return (
    <div
      style={{
        background: '#0D1610',
        border: '1px solid rgba(0,255,156,0.2)',
        padding: '8px 12px',
        display: 'flex',
        flexDirection: 'column',
        gap: 6,
      }}
    >
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <span style={{ ...orbitron, color: '#FFB300', fontSize: 9, letterSpacing: '0.15em' }}>
          MISSION REPLAY
        </span>
        <button
          onClick={onClose}
          style={{ ...mono, color: 'rgba(200,230,201,0.45)', background: 'none', border: 'none', cursor: 'pointer', fontSize: 10 }}
        >
          ✕
        </button>
      </div>

      {/* Timeline scrubber */}
      <input
        type="range"
        min={0}
        max={timeline.count - 1}
        value={index}
        onChange={(e) => seek(Number(e.target.value))}
        style={{ width: '100%', accentColor: '#FFB300', cursor: 'pointer' }}
      />

      {/* Time display */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <span style={{ ...mono, color: 'rgba(200,230,201,0.45)', fontSize: 10 }}>
          {timeline.start ? new Date(timeline.start).toLocaleTimeString() : ''}
        </span>
        <span style={{ ...mono, color: '#FFB300', fontSize: 11, fontWeight: 'bold' }}>
          {currentTs}
        </span>
        <span style={{ ...mono, color: 'rgba(200,230,201,0.45)', fontSize: 10 }}>
          {timeline.end ? new Date(timeline.end).toLocaleTimeString() : ''}
        </span>
      </div>

      {/* Transport controls */}
      <div style={{ display: 'flex', gap: 4, justifyContent: 'center' }}>
        {[
          { label: '|◀', action: () => seek(0), title: 'Start' },
          { label: '◀', action: () => step(-10), title: '-10s' },
          { label: isPlaying ? '⏸' : '▶', action: isPlaying ? pause : play, title: isPlaying ? 'Pause' : 'Play', active: isPlaying },
          { label: '▶', action: () => step(10), title: '+10s' },
          { label: '▶|', action: () => seek(timeline.count - 1), title: 'End' },
        ].map(({ label, action, title, active }) => (
          <button
            key={title}
            title={title}
            onClick={action}
            style={{
              ...mono,
              fontSize: 12,
              padding: '3px 10px',
              background: active ? 'rgba(255,179,0,0.15)' : 'rgba(0,255,156,0.05)',
              border: `1px solid ${active ? 'rgba(255,179,0,0.4)' : 'rgba(0,255,156,0.15)'}`,
              color: active ? '#FFB300' : '#00FF9C',
              cursor: 'pointer',
            }}
          >
            {label}
          </button>
        ))}
      </div>

      {/* Frame counter */}
      <div style={{ ...mono, color: 'rgba(200,230,201,0.35)', fontSize: 9, textAlign: 'center' }}>
        FRAME {index + 1} / {timeline.count}
      </div>

      {/* Events in current snapshot */}
      {snapshot?.events && snapshot.events.length > 0 && (
        <div style={{ borderTop: '1px solid rgba(0,255,156,0.1)', paddingTop: 4 }}>
          {snapshot.events.slice(0, 3).map((ev, i) => (
            <div key={i} style={{ ...mono, fontSize: 9, color: '#FFB300', marginBottom: 2 }}>
              {ev.type}: {ev.description}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
