'use client';

import React from 'react';
import type { EntityData, TrackData } from '../../hooks/useEntityStream';

interface EntityDetailProps {
  entity: EntityData | null;
  track?: TrackData | null;
  onClose: () => void;
}

const TYPE_COLORS: Record<string, string> = {
  friendly: 'text-blue-400',
  alert: 'text-red-400',
  neutral: 'text-gray-400',
  unknown: 'text-yellow-400',
};

const DOMAIN_ICONS: Record<string, string> = {
  aerial: '✈',
  ground: '🚗',
  maritime: '🚢',
  fixed: '🏗',
  sensor: '📡',
};

export default function EntityDetail({ entity, track, onClose }: EntityDetailProps) {
  if (!entity) return null;

  const typeColor = TYPE_COLORS[entity.entity_type] || 'text-gray-400';
  const domainIcon = DOMAIN_ICONS[entity.domain] || '?';

  return (
    <div className="absolute right-0 top-0 h-full w-80 bg-gray-900/95 border-l border-gray-700 p-4 overflow-y-auto z-50">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <span className="text-xl">{domainIcon}</span>
          <div>
            <h3 className={`font-bold text-sm ${typeColor}`}>
              {entity.callsign || entity.entity_id.slice(0, 8)}
            </h3>
            <p className="text-xs text-gray-500">{entity.entity_id}</p>
          </div>
        </div>
        <button
          onClick={onClose}
          className="text-gray-500 hover:text-white text-lg"
        >
          ✕
        </button>
      </div>

      {/* Classification badge */}
      <div className="flex gap-2 mb-4">
        <span className={`px-2 py-0.5 rounded text-xs font-medium ${typeColor} bg-gray-800`}>
          {entity.entity_type.toUpperCase()}
        </span>
        <span className="px-2 py-0.5 rounded text-xs font-medium text-gray-300 bg-gray-800">
          {entity.domain}
        </span>
        {entity.classification && (
          <span className="px-2 py-0.5 rounded text-xs font-medium text-gray-300 bg-gray-800">
            {entity.classification}
          </span>
        )}
      </div>

      {/* Position */}
      <Section title="Position">
        <Row label="Lat" value={entity.position.lat.toFixed(6)} />
        <Row label="Lon" value={entity.position.lon.toFixed(6)} />
        <Row label="Alt" value={`${entity.position.alt.toFixed(1)} m`} />
        <Row label="Heading" value={`${entity.position.heading_deg.toFixed(1)}°`} />
        <Row label="Speed" value={`${entity.speed_mps.toFixed(1)} m/s`} />
      </Section>

      {/* Track info */}
      {track && (
        <Section title="Track">
          <Row label="Track ID" value={track.track_id.slice(0, 8)} />
          <Row label="State" value={
            <span className={
              track.state === 'confirmed' ? 'text-green-400' :
              track.state === 'coasting' ? 'text-yellow-400' : 'text-gray-400'
            }>
              {track.state.toUpperCase()}
            </span>
          } />
          <Row label="Uncertainty" value={`${track.uncertainty_m.toFixed(1)} m`} />
          <Row label="Confidence" value={
            <ConfidenceBar value={track.confidence} />
          } />
          <Row label="Hits / Misses" value={`${track.hits} / ${track.misses}`} />
          <Row label="Sensors" value={track.contributing_sensors.join(', ')} />
        </Section>
      )}

      {/* Confidence */}
      <Section title="Confidence">
        <ConfidenceBar value={entity.confidence} />
        <p className="text-xs text-gray-500 mt-1">
          {entity.source_sensors.length} sensor(s): {entity.source_sensors.join(', ')}
        </p>
      </Section>

      {/* Timing */}
      <Section title="Timing">
        <Row label="Last seen" value={new Date(entity.last_seen * 1000).toLocaleTimeString()} />
      </Section>

      {/* Battery (if available) */}
      {entity.battery_pct !== undefined && (
        <Section title="Battery">
          <div className="flex items-center gap-2">
            <div className="flex-1 h-2 bg-gray-700 rounded">
              <div
                className={`h-full rounded ${
                  entity.battery_pct > 50 ? 'bg-green-500' :
                  entity.battery_pct > 20 ? 'bg-yellow-500' : 'bg-red-500'
                }`}
                style={{ width: `${entity.battery_pct}%` }}
              />
            </div>
            <span className="text-xs text-gray-400">{entity.battery_pct.toFixed(0)}%</span>
          </div>
        </Section>
      )}
    </div>
  );
}

// ── Helpers ──────────────────────────────────────────────────

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="mb-4">
      <h4 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2">
        {title}
      </h4>
      {children}
    </div>
  );
}

function Row({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="flex justify-between items-center py-0.5">
      <span className="text-xs text-gray-500">{label}</span>
      <span className="text-xs text-gray-300 font-mono">{value}</span>
    </div>
  );
}

function ConfidenceBar({ value }: { value: number }) {
  const pct = Math.round(value * 100);
  const color = pct > 70 ? 'bg-green-500' : pct > 40 ? 'bg-yellow-500' : 'bg-red-500';
  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 h-1.5 bg-gray-700 rounded">
        <div className={`h-full rounded ${color}`} style={{ width: `${pct}%` }} />
      </div>
      <span className="text-xs text-gray-400 w-8 text-right">{pct}%</span>
    </div>
  );
}
