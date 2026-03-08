'use client';

import React from 'react';
import { useEntityStream, EntityData } from '../../hooks/useEntityStream';
import { MapLayer } from './MapLayerControls';
import { Geofence } from './GeofenceEditor';

const DOMAIN_LABELS: Record<string, string> = {
  aerial: 'UAV',
  ground: 'GND',
  maritime: 'MAR',
  fixed: 'FIX',
  sensor: 'SEN',
};

function deriveStatus(e: EntityData): 'ACTIVE' | 'IDLE' | 'WARNING' | 'OFFLINE' {
  const ageSec = (Date.now() - e.last_seen * 1000) / 1000;
  if (ageSec > 300) return 'OFFLINE';
  if (e.track_state === 'coasting') return 'WARNING';
  if ((e.battery_pct ?? 100) < 25) return 'WARNING';
  if (e.speed_mps > 0.5 || e.mission_id) return 'ACTIVE';
  return 'IDLE';
}

interface LeftSidebarProps {
  layers: MapLayer[];
  onToggleLayer: (layerId: string) => void;
  geofences: Geofence[];
  editingGeofence: Geofence | null;
  onCreateGeofence: () => void;
  onSelectGeofence: (id: string) => void;
  onDeleteGeofence: (id: string) => void;
  onSaveGeofence: (geofence: Geofence) => void;
  onCancelEdit: () => void;
  onToggleGeofenceActive: (id: string) => void;
}

export default function LeftSidebar({
  layers,
  onToggleLayer,
  geofences,
  editingGeofence,
  onCreateGeofence,
  onSelectGeofence,
  onDeleteGeofence,
  onSaveGeofence,
  onCancelEdit,
  onToggleGeofenceActive,
}: LeftSidebarProps) {
  const { entityList, connected, entityCount } = useEntityStream();

  return (
    <div className="w-72 bg-[#0F0F0F] border-r-2 border-[#00FF91]/20 flex flex-col overflow-hidden">
      {/* ── ASSETS ── */}
      <div className="h-10 border-b border-[#00FF91]/20 flex items-center px-4 bg-[#0A0A0A]">
        <div className="text-[#00FF91] text-sm font-semibold tracking-wider uppercase">
          ASSETS
        </div>
        <div className="ml-auto flex items-center gap-2">
          <div
            className={`w-1.5 h-1.5 rounded-full ${connected ? 'bg-[#00FF91]' : 'bg-red-500'}`}
            style={connected ? { boxShadow: '0 0 4px #00FF91' } : {}}
          />
          <div className="text-[10px] text-[#006644] font-mono">
            {entityCount > 0 ? entityCount : '—'}
          </div>
        </div>
      </div>
      <div className="flex-1 overflow-y-auto overflow-x-hidden">
        {entityList.length === 0 ? (
          <div className="px-4 py-6">
            <div className="text-[11px] text-[#00FF91] font-mono tracking-wider mb-4">
              AWAITING CONNECTIONS
            </div>
            {!connected && (
              <div className="mb-4 px-2 py-1.5 border border-red-500/30 bg-red-500/5">
                <div className="text-[10px] text-red-400 font-mono">WS DISCONNECTED</div>
              </div>
            )}
            <div className="mb-4">
              <div className="text-[10px] text-[#006644] font-mono mb-2">Connect your first asset:</div>
              <div className="bg-[#0A0A0A] border border-[#00FF91]/20 px-2 py-1.5">
                <code className="text-[10px] text-[#00CC74] font-mono">$ pip install summit-os-sdk</code>
              </div>
            </div>
            <div className="text-[10px] text-[#006644] font-mono mb-2">Expecting:</div>
            <div className="space-y-1.5">
              {[
                { type: 'DRONE', icon: '○' },
                { type: 'UGV', icon: '○' },
                { type: 'TOWER', icon: '○' },
                { type: 'SENSOR', icon: '○' },
              ].map(({ type, icon }) => (
                <div key={type} className="flex items-center gap-2 text-[10px] font-mono">
                  <span className="text-[#006644]">{icon}</span>
                  <span className="text-[#006644]">{type}</span>
                  <span className="text-[#004422] ml-auto">— 0 connected</span>
                </div>
              ))}
            </div>
          </div>
        ) : (
          entityList.map((entity) => (
            <AssetRow key={entity.entity_id} entity={entity} />
          ))
        )}

        {/* ── LAYERS ── */}
        <div className="border-t border-[#00FF91]/20">
          <div className="h-10 flex items-center px-4 bg-[#0A0A0A]">
            <div className="text-[#00FF91] text-sm font-semibold tracking-wider uppercase">
              LAYERS
            </div>
            <div className="ml-auto text-[10px] text-[#006644] font-mono">
              {layers.filter((l) => l.enabled).length}/{layers.length}
            </div>
          </div>
          <div className="px-2 py-1.5 space-y-0.5">
            {layers.map((layer) => (
              <div
                key={layer.id}
                className="flex items-center gap-2 px-2 py-1.5 hover:bg-[#00FF91]/5 transition-colors"
              >
                <button
                  onClick={() => onToggleLayer(layer.id)}
                  className="flex-shrink-0 w-4 h-4 border flex items-center justify-center transition-colors"
                  style={{
                    borderColor: layer.enabled ? layer.color : '#006644',
                    backgroundColor: layer.enabled ? `${layer.color}20` : 'transparent',
                  }}
                >
                  {layer.enabled && (
                    <span className="text-[10px]" style={{ color: layer.color }}>✓</span>
                  )}
                </button>
                <div
                  className="flex-shrink-0 w-5 h-5 flex items-center justify-center text-xs"
                  style={{ color: layer.enabled ? layer.color : '#006644' }}
                >
                  {layer.icon}
                </div>
                <span
                  className={`flex-1 text-[11px] font-mono ${
                    layer.enabled ? 'text-[#00CC74]' : 'text-[#006644]'
                  }`}
                >
                  {layer.name}
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* ── GEOFENCES ── */}
        <div className="border-t border-[#00FF91]/20">
          <div className="h-10 flex items-center px-4 bg-[#0A0A0A]">
            <div className="text-[#00FF91] text-sm font-semibold tracking-wider uppercase">
              GEOFENCES
            </div>
            <div className="ml-auto flex items-center gap-2">
              <button
                onClick={onCreateGeofence}
                className="text-[#00FF91] hover:text-[#00CC74] text-xs font-bold"
                title="Create New Geofence"
              >
                +
              </button>
              <div className="text-[10px] text-[#006644] font-mono">
                {geofences.filter((g) => g.active).length}/{geofences.length}
              </div>
            </div>
          </div>

          {/* Editor Mode */}
          {editingGeofence && (
            <GeofenceEditForm
              editingGeofence={editingGeofence}
              onSaveGeofence={onSaveGeofence}
              onCancelEdit={onCancelEdit}
            />
          )}

          {/* Geofence List */}
          <div className="px-2 py-1.5 space-y-0.5">
            {geofences.length === 0 ? (
              <div className="text-center py-3 text-[#006644] text-[10px] font-mono">
                No geofences defined
                <br />
                Click + to create
              </div>
            ) : (
              geofences.map((geofence) => (
                <GeofenceRow
                  key={geofence.id}
                  geofence={geofence}
                  onSelect={onSelectGeofence}
                  onDelete={onDeleteGeofence}
                  onToggleActive={onToggleGeofenceActive}
                />
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

/* ── Sub-components ─────────────────────────────────────── */

function AssetRow({ entity }: { entity: EntityData }) {
  const status = deriveStatus(entity);
  const label = entity.callsign || entity.entity_id.slice(0, 12);
  const domainTag = DOMAIN_LABELS[entity.domain] || entity.domain?.toUpperCase() || '?';
  const battery = entity.battery_pct;
  const conf = Math.round(entity.confidence * 100);
  const speed = entity.speed_mps;

  const statusColor = {
    ACTIVE: '#00FF91',
    IDLE: '#006644',
    WARNING: '#FF9933',
    OFFLINE: '#FF3333',
  };
  const color = statusColor[status];

  return (
    <div className="border-b border-[#00FF91]/10 px-4 py-2 hover:bg-[#00FF91]/5 transition-colors cursor-pointer">
      <div className="flex items-center justify-between mb-1">
        <div className="flex items-center gap-2">
          <div
            className="w-1.5 h-1.5 rounded-full"
            style={{ backgroundColor: color, boxShadow: `0 0 4px ${color}80` }}
          />
          <div className="text-[#00CC74] text-xs font-mono">{label}</div>
        </div>
        <div
          className="text-[8px] px-1.5 py-0.5 font-semibold tracking-wider border"
          style={{ color, borderColor: `${color}40`, backgroundColor: `${color}10` }}
        >
          {domainTag}
        </div>
      </div>
      <div className="flex gap-3 text-[10px] font-mono">
        {battery !== undefined && battery !== null && (
          <span className={battery < 30 ? 'text-[#FF9933]' : 'text-[#006644]'}>
            {Math.round(battery)}%
          </span>
        )}
        <span className={conf < 70 ? 'text-[#FF9933]' : 'text-[#006644]'}>
          {conf}% conf
        </span>
        {speed > 0.1 && (
          <span className="text-[#006644]">
            {speed.toFixed(1)} m/s
          </span>
        )}
      </div>
    </div>
  );
}

function GeofenceRow({
  geofence,
  onSelect,
  onDelete,
  onToggleActive,
}: {
  geofence: Geofence;
  onSelect: (id: string) => void;
  onDelete: (id: string) => void;
  onToggleActive: (id: string) => void;
}) {
  const getTypeColor = (type: string) => {
    switch (type) {
      case 'exclusion': return '#FF3333';
      case 'inclusion': return '#00FF91';
      case 'warning': return '#FF9933';
      default: return '#00FF91';
    }
  };

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'exclusion': return '⊘';
      case 'inclusion': return '✓';
      case 'warning': return '⚠';
      default: return '○';
    }
  };

  const color = getTypeColor(geofence.type);

  return (
    <div className="flex items-center gap-2 px-2 py-1.5 hover:bg-[#00FF91]/5 transition-colors">
      <button
        onClick={() => onToggleActive(geofence.id)}
        className="flex-shrink-0 w-4 h-4 border flex items-center justify-center hover:border-[#00FF91] transition-colors"
        style={{
          borderColor: geofence.active ? color : '#006644',
          backgroundColor: geofence.active ? `${color}20` : 'transparent',
        }}
      >
        {geofence.active && (
          <span className="text-[10px]" style={{ color }}>✓</span>
        )}
      </button>
      <div
        className="flex-shrink-0 w-6 h-6 flex items-center justify-center text-sm"
        style={{
          color: geofence.active ? color : '#006644',
          filter: geofence.active ? `drop-shadow(0 0 2px ${color})` : 'none',
        }}
      >
        {getTypeIcon(geofence.type)}
      </div>
      <button
        onClick={() => onSelect(geofence.id)}
        className={`flex-1 text-left text-[11px] font-mono hover:text-[#00FF91] transition-colors ${
          geofence.active ? 'text-[#00FF91]' : 'text-[#006644]'
        }`}
      >
        {geofence.name}
        <div className="text-[9px] text-[#006644]">
          {geofence.points.length} pts
          {geofence.altitude_min !== undefined &&
            ` • ${geofence.altitude_min}-${geofence.altitude_max}m`}
        </div>
      </button>
      <button
        onClick={() => onDelete(geofence.id)}
        className="flex-shrink-0 text-[#FF3333] hover:text-[#FF6666] text-[10px] px-1"
        title="Delete Geofence"
      >
        ✕
      </button>
    </div>
  );
}

function GeofenceEditForm({
  editingGeofence,
  onSaveGeofence,
  onCancelEdit,
}: {
  editingGeofence: Geofence;
  onSaveGeofence: (geofence: Geofence) => void;
  onCancelEdit: () => void;
}) {
  const [editForm, setEditForm] = React.useState<Partial<Geofence>>(editingGeofence);

  React.useEffect(() => {
    setEditForm(editingGeofence);
  }, [editingGeofence]);

  return (
    <div className="p-3 border-b border-[#00FF91]/30 space-y-2">
      <div className="text-[10px] text-[#00FF91] font-mono mb-2">
        {editingGeofence.id === 'new' ? 'NEW GEOFENCE' : 'EDIT GEOFENCE'}
      </div>
      <div>
        <label className="text-[10px] text-[#006644] font-mono">NAME</label>
        <input
          type="text"
          value={editForm.name || ''}
          onChange={(e) => setEditForm({ ...editForm, name: e.target.value })}
          className="w-full bg-[#0A0A0A] border border-[#00FF91]/40 text-[#00FF91] text-xs font-mono px-2 py-1 focus:outline-none focus:border-[#00FF91]"
          placeholder="Geofence name"
        />
      </div>
      <div>
        <label className="text-[10px] text-[#006644] font-mono">TYPE</label>
        <select
          value={editForm.type || 'warning'}
          onChange={(e) =>
            setEditForm({ ...editForm, type: e.target.value as Geofence['type'] })
          }
          className="w-full bg-[#0A0A0A] border border-[#00FF91]/40 text-[#00FF91] text-xs font-mono px-2 py-1 focus:outline-none focus:border-[#00FF91]"
        >
          <option value="inclusion">Inclusion Zone</option>
          <option value="exclusion">Exclusion Zone</option>
          <option value="warning">Warning Zone</option>
        </select>
      </div>
      <div className="grid grid-cols-2 gap-2">
        <div>
          <label className="text-[10px] text-[#006644] font-mono">ALT MIN (m)</label>
          <input
            type="number"
            value={editForm.altitude_min || ''}
            onChange={(e) =>
              setEditForm({ ...editForm, altitude_min: parseInt(e.target.value) || undefined })
            }
            className="w-full bg-[#0A0A0A] border border-[#00FF91]/40 text-[#00FF91] text-xs font-mono px-2 py-1 focus:outline-none focus:border-[#00FF91]"
            placeholder="0"
          />
        </div>
        <div>
          <label className="text-[10px] text-[#006644] font-mono">ALT MAX (m)</label>
          <input
            type="number"
            value={editForm.altitude_max || ''}
            onChange={(e) =>
              setEditForm({ ...editForm, altitude_max: parseInt(e.target.value) || undefined })
            }
            className="w-full bg-[#0A0A0A] border border-[#00FF91]/40 text-[#00FF91] text-xs font-mono px-2 py-1 focus:outline-none focus:border-[#00FF91]"
            placeholder="500"
          />
        </div>
      </div>
      <div className="text-[10px] text-[#006644] font-mono">
        VERTICES: {editForm.points?.length || 0} (click map to add)
      </div>
      <div className="flex gap-2 pt-2">
        <button
          onClick={() => onSaveGeofence(editForm as Geofence)}
          disabled={!editForm.name || (editForm.points?.length || 0) < 3}
          className="flex-1 bg-[#00FF91] text-[#0A0A0A] text-xs font-mono px-3 py-1.5 hover:bg-[#00CC74] disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
        >
          SAVE
        </button>
        <button
          onClick={onCancelEdit}
          className="flex-1 bg-[#0A0A0A] border border-[#00FF91]/40 text-[#00FF91] text-xs font-mono px-3 py-1.5 hover:bg-[#00FF91]/10 transition-colors"
        >
          CANCEL
        </button>
      </div>
    </div>
  );
}
