'use client';

import React, { useState } from 'react';

export interface Geofence {
  id: string;
  name: string;
  points: Array<{ lat: number; lon: number }>;
  type: 'exclusion' | 'inclusion' | 'warning';
  altitude_min?: number;
  altitude_max?: number;
  active: boolean;
}

interface GeofenceEditorProps {
  geofences: Geofence[];
  editingGeofence: Geofence | null;
  onCreateNew: () => void;
  onSelectGeofence: (id: string) => void;
  onDeleteGeofence: (id: string) => void;
  onSaveGeofence: (geofence: Geofence) => void;
  onCancelEdit: () => void;
  onToggleActive: (id: string) => void;
}

export default function GeofenceEditor({
  geofences,
  editingGeofence,
  onCreateNew,
  onSelectGeofence,
  onDeleteGeofence,
  onSaveGeofence,
  onCancelEdit,
  onToggleActive,
}: GeofenceEditorProps) {
  const [collapsed, setCollapsed] = useState(false);
  const [editForm, setEditForm] = useState<Partial<Geofence>>(
    editingGeofence || {}
  );

  // Update form when editingGeofence changes
  React.useEffect(() => {
    if (editingGeofence) {
      setEditForm(editingGeofence);
    }
  }, [editingGeofence]);

  const getTypeColor = (type: string) => {
    switch (type) {
      case 'exclusion':
        return '#FF3333';
      case 'inclusion':
        return '#00FF91';
      case 'warning':
        return '#FF9933';
      default:
        return '#00FF91';
    }
  };

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'exclusion':
        return '⊘';
      case 'inclusion':
        return '✓';
      case 'warning':
        return '⚠';
      default:
        return '○';
    }
  };

  if (collapsed) {
    return (
      <div className="absolute top-20 left-80 z-10">
        <button
          onClick={() => setCollapsed(false)}
          className="px-3 py-2 bg-[#0F0F0F]/90 border border-[#00FF91]/40 text-[#00FF91] text-xs font-mono hover:bg-[#00FF91]/10 transition-colors backdrop-blur-sm"
          style={{
            boxShadow: '0 0 8px rgba(0, 255, 145, 0.2)',
          }}
        >
          GEOFENCES ▸
        </button>
      </div>
    );
  }

  return (
    <div className="absolute top-20 left-80 z-10 w-80">
      <div
        className="bg-[#0F0F0F]/95 border border-[#00FF91]/40 backdrop-blur-sm"
        style={{
          boxShadow: '0 0 12px rgba(0, 255, 145, 0.2)',
        }}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-3 py-2 border-b border-[#00FF91]/30">
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 bg-[#FF9933] animate-pulse" />
            <span className="text-[#00FF91] text-xs font-mono font-bold tracking-wider">
              GEOFENCES
            </span>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={onCreateNew}
              className="text-[#00FF91] hover:text-[#00CC74] text-xs font-bold"
              title="Create New Geofence"
            >
              +
            </button>
            <button
              onClick={() => setCollapsed(true)}
              className="text-[#00FF91] hover:text-[#00CC74] text-xs"
            >
              ◂
            </button>
          </div>
        </div>

        {/* Editor Mode */}
        {editingGeofence && (
          <div className="p-3 border-b border-[#00FF91]/30 space-y-2">
            <div className="text-[10px] text-[#00FF91] font-mono mb-2">
              {editingGeofence.id === 'new' ? 'NEW GEOFENCE' : 'EDIT GEOFENCE'}
            </div>

            {/* Name */}
            <div>
              <label className="text-[10px] text-[#006644] font-mono">
                NAME
              </label>
              <input
                type="text"
                value={editForm.name || ''}
                onChange={(e) =>
                  setEditForm({ ...editForm, name: e.target.value })
                }
                className="w-full bg-[#0A0A0A] border border-[#00FF91]/40 text-[#00FF91] text-xs font-mono px-2 py-1 focus:outline-none focus:border-[#00FF91]"
                placeholder="Geofence name"
              />
            </div>

            {/* Type */}
            <div>
              <label className="text-[10px] text-[#006644] font-mono">
                TYPE
              </label>
              <select
                value={editForm.type || 'warning'}
                onChange={(e) =>
                  setEditForm({
                    ...editForm,
                    type: e.target.value as Geofence['type'],
                  })
                }
                className="w-full bg-[#0A0A0A] border border-[#00FF91]/40 text-[#00FF91] text-xs font-mono px-2 py-1 focus:outline-none focus:border-[#00FF91]"
              >
                <option value="inclusion">Inclusion Zone</option>
                <option value="exclusion">Exclusion Zone</option>
                <option value="warning">Warning Zone</option>
              </select>
            </div>

            {/* Altitude Range */}
            <div className="grid grid-cols-2 gap-2">
              <div>
                <label className="text-[10px] text-[#006644] font-mono">
                  ALT MIN (m)
                </label>
                <input
                  type="number"
                  value={editForm.altitude_min || ''}
                  onChange={(e) =>
                    setEditForm({
                      ...editForm,
                      altitude_min: parseInt(e.target.value) || undefined,
                    })
                  }
                  className="w-full bg-[#0A0A0A] border border-[#00FF91]/40 text-[#00FF91] text-xs font-mono px-2 py-1 focus:outline-none focus:border-[#00FF91]"
                  placeholder="0"
                />
              </div>
              <div>
                <label className="text-[10px] text-[#006644] font-mono">
                  ALT MAX (m)
                </label>
                <input
                  type="number"
                  value={editForm.altitude_max || ''}
                  onChange={(e) =>
                    setEditForm({
                      ...editForm,
                      altitude_max: parseInt(e.target.value) || undefined,
                    })
                  }
                  className="w-full bg-[#0A0A0A] border border-[#00FF91]/40 text-[#00FF91] text-xs font-mono px-2 py-1 focus:outline-none focus:border-[#00FF91]"
                  placeholder="500"
                />
              </div>
            </div>

            {/* Points Info */}
            <div className="text-[10px] text-[#006644] font-mono">
              VERTICES: {editForm.points?.length || 0} (click map to add)
            </div>

            {/* Actions */}
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
        )}

        {/* Geofence List */}
        <div className="p-2 space-y-1 max-h-96 overflow-y-auto">
          {geofences.length === 0 ? (
            <div className="text-center py-4 text-[#006644] text-[10px] font-mono">
              No geofences defined
              <br />
              Click + to create
            </div>
          ) : (
            geofences.map((geofence) => (
              <div
                key={geofence.id}
                className="flex items-center gap-2 px-2 py-1.5 hover:bg-[#00FF91]/5 transition-colors"
              >
                {/* Active Toggle */}
                <button
                  onClick={() => onToggleActive(geofence.id)}
                  className="flex-shrink-0 w-4 h-4 border flex items-center justify-center hover:border-[#00FF91] transition-colors"
                  style={{
                    borderColor: geofence.active
                      ? getTypeColor(geofence.type)
                      : '#006644',
                    backgroundColor: geofence.active
                      ? `${getTypeColor(geofence.type)}20`
                      : 'transparent',
                  }}
                >
                  {geofence.active && (
                    <span
                      className="text-[10px]"
                      style={{ color: getTypeColor(geofence.type) }}
                    >
                      ✓
                    </span>
                  )}
                </button>

                {/* Type Icon */}
                <div
                  className="flex-shrink-0 w-6 h-6 flex items-center justify-center text-sm"
                  style={{
                    color: geofence.active
                      ? getTypeColor(geofence.type)
                      : '#006644',
                    filter: geofence.active
                      ? `drop-shadow(0 0 2px ${getTypeColor(geofence.type)})`
                      : 'none',
                  }}
                >
                  {getTypeIcon(geofence.type)}
                </div>

                {/* Name */}
                <button
                  onClick={() => onSelectGeofence(geofence.id)}
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

                {/* Delete Button */}
                <button
                  onClick={() => onDeleteGeofence(geofence.id)}
                  className="flex-shrink-0 text-[#FF3333] hover:text-[#FF6666] text-[10px] px-1"
                  title="Delete Geofence"
                >
                  ✕
                </button>
              </div>
            ))
          )}
        </div>

        {/* Footer */}
        <div className="px-3 py-2 border-t border-[#00FF91]/30 text-[10px] text-[#006644] font-mono">
          {geofences.filter((g) => g.active).length}/{geofences.length} active
        </div>
      </div>
    </div>
  );
}
