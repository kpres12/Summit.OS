'use client';

import React, { useState } from 'react';

export interface MapLayer {
  id: string;
  name: string;
  enabled: boolean;
  color: string;
  icon: string;
}

interface MapLayerControlsProps {
  layers: MapLayer[];
  onToggleLayer: (layerId: string) => void;
  onLayerSettings?: (layerId: string) => void;
}

export default function MapLayerControls({
  layers,
  onToggleLayer,
  onLayerSettings,
}: MapLayerControlsProps) {
  const [collapsed, setCollapsed] = useState(false);

  if (collapsed) {
    return (
      <div className="absolute top-20 right-4 z-10">
        <button
          onClick={() => setCollapsed(false)}
          className="px-3 py-2 bg-zinc-900/90 border border-zinc-700 text-zinc-300 text-xs font-mono hover:bg-zinc-800 transition-colors backdrop-blur-sm rounded"
        >
          ◂ LAYERS
        </button>
      </div>
    );
  }

  return (
    <div className="absolute top-20 right-4 z-10 w-56">
      <div className="bg-zinc-900/95 border border-zinc-700 backdrop-blur-sm rounded">
        {/* Header */}
        <div className="flex items-center justify-between px-3 py-2 border-b border-zinc-700">
          <span className="text-zinc-400 text-xs font-mono font-medium tracking-wider">
            LAYERS
          </span>
          <button
            onClick={() => setCollapsed(true)}
            className="text-zinc-500 hover:text-zinc-300 text-xs"
          >
            ▸
          </button>
        </div>

        {/* Layer List */}
        <div className="p-1.5 space-y-0.5">
          {layers.map((layer) => (
            <div
              key={layer.id}
              className="flex items-center gap-2 px-2 py-1.5 hover:bg-zinc-800 rounded transition-colors"
            >
              <button
                onClick={() => onToggleLayer(layer.id)}
                className={`flex-shrink-0 w-4 h-4 border rounded-sm flex items-center justify-center transition-colors ${
                  layer.enabled
                    ? 'border-emerald-500 bg-emerald-500/20'
                    : 'border-zinc-600 bg-transparent'
                }`}
              >
                {layer.enabled && (
                  <span className="text-emerald-400 text-[10px]">✓</span>
                )}
              </button>

              <div
                className="flex-shrink-0 w-5 h-5 flex items-center justify-center text-xs"
                style={{ color: layer.enabled ? layer.color : '#52525b' }}
              >
                {layer.icon}
              </div>

              <span
                className={`flex-1 text-[11px] font-mono ${
                  layer.enabled ? 'text-zinc-300' : 'text-zinc-600'
                }`}
              >
                {layer.name}
              </span>

              {onLayerSettings && (
                <button
                  onClick={() => onLayerSettings(layer.id)}
                  className="flex-shrink-0 text-zinc-600 hover:text-zinc-300 text-[10px] px-1"
                  title="Layer Settings"
                >
                  ⚙
                </button>
              )}
            </div>
          ))}
        </div>

        {/* Footer */}
        <div className="px-3 py-1.5 border-t border-zinc-700 text-[10px] text-zinc-500 font-mono">
          {layers.filter((l) => l.enabled).length}/{layers.length} active
        </div>
      </div>
    </div>
  );
}
