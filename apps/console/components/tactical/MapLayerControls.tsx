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
          className="px-3 py-2 bg-[#0F0F0F]/90 border border-[#00FF91]/40 text-[#00FF91] text-xs font-mono hover:bg-[#00FF91]/10 transition-colors backdrop-blur-sm"
          style={{
            boxShadow: '0 0 8px rgba(0, 255, 145, 0.2)',
          }}
        >
          ◂ LAYERS
        </button>
      </div>
    );
  }

  return (
    <div className="absolute top-20 right-4 z-10 w-64">
      <div
        className="bg-[#0F0F0F]/95 border border-[#00FF91]/40 backdrop-blur-sm"
        style={{
          boxShadow: '0 0 12px rgba(0, 255, 145, 0.2)',
        }}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-3 py-2 border-b border-[#00FF91]/30">
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 bg-[#00FF91] animate-pulse" />
            <span className="text-[#00FF91] text-xs font-mono font-bold tracking-wider">
              MAP LAYERS
            </span>
          </div>
          <button
            onClick={() => setCollapsed(true)}
            className="text-[#00FF91] hover:text-[#00CC74] text-xs"
          >
            ▸
          </button>
        </div>

        {/* Layer List */}
        <div className="p-2 space-y-1 max-h-96 overflow-y-auto">
          {layers.map((layer) => (
            <div
              key={layer.id}
              className="flex items-center gap-2 px-2 py-1.5 hover:bg-[#00FF91]/5 transition-colors"
            >
              {/* Toggle Checkbox */}
              <button
                onClick={() => onToggleLayer(layer.id)}
                className="flex-shrink-0 w-4 h-4 border border-[#00FF91]/60 flex items-center justify-center hover:border-[#00FF91] transition-colors"
                style={{
                  backgroundColor: layer.enabled
                    ? 'rgba(0, 255, 145, 0.2)'
                    : 'transparent',
                }}
              >
                {layer.enabled && (
                  <span className="text-[#00FF91] text-[10px]">✓</span>
                )}
              </button>

              {/* Layer Icon */}
              <div
                className="flex-shrink-0 w-6 h-6 flex items-center justify-center text-sm"
                style={{
                  color: layer.enabled ? layer.color : '#006644',
                  filter: layer.enabled
                    ? `drop-shadow(0 0 2px ${layer.color})`
                    : 'none',
                }}
              >
                {layer.icon}
              </div>

              {/* Layer Name */}
              <span
                className={`flex-1 text-[11px] font-mono ${
                  layer.enabled ? 'text-[#00FF91]' : 'text-[#006644]'
                }`}
              >
                {layer.name}
              </span>

              {/* Settings Button */}
              {onLayerSettings && (
                <button
                  onClick={() => onLayerSettings(layer.id)}
                  className="flex-shrink-0 text-[#006644] hover:text-[#00FF91] text-[10px] px-1"
                  title="Layer Settings"
                >
                  ⚙
                </button>
              )}
            </div>
          ))}
        </div>

        {/* Footer */}
        <div className="px-3 py-2 border-t border-[#00FF91]/30 text-[10px] text-[#006644] font-mono">
          {layers.filter((l) => l.enabled).length}/{layers.length} layers active
        </div>
      </div>
    </div>
  );
}
