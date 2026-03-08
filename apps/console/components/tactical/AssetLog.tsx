'use client';

import React from 'react';
import { useEntityStream, EntityData } from '../../hooks/useEntityStream';

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

export default function AssetLog() {
  const { entityList, connected, entityCount } = useEntityStream();

  return (
    <div className="w-72 bg-[#0F0F0F] border-r-2 border-[#00FF91]/20 flex flex-col overflow-hidden">
      <div className="h-10 border-b border-[#00FF91]/20 flex items-center px-4 bg-[#0A0A0A]">
        <div className="text-[#00FF91] text-sm font-semibold tracking-wider uppercase">
          ASSETS
        </div>
        <div className="ml-auto flex items-center gap-2">
          <div className={`w-1.5 h-1.5 rounded-full ${connected ? 'bg-[#00FF91]' : 'bg-red-500'}`}
               style={connected ? { boxShadow: '0 0 4px #00FF91' } : {}} />
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
      </div>
    </div>
  );
}

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
