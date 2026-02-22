'use client';

import React from 'react';
import type { MeshStatus } from '../../hooks/useEntityStream';

interface MeshStatusPanelProps {
  status: MeshStatus | null;
  connected: boolean;
}

export default function MeshStatusPanel({ status, connected }: MeshStatusPanelProps) {
  return (
    <div className="bg-gray-900/80 border border-gray-700 rounded-lg p-3 text-xs">
      <div className="flex items-center justify-between mb-2">
        <h4 className="font-semibold text-gray-400 uppercase tracking-wider">Mesh Network</h4>
        <span className={`w-2 h-2 rounded-full ${connected ? 'bg-green-500' : 'bg-red-500'}`} />
      </div>

      {!status ? (
        <p className="text-gray-600">No mesh data</p>
      ) : (
        <>
          <div className="grid grid-cols-3 gap-2 mb-2">
            <MetricBox label="Alive" value={status.peers_alive} color="text-green-400" />
            <MetricBox label="Suspect" value={status.peers_suspect} color="text-yellow-400" />
            <MetricBox label="Dead" value={status.peers_dead} color="text-red-400" />
          </div>

          {status.partitioned && (
            <div className="bg-red-900/50 border border-red-700 rounded px-2 py-1 text-red-300 text-center animate-pulse">
              ⚠ NETWORK PARTITION
            </div>
          )}

          <div className="mt-2 text-gray-600">
            Node: {status.node_id} · {status.crdt_keys} CRDT keys
          </div>
        </>
      )}
    </div>
  );
}

function MetricBox({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <div className="text-center">
      <div className={`text-lg font-bold ${color}`}>{value}</div>
      <div className="text-gray-600">{label}</div>
    </div>
  );
}
