import React from 'react';

interface Asset {
  id: string;
  type: string;
  battery: number;
  temp: number;
  signal: number;
  status: 'ACTIVE' | 'IDLE' | 'WARNING' | 'OFFLINE';
}

const mockAssets: Asset[] = [
  { id: 'UAV-01', type: 'DRONE', battery: 87, temp: 42, signal: 95, status: 'ACTIVE' },
  { id: 'UAV-02', type: 'DRONE', battery: 76, temp: 39, signal: 89, status: 'ACTIVE' },
  { id: 'UAV-03', type: 'DRONE', battery: 92, temp: 38, signal: 94, status: 'IDLE' },
  { id: 'GND-01', type: 'ROVER', battery: 45, temp: 51, signal: 78, status: 'WARNING' },
  { id: 'GND-02', type: 'ROVER', battery: 68, temp: 44, signal: 82, status: 'ACTIVE' },
  { id: 'TWR-01', type: 'RELAY', battery: 100, temp: 31, signal: 100, status: 'ACTIVE' },
  { id: 'TWR-02', type: 'RELAY', battery: 98, temp: 33, signal: 98, status: 'ACTIVE' },
  { id: 'SEN-01', type: 'SENSOR', battery: 62, temp: 35, signal: 91, status: 'ACTIVE' },
  { id: 'UAV-04', type: 'DRONE', battery: 23, temp: 47, signal: 71, status: 'WARNING' },
  { id: 'SEN-02', type: 'SENSOR', battery: 0, temp: 0, signal: 0, status: 'OFFLINE' },
];

export default function AssetLog() {
  return (
    <div className="w-80 bg-[#0F0F0F] border-r-2 border-[#00FF91]/20 flex flex-col overflow-hidden">
      <div className="h-10 border-b border-[#00FF91]/20 flex items-center px-4 bg-[#0A0A0A]">
        <div className="text-[#00FF91] text-sm font-semibold tracking-wider uppercase">
          ASSET LOG
        </div>
        <div className="ml-auto text-[10px] text-[#006644] font-mono">
          [{mockAssets.length} NODES]
        </div>
      </div>

      <div className="flex-1 overflow-y-auto overflow-x-hidden">
        {mockAssets.map((asset, idx) => (
          <AssetRow key={asset.id} asset={asset} index={idx} />
        ))}
      </div>
    </div>
  );
}

interface AssetRowProps {
  asset: Asset;
  index: number;
}

function AssetRow({ asset, index }: AssetRowProps) {
  const statusColors = {
    ACTIVE: '#00FF91',
    IDLE: '#00CC74',
    WARNING: '#FF9933',
    OFFLINE: '#FF3333',
  } as const;

  const statusColor = statusColors[asset.status];

  return (
    <div 
      className="border-b border-[#00FF91]/10 p-3 hover:bg-[#00FF91]/5 transition-colors"
      style={{ animationDelay: `${index * 0.05}s` }}
    >
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <div 
            className="w-1.5 h-1.5 rounded-full"
            style={{ 
              backgroundColor: statusColor,
              boxShadow: `0 0 4px ${statusColor}`
            }}
          />
          <div className="text-[#00FF91] text-sm font-mono font-semibold tracking-wide">
            {asset.id}
          </div>
        </div>
        <div className="text-[10px] text-[#006644] uppercase tracking-wider px-2 py-0.5 border border-[#006644]/50">
          {asset.type}
        </div>
      </div>

      <div className="grid grid-cols-3 gap-2 text-[10px]">
        <TelemetryItem label="BAT" value={`${asset.battery}%`} warning={asset.battery < 30} />
        <TelemetryItem label="TMP" value={`${asset.temp}°C`} warning={asset.temp > 50} />
        <TelemetryItem label="SIG" value={`${asset.signal}%`} warning={asset.signal < 75} />
      </div>

      <div className="mt-2 text-[10px]">
        <span className="text-[#006644]">STATUS: </span>
        <span style={{ color: statusColor }} className="font-semibold">
          {asset.status}
        </span>
      </div>
    </div>
  );
}

interface TelemetryItemProps {
  label: string;
  value: string;
  warning?: boolean;
}

function TelemetryItem({ label, value, warning }: TelemetryItemProps) {
  return (
    <div className="flex flex-col">
      <div className="text-[#006644] uppercase tracking-wider">{label}</div>
      <div className={`font-mono ${warning ? 'text-[#FF9933]' : 'text-[#00CC74]'}`}>
        {value}
      </div>
    </div>
  );
}
