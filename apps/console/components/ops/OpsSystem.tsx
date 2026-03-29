'use client';

import React from 'react';
import PanelHeader from '@/components/ui/PanelHeader';
import DataRow from '@/components/ui/DataRow';

export default function OpsSystem() {
  return (
    <div className="flex flex-col h-full panel-scanline">
      <PanelHeader title="SYSTEM" />
      <div className="flex-1 overflow-y-auto p-3">
        <DataRow label="NODE" value="CONSOLE-01" />
        <DataRow label="VERSION" value="1.0.0" />
        <DataRow label="ENV" value={process.env.NODE_ENV?.toUpperCase() || 'PRODUCTION'} />
        <DataRow label="API" value={process.env.NEXT_PUBLIC_API_URL || 'localhost:8000'} />
        <DataRow label="WS" value={process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8001'} />
      </div>
    </div>
  );
}
