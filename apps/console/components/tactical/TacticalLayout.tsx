'use client';

import React from 'react';
import TopOverlay from './TopOverlay';
import AssetLog from './AssetLog';
import TacticalMap from './TacticalMap';
import CommandBar from './CommandBar';
import RightSidebar from './RightSidebar';

export default function TacticalLayout() {
  return (
    <div className="fixed inset-0 flex flex-col bg-zinc-950 overflow-hidden">
      <TopOverlay />
      <div className="flex flex-1 overflow-hidden relative">
        <AssetLog />
        <TacticalMap />
        <RightSidebar />
      </div>
      <CommandBar />
    </div>
  );
}
