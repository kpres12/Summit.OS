'use client';

import React from 'react';
import TopOverlay from './TopOverlay';
import AssetLog from './AssetLog';
import TacticalMap from './TacticalMap';
import CommandBar from './CommandBar';
import RightSidebar from './RightSidebar';

export default function TacticalLayout() {
  return (
    <div className="fixed inset-0 flex flex-col bg-[#0A0A0A] overflow-hidden">
      {/* Top Overlay */}
      <TopOverlay />

      {/* Main Content Area */}
      <div className="flex flex-1 overflow-hidden relative">
        {/* Left Sidebar - Asset Log */}
        <AssetLog />

        {/* Center - Tactical Map */}
        <TacticalMap />

        {/* Right Sidebar - Feed + Timeline */}
        <RightSidebar />
      </div>

      {/* Bottom Command Bar */}
      <CommandBar />

      {/* Corner dust/wear texture overlays */}
      <div className="pointer-events-none fixed inset-0 opacity-10">
        <div className="absolute top-0 left-0 w-32 h-32 bg-gradient-radial from-transparent to-black/50" />
        <div className="absolute top-0 right-0 w-32 h-32 bg-gradient-radial from-transparent to-black/50" />
        <div className="absolute bottom-0 left-0 w-32 h-32 bg-gradient-radial from-transparent to-black/50" />
        <div className="absolute bottom-0 right-0 w-32 h-32 bg-gradient-radial from-transparent to-black/50" />
      </div>
    </div>
  );
}
