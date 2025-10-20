'use client';

import React, { useEffect, useRef } from 'react';

interface FeedEvent {
  id: string;
  timestamp: string;
  type: 'INFO' | 'ALERT' | 'THREAT' | 'TASK' | 'AI';
  message: string;
}

const mockEvents: FeedEvent[] = [
  { id: '001', timestamp: '03:42:15', type: 'TASK', message: 'UAV-01 DEPLOYED TO SECTOR B7' },
  { id: '002', timestamp: '03:42:18', type: 'INFO', message: 'PERIMETER SCAN INITIATED' },
  { id: '003', timestamp: '03:42:22', type: 'AI', message: 'PATTERN RECOGNITION: ANOMALY DETECTED' },
  { id: '004', timestamp: '03:42:29', type: 'ALERT', message: 'GND-01 BATTERY THRESHOLD WARNING' },
  { id: '005', timestamp: '03:42:35', type: 'INFO', message: 'MESH NETWORK OPTIMIZATION COMPLETE' },
  { id: '006', timestamp: '03:42:41', type: 'THREAT', message: 'THERMAL SIGNATURE DETECTED SECTOR A3' },
  { id: '007', timestamp: '03:42:48', type: 'TASK', message: 'FIRELINE BOT DISPATCHED TO ZONE 5' },
  { id: '008', timestamp: '03:42:52', type: 'AI', message: 'PREDICTIVE MODEL UPDATED: 94% CONFIDENCE' },
  { id: '009', timestamp: '03:43:01', type: 'INFO', message: 'DATA FUSION CYCLE COMPLETE' },
  { id: '010', timestamp: '03:43:07', type: 'TASK', message: 'RECON WAYPOINT ALPHA REACHED' },
];

export default function MissionFeed() {
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, []);

  return (
    <div className="w-80 bg-[#0F0F0F] border-l-2 border-[#00FF91]/20 flex flex-col overflow-hidden">
      {/* Header */}
      <div className="h-10 border-b border-[#00FF91]/20 flex items-center px-4 bg-[#0A0A0A]">
        <div className="text-[#00FF91] text-sm font-semibold tracking-wider uppercase">
          MISSION FEED
        </div>
        <div className="ml-auto text-[10px] text-[#006644] font-mono">
          [LIVE]
        </div>
      </div>

      {/* Event List */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto overflow-x-hidden p-2 space-y-1">
        {mockEvents.map((event, idx) => (
          <EventRow key={event.id} event={event} index={idx} />
        ))}
      </div>
    </div>
  );
}

interface EventRowProps {
  event: FeedEvent;
  index: number;
}

function EventRow({ event, index }: EventRowProps) {
  const typeStyles = {
    INFO: { color: '#00CC74', icon: '●', bg: '#00CC74' },
    ALERT: { color: '#FF9933', icon: '▲', bg: '#FF9933' },
    THREAT: { color: '#FF3333', icon: '⬢', bg: '#FF3333' },
    TASK: { color: '#00FF91', icon: '▶', bg: '#00FF91' },
    AI: { color: '#00DDFF', icon: '◆', bg: '#00DDFF' },
  };

  const style = typeStyles[event.type];

  return (
    <div 
      className="p-2 border-l-2 hover:bg-[#00FF91]/5 transition-colors"
      style={{ 
        borderLeftColor: style.color,
        animationDelay: `${index * 0.05}s` 
      }}
    >
      {/* Timestamp and Type */}
      <div className="flex items-center gap-2 mb-1">
        <div className="text-[10px] text-[#006644] font-mono tracking-wider">
          {event.timestamp}
        </div>
        <div 
          className="text-[8px] px-1.5 py-0.5 font-semibold tracking-wider border"
          style={{ 
            color: style.color,
            borderColor: `${style.color}40`,
            backgroundColor: `${style.bg}10`
          }}
        >
          {event.type}
        </div>
      </div>

      {/* Message */}
      <div className="flex items-start gap-2">
        <div 
          className="text-xs mt-0.5" 
          style={{ color: style.color }}
        >
          {style.icon}
        </div>
        <div className="text-xs text-[#00CC74] font-mono leading-relaxed">
          {event.message}
        </div>
      </div>
    </div>
  );
}
