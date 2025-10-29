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
];

export default function RightSidebar() {
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, []);

  return (
    <div className="w-80 bg-[#0F0F0F] border-l-2 border-[#00FF91]/20 flex flex-col overflow-hidden">
      {/* Mission Feed */}
      <div className="flex-1 flex flex-col">
        <div className="h-10 border-b border-[#00FF91]/20 flex items-center px-4 bg-[#0A0A0A]">
          <div className="text-[#00FF91] text-sm font-semibold tracking-wider uppercase">
            MISSION FEED
          </div>
          <div className="ml-auto text-[10px] text-[#006644] font-mono">[LIVE]</div>
        </div>
        <div ref={scrollRef} className="flex-1 overflow-y-auto overflow-x-hidden p-2 space-y-1">
          {mockEvents.map((e) => (
            <div key={e.id} className="p-2 border-l-2 hover:bg-[#00FF91]/5 transition-colors" style={{ borderLeftColor: '#00FF91' }}>
              <div className="flex items-center gap-2 mb-1">
                <div className="text-[10px] text-[#006644] font-mono tracking-wider">{e.timestamp}</div>
                <div className="text-[8px] px-1.5 py-0.5 font-semibold tracking-wider border" style={{ color: '#00FF91', borderColor: '#00FF9140', backgroundColor: '#00FF9110' }}>{e.type}</div>
              </div>
              <div className="text-xs text-[#00CC74] font-mono leading-relaxed">{e.message}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Mission Timeline */}
      <div className="h-64">
        {/** dynamic import avoided; directly include to keep simple **/}
        {/* eslint-disable-next-line @typescript-eslint/ban-ts-comment */}
        {/* @ts-expect-error: imported via next transpilation */}
        {require('./MissionTimeline').default()}
      </div>
    </div>
  );
}
