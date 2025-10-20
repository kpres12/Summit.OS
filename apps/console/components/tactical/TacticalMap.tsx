'use client';

import React, { useEffect, useRef } from 'react';

interface MapNode {
  id: string;
  x: number;
  y: number;
  type: 'UAV' | 'GND' | 'TWR' | 'SEN';
  active: boolean;
}

const mockNodes: MapNode[] = [
  { id: 'UAV-01', x: 35, y: 25, type: 'UAV', active: true },
  { id: 'UAV-02', x: 65, y: 40, type: 'UAV', active: true },
  { id: 'UAV-03', x: 50, y: 60, type: 'UAV', active: false },
  { id: 'GND-01', x: 25, y: 55, type: 'GND', active: true },
  { id: 'GND-02', x: 70, y: 70, type: 'GND', active: true },
  { id: 'TWR-01', x: 50, y: 50, type: 'TWR', active: true },
  { id: 'TWR-02', x: 80, y: 30, type: 'TWR', active: true },
  { id: 'SEN-01', x: 40, y: 75, type: 'SEN', active: true },
];

const connections = [
  ['UAV-01', 'TWR-01'],
  ['UAV-02', 'TWR-02'],
  ['UAV-03', 'TWR-01'],
  ['GND-01', 'TWR-01'],
  ['GND-02', 'TWR-02'],
  ['SEN-01', 'TWR-01'],
  ['TWR-01', 'TWR-02'],
];

export default function TacticalMap() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const resizeCanvas = () => {
      const rect = canvas.getBoundingClientRect();
      canvas.width = rect.width * window.devicePixelRatio;
      canvas.height = rect.height * window.devicePixelRatio;
      ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
      drawMap();
    };

    const drawMap = () => {
      const w = canvas.width / window.devicePixelRatio;
      const h = canvas.height / window.devicePixelRatio;

      // Clear
      ctx.fillStyle = '#0A0A0A';
      ctx.fillRect(0, 0, w, h);

      // Draw grid
      ctx.strokeStyle = 'rgba(0, 255, 145, 0.15)';
      ctx.lineWidth = 1;

      const gridSize = 50;
      for (let x = 0; x <= w; x += gridSize) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, h);
        ctx.stroke();
      }

      for (let y = 0; y <= h; y += gridSize) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(w, y);
        ctx.stroke();
      }

      // Draw connections
      ctx.strokeStyle = 'rgba(0, 255, 145, 0.3)';
      ctx.lineWidth = 1;
      ctx.setLineDash([5, 5]);

      connections.forEach(([from, to]) => {
        const nodeFrom = mockNodes.find(n => n.id === from);
        const nodeTo = mockNodes.find(n => n.id === to);
        if (nodeFrom && nodeTo) {
          ctx.beginPath();
          ctx.moveTo((nodeFrom.x / 100) * w, (nodeFrom.y / 100) * h);
          ctx.lineTo((nodeTo.x / 100) * w, (nodeTo.y / 100) * h);
          ctx.stroke();
        }
      });

      ctx.setLineDash([]);

      // Draw terrain contours (pseudo-3D effect)
      ctx.strokeStyle = 'rgba(0, 255, 145, 0.1)';
      ctx.lineWidth = 1;
      
      const contours = 8;
      for (let i = 0; i < contours; i++) {
        const offset = (i / contours) * 100;
        const amplitude = 30 + Math.sin(i) * 20;
        ctx.beginPath();
        for (let x = 0; x <= w; x += 5) {
          const y = h / 2 + Math.sin((x + offset) * 0.02) * amplitude + 
                     Math.cos((x - offset) * 0.01) * (amplitude * 0.5);
          if (x === 0) {
            ctx.moveTo(x, y);
          } else {
            ctx.lineTo(x, y);
          }
        }
        ctx.stroke();
      }
    };

    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);

    return () => {
      window.removeEventListener('resize', resizeCanvas);
    };
  }, []);

  return (
    <div className="flex-1 relative overflow-hidden bg-[#0A0A0A]">
      {/* Canvas */}
      <canvas 
        ref={canvasRef} 
        className="absolute inset-0 w-full h-full"
      />

      {/* Map Nodes Overlay */}
      <div className="absolute inset-0 pointer-events-none">
        {mockNodes.map((node) => (
          <MapNode key={node.id} node={node} />
        ))}
      </div>

      {/* Grid Coordinates Overlay */}
      <div className="absolute top-4 left-4 text-[#006644] text-xs font-mono space-y-1 pointer-events-none">
        <div>GRID: 34.05°N 118.24°W</div>
        <div>SCALE: 1:25000</div>
        <div>ALT: 120M MSL</div>
      </div>

      {/* Compass/Orientation */}
      <div className="absolute top-4 right-4 pointer-events-none">
        <div className="w-16 h-16 border-2 border-[#00FF91]/40 rounded-full flex items-center justify-center relative">
          <div className="text-[#00FF91] text-xs font-bold">N</div>
          <div className="absolute w-0.5 h-6 bg-[#00FF91]/60 top-1 left-1/2 -translate-x-1/2" />
        </div>
      </div>
    </div>
  );
}

interface MapNodeProps {
  node: MapNode;
}

function MapNode({ node }: MapNodeProps) {
  const nodeStyles = {
    UAV: { icon: '▲', color: '#00FF91', size: 16 },
    GND: { icon: '■', color: '#00CC74', size: 14 },
    TWR: { icon: '⬢', color: '#FF9933', size: 18 },
    SEN: { icon: '●', color: '#00DDFF', size: 12 },
  };

  const style = nodeStyles[node.type];

  return (
    <div
      className="absolute pointer-events-auto cursor-pointer transform -translate-x-1/2 -translate-y-1/2 group"
      style={{
        left: `${node.x}%`,
        top: `${node.y}%`,
      }}
    >
      {/* Pulsing ring for active nodes */}
      {node.active && (
        <div
          className="absolute inset-0 rounded-full animate-ping opacity-75"
          style={{
            backgroundColor: style.color,
            width: `${style.size + 8}px`,
            height: `${style.size + 8}px`,
            left: '50%',
            top: '50%',
            transform: 'translate(-50%, -50%)',
          }}
        />
      )}

      {/* Node marker */}
      <div
        className="relative z-10 flex items-center justify-center font-bold transition-transform group-hover:scale-125"
        style={{
          color: style.color,
          fontSize: `${style.size}px`,
          filter: `drop-shadow(0 0 4px ${style.color}) drop-shadow(0 0 8px ${style.color})`,
        }}
      >
        {style.icon}
      </div>

      {/* Label */}
      <div
        className="absolute top-full mt-1 left-1/2 -translate-x-1/2 text-[10px] font-mono whitespace-nowrap opacity-0 group-hover:opacity-100 transition-opacity px-2 py-1 border"
        style={{
          color: style.color,
          borderColor: `${style.color}40`,
          backgroundColor: '#0A0A0A',
          boxShadow: `0 0 8px ${style.color}40`,
        }}
      >
        {node.id}
      </div>
    </div>
  );
}
