'use client';

/**
 * OpsVideoPane — Live HLS video overlay for the OPS map view.
 *
 * Plays an HLS stream (served by Heli.OS fusion/hls_router) using hls.js
 * with a graceful fallback to native HLS on Safari.
 *
 * Props:
 *   streamId  — which HLS stream to play (maps to /api/v1/video/hls/{id}/index.m3u8)
 *   onClose   — called when operator dismisses the pane
 */

import React, { useEffect, useRef, useState } from 'react';

const FUSION_URL = process.env.NEXT_PUBLIC_FUSION_URL || 'http://localhost:8003';

interface OpsVideoPaneProps {
  streamId: string;
  onClose: () => void;
}

export default function OpsVideoPane({ streamId, onClose }: OpsVideoPaneProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [error, setError]   = useState<string | null>(null);
  const [ready, setReady]   = useState(false);

  const playlistUrl = `${FUSION_URL}/api/v1/video/hls/${streamId}/index.m3u8`;

  useEffect(() => {
    if (!videoRef.current) return;
    const video = videoRef.current;

    let hlsInstance: { destroy: () => void } | null = null;

    async function init() {
      try {
        const Hls = (await import('hls.js')).default;
        if (Hls.isSupported()) {
          const hls = new Hls({ lowLatencyMode: true });
          hlsInstance = hls;
          hls.loadSource(playlistUrl);
          hls.attachMedia(video);
          hls.on(Hls.Events.MANIFEST_PARSED, () => {
            setReady(true);
            video.play().catch(() => { /* autoplay policy */ });
          });
          hls.on(Hls.Events.ERROR, (_: unknown, data: { fatal?: boolean; details?: string }) => {
            if (data.fatal) setError(`HLS error: ${data.details}`);
          });
        } else if (video.canPlayType('application/vnd.apple.mpegurl')) {
          // Native HLS (Safari)
          video.src = playlistUrl;
          video.addEventListener('canplay', () => setReady(true));
          video.play().catch(() => { /* autoplay policy */ });
        } else {
          setError('HLS not supported in this browser — install hls.js');
        }
      } catch (e) {
        setError(`Failed to load hls.js: ${e}`);
      }
    }

    init();

    return () => {
      hlsInstance?.destroy();
    };
  }, [playlistUrl]);

  const mono: React.CSSProperties = {
    fontFamily: 'var(--font-ibm-plex-mono), monospace',
  };
  const orbitron: React.CSSProperties = {
    fontFamily: 'var(--font-ibm-plex-mono), monospace',
  };

  return (
    <div
      style={{
        position: 'absolute',
        bottom: 64,
        right: 16,
        width: 420,
        background: '#080C0A',
        border: '1px solid rgba(0,232,150,0.25)',
        boxShadow: '0 4px 32px rgba(0,0,0,0.7)',
        zIndex: 200,
      }}
    >
      {/* Header bar */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '4px 8px',
          borderBottom: '1px solid rgba(0,232,150,0.15)',
          background: 'rgba(0,232,150,0.04)',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          {/* Live dot */}
          <div
            style={{
              width: 6,
              height: 6,
              borderRadius: '50%',
              background: ready ? '#FF3B3B' : 'rgba(200,230,201,0.3)',
              boxShadow: ready ? '0 0 6px #FF3B3B' : 'none',
              animation: ready ? 'pulse 1.5s infinite' : 'none',
            }}
          />
          <span style={{ ...orbitron, color: '#00E896', fontSize: 9, letterSpacing: '0.15em' }}>
            LIVE FEED — {streamId.toUpperCase()}
          </span>
        </div>
        <button
          onClick={onClose}
          style={{
            ...mono,
            color: 'rgba(200,230,201,0.45)',
            background: 'none',
            border: 'none',
            cursor: 'pointer',
            fontSize: 12,
            padding: '0 4px',
          }}
        >
          ✕
        </button>
      </div>

      {/* Video */}
      <div style={{ position: 'relative', aspectRatio: '16/9', background: '#000' }}>
        <video
          ref={videoRef}
          style={{ width: '100%', height: '100%', display: 'block' }}
          muted
          playsInline
        />
        {!ready && !error && (
          <div
            style={{
              position: 'absolute', inset: 0,
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              ...mono, color: 'rgba(200,230,201,0.45)', fontSize: 10,
            }}
          >
            BUFFERING…
          </div>
        )}
        {error && (
          <div
            style={{
              position: 'absolute', inset: 0,
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              flexDirection: 'column', gap: 4,
              ...mono, color: '#FF3B3B', fontSize: 10, padding: 12, textAlign: 'center',
            }}
          >
            <span>⚠ STREAM UNAVAILABLE</span>
            <span style={{ color: 'rgba(200,230,201,0.35)', fontSize: 9 }}>{error}</span>
          </div>
        )}
      </div>

      {/* Footer */}
      <div
        style={{
          ...mono, fontSize: 9,
          color: 'rgba(200,230,201,0.35)',
          padding: '3px 8px',
          borderTop: '1px solid rgba(0,232,150,0.08)',
        }}
      >
        {playlistUrl}
      </div>
    </div>
  );
}
