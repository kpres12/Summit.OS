'use client';

import React, { useState, useEffect, useCallback } from 'react';
import { fetchAlerts, connectWebSocket, AlertAPI } from '@/lib/api';

function ageString(isoString: string): string {
  const ts = new Date(isoString).getTime();
  const diff = Math.floor((Date.now() - ts) / 1000);
  if (diff < 60) return `${diff}s ago`;
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  return `${Math.floor(diff / 86400)}d ago`;
}

function severityColor(severity: string): string {
  const s = severity.toUpperCase();
  if (s === 'CRITICAL') return '#FF3B3B';
  if (s === 'HIGH') return 'rgba(255,59,59,0.8)';
  if (s === 'MED' || s === 'MEDIUM') return '#FFB300';
  return 'rgba(200,230,201,0.45)';
}

interface OpsAlertQueueProps {
  onInvestigate?: (alert: AlertAPI) => void;
}

export default function OpsAlertQueue({ onInvestigate }: OpsAlertQueueProps) {
  const [alerts, setAlerts] = useState<AlertAPI[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchAlerts(50)
      .then((res) => setAlerts(res.alerts || []))
      .catch(() => setAlerts([]))
      .finally(() => setLoading(false));
  }, []);

  const handleWsMessage = useCallback((data: unknown) => {
    const msg = data as { type?: string; data?: AlertAPI };
    if (msg.type === 'alert' && msg.data) {
      setAlerts((prev) => [msg.data as AlertAPI, ...prev].slice(0, 100));
    }
  }, []);

  useEffect(() => {
    const ws = connectWebSocket(handleWsMessage);
    return () => {
      ws?.close();
    };
  }, [handleWsMessage]);

  const dismiss = (alertId: string) => {
    setAlerts((prev) => prev.filter((a) => a.alert_id !== alertId));
  };

  return (
    <div className="flex flex-col h-full">
      {/* Panel header */}
      <div
        className="flex-none flex items-center justify-between px-3 py-2"
        style={{ borderBottom: '1px solid rgba(0,255,156,0.15)' }}
      >
        <span
          className="text-xs font-bold tracking-widest"
          style={{ fontFamily: 'var(--font-orbitron), Orbitron, sans-serif', color: '#00FF9C' }}
        >
          ALERT QUEUE
        </span>
        {alerts.length > 0 && (
          <span
            className="text-[10px] px-1.5 py-0.5"
            style={{
              background: 'rgba(255,59,59,0.15)',
              color: '#FF3B3B',
              border: '1px solid rgba(255,59,59,0.3)',
              fontFamily: 'var(--font-ibm-plex-mono), monospace',
            }}
          >
            {alerts.length}
          </span>
        )}
      </div>

      {/* Alert list */}
      <div className="flex-1 overflow-y-auto">
        {loading && (
          <div
            className="flex items-center justify-center h-20 text-[10px]"
            style={{ color: 'rgba(200,230,201,0.35)', fontFamily: 'var(--font-ibm-plex-mono), monospace' }}
          >
            LOADING...
          </div>
        )}
        {!loading && alerts.length === 0 && (
          <div
            className="flex items-center justify-center h-full text-[10px] tracking-widest"
            style={{ color: 'rgba(200,230,201,0.35)', fontFamily: 'var(--font-ibm-plex-mono), monospace' }}
          >
            NO ACTIVE ALERTS
          </div>
        )}
        {alerts.map((alert) => {
          const sColor = severityColor(alert.severity);
          return (
            <div
              key={alert.alert_id}
              className="mx-2 my-1.5"
              style={{
                background: '#111916',
                border: '1px solid rgba(0,255,156,0.15)',
                borderLeft: `3px solid ${sColor}`,
              }}
            >
              <div className="px-2 pt-2">
                {/* Top row */}
                <div className="flex items-center justify-between mb-1">
                  <span
                    className="text-[9px] font-bold tracking-widest px-1 py-0.5"
                    style={{
                      color: sColor,
                      border: `1px solid ${sColor}`,
                      fontFamily: 'var(--font-ibm-plex-mono), monospace',
                      background: `${sColor}15`,
                    }}
                  >
                    {alert.severity.toUpperCase()}
                  </span>
                  <span
                    className="text-[10px]"
                    style={{ color: 'rgba(200,230,201,0.45)', fontFamily: 'var(--font-ibm-plex-mono), monospace' }}
                  >
                    {ageString(alert.ts_iso)}
                  </span>
                </div>
                {/* Description */}
                <div
                  className="text-xs font-bold mb-0.5 leading-tight"
                  style={{ color: '#00FF9C', fontFamily: 'var(--font-ibm-plex-mono), monospace' }}
                >
                  {alert.description.slice(0, 60)}
                </div>
                {/* Source */}
                <div
                  className="text-[10px] mb-2"
                  style={{ color: 'rgba(200,230,201,0.45)', fontFamily: 'var(--font-ibm-plex-mono), monospace' }}
                >
                  {alert.source}
                </div>
              </div>
              {/* Buttons */}
              <div
                className="flex gap-0"
                style={{ borderTop: '1px solid rgba(0,255,156,0.1)' }}
              >
                <button
                  className="flex-1 text-[10px] py-1.5 tracking-widest transition-colors"
                  style={{
                    fontFamily: 'var(--font-ibm-plex-mono), monospace',
                    color: '#00FF9C',
                    background: 'transparent',
                    border: 'none',
                    borderRight: '1px solid rgba(0,255,156,0.1)',
                    cursor: 'pointer',
                  }}
                  onClick={() => onInvestigate?.(alert)}
                  onMouseEnter={(e) => ((e.currentTarget as HTMLButtonElement).style.background = 'rgba(0,255,156,0.08)')}
                  onMouseLeave={(e) => ((e.currentTarget as HTMLButtonElement).style.background = 'transparent')}
                >
                  INVESTIGATE
                </button>
                <button
                  className="flex-1 text-[10px] py-1.5 tracking-widest transition-colors"
                  style={{
                    fontFamily: 'var(--font-ibm-plex-mono), monospace',
                    color: 'rgba(200,230,201,0.45)',
                    background: 'transparent',
                    border: 'none',
                    cursor: 'pointer',
                  }}
                  onClick={() => dismiss(alert.alert_id)}
                  onMouseEnter={(e) => ((e.currentTarget as HTMLButtonElement).style.background = 'rgba(200,230,201,0.05)')}
                  onMouseLeave={(e) => ((e.currentTarget as HTMLButtonElement).style.background = 'transparent')}
                >
                  DISMISS
                </button>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
