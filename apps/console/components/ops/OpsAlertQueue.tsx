'use client';

import React, { useState, useEffect, useCallback } from 'react';
import { fetchAlerts, connectWebSocket, acknowledgeAlert, AlertAPI } from '@/lib/api';
import PanelHeader from '@/components/ui/PanelHeader';
import StatusBadge from '@/components/ui/StatusBadge';
import { ageFromISO, severityColor } from '@/lib/format';

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

  const ack = async (alertId: string) => {
    try {
      await acknowledgeAlert(alertId);
    } catch { /* best-effort */ }
    setAlerts((prev) =>
      prev.map((a) => a.alert_id === alertId ? { ...a, acknowledged: true } : a)
    );
  };

  return (
    <div className="flex flex-col h-full panel-scanline">
      <PanelHeader title="ALERT QUEUE" count={alerts.length > 0 ? alerts.length : undefined} badgeVariant="critical" />

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
          const isCritical = alert.severity.toUpperCase() === 'CRITICAL';
          return (
            <div
              key={alert.alert_id}
              className={`mx-2 my-1.5 ${isCritical ? 'severity-critical' : ''}`}
              style={{
                background: 'var(--background-card)',
                border: '1px solid var(--border)',
                borderLeft: `3px solid ${sColor}`,
              }}
            >
              <div className="px-2 pt-2">
                {/* Top row */}
                <div className="flex items-center justify-between mb-1">
                  <StatusBadge label={alert.severity.toUpperCase()} color={sColor} />
                  <span
                    className="text-[10px]"
                    style={{ color: 'var(--text-dim)', fontFamily: 'var(--font-ibm-plex-mono), monospace' }}
                  >
                    {ageFromISO(alert.ts_iso)}
                  </span>
                </div>
                {/* Description */}
                <div
                  className="text-xs font-bold mb-0.5 leading-tight"
                  style={{ color: 'var(--accent)', fontFamily: 'var(--font-ibm-plex-mono), monospace' }}
                >
                  {alert.description.slice(0, 60)}
                </div>
                {/* Source */}
                <div
                  className="text-[10px] mb-2"
                  style={{ color: 'var(--text-dim)', fontFamily: 'var(--font-ibm-plex-mono), monospace' }}
                >
                  {alert.source}
                </div>
              </div>
              {/* Buttons */}
              <div
                className="flex gap-0"
                style={{ borderTop: '1px solid var(--accent-10)' }}
              >
                <button
                  className="summit-btn flex-1 text-[10px] py-1.5 tracking-widest"
                  aria-label={`Investigate alert: ${alert.description.slice(0, 30)}`}
                  style={{
                    fontFamily: 'var(--font-ibm-plex-mono), monospace',
                    color: 'var(--accent)',
                    background: 'transparent',
                    border: 'none',
                    borderRight: '1px solid var(--accent-10)',
                  }}
                  onClick={() => onInvestigate?.(alert)}
                >
                  INVESTIGATE
                </button>
                <button
                  className="summit-btn flex-1 text-[10px] py-1.5 tracking-widest"
                  aria-label={`Acknowledge alert`}
                  style={{
                    fontFamily: 'var(--font-ibm-plex-mono), monospace',
                    color: (alert as AlertAPI & { acknowledged?: boolean }).acknowledged ? 'var(--text-muted)' : 'var(--color-active)',
                    background: 'transparent',
                    border: 'none',
                    borderRight: '1px solid var(--accent-10)',
                  }}
                  onClick={() => ack(alert.alert_id)}
                >
                  {(alert as AlertAPI & { acknowledged?: boolean }).acknowledged ? 'ACK\'D' : 'ACK'}
                </button>
                <button
                  className="summit-btn flex-1 text-[10px] py-1.5 tracking-widest"
                  aria-label={`Dismiss alert`}
                  style={{
                    fontFamily: 'var(--font-ibm-plex-mono), monospace',
                    color: 'var(--text-dim)',
                    background: 'transparent',
                    border: 'none',
                  }}
                  onClick={() => dismiss(alert.alert_id)}
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
