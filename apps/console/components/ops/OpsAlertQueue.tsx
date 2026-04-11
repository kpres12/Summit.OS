'use client';

import React, { useState, useEffect, useCallback } from 'react';
import { fetchAlerts, connectWebSocket, acknowledgeAlert, AlertAPI } from '@/lib/api';
import PanelHeader from '@/components/ui/PanelHeader';
import StatusBadge from '@/components/ui/StatusBadge';
import { ageFromISO, severityColor } from '@/lib/format';
import OpsDecisionTimer from '@/components/ops/OpsDecisionTimer';

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
    return () => { ws?.close(); };
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
      <PanelHeader title="ALERTS" count={alerts.length > 0 ? alerts.length : undefined} badgeVariant="critical" />

      <div className="flex-1 overflow-y-auto">
        {loading && (
          <div
            className="flex items-center justify-center h-20 text-[10px]"
            style={{ color: 'var(--text-muted)', fontFamily: 'var(--font-ibm-plex-mono), monospace' }}
          >
            LOADING...
          </div>
        )}

        {/* Empty state — intentional, not an error */}
        {!loading && alerts.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full gap-3 px-6">
            <div style={{ fontSize: '28px', opacity: 0.15 }} aria-hidden="true">✓</div>
            <span
              className="text-[11px] tracking-widest"
              style={{ color: 'var(--accent-30)', fontFamily: 'var(--font-ibm-plex-mono), monospace' }}
            >
              ALL CLEAR
            </span>
            <span
              className="text-[9px] text-center leading-relaxed"
              style={{ color: 'var(--text-muted)', fontFamily: 'var(--font-ibm-plex-mono), monospace' }}
            >
              No active alerts. All assets nominal.
            </span>
          </div>
        )}

        {alerts.map((alert) => {
          const sColor = severityColor(alert.severity);
          const isCritical = alert.severity.toUpperCase() === 'CRITICAL';
          const isAcked = (alert as AlertAPI & { acknowledged?: boolean }).acknowledged;
          return (
            <div
              key={alert.alert_id}
              className={`mx-2 my-1.5 ${isCritical ? 'severity-critical' : ''}`}
              style={{
                background: 'var(--background-card)',
                border: '1px solid var(--border)',
                borderLeft: `3px solid ${sColor}`,
                opacity: isAcked ? 0.6 : 1,
                transition: 'opacity 0.2s',
              }}
            >
              <div className="px-2 pt-2">
                {/* Top row */}
                <div className="flex items-center justify-between mb-1">
                  <div className="flex items-center gap-2">
                    <StatusBadge label={alert.severity.toUpperCase()} color={sColor} />
                    <OpsDecisionTimer startIso={alert.ts_iso} />
                  </div>
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

              {/* Action row — INVESTIGATE is the primary action, visually dominant */}
              <div style={{ borderTop: '1px solid var(--accent-10)' }}>
                {/* Primary: INVESTIGATE — full-bleed, high contrast */}
                <button
                  className="summit-btn w-full text-[10px] py-2 tracking-widest font-bold"
                  aria-label={`Investigate alert: ${alert.description.slice(0, 40)}`}
                  style={{
                    fontFamily: 'var(--font-ibm-plex-mono), monospace',
                    color: 'var(--background)',
                    background: sColor,
                    border: 'none',
                    letterSpacing: '0.15em',
                  }}
                  onClick={() => onInvestigate?.(alert)}
                >
                  INVESTIGATE ›
                </button>

                {/* Secondary: ACK + DISMISS — ghost/text only */}
                <div className="flex" style={{ borderTop: '1px solid var(--accent-5)' }}>
                  <button
                    className="summit-btn flex-1 text-[9px] py-1.5 tracking-wider"
                    aria-label="Acknowledge alert"
                    style={{
                      fontFamily: 'var(--font-ibm-plex-mono), monospace',
                      color: isAcked ? 'var(--text-muted)' : 'var(--text-dim)',
                      background: 'transparent',
                      border: 'none',
                      borderRight: '1px solid var(--accent-5)',
                    }}
                    onClick={() => ack(alert.alert_id)}
                  >
                    {isAcked ? 'ACK\'D' : 'ACK'}
                    {!isAcked && (
                      <span style={{ color: 'var(--text-muted)', fontSize: '8px', marginLeft: '4px' }}>[SPC]</span>
                    )}
                  </button>
                  <button
                    className="summit-btn flex-1 text-[9px] py-1.5 tracking-wider"
                    aria-label="Dismiss alert"
                    style={{
                      fontFamily: 'var(--font-ibm-plex-mono), monospace',
                      color: 'var(--text-muted)',
                      background: 'transparent',
                      border: 'none',
                    }}
                    onClick={() => dismiss(alert.alert_id)}
                  >
                    DISMISS
                  </button>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
