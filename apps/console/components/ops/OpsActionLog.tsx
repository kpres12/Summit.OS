'use client';

import React from 'react';
import { useActionLog, type ActionEntry } from '@/contexts/ActionLogContext';
import PanelHeader from '@/components/ui/PanelHeader';
import { toUTCShort } from '@/lib/format';

const ACTION_COLOR: Record<ActionEntry['action'], string> = {
  DISPATCH:        'var(--accent)',
  HALT:            'var(--critical)',
  RTB:             'var(--warning)',
  CAMERA:          'var(--color-active)',
  ALERT_ACK:       'var(--text-dim)',
  ALERT_DISMISS:   'var(--text-muted)',
  MISSION_CREATE:  'var(--accent)',
  GEOFENCE_CREATE: 'var(--warning)',
  GEOFENCE_DELETE: 'var(--critical)',
  ENGAGE:          'var(--critical)',
};

const STATUS_COLOR: Record<ActionEntry['status'], string> = {
  success: 'var(--accent)',
  failed:  'var(--critical)',
  pending: 'var(--warning)',
};

export default function OpsActionLog() {
  const { actions, clearLog } = useActionLog();

  return (
    <div className="flex flex-col h-full panel-scanline">
      <PanelHeader
        title="ACTION LOG"
        count={actions.length > 0 ? actions.length : undefined}
      />

      <div className="flex-1 overflow-y-auto">
        {actions.length === 0 ? (
          <div
            className="flex items-center justify-center h-20 text-[10px]"
            style={{ color: 'var(--text-muted)', fontFamily: 'var(--font-ibm-plex-mono), monospace' }}
          >
            NO ACTIONS THIS SESSION
          </div>
        ) : (
          actions.map(entry => (
            <div
              key={entry.id}
              className="mx-2 my-1 px-2 py-2"
              style={{
                background: 'var(--background-card)',
                border: '1px solid var(--border)',
                borderLeft: `3px solid ${ACTION_COLOR[entry.action] ?? 'var(--text-dim)'}`,
              }}
            >
              <div className="flex items-center justify-between mb-0.5">
                <span style={{
                  fontFamily: 'var(--font-ibm-plex-mono), monospace',
                  fontSize: '10px',
                  fontWeight: 700,
                  color: ACTION_COLOR[entry.action] ?? 'var(--text-dim)',
                  letterSpacing: '0.1em',
                }}>
                  {entry.action}
                </span>
                <span style={{
                  fontFamily: 'var(--font-ibm-plex-mono), monospace',
                  fontSize: '8px',
                  color: STATUS_COLOR[entry.status],
                  letterSpacing: '0.1em',
                }}>
                  {entry.status.toUpperCase()}
                </span>
              </div>
              <div style={{
                fontFamily: 'var(--font-ibm-plex-mono), monospace',
                fontSize: '10px',
                color: 'var(--text-dim)',
              }}>
                {entry.target}
              </div>
              {entry.detail && (
                <div style={{
                  fontFamily: 'var(--font-ibm-plex-mono), monospace',
                  fontSize: '9px',
                  color: 'var(--text-muted)',
                  marginTop: '2px',
                }}>
                  {entry.detail}
                </div>
              )}
              <div style={{
                fontFamily: 'var(--font-ibm-plex-mono), monospace',
                fontSize: '8px',
                color: 'var(--text-muted)',
                marginTop: '2px',
              }}>
                {toUTCShort(entry.timestamp / 1000)}
              </div>
            </div>
          ))
        )}
      </div>

      {actions.length > 0 && (
        <div style={{ borderTop: '1px solid var(--border)', padding: '8px' }}>
          <button
            onClick={clearLog}
            className="summit-btn w-full text-[9px] py-1.5 tracking-widest"
            style={{
              fontFamily: 'var(--font-ibm-plex-mono), monospace',
              color: 'var(--text-muted)',
              border: '1px solid var(--border)',
              background: 'transparent',
              cursor: 'pointer',
            }}
          >
            CLEAR LOG
          </button>
        </div>
      )}
    </div>
  );
}
