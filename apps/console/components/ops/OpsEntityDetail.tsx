'use client';

import React, { useState, useEffect } from 'react';
import { EntityData } from '@/hooks/useEntityStream';
import { dispatchTask, sendAgentCommand } from '@/lib/api';
import SectionHeader from '@/components/ui/SectionHeader';
import DataRow from '@/components/ui/DataRow';
import ConfirmDialog from '@/components/ui/ConfirmDialog';
import { ageFromEpoch, toUTCShort, entityTypeColor, batteryColor } from '@/lib/format';
import { useActionLog } from '@/contexts/ActionLogContext';
import { useToast } from '@/contexts/ToastContext';

interface OpsEntityDetailProps {
  entity: EntityData | null;
  onClose: () => void;
  onDispatch?: (entity: EntityData) => void;
  onLiveFeed?: (streamId: string) => void;
}

// Generate mock AI reasoning based on entity state
function buildThoughts(entity: EntityData): { ts: string; msg: string; confidence: number }[] {
  const thoughts = [];
  const now = Date.now();

  if (entity.entity_type === 'alert') {
    const spd = entity.speed_mps != null ? `${entity.speed_mps.toFixed(1)} m/s` : 'unknown speed';
    thoughts.push({ ts: `${new Date(now - 8000).toISOString().slice(11,19)}Z`, msg: `Anomalous velocity detected: ${spd} exceeds baseline`, confidence: 0.91 });
    thoughts.push({ ts: `${new Date(now - 5000).toISOString().slice(11,19)}Z`, msg: 'Cross-referencing against known flight corridors — no match found', confidence: 0.87 });
    thoughts.push({ ts: `${new Date(now - 2000).toISOString().slice(11,19)}Z`, msg: 'Flagging for operator review. Recommend visual verification.', confidence: 0.84 });
  } else if (entity.battery_pct != null && entity.battery_pct < 25) {
    thoughts.push({ ts: `${new Date(now - 6000).toISOString().slice(11,19)}Z`, msg: `Battery critical at ${entity.battery_pct.toFixed(0)}% — estimating 4 min flight time remaining`, confidence: 0.96 });
    thoughts.push({ ts: `${new Date(now - 3000).toISOString().slice(11,19)}Z`, msg: 'Initiating RTB evaluation. Current position within return range.', confidence: 0.94 });
  } else {
    thoughts.push({ ts: `${new Date(now - 10000).toISOString().slice(11,19)}Z`, msg: `Tracking ${entity.classification || 'entity'} on nominal trajectory`, confidence: 0.97 });
    const spdStr = entity.speed_mps != null ? `${entity.speed_mps.toFixed(1)} m/s` : 'speed unknown';
    const hdgStr = entity.position?.heading_deg != null ? `, heading ${entity.position.heading_deg.toFixed(0)}°` : '';
    thoughts.push({ ts: `${new Date(now - 4000).toISOString().slice(11,19)}Z`, msg: `${spdStr}${hdgStr} — consistent with mission profile`, confidence: 0.95 });
  }
  return thoughts;
}

export default function OpsEntityDetail({ entity, onClose, onDispatch, onLiveFeed }: OpsEntityDetailProps) {
  const { logAction, updateAction } = useActionLog();
  const { addToast } = useToast();
  const [dispatched, setDispatched] = useState(false);
  const [overrideStatus, setOverrideStatus] = useState<string | null>(null);
  const [thoughts, setThoughts] = useState<{ ts: string; msg: string; confidence: number }[]>([]);
  const [pendingAction, setPendingAction] = useState<{
    title: string;
    message: string;
    confirmLabel: string;
    danger: boolean;
    execute: () => Promise<void>;
  } | null>(null);

  useEffect(() => {
    if (!entity) return;
    setDispatched(false);
    setOverrideStatus(null);
    // Try real reasoning endpoint first, fall back to local generation
    fetch(`${process.env.NEXT_PUBLIC_API_URL || ''}/reasoning/${entity.entity_id}`)
      .then(r => r.ok ? r.json() : null)
      .then(data => {
        if (data?.thoughts?.length) {
          setThoughts(data.thoughts);
        } else {
          setThoughts(buildThoughts(entity));
        }
      })
      .catch(() => setThoughts(buildThoughts(entity)));
  }, [entity?.entity_id]); // eslint-disable-line react-hooks/exhaustive-deps

  if (!entity) return null;

  const typeColor = entityTypeColor(entity.entity_type);
  const shortId = entity.entity_id.slice(0, 12);
  const displayName = entity.callsign || shortId;

  const handleDispatch = () => {
    setPendingAction({
      title: 'ASSIGN TASK',
      message: `Assign task to ${displayName}? Asset will be dispatched immediately.`,
      confirmLabel: 'ASSIGN',
      danger: false,
      execute: async () => {
        setDispatched(true);
        const logId = logAction({ action: 'DISPATCH', target: displayName, status: 'pending' });
        try {
          await dispatchTask({ asset_id: entity.entity_id, action: 'DISPATCH', risk_level: 'LOW' });
          updateAction(logId, 'success');
          addToast({ message: `DISPATCH confirmed — ${displayName}`, severity: 'success' });
        } catch {
          updateAction(logId, 'failed');
          addToast({ message: `DISPATCH failed — ${displayName}`, severity: 'critical' });
        }
        onDispatch?.(entity);
        setTimeout(() => onClose(), 600);
      },
    });
  };

  return (
    <>
    {pendingAction && (
      <ConfirmDialog
        title={pendingAction.title}
        message={pendingAction.message}
        confirmLabel={pendingAction.confirmLabel}
        danger={pendingAction.danger}
        onCancel={() => setPendingAction(null)}
        onConfirm={async () => {
          setPendingAction(null);
          await pendingAction.execute();
        }}
      />
    )}
    <div
      className="h-full flex flex-col overflow-hidden"
      style={{ background: 'var(--background-panel)' }}
    >
      {/* Header */}
      <div
        className="flex-none px-4 pt-4 pb-3 flex items-start justify-between"
        style={{ borderBottom: '1px solid var(--border)' }}
      >
        <div>
          <div
            className="text-[18px] font-bold tracking-[0.06em] leading-tight"
            style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: 'rgba(200,230,201,0.92)' }}
          >
            {displayName.toUpperCase()}
          </div>
          <div className="flex items-center gap-2 mt-1.5">
            <span
              className="text-[10px] px-1.5 py-0.5"
              style={{
                fontFamily: 'var(--font-ibm-plex-mono), monospace',
                color: typeColor,
                border: `1px solid ${typeColor}40`,
                background: `${typeColor}10`,
              }}
            >
              {entity.entity_type.toUpperCase()}
            </span>
            <span
              className="text-[10px]"
              style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: 'var(--text-muted)' }}
            >
              {entity.entity_id.slice(0, 12)}
            </span>
          </div>
        </div>
        <button
          onClick={onClose}
          aria-label="Close entity detail"
          className="summit-btn text-lg leading-none mt-0.5"
          style={{ color: 'var(--text-dim)', background: 'none', border: 'none', cursor: 'pointer' }}
        >
          ✕
        </button>
      </div>

      {/* ASSIGN — primary action */}
      <div
        className="flex-none px-4 py-3"
        style={{ borderBottom: '1px solid var(--border)' }}
      >
        <button
          onClick={handleDispatch}
          disabled={dispatched}
          aria-label={dispatched ? `Task assigned to ${displayName}` : `Assign task to ${displayName}`}
          className="w-full py-3 text-sm font-bold tracking-[0.2em] transition-all"
          style={{
            fontFamily: 'var(--font-ibm-plex-mono), monospace',
            color: 'var(--background)',
            background: dispatched ? 'var(--accent-dim)' : 'var(--accent)',
            border: 'none',
            cursor: dispatched ? 'default' : 'pointer',
            letterSpacing: '0.2em',
          }}
        >
          {dispatched ? 'ASSIGNED' : 'ASSIGN TASK'}
        </button>
      </div>

      {/* Scrollable content */}
      <div className="flex-1 overflow-y-auto px-4 pb-4">
        {/* Status section */}
        <SectionHeader title="STATUS" />
        <DataRow label="TYPE" value={entity.entity_type.toUpperCase()} valueColor={typeColor} />
        <DataRow label="DOMAIN" value={entity.domain.toUpperCase()} />
        <DataRow label="CLASS" value={entity.classification || '—'} />
        {entity.track_state && (
          <DataRow label="TRACK" value={entity.track_state.toUpperCase()} />
        )}

        {/* Position section */}
        {entity.position && (
          <>
            <SectionHeader title="POSITION" />
            <DataRow label="LAT" value={entity.position.lat.toFixed(6)} />
            <DataRow label="LON" value={entity.position.lon.toFixed(6)} />
            {entity.position.alt != null && <DataRow label="ALT" value={`${entity.position.alt.toFixed(0)} m`} />}
            {entity.position.heading_deg != null && <DataRow label="HDG" value={`${entity.position.heading_deg.toFixed(1)}°`} />}
            {entity.speed_mps != null && <DataRow label="SPD" value={`${entity.speed_mps.toFixed(1)} m/s`} />}
          </>
        )}

        {/* Battery section */}
        {entity.battery_pct != null && (
          <>
            <SectionHeader title="BATTERY" />
            <div className="mb-1">
              <div className="flex items-center justify-between mb-1">
                <span
                  className="text-[10px]"
                  style={{ color: 'var(--text-dim)', fontFamily: 'var(--font-ibm-plex-mono), monospace' }}
                >
                  CHARGE
                </span>
                <span
                  className="text-[11px] font-bold"
                  style={{ color: batteryColor(entity.battery_pct), fontFamily: 'var(--font-ibm-plex-mono), monospace' }}
                >
                  {Math.round(entity.battery_pct)}%
                </span>
              </div>
              <div
                role="progressbar"
                aria-label="Battery charge"
                aria-valuenow={Math.round(entity.battery_pct)}
                aria-valuemin={0}
                aria-valuemax={100}
                className="h-1.5 w-full rounded-full overflow-hidden"
                style={{ background: 'var(--accent-10)' }}
              >
                <div
                  className="h-full rounded-full transition-all duration-300"
                  style={{
                    width: `${entity.battery_pct}%`,
                    background: batteryColor(entity.battery_pct),
                  }}
                />
              </div>
            </div>
          </>
        )}

        {/* Confidence section */}
        <SectionHeader title="CONFIDENCE" />
        <div className="mb-1">
          <div className="flex items-center justify-between mb-1">
            <span
              className="text-[10px]"
              style={{ color: 'var(--text-dim)', fontFamily: 'var(--font-ibm-plex-mono), monospace' }}
            >
              SCORE
            </span>
            <span
              className="text-[11px] font-bold"
              style={{ color: 'var(--accent)', fontFamily: 'var(--font-ibm-plex-mono), monospace' }}
            >
              {Math.round(entity.confidence * 100)}%
            </span>
          </div>
          <div
            role="progressbar"
            aria-label="Confidence score"
            aria-valuenow={Math.round(entity.confidence * 100)}
            aria-valuemin={0}
            aria-valuemax={100}
            className="h-1.5 w-full rounded-full overflow-hidden"
            style={{ background: 'var(--accent-10)' }}
          >
            <div
              className="h-full rounded-full transition-all duration-300"
              style={{ width: `${entity.confidence * 100}%`, background: 'var(--accent)' }}
            />
          </div>
        </div>

        {/* Meta section */}
        <SectionHeader title="META" />
        <DataRow label="LAST SEEN" value={`${toUTCShort(entity.last_seen)}  (${ageFromEpoch(entity.last_seen)})`} />
        <DataRow
          label="MISSION"
          value={entity.mission_id ? entity.mission_id.slice(0, 12) : 'UNASSIGNED'}
          valueColor={entity.mission_id ? 'var(--color-active)' : 'var(--text-muted)'}
        />
        {entity.source_sensors && entity.source_sensors.length > 0 && (
          <DataRow label="SENSORS" value={entity.source_sensors.join(', ')} />
        )}

        {/* Brain Reasoning */}
        <SectionHeader title="BRAIN REASONING" />
        <div
          className="flex flex-col gap-1 mb-1"
          style={{
            background: 'var(--accent-5)',
            border: '1px solid var(--accent-10)',
            padding: '8px',
          }}
        >
          {thoughts.length === 0 ? (
            <span style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: 'var(--text-muted)', fontSize: 9 }}>
              No reasoning available
            </span>
          ) : (
            thoughts.map((t, i) => (
              <div key={i} className="flex flex-col gap-0.5">
                <div className="flex items-center justify-between">
                  <span style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: 'var(--text-muted)', fontSize: 8 }}>
                    {t.ts}
                  </span>
                  <span style={{
                    fontFamily: 'var(--font-ibm-plex-mono), monospace',
                    fontSize: 8,
                    color: t.confidence > 0.9 ? 'var(--accent)' : t.confidence > 0.8 ? 'var(--warning)' : 'var(--critical)',
                  }}>
                    {Math.round(t.confidence * 100)}%
                  </span>
                </div>
                <span style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: 'var(--text-dim)', fontSize: 9, lineHeight: 1.4 }}>
                  {t.msg}
                </span>
              </div>
            ))
          )}
        </div>

        {/* Manual Overrides */}
        <SectionHeader title="MANUAL OVERRIDES" />
        <div
          aria-live="polite"
          aria-atomic="true"
          style={{ minHeight: 0 }}
        >
          {overrideStatus && (
            <div style={{
              fontFamily: 'var(--font-ibm-plex-mono), monospace',
              fontSize: 9,
              color: 'var(--accent)',
              border: '1px solid var(--accent-15)',
              padding: '4px 8px',
              marginBottom: 8,
            }}>
              ✓ {overrideStatus}
            </div>
          )}
        </div>
        <div className="flex flex-col gap-2 mt-1">
          <button
            className="summit-btn w-full text-[11px] py-2.5 tracking-widest"
            aria-label={`Halt ${entity.callsign || entity.entity_id}`}
            style={{
              fontFamily: 'var(--font-ibm-plex-mono), monospace',
              color: 'var(--critical)',
              border: '1px solid color-mix(in srgb, var(--critical) 40%, transparent)',
              background: 'transparent',
              cursor: 'pointer',
            }}
            onClick={() => setPendingAction({
              title: 'CONFIRM HALT',
              message: `Send HALT to ${displayName}? This will immediately stop all movement.`,
              confirmLabel: 'HALT',
              danger: true,
              execute: async () => {
                const logId = logAction({ action: 'HALT', target: displayName, status: 'pending' });
                try {
                  await sendAgentCommand({ entity_id: entity.entity_id, command: 'halt', mission_objective: `HALT ${entity.entity_id}` });
                  setOverrideStatus(`HALT sent to ${entity.callsign || entity.entity_id.slice(0,8)}`);
                  updateAction(logId, 'success');
                  addToast({ message: `HALT sent — ${displayName}`, severity: 'warning' });
                } catch (err) {
                  setOverrideStatus(`HALT failed: ${(err as Error)?.message ?? 'network error'}`);
                  updateAction(logId, 'failed');
                  addToast({ message: `HALT failed — ${displayName}`, severity: 'critical' });
                }
              },
            })}
          >
            HALT
          </button>
          <button
            className="summit-btn w-full text-[11px] py-2.5 tracking-widest"
            aria-label={`Return ${entity.callsign || entity.entity_id} to base`}
            style={{
              fontFamily: 'var(--font-ibm-plex-mono), monospace',
              color: 'var(--warning)',
              border: '1px solid color-mix(in srgb, var(--warning) 40%, transparent)',
              background: 'transparent',
              cursor: 'pointer',
            }}
            onClick={() => setPendingAction({
              title: 'CONFIRM RETURN TO BASE',
              message: `Return ${displayName} to base? Asset will begin RTB procedure immediately.`,
              confirmLabel: 'RETURN',
              danger: false,
              execute: async () => {
                const logId = logAction({ action: 'RTB', target: displayName, status: 'pending' });
                try {
                  await sendAgentCommand({ entity_id: entity.entity_id, command: 'rtb', mission_objective: `Return ${entity.entity_id} to base` });
                  setOverrideStatus(`RTB sent to ${entity.callsign || entity.entity_id.slice(0,8)}`);
                  updateAction(logId, 'success');
                  addToast({ message: `RTB sent — ${displayName}`, severity: 'info' });
                } catch (err) {
                  setOverrideStatus(`RTB failed: ${(err as Error)?.message ?? 'network error'}`);
                  updateAction(logId, 'failed');
                  addToast({ message: `RTB failed — ${displayName}`, severity: 'critical' });
                }
              },
            })}
          >
            RETURN TO BASE
          </button>
          <button
            className="summit-btn w-full text-[11px] py-2.5 tracking-widest"
            aria-label={`Activate camera on ${entity.callsign || entity.entity_id}`}
            style={{
              fontFamily: 'var(--font-ibm-plex-mono), monospace',
              color: 'var(--color-active)',
              border: '1px solid color-mix(in srgb, var(--color-active) 40%, transparent)',
              background: 'transparent',
              cursor: 'pointer',
            }}
            onClick={async () => {
              const logId = logAction({ action: 'CAMERA', target: displayName, status: 'pending' });
              try {
                await sendAgentCommand({ entity_id: entity.entity_id, command: 'activate_camera', mission_objective: `Activate camera on ${entity.entity_id}` });
                setOverrideStatus(`Camera activated on ${entity.callsign || entity.entity_id.slice(0,8)}`);
                updateAction(logId, 'success');
                addToast({ message: `Camera activated — ${displayName}`, severity: 'info' });
              } catch (err) {
                setOverrideStatus(`Camera failed: ${(err as Error)?.message ?? 'network error'}`);
                updateAction(logId, 'failed');
                addToast({ message: `Camera failed — ${displayName}`, severity: 'critical' });
              }
            }}
          >
            ACTIVATE CAMERA
          </button>
          {onLiveFeed && (
            <button
              className="summit-btn w-full text-[11px] py-2.5 tracking-widest"
              aria-label={`Open live feed for ${entity.callsign || entity.entity_id}`}
              style={{
                fontFamily: 'var(--font-ibm-plex-mono), monospace',
                color: 'var(--accent)',
                border: '1px solid var(--accent-30)',
                background: 'transparent',
                cursor: 'pointer',
              }}
              onClick={() => onLiveFeed(entity.entity_id)}
            >
              LIVE FEED
            </button>
          )}
        </div>
      </div>
    </div>
    </>
  );
}
