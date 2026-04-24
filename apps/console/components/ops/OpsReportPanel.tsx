'use client';

import React, { useState, useCallback } from 'react';
import PanelHeader from '@/components/ui/PanelHeader';
import { apiFetch } from '@/lib/api';

// ─── Types ────────────────────────────────────────────────────────────────────

type ReportFormat = 'SALUTE' | 'SPOT' | '9LINE' | 'SITREP' | 'INSPECTION';

interface ReportResult {
  format: ReportFormat;
  report_id: string;
  ts_iso: string;
  text: string;
  structured: Record<string, unknown>;
}

interface OpsReportPanelProps {
  selectedEntityId?: string | null;
  onClose: () => void;
}

// ─── Sub-components ───────────────────────────────────────────────────────────

function FieldLabel({ children }: { children: React.ReactNode }) {
  return (
    <div className="text-[9px] tracking-widest mb-1"
      style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: 'rgba(0,232,150,0.45)' }}>
      {children}
    </div>
  );
}

function FormatButton({
  active, label, onClick,
}: { active: boolean; label: string; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className="px-2 py-1 text-[9px] tracking-widest transition-all"
      style={{
        fontFamily: 'var(--font-ibm-plex-mono), monospace',
        background: active ? 'rgba(0,232,150,0.12)' : 'transparent',
        border: `1px solid ${active ? 'rgba(0,232,150,0.4)' : 'rgba(0,232,150,0.12)'}`,
        color: active ? '#00E896' : 'rgba(0,232,150,0.4)',
        cursor: 'pointer',
      }}
    >
      {label}
    </button>
  );
}

const FORMAT_DESCRIPTIONS: Record<ReportFormat, string> = {
  SALUTE:     'Size · Activity · Location · Unit · Time · Equipment',
  SPOT:       'Size · Position · Other · Time (rapid)',
  '9LINE':    '9-Line MEDEVAC / CASEVAC request',
  SITREP:     'Situation report — all entities + active alerts',
  INSPECTION: 'Infrastructure inspection report',
};

// ─── Main Component ───────────────────────────────────────────────────────────

export default function OpsReportPanel({ selectedEntityId, onClose }: OpsReportPanelProps) {
  const [format, setFormat] = useState<ReportFormat>('SALUTE');
  const [callsign, setCallsign] = useState('HELI-1');
  const [opArea, setOpArea] = useState('AO ALPHA');
  const [assetId, setAssetId] = useState(selectedEntityId || '');
  const [generating, setGenerating] = useState(false);
  const [result, setResult] = useState<ReportResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  // 9-line specific
  const [patientCount, setPatientCount] = useState(1);
  const [precedence, setPrecedence] = useState<'U' | 'P' | 'R'>('U');
  const [security9L, setSecurity9L] = useState<'N' | 'P' | 'E'>('N');

  // Inspection specific
  const [assetType, setAssetType] = useState('POWER_LINE');
  const [inspAssetId, setInspAssetId] = useState('');

  const generate = useCallback(async () => {
    setGenerating(true);
    setError(null);
    setResult(null);

    try {
      let endpoint = '';
      let body: Record<string, unknown> = { observer_callsign: callsign };

      switch (format) {
        case 'SALUTE':
          endpoint = '/v1/reports/salute';
          body = { ...body, entity_id: assetId || selectedEntityId };
          break;
        case 'SPOT':
          endpoint = '/v1/reports/spot';
          body = { ...body, entity_id: assetId || selectedEntityId };
          break;
        case '9LINE':
          endpoint = '/v1/reports/nineline';
          body = {
            ...body,
            patient_count: patientCount,
            precedence,
            security: security9L,
            entity_id: assetId || selectedEntityId,
          };
          break;
        case 'SITREP':
          endpoint = '/v1/reports/sitrep';
          body = { ...body, operational_area: opArea };
          break;
        case 'INSPECTION':
          endpoint = '/v1/reports/inspection';
          body = { ...body, asset_id: inspAssetId || assetId, asset_type: assetType };
          break;
      }

      const resp = await apiFetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });

      if (!resp.ok) throw new Error(`${resp.status} ${resp.statusText}`);
      const data = await resp.json();
      setResult(data);
    } catch (e) {
      setError((e as Error)?.message || 'Generation failed');
    } finally {
      setGenerating(false);
    }
  }, [format, callsign, assetId, selectedEntityId, opArea, patientCount, precedence, security9L, assetType, inspAssetId]);

  const copyToClipboard = useCallback(() => {
    if (!result?.text) return;
    navigator.clipboard.writeText(result.text).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  }, [result]);

  return (
    <div className="flex flex-col h-full" style={{ background: '#0D1210' }}>
      <PanelHeader
        title="REPORT GENERATOR"
        subtitle="SALUTE // 9LINE // SITREP"
        onClose={onClose}
      />

      <div className="flex-1 overflow-y-auto">
        {/* Format selector */}
        <div className="px-3 py-3" style={{ borderBottom: '1px solid rgba(0,232,150,0.08)' }}>
          <FieldLabel>FORMAT</FieldLabel>
          <div className="flex flex-wrap gap-1.5 mb-2">
            {(['SALUTE', 'SPOT', '9LINE', 'SITREP', 'INSPECTION'] as ReportFormat[]).map((f) => (
              <FormatButton key={f} active={format === f} label={f} onClick={() => setFormat(f)} />
            ))}
          </div>
          <div className="text-[9px]"
            style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: 'rgba(200,230,201,0.35)' }}>
            {FORMAT_DESCRIPTIONS[format]}
          </div>
        </div>

        {/* Common fields */}
        <div className="px-3 py-3" style={{ borderBottom: '1px solid rgba(0,232,150,0.08)' }}>
          <div className="grid grid-cols-2 gap-2">
            <div>
              <FieldLabel>OBSERVER CALLSIGN</FieldLabel>
              <input
                value={callsign}
                onChange={(e) => setCallsign(e.target.value.toUpperCase())}
                className="w-full text-[11px] px-2 py-1.5 outline-none"
                style={{
                  fontFamily: 'var(--font-ibm-plex-mono), monospace',
                  background: '#0A0F0C', border: '1px solid rgba(0,232,150,0.2)', color: '#00E896',
                }}
              />
            </div>
            {(format === 'SALUTE' || format === 'SPOT' || format === '9LINE') && (
              <div>
                <FieldLabel>ENTITY ID</FieldLabel>
                <input
                  value={assetId}
                  onChange={(e) => setAssetId(e.target.value)}
                  placeholder={selectedEntityId || 'entity-id'}
                  className="w-full text-[11px] px-2 py-1.5 outline-none"
                  style={{
                    fontFamily: 'var(--font-ibm-plex-mono), monospace',
                    background: '#0A0F0C', border: '1px solid rgba(0,232,150,0.2)',
                    color: 'rgba(200,230,201,0.8)',
                  }}
                />
              </div>
            )}
            {format === 'SITREP' && (
              <div>
                <FieldLabel>AO / AREA NAME</FieldLabel>
                <input
                  value={opArea}
                  onChange={(e) => setOpArea(e.target.value.toUpperCase())}
                  className="w-full text-[11px] px-2 py-1.5 outline-none"
                  style={{
                    fontFamily: 'var(--font-ibm-plex-mono), monospace',
                    background: '#0A0F0C', border: '1px solid rgba(0,232,150,0.2)', color: '#00E896',
                  }}
                />
              </div>
            )}
          </div>
        </div>

        {/* Format-specific fields */}
        {format === '9LINE' && (
          <div className="px-3 py-3" style={{ borderBottom: '1px solid rgba(0,232,150,0.08)' }}>
            <div className="text-[9px] font-bold tracking-widest mb-2"
              style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: 'rgba(0,232,150,0.5)' }}>
              9-LINE PARAMETERS
            </div>
            <div className="grid grid-cols-3 gap-2">
              <div>
                <FieldLabel>PATIENTS</FieldLabel>
                <input type="number" min={1} max={99} value={patientCount}
                  onChange={(e) => setPatientCount(Math.max(1, Number(e.target.value)))}
                  className="w-full text-[11px] px-2 py-1.5 outline-none"
                  style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', background: '#0A0F0C', border: '1px solid rgba(0,232,150,0.2)', color: '#00E896' }}
                />
              </div>
              <div>
                <FieldLabel>PRECEDENCE</FieldLabel>
                <select value={precedence} onChange={(e) => setPrecedence(e.target.value as 'U' | 'P' | 'R')}
                  className="w-full text-[11px] px-2 py-1.5 outline-none appearance-none"
                  style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', background: '#0A0F0C', border: '1px solid rgba(0,232,150,0.2)', color: '#00E896', cursor: 'pointer' }}>
                  <option value="U">U — Urgent</option>
                  <option value="P">P — Priority</option>
                  <option value="R">R — Routine</option>
                </select>
              </div>
              <div>
                <FieldLabel>SECURITY</FieldLabel>
                <select value={security9L} onChange={(e) => setSecurity9L(e.target.value as 'N' | 'P' | 'E')}
                  className="w-full text-[11px] px-2 py-1.5 outline-none appearance-none"
                  style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', background: '#0A0F0C', border: '1px solid rgba(0,232,150,0.2)', color: '#00E896', cursor: 'pointer' }}>
                  <option value="N">N — Clear</option>
                  <option value="P">P — Possible</option>
                  <option value="E">E — Enemy</option>
                </select>
              </div>
            </div>
          </div>
        )}

        {format === 'INSPECTION' && (
          <div className="px-3 py-3" style={{ borderBottom: '1px solid rgba(0,232,150,0.08)' }}>
            <div className="text-[9px] font-bold tracking-widest mb-2"
              style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: 'rgba(0,232,150,0.5)' }}>
              ASSET DETAILS
            </div>
            <div className="grid grid-cols-2 gap-2">
              <div>
                <FieldLabel>ASSET ID</FieldLabel>
                <input value={inspAssetId} onChange={(e) => setInspAssetId(e.target.value)}
                  placeholder="SEGMENT-47A"
                  className="w-full text-[11px] px-2 py-1.5 outline-none"
                  style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', background: '#0A0F0C', border: '1px solid rgba(0,232,150,0.2)', color: 'rgba(200,230,201,0.8)' }}
                />
              </div>
              <div>
                <FieldLabel>ASSET TYPE</FieldLabel>
                <select value={assetType} onChange={(e) => setAssetType(e.target.value)}
                  className="w-full text-[11px] px-2 py-1.5 outline-none appearance-none"
                  style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', background: '#0A0F0C', border: '1px solid rgba(0,232,150,0.2)', color: '#00E896', cursor: 'pointer' }}>
                  <option value="POWER_LINE">POWER LINE</option>
                  <option value="PIPELINE">PIPELINE</option>
                  <option value="BRIDGE">BRIDGE</option>
                  <option value="CONSTRUCTION">CONSTRUCTION</option>
                  <option value="OIL_GAS">OIL & GAS</option>
                </select>
              </div>
            </div>
          </div>
        )}

        {/* Result */}
        {result && (
          <div className="px-3 py-3" style={{ borderBottom: '1px solid rgba(0,232,150,0.08)' }}>
            <div className="flex items-center justify-between mb-2">
              <div className="text-[9px] font-bold tracking-widest"
                style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: 'rgba(0,232,150,0.5)' }}>
                {result.report_id}
              </div>
              <button onClick={copyToClipboard}
                className="text-[9px] px-2 py-0.5 tracking-widest transition-all"
                style={{
                  fontFamily: 'var(--font-ibm-plex-mono), monospace',
                  color: copied ? '#00E896' : 'rgba(0,232,150,0.5)',
                  border: '1px solid rgba(0,232,150,0.2)',
                  background: 'transparent', cursor: 'pointer',
                }}>
                {copied ? '✓ COPIED' : 'COPY'}
              </button>
            </div>
            <pre className="text-[10px] leading-relaxed whitespace-pre-wrap"
              style={{
                fontFamily: 'var(--font-ibm-plex-mono), monospace',
                color: 'rgba(200,230,201,0.75)',
                background: '#080C0A',
                padding: '10px',
                border: '1px solid rgba(0,232,150,0.08)',
              }}>
              {result.text}
            </pre>
          </div>
        )}

        {error && (
          <div className="mx-3 mt-3 px-2 py-2"
            style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', fontSize: '10px', color: '#FF3B3B', border: '1px solid rgba(255,59,59,0.2)', background: 'rgba(255,59,59,0.04)' }}>
            ✗ {error}
          </div>
        )}
      </div>

      {/* Generate button */}
      <div className="flex-none px-3 py-3" style={{ borderTop: '1px solid rgba(0,232,150,0.15)' }}>
        <button onClick={generate} disabled={generating}
          className="w-full py-2.5 text-xs font-bold tracking-widest transition-all"
          style={{
            fontFamily: 'var(--font-ibm-plex-mono), monospace',
            color: '#080C0A',
            background: generating ? 'rgba(0,232,150,0.3)' : '#00E896',
            border: 'none',
            cursor: generating ? 'wait' : 'pointer',
            letterSpacing: '0.2em',
          }}>
          {generating ? 'GENERATING...' : `GENERATE ${format}`}
        </button>
      </div>
    </div>
  );
}
