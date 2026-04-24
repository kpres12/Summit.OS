'use client';

import React, { useState, useEffect, useCallback } from 'react';
import {
  parseMissionNlp,
  previewMissionWaypoints,
  createMission,
  fetchAssets,
  type NlpParseResponse,
  type WaypointPreview,
  type AssetAPI,
} from '@/lib/api';

const MISSION_TYPES = [
  'SURVEY', 'MONITOR', 'SEARCH', 'PERIMETER', 'ORBIT', 'DELIVER', 'INSPECT',
  'PATROL', 'RECON', 'ESCORT',
  // Military / government
  'HADR', 'FORCE_PROTECT', 'CASEVAC_ESCORT', 'ACE',
  // Domain-specific
  'MARITIME_SAR', 'ANTI_POACH', 'PRECISION_AG', 'PIPELINE_PATROL',
];

const MISSION_TYPE_LABELS: Record<string, string> = {
  SURVEY: 'SURVEY', MONITOR: 'MONITOR', SEARCH: 'SEARCH', PERIMETER: 'PERIMETER',
  ORBIT: 'ORBIT', DELIVER: 'DELIVER', INSPECT: 'INSPECT', PATROL: 'PATROL',
  RECON: 'RECON', ESCORT: 'ESCORT',
  HADR: 'HADR', FORCE_PROTECT: 'FORCE PROTECT', CASEVAC_ESCORT: 'CASEVAC ESCORT', ACE: 'ACE',
  MARITIME_SAR: 'MARITIME SAR', ANTI_POACH: 'ANTI-POACH', PRECISION_AG: 'PRECISION AG',
  PIPELINE_PATROL: 'PIPELINE PATROL',
};

const MISSION_TYPE_DOMAIN: Record<string, string> = {
  HADR: 'military', FORCE_PROTECT: 'military', CASEVAC_ESCORT: 'military', ACE: 'military',
  MARITIME_SAR: 'maritime', ANTI_POACH: 'wildlife',
  PRECISION_AG: 'agriculture', PIPELINE_PATROL: 'oilgas',
  INSPECT: 'utilities',
};

const PATTERNS = ['lawnmower', 'grid', 'expanding_square', 'orbit', 'parallel_track', 'direct', 'spiral', 'perimeter'];
const PATTERN_LABELS: Record<string, string> = {
  lawnmower: 'LAWNMOWER', grid: 'GRID', spiral: 'SPIRAL',
  expanding_square: 'EXPANDING SQ', orbit: 'ORBIT',
  parallel_track: 'PARALLEL TRACK', direct: 'DIRECT', perimeter: 'PERIMETER',
};
const PATTERN_DEFAULTS: Record<string, string> = {
  SURVEY: 'lawnmower', MONITOR: 'orbit', SEARCH: 'expanding_square',
  PERIMETER: 'perimeter', ORBIT: 'orbit', DELIVER: 'direct', INSPECT: 'lawnmower',
  PATROL: 'lawnmower', RECON: 'lawnmower', ESCORT: 'direct',
  HADR: 'lawnmower', FORCE_PROTECT: 'orbit', CASEVAC_ESCORT: 'direct', ACE: 'direct',
  MARITIME_SAR: 'expanding_square', ANTI_POACH: 'expanding_square',
  PRECISION_AG: 'lawnmower', PIPELINE_PATROL: 'lawnmower',
};

interface OpsMissionBuilderProps {
  onMissionLaunched: () => void;
  onRequestDrawArea: () => void;
  missionPolygon: { lat: number; lon: number }[] | null;
  onWaypointsChanged: (wps: WaypointPreview[]) => void;
}

function FieldLabel({ children }: { children: React.ReactNode }) {
  return (
    <div className="text-[9px] tracking-widest mb-1"
      style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: 'rgba(0,232,150,0.45)' }}>
      {children}
    </div>
  );
}

function SelectField({
  value, options, optionLabels, onChange,
}: {
  value: string;
  options: string[];
  optionLabels?: Record<string, string>;
  onChange: (v: string) => void;
}) {
  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className="w-full text-[11px] px-2 py-1.5 outline-none appearance-none"
      style={{
        fontFamily: 'var(--font-ibm-plex-mono), monospace',
        background: '#0A0F0C',
        border: '1px solid rgba(0,232,150,0.2)',
        color: '#00E896',
        cursor: 'pointer',
      }}
    >
      {options.map((o) => (
        <option key={o} value={o} style={{ background: '#0A0F0C' }}>
          {optionLabels ? optionLabels[o] || o : o}
        </option>
      ))}
    </select>
  );
}

export default function OpsMissionBuilder({
  onMissionLaunched,
  onRequestDrawArea,
  missionPolygon,
  onWaypointsChanged,
}: OpsMissionBuilderProps) {
  const [nlpText, setNlpText] = useState('');
  const [parsing, setParsing] = useState(false);
  const [parsed, setParsed] = useState<NlpParseResponse | null>(null);

  const [missionType, setMissionType] = useState('SURVEY');
  const [pattern, setPattern] = useState('grid');
  const [altitudeM, setAltitudeM] = useState(120);
  const [priority, setPriority] = useState('MEDIUM');
  const [selectedAssetId, setSelectedAssetId] = useState<string | 'auto'>('auto');

  const [assets, setAssets] = useState<AssetAPI[]>([]);
  const [assetsLoading, setAssetsLoading] = useState(false);

  const [waypoints, setWaypoints] = useState<WaypointPreview[]>([]);
  const [previewing, setPreviewing] = useState(false);

  const [launching, setLaunching] = useState(false);
  const [launchError, setLaunchError] = useState<string | null>(null);

  // Load assets once
  useEffect(() => {
    setAssetsLoading(true);
    fetchAssets()
      .then((r) => setAssets(r.assets || []))
      .catch(() => setAssets([]))
      .finally(() => setAssetsLoading(false));
  }, []);

  // Auto-preview when polygon or key params change
  useEffect(() => {
    if (!missionPolygon || missionPolygon.length < 3) return;
    setPreviewing(true);
    previewMissionWaypoints({ pattern, altitude_m: altitudeM, area: missionPolygon })
      .then((r) => {
        setWaypoints(r.waypoints);
        onWaypointsChanged(r.waypoints);
      })
      .catch(() => { setWaypoints([]); onWaypointsChanged([]); })
      .finally(() => setPreviewing(false));
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [missionPolygon, pattern, altitudeM]);

  const handleParse = useCallback(async () => {
    if (!nlpText.trim()) return;
    setParsing(true);
    try {
      const result = await parseMissionNlp(nlpText);
      setParsed(result);
      setMissionType(result.mission_type);
      setPattern(result.pattern);
      setAltitudeM(result.altitude_m);
    } catch {
      // ignore — user can still fill manually
    } finally {
      setParsing(false);
    }
  }, [nlpText]);

  const handleMissionTypeChange = (mt: string) => {
    setMissionType(mt);
    setPattern(PATTERN_DEFAULTS[mt] || 'grid');
  };

  const handleLaunch = async () => {
    setLaunchError(null);
    setLaunching(true);
    try {
      const payload: Record<string, unknown> = {
        mission_type: missionType,
        objectives: parsed?.objectives || [`${missionType} mission`],
        priority,
        pattern,
        altitude_m: altitudeM,
      };
      if (selectedAssetId !== 'auto') payload.asset_ids = [selectedAssetId];
      if (missionPolygon && missionPolygon.length >= 3) {
        payload.area = missionPolygon;
        const center = missionPolygon.reduce(
          (acc, p) => ({ lat: acc.lat + p.lat / missionPolygon.length, lon: acc.lon + p.lon / missionPolygon.length }),
          { lat: 0, lon: 0 }
        );
        payload.target_lat = center.lat;
        payload.target_lon = center.lon;
      }
      await createMission(payload);
      onWaypointsChanged([]);
      onMissionLaunched();
    } catch (e) {
      setLaunchError((e as Error)?.message || 'Launch failed');
    } finally {
      setLaunching(false);
    }
  };

  const canLaunch = missionPolygon && missionPolygon.length >= 3;

  return (
    <div className="flex flex-col h-full overflow-hidden" style={{ background: '#0D1210' }}>
      {/* Header */}
      <div className="flex-none px-3 py-2 flex items-center justify-between"
        style={{ borderBottom: '1px solid rgba(0,232,150,0.15)' }}>
        <span className="text-xs font-bold tracking-widest"
          style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: '#00E896' }}>
          MISSION BUILDER
        </span>
        <span className="text-[9px] tracking-wider"
          style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: 'rgba(0,232,150,0.35)' }}>
          PLAN // LAUNCH
        </span>
      </div>

      <div className="flex-1 overflow-y-auto">
        {/* NLP Input */}
        <div className="px-3 py-3" style={{ borderBottom: '1px solid rgba(0,232,150,0.08)' }}>
          <FieldLabel>NATURAL LANGUAGE COMMAND</FieldLabel>
          <div className="flex gap-1.5">
            <textarea
              value={nlpText}
              onChange={(e) => setNlpText(e.target.value)}
              onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleParse(); } }}
              placeholder="Grid search northwest sector at 150m, use Echo-1..."
              rows={2}
              className="flex-1 text-[11px] px-2 py-1.5 resize-none outline-none"
              style={{
                fontFamily: 'var(--font-ibm-plex-mono), monospace',
                background: '#0A0F0C',
                border: '1px solid rgba(0,232,150,0.2)',
                color: 'rgba(200,230,201,0.85)',
                lineHeight: 1.5,
              }}
            />
            <button
              onClick={handleParse}
              disabled={parsing || !nlpText.trim()}
              className="flex-none px-2 text-[10px] tracking-widest transition-all"
              style={{
                fontFamily: 'var(--font-ibm-plex-mono), monospace',
                color: parsing ? 'rgba(0,232,150,0.4)' : '#00E896',
                border: '1px solid rgba(0,232,150,0.3)',
                background: 'transparent',
                cursor: parsing ? 'wait' : 'pointer',
                alignSelf: 'stretch',
              }}
              onMouseEnter={(e) => { if (!parsing) (e.currentTarget as HTMLButtonElement).style.background = 'rgba(0,232,150,0.06)'; }}
              onMouseLeave={(e) => { (e.currentTarget as HTMLButtonElement).style.background = 'transparent'; }}
            >
              {parsing ? '...' : 'PARSE'}
            </button>
          </div>

          {parsed && (
            <div className="mt-2 px-2 py-1.5"
              style={{ background: 'rgba(0,232,150,0.03)', border: '1px solid rgba(0,232,150,0.1)' }}>
              <div className="flex items-start justify-between gap-2">
                <span className="text-[9px] leading-relaxed"
                  style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: 'rgba(200,230,201,0.6)' }}>
                  {parsed.interpretation}
                </span>
                <span className="flex-none text-[9px] font-bold"
                  style={{
                    fontFamily: 'var(--font-ibm-plex-mono), monospace',
                    color: parsed.confidence >= 0.8 ? '#00E896' : parsed.confidence >= 0.65 ? '#FFB300' : '#FF3B3B',
                  }}>
                  {Math.round(parsed.confidence * 100)}%
                </span>
              </div>
            </div>
          )}
        </div>

        {/* Mission Parameters */}
        <div className="px-3 py-3" style={{ borderBottom: '1px solid rgba(0,232,150,0.08)' }}>
          <div className="text-[9px] font-bold tracking-widest mb-2"
            style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: 'rgba(0,232,150,0.5)' }}>
            PARAMETERS
          </div>

          <div className="grid grid-cols-2 gap-2 mb-2">
            <div>
              <FieldLabel>MISSION TYPE</FieldLabel>
              <SelectField value={missionType} options={MISSION_TYPES} optionLabels={MISSION_TYPE_LABELS} onChange={handleMissionTypeChange} />
            </div>
            <div>
              <FieldLabel>PATTERN</FieldLabel>
              <SelectField value={pattern} options={PATTERNS} optionLabels={PATTERN_LABELS} onChange={setPattern} />
            </div>
          </div>
          {MISSION_TYPE_DOMAIN[missionType] && (
            <div className="mb-2 px-2 py-1"
              style={{ background: 'rgba(79,195,247,0.04)', border: '1px solid rgba(79,195,247,0.15)' }}>
              <span className="text-[9px]" style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: 'rgba(79,195,247,0.6)' }}>
                DOMAIN: {MISSION_TYPE_DOMAIN[missionType].toUpperCase()}
              </span>
            </div>
          )}

          <div className="grid grid-cols-2 gap-2">
            <div>
              <FieldLabel>ALTITUDE (m)</FieldLabel>
              <input
                type="number"
                value={altitudeM}
                min={20}
                max={500}
                step={10}
                onChange={(e) => setAltitudeM(Math.max(20, Math.min(500, Number(e.target.value))))}
                className="w-full text-[11px] px-2 py-1.5 outline-none"
                style={{
                  fontFamily: 'var(--font-ibm-plex-mono), monospace',
                  background: '#0A0F0C',
                  border: '1px solid rgba(0,232,150,0.2)',
                  color: '#00E896',
                }}
              />
            </div>
            <div>
              <FieldLabel>PRIORITY</FieldLabel>
              <SelectField value={priority} options={['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']} onChange={setPriority} />
            </div>
          </div>
        </div>

        {/* Asset Selection */}
        <div className="px-3 py-3" style={{ borderBottom: '1px solid rgba(0,232,150,0.08)' }}>
          <div className="text-[9px] font-bold tracking-widest mb-2"
            style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: 'rgba(0,232,150,0.5)' }}>
            ASSET
          </div>

          {assetsLoading ? (
            <div className="text-[9px]" style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: 'rgba(0,232,150,0.3)' }}>
              LOADING...
            </div>
          ) : (
            <div className="flex flex-col gap-1">
              {/* Auto-select option */}
              <button
                onClick={() => setSelectedAssetId('auto')}
                className="flex items-center gap-2 px-2 py-1.5 text-left transition-colors"
                style={{
                  background: selectedAssetId === 'auto' ? 'rgba(0,232,150,0.06)' : 'transparent',
                  border: `1px solid ${selectedAssetId === 'auto' ? 'rgba(0,232,150,0.3)' : 'rgba(0,232,150,0.08)'}`,
                  cursor: 'pointer',
                }}
              >
                <div className="w-2 h-2 rounded-full flex-none"
                  style={{ background: selectedAssetId === 'auto' ? '#00E896' : 'transparent', border: '1px solid #00E896' }} />
                <span className="text-[10px]"
                  style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: selectedAssetId === 'auto' ? '#00E896' : 'rgba(200,230,201,0.55)' }}>
                  AUTO-SELECT (nearest available)
                </span>
              </button>

              {assets.slice(0, 6).map((asset) => {
                const isSelected = selectedAssetId === asset.asset_id;
                const bat = asset.battery ?? null;
                return (
                  <button
                    key={asset.asset_id}
                    onClick={() => setSelectedAssetId(asset.asset_id)}
                    className="flex items-center gap-2 px-2 py-1.5 text-left transition-colors"
                    style={{
                      background: isSelected ? 'rgba(0,232,150,0.06)' : 'transparent',
                      border: `1px solid ${isSelected ? 'rgba(0,232,150,0.3)' : 'rgba(0,232,150,0.08)'}`,
                      cursor: 'pointer',
                    }}
                  >
                    <div className="w-2 h-2 rounded-full flex-none"
                      style={{ background: isSelected ? '#00E896' : 'transparent', border: '1px solid #00E896' }} />
                    <span className="flex-1 text-[10px]"
                      style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: isSelected ? '#00E896' : 'rgba(200,230,201,0.55)' }}>
                      {asset.asset_id}
                    </span>
                    <span className="text-[9px]"
                      style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: 'rgba(200,230,201,0.3)' }}>
                      {asset.type}
                    </span>
                    {bat !== null && (
                      <span className="text-[9px]"
                        style={{
                          fontFamily: 'var(--font-ibm-plex-mono), monospace',
                          color: bat > 40 ? '#00E896' : bat > 20 ? '#FFB300' : '#FF3B3B',
                        }}>
                        {Math.round(bat)}%
                      </span>
                    )}
                  </button>
                );
              })}

              {assets.length === 0 && (
                <div className="text-[9px] px-1"
                  style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: 'rgba(200,230,201,0.3)' }}>
                  No assets registered — auto-select will be used
                </div>
              )}
            </div>
          )}
        </div>

        {/* Mission Area */}
        <div className="px-3 py-3" style={{ borderBottom: '1px solid rgba(0,232,150,0.08)' }}>
          <div className="text-[9px] font-bold tracking-widest mb-2"
            style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: 'rgba(0,232,150,0.5)' }}>
            MISSION AREA
          </div>

          {missionPolygon && missionPolygon.length >= 3 ? (
            <div className="flex items-center gap-2">
              <div className="flex-1 text-[9px]"
                style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: '#4FC3F7' }}>
                ✓ {missionPolygon.length}-VERTEX POLYGON DRAWN
                {waypoints.length > 0 && (
                  <span style={{ color: 'rgba(79,195,247,0.6)' }}> · {waypoints.length} waypoints</span>
                )}
                {previewing && <span style={{ color: 'rgba(200,230,201,0.4)' }}> · computing...</span>}
              </div>
            </div>
          ) : (
            <div className="text-[9px] mb-2"
              style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: 'rgba(200,230,201,0.4)' }}>
              Draw the mission area on the map. Click to place vertices, double-click to close.
            </div>
          )}

          <button
            onClick={onRequestDrawArea}
            className="w-full text-[10px] py-2 tracking-widest mt-1 transition-all"
            style={{
              fontFamily: 'var(--font-ibm-plex-mono), monospace',
              color: '#4FC3F7',
              border: '1px solid rgba(79,195,247,0.35)',
              background: 'transparent',
              cursor: 'pointer',
            }}
            onMouseEnter={(e) => { (e.currentTarget as HTMLButtonElement).style.background = 'rgba(79,195,247,0.06)'; }}
            onMouseLeave={(e) => { (e.currentTarget as HTMLButtonElement).style.background = 'transparent'; }}
          >
            {missionPolygon && missionPolygon.length >= 3 ? '⬡ REDRAW AREA' : '⬡ DRAW AREA ON MAP'}
          </button>
        </div>
      </div>

      {/* Launch */}
      <div className="flex-none px-3 py-3" style={{ borderTop: '1px solid rgba(0,232,150,0.15)' }}>
        {launchError && (
          <div className="text-[9px] mb-2 px-2 py-1"
            style={{
              fontFamily: 'var(--font-ibm-plex-mono), monospace',
              color: '#FF3B3B',
              border: '1px solid rgba(255,59,59,0.2)',
              background: 'rgba(255,59,59,0.04)',
            }}>
            ✗ {launchError}
          </div>
        )}
        {!canLaunch && (
          <div className="text-[9px] mb-2 text-center"
            style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: 'rgba(200,230,201,0.3)' }}>
            Draw a mission area to enable launch
          </div>
        )}
        <button
          onClick={handleLaunch}
          disabled={!canLaunch || launching}
          className="w-full py-3 text-sm font-bold tracking-widest transition-all"
          style={{
            fontFamily: 'var(--font-ibm-plex-mono), monospace',
            color: '#080C0A',
            background: !canLaunch || launching ? 'rgba(0,232,150,0.3)' : '#00E896',
            border: 'none',
            cursor: !canLaunch || launching ? 'not-allowed' : 'pointer',
            letterSpacing: '0.2em',
          }}
          onMouseEnter={(e) => { if (canLaunch && !launching) (e.currentTarget as HTMLButtonElement).style.background = '#00CC74'; }}
          onMouseLeave={(e) => { if (canLaunch && !launching) (e.currentTarget as HTMLButtonElement).style.background = '#00E896'; }}
        >
          {launching ? 'LAUNCHING...' : 'LAUNCH MISSION'}
        </button>
      </div>
    </div>
  );
}
