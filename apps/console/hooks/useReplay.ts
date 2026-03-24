'use client';

import { useState, useEffect, useCallback, useRef } from 'react';
import { fetchReplayTimeline, fetchReplaySnapshot, ReplayTimeline } from '@/lib/api';

export interface ReplaySnapshot {
  ts_iso: string;
  mission_id: string;
  assignments?: Array<{
    asset_id: string;
    lat: number;
    lon: number;
    status: string;
    completed_seq: number;
  }>;
  events?: Array<{ type: string; description: string; ts_iso: string }>;
}

export interface UseReplayResult {
  timeline: ReplayTimeline | null;
  snapshot: ReplaySnapshot | null;
  index: number;
  isPlaying: boolean;
  isLoading: boolean;
  error: string | null;
  play: () => void;
  pause: () => void;
  seek: (index: number) => void;
  step: (delta: number) => void;
}

export function useReplay(missionId: string | null): UseReplayResult {
  const [timeline, setTimeline]   = useState<ReplayTimeline | null>(null);
  const [snapshot, setSnapshot]   = useState<ReplaySnapshot | null>(null);
  const [index, setIndex]         = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError]         = useState<string | null>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Load timeline when missionId changes
  useEffect(() => {
    if (!missionId) {
      setTimeline(null);
      setSnapshot(null);
      setIndex(0);
      return;
    }
    setIsLoading(true);
    setError(null);
    fetchReplayTimeline(missionId)
      .then((tl) => {
        setTimeline(tl);
        setIndex(0);
      })
      .catch((e) => setError(String(e)))
      .finally(() => setIsLoading(false));
  }, [missionId]);

  // Fetch snapshot whenever index changes
  useEffect(() => {
    if (!missionId || !timeline || timeline.count === 0) return;
    const clampedIndex = Math.max(0, Math.min(index, timeline.count - 1));
    fetchReplaySnapshot(missionId, undefined, clampedIndex)
      .then((snap) => setSnapshot(snap as ReplaySnapshot))
      .catch(() => { /* best-effort */ });
  }, [missionId, index, timeline]);

  // Playback interval
  useEffect(() => {
    if (isPlaying && timeline && index < timeline.count - 1) {
      intervalRef.current = setInterval(() => {
        setIndex((i) => {
          const next = i + 1;
          if (next >= (timeline?.count ?? 0) - 1) {
            setIsPlaying(false);
          }
          return next;
        });
      }, 200);   // 5× speed (200ms per 1-second snapshot)
    } else {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    }
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [isPlaying, timeline, index]);

  const play   = useCallback(() => setIsPlaying(true), []);
  const pause  = useCallback(() => setIsPlaying(false), []);
  const seek   = useCallback((i: number) => { setIsPlaying(false); setIndex(i); }, []);
  const step   = useCallback((delta: number) => {
    setIsPlaying(false);
    setIndex((i) => Math.max(0, Math.min(i + delta, (timeline?.count ?? 1) - 1)));
  }, [timeline]);

  return { timeline, snapshot, index, isPlaying, isLoading, error, play, pause, seek, step };
}
