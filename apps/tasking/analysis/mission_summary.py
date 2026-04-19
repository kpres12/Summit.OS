"""
Mission Summary Generator — Heli.OS

Generates structured after-action reports from mission execution data.
Covers: timeline, asset utilization, waypoint completion rate, alerts triggered,
replanning events, and performance metrics.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger("tasking.mission_summary")


class MissionSummaryGenerator:
    """Generates after-action summary reports from replay snapshots and events."""

    def generate(
        self,
        mission_id: str,
        snapshots: List[dict],
        events: List[dict],
    ) -> dict:
        """
        Generate a structured mission summary.

        Parameters
        ----------
        mission_id:
            The mission identifier.
        snapshots:
            Ordered list of world-state snapshots (from ReplayPersistence or in-memory store).
        events:
            Flat list of event dicts with {type, description, ts_iso}.

        Returns
        -------
        dict with keys:
            mission_id, duration_s, assets_deployed, waypoints_total,
            waypoints_completed, completion_rate, alerts_triggered,
            replan_count, asset_utilization, timeline
        """
        if not snapshots:
            return self._empty_summary(mission_id)

        # ── Duration ─────────────────────────────────────────────────────────
        start_ts = self._parse_ts(snapshots[0].get("ts_iso", ""))
        end_ts   = self._parse_ts(snapshots[-1].get("ts_iso", ""))
        duration_s = max(0.0, (end_ts - start_ts)) if (start_ts and end_ts) else 0.0

        # ── Assets ───────────────────────────────────────────────────────────
        asset_ids: set[str] = set()
        for snap in snapshots:
            for a in snap.get("assignments", []):
                asset_id = a.get("asset_id")
                if asset_id:
                    asset_ids.add(asset_id)
        assets_deployed = len(asset_ids)

        # ── Waypoints ────────────────────────────────────────────────────────
        # completed_seq is a count; use max across all assets/snapshots
        max_seq_per_asset: Dict[str, int] = {}
        total_waypoints = 0
        for snap in snapshots:
            for a in snap.get("assignments", []):
                aid = a.get("asset_id")
                if not aid:
                    continue
                seq = int(a.get("completed_seq", 0))
                max_seq_per_asset[aid] = max(max_seq_per_asset.get(aid, 0), seq)
                total_wp = int(a.get("total_waypoints", seq))  # fallback: completed = total
                total_waypoints = max(total_waypoints, total_wp)

        waypoints_completed = sum(max_seq_per_asset.values())
        if total_waypoints == 0:
            total_waypoints = waypoints_completed  # if never set, assume all done
        completion_rate = (
            round(waypoints_completed / total_waypoints, 4)
            if total_waypoints > 0
            else 1.0
        )

        # ── Asset utilization ────────────────────────────────────────────────
        # Fraction of snapshots each asset appears with status != IDLE/FAILED
        active_counts: Dict[str, int] = {aid: 0 for aid in asset_ids}
        snap_count = len(snapshots)
        for snap in snapshots:
            for a in snap.get("assignments", []):
                aid = a.get("asset_id")
                status = str(a.get("status", "")).upper()
                if aid and status not in ("IDLE", "FAILED", ""):
                    active_counts[aid] = active_counts.get(aid, 0) + 1

        asset_utilization = {
            aid: round(active_counts[aid] / snap_count, 4) if snap_count > 0 else 0.0
            for aid in asset_ids
        }

        # ── Alerts & replans ─────────────────────────────────────────────────
        # Count from inline snapshot events + provided events list
        all_events: List[dict] = list(events)
        for snap in snapshots:
            all_events.extend(snap.get("events", []))

        alerts_triggered = sum(
            1 for e in all_events if str(e.get("type", "")).upper() in ("ALERT", "ALERT_RAISED")
        )
        replan_count = sum(
            1 for e in all_events if str(e.get("type", "")).upper() in ("REPLAN", "REPLANNING")
        )

        # ── Timeline ─────────────────────────────────────────────────────────
        timeline = sorted(
            [
                {
                    "ts_iso": e.get("ts_iso", ""),
                    "type":   e.get("type", ""),
                    "description": e.get("description", ""),
                }
                for e in all_events
            ],
            key=lambda x: x["ts_iso"],
        )

        return {
            "mission_id":          mission_id,
            "duration_s":          round(duration_s, 1),
            "assets_deployed":     assets_deployed,
            "waypoints_total":     total_waypoints,
            "waypoints_completed": waypoints_completed,
            "completion_rate":     completion_rate,
            "alerts_triggered":    alerts_triggered,
            "replan_count":        replan_count,
            "asset_utilization":   asset_utilization,
            "timeline":            timeline,
        }

    # ------------------------------------------------------------------
    @staticmethod
    def _empty_summary(mission_id: str) -> dict:
        return {
            "mission_id":          mission_id,
            "duration_s":          0.0,
            "assets_deployed":     0,
            "waypoints_total":     0,
            "waypoints_completed": 0,
            "completion_rate":     0.0,
            "alerts_triggered":    0,
            "replan_count":        0,
            "asset_utilization":   {},
            "timeline":            [],
        }

    @staticmethod
    def _parse_ts(ts_iso: str) -> Optional[float]:
        if not ts_iso:
            return None
        try:
            return datetime.fromisoformat(ts_iso.replace("Z", "+00:00")).timestamp()
        except Exception:
            return None
