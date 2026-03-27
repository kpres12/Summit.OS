"""
Summit.OS Ontology Sync

Bridges raw service data into the ontology object store.

  from_entity(entity_dict)      → Asset or Track ObjectInstance
  from_alert(alert_dict)        → Alert ObjectInstance
  from_mission(mission_dict)    → Mission ObjectInstance
  from_observation(obs_dict)    → Observation ObjectInstance
  from_sitrep(sitrep_dict)      → SitRep ObjectInstance + links to incidents

All methods are idempotent — safe to call on every incoming event.
They call store._upsert() directly (not through ActionRunner) because sync
is internal pipeline flow, not a governed user action.

The canonical flow:
  Fusion      → entity/track telemetry   → sync.from_entity()
  Intelligence → advisory stream         → sync.from_alert() + sync.from_observation()
  Tasking     → mission status updates   → sync.from_mission()
  Intelligence → SITREP endpoint         → sync.from_sitrep()
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from .store import ObjectStore, get_store
from .types import LinkInstance, ObjectInstance

logger = logging.getLogger("ontology.sync")


class OntologySync:

    def __init__(self, store: Optional[ObjectStore] = None) -> None:
        self._store = store or get_store()

    # ── entity → Asset / Track ────────────────────────────────────────────────

    def from_entity(self, entity: Dict[str, Any]) -> ObjectInstance:
        """
        Sync a raw Entity dict (from Fusion / world model) into the ontology.
        Entities of type ASSET → Asset object type.
        All others (TRACK, OBSERVATION) → Track object type.
        """
        entity_type = entity.get("entity_type", "TRACK").upper()
        eid = entity.get("id", entity.get("entity_id", ""))
        domain = entity.get("domain", "AERIAL").upper()
        state = entity.get("state", "ACTIVE")
        kinematics = entity.get("kinematics") or {}
        position = kinematics.get("position") or {}

        if entity_type == "ASSET":
            instance = ObjectInstance(
                object_type="Asset",
                object_id=eid,
                properties={
                    "id": eid,
                    "name": entity.get("name", eid[:8]),
                    "asset_type": _map_asset_type(entity),
                    "domain": (
                        domain
                        if domain in ("AERIAL", "GROUND", "MARITIME", "FIXED", "CYBER")
                        else "AERIAL"
                    ),
                    "status": _map_asset_status(state),
                    "lat": position.get("lat"),
                    "lon": position.get("lon"),
                    "alt_m": position.get("altitude_msl"),
                    "heading_deg": kinematics.get("heading"),
                    "speed_mps": kinematics.get("speed"),
                    "battery_pct": (
                        entity.get("aerial", {}).get("battery_pct")
                        if isinstance(entity.get("aerial"), dict)
                        else None
                    ),
                    "org_id": entity.get("provenance", {}).get("org_id", ""),
                    "metadata": {"_source": "fusion"},
                },
            )
        else:
            # Track
            instance = ObjectInstance(
                object_type="Track",
                object_id=eid,
                properties={
                    "id": eid,
                    "state": _map_track_state(state),
                    "class_label": entity.get("class_label", "unknown"),
                    "confidence": entity.get("confidence", 0.0),
                    "lat": position.get("lat"),
                    "lon": position.get("lon"),
                    "alt_m": position.get("altitude_msl"),
                    "speed_mps": kinematics.get("speed"),
                    "heading_deg": kinematics.get("heading"),
                    "last_seen": entity.get("provenance", {}).get("updated_at"),
                    "org_id": entity.get("provenance", {}).get("org_id", ""),
                    "metadata": {"_source": "fusion"},
                },
            )

        return self._store._upsert(instance)

    # ── alert → Alert ─────────────────────────────────────────────────────────

    def from_alert(self, alert: Dict[str, Any]) -> ObjectInstance:
        """Sync a raw advisory/alert dict from the Intelligence service."""
        alert_id = alert.get("alert_id", alert.get("id", ""))
        instance = ObjectInstance(
            object_type="Alert",
            object_id=alert_id,
            properties={
                "id": alert_id,
                "severity": alert.get("risk_level", alert.get("severity", "LOW")),
                "description": alert.get("message", alert.get("description", "")),
                "source": alert.get("source", ""),
                "acknowledged": alert.get("acknowledged", False),
                "ts": alert.get(
                    "ts",
                    alert.get("created_at", datetime.now(timezone.utc).isoformat()),
                ),
                "org_id": alert.get("org_id", ""),
                "metadata": {"_source": "intelligence"},
            },
        )
        return self._store._upsert(instance)

    # ── observation → Observation ─────────────────────────────────────────────

    def from_observation(
        self, obs: Dict[str, Any], alert_id: Optional[str] = None
    ) -> ObjectInstance:
        """Sync a raw observation dict. Optionally link it to an Alert."""
        obs_id = obs.get("id", obs.get("obs_id", ""))
        instance = ObjectInstance(
            object_type="Observation",
            object_id=obs_id,
            properties={
                "id": obs_id,
                "class_label": obs.get("class_label", obs.get("class", "unknown")),
                "confidence": float(obs.get("confidence", 0.0)),
                "risk_level": obs.get("risk_level", "LOW"),
                "domain": obs.get("domain", "other"),
                "lat": obs.get("lat"),
                "lon": obs.get("lon"),
                "alt_m": obs.get("alt_m"),
                "sensor_id": obs.get("sensor_id", ""),
                "asset_id": obs.get("asset_id", obs.get("source", "")),
                "ts": obs.get("ts", datetime.now(timezone.utc).isoformat()),
                "is_fp": obs.get("is_fp", False),
                "features": obs.get("features"),
                "metadata": {"_source": "intelligence"},
            },
        )
        result = self._store._upsert(instance)

        # Create Observation → Alert link if provided
        if alert_id:
            self._store._upsert_link(
                LinkInstance(
                    link_type="observation_triggered_alert",
                    source_id=obs_id,
                    target_id=alert_id,
                )
            )

        return result

    # ── mission → Mission ─────────────────────────────────────────────────────

    def from_mission(self, mission: Dict[str, Any]) -> ObjectInstance:
        """Sync a MissionPlan/MissionStatus dict from the Tasking service."""
        mission_id = mission.get("id", mission.get("mission_id", ""))
        raw_obs = mission.get("raw_observation", {})
        instance = ObjectInstance(
            object_type="Mission",
            object_id=mission_id,
            properties={
                "id": mission_id,
                "mission_type": mission.get("mission_type", "SURVEY"),
                "status": mission.get("status", "PENDING"),
                "priority": mission.get("priority", "ROUTINE"),
                "lat": mission.get("lat", 0.0),
                "lon": mission.get("lon", 0.0),
                "alt_m": mission.get("alt_m", 80.0),
                "asset_id": mission.get("asset_id", raw_obs.get("asset_id", "")),
                "swarm_id": raw_obs.get("_swarm_id", mission.get("swarm_id", "")),
                "sector_id": raw_obs.get("_sector_id", ""),
                "rationale": mission.get("rationale", ""),
                "outcome_prob": mission.get("kofa_outcome_prob"),
                "org_id": mission.get("org_id", ""),
                "waypoints": raw_obs.get("_waypoints", []),
                "metadata": {"_source": "tasking"},
            },
        )
        result = self._store._upsert(instance)

        # Auto-create Asset → Mission link
        asset_id = instance.properties.get("asset_id")
        if asset_id:
            self._store._upsert_link(
                LinkInstance(
                    link_type="asset_executing_mission",
                    source_id=asset_id,
                    target_id=mission_id,
                )
            )

        # Auto-create Mission → Swarm link
        swarm_id = instance.properties.get("swarm_id")
        if swarm_id:
            self._store._upsert_link(
                LinkInstance(
                    link_type="mission_part_of_swarm",
                    source_id=mission_id,
                    target_id=swarm_id,
                )
            )

        return result

    # ── sitrep → SitRep ───────────────────────────────────────────────────────

    def from_sitrep(self, sitrep: Dict[str, Any]) -> ObjectInstance:
        """Sync a SitRep dict from the Intelligence /sitrep endpoint."""
        sitrep_id = sitrep.get("sitrep_id", sitrep.get("id", ""))
        instance = ObjectInstance(
            object_type="SitRep",
            object_id=sitrep_id,
            properties={
                "id": sitrep_id,
                "generated_at": sitrep.get("generated_at", ""),
                "generated_by": sitrep.get("generated_by", "kofa-template"),
                "time_window_s": sitrep.get("time_window_s"),
                "advisory_count": sitrep.get("advisory_count"),
                "highest_risk": sitrep.get("highest_risk", "LOW"),
                "summary": sitrep.get("summary", ""),
                "recommended_action": sitrep.get("recommended_action", ""),
                "findings": sitrep.get("findings", []),
                "org_id": sitrep.get("org_id", ""),
                "metadata": {"_source": "intelligence"},
            },
        )
        return self._store._upsert(instance)


# ── mapping helpers ────────────────────────────────────────────────────────────


def _map_asset_type(entity: dict) -> str:
    aerial = entity.get("aerial") or {}
    if isinstance(aerial, dict):
        ac_type = aerial.get("aircraft_type", "")
        mapping = {
            "MULTIROTOR": "UAV_MULTIROTOR",
            "FIXED_WING": "UAV_FIXED_WING",
            "VTOL": "UAV_VTOL",
        }
        if ac_type in mapping:
            return mapping[ac_type]
    domain = entity.get("domain", "").upper()
    if domain == "MARITIME":
        return "VESSEL"
    if domain == "FIXED":
        return "SENSOR_STATION"
    return "UNKNOWN"


def _map_asset_status(state: str) -> str:
    mapping = {
        "ACTIVE": "IN_FLIGHT",
        "INACTIVE": "OFFLINE",
        "TENTATIVE": "AVAILABLE",
        "COMPLETED": "RETURNING",
        "DELETED": "OFFLINE",
        "COASTING": "IN_FLIGHT",
    }
    return mapping.get(state.upper(), "AVAILABLE")


def _map_track_state(state: str) -> str:
    valid = ("TENTATIVE", "CONFIRMED", "COASTING", "DELETED")
    return state.upper() if state.upper() in valid else "TENTATIVE"


# ── singleton ──────────────────────────────────────────────────────────────────

_sync: OntologySync | None = None


def get_sync() -> OntologySync:
    global _sync
    if _sync is None:
        _sync = OntologySync()
    return _sync
