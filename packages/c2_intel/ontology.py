"""
C2 Domain Ontology System

The engine is domain-agnostic. This module makes it intelligent for specific contexts.

Same raw observations → different C2Ontology → different intelligence output.

    WILDFIRE_OPS:     "ASSET_OFFLINE" → "Coverage gap, re-task retardant drop asset"
    URBAN_SAR:        "SENSOR_LOSS"   → "Search area unmonitored, re-route ground team"
    MILITARY_ACE:     "NODE_FAILED"   → "Distributed C2 node lost, verify fallback auth"
    BORDER_PATROL:    "GEOFENCE_BREACH" → "Incursion detected, intercept window open"
    DISASTER_RESPONSE: "COMMS_DEGRADED" → "EOC link at risk, activate backup comms"

Each domain defines:
    - relevant_event_types: which C2EventTypes matter
    - chains:               what event sequences predict
    - action_plays:         what operator should DO about it
    - headline_format:      how to frame the brief

Adding a new domain = adding a new subclass. No changes to core engine required.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any



# ---------------------------------------------------------------------------
# Action Play — what an operator should DO when a pattern is detected
# ---------------------------------------------------------------------------

@dataclass
class C2ActionPlay:
    """
    Operator action play for a detected C2 signal pattern.
    Domain-specific. Replaces Mira's SalesPlay for C2 context.
    """
    domain:             str
    play_type:          str
    target_role:        str          # Who should act (e.g. "Mission Commander", "Watch Officer")
    urgency:            str          # "immediate", "90s", "5min", "monitor"
    recommended_action: str
    rationale:          str
    follow_on_steps:    List[str] = field(default_factory=list)
    escalation_threshold: Optional[str] = None


# ---------------------------------------------------------------------------
# Domain chain — event sequences that predict future events
# ---------------------------------------------------------------------------

@dataclass
class DomainChain:
    name:          str
    domain:        str
    trigger_types: List[str]    # C2EventType strings
    min_signals:   int = 1
    predictions:   List[Dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Base ontology
# ---------------------------------------------------------------------------

class C2DomainOntology:
    DOMAIN_ID:          str = "base"
    DOMAIN_NAME:        str = "Base C2"
    DOMAIN_DESCRIPTION: str = ""

    RELEVANT_EVENT_TYPES: List[str] = []
    CHAINS:               List[DomainChain] = []

    @classmethod
    def get_action_play(
        cls,
        event_types: List[str],
        entity_id: str,
        composite_score: int,
    ) -> Optional[C2ActionPlay]:
        return None

    @classmethod
    def format_brief_headline(cls, entity_id: str, top_event: str, score: int) -> str:
        return f"{entity_id} — {top_event.replace('_', ' ').title()} detected (score: {score})"

    @classmethod
    def is_relevant(cls, event_type: str) -> bool:
        if not cls.RELEVANT_EVENT_TYPES:
            return True
        return event_type in cls.RELEVANT_EVENT_TYPES


# ---------------------------------------------------------------------------
# Wildfire Operations
# ---------------------------------------------------------------------------

class WildfireOntology(C2DomainOntology):
    DOMAIN_ID          = "wildfire"
    DOMAIN_NAME        = "Wildfire Operations"
    DOMAIN_DESCRIPTION = "UAS/air asset coordination for wildfire suppression and monitoring"

    RELEVANT_EVENT_TYPES = [
        "ASSET_OFFLINE", "ASSET_ONLINE", "BATTERY_CRITICAL", "BATTERY_LOW",
        "SENSOR_LOSS", "SENSOR_RESTORED", "COMMS_DEGRADED", "COMMS_RESTORED",
        "GEOFENCE_BREACH", "GEOFENCE_CLEARED", "WEATHER_ALERT", "AIRSPACE_CONFLICT",
        "MISSION_STARTED", "MISSION_COMPLETED", "MISSION_ABORTED",
        "NODE_DEGRADED", "NODE_FAILED", "NODE_RECOVERED",
        "AUTHORITY_DELEGATED", "HANDOFF_INITIATED", "HANDOFF_COMPLETE",
    ]

    CHAINS = [
        DomainChain(
            name="Asset Offline → Coverage Gap",
            domain="wildfire",
            trigger_types=["ASSET_OFFLINE", "BATTERY_CRITICAL"],
            predictions=[
                {"event": "Unmonitored fire perimeter segment", "minutes": (0, 5), "confidence": 0.90,
                 "action": "Re-task nearest available air asset to cover gap"},
                {"event": "Fire spread into unmonitored zone", "minutes": (5, 30), "confidence": 0.50,
                 "action": "Notify incident commander of intelligence gap"},
            ],
        ),
        DomainChain(
            name="Weather Alert → Operations Hold",
            domain="wildfire",
            trigger_types=["WEATHER_ALERT"],
            predictions=[
                {"event": "Wind shift affecting fire behavior", "minutes": (10, 60), "confidence": 0.75,
                 "action": "Update perimeter projections and re-position retardant assets"},
                {"event": "Air operations ground stop", "minutes": (5, 20), "confidence": 0.65,
                 "action": "RTB all aerial assets and activate ground observation teams"},
            ],
        ),
    ]

    @classmethod
    def format_brief_headline(cls, entity_id, top_event, score):
        urgency = "CRITICAL" if score >= 80 else "HIGH" if score >= 60 else "ELEVATED"
        return f"[{urgency}] {entity_id} — {top_event.replace('_', ' ').title()}"

    @classmethod
    def get_action_play(cls, event_types, entity_id, composite_score):
        urgency = "immediate" if composite_score >= 80 else "5min" if composite_score >= 60 else "monitor"

        if "ASSET_OFFLINE" in event_types or "BATTERY_CRITICAL" in event_types:
            return C2ActionPlay(
                domain="wildfire", play_type="coverage_gap",
                target_role="Air Tactical Group Supervisor (ATGS)",
                urgency=urgency,
                recommended_action=f"Re-task nearest available air asset to cover {entity_id} position",
                rationale="Offline asset creates unmonitored fire perimeter — spread risk increases",
                follow_on_steps=[
                    "Query asset inventory for nearest available UAS with sufficient battery",
                    "Issue re-tasking command via Air Operations Branch",
                    "Notify Incident Commander of coverage gap duration",
                    "Update Common Operating Picture with gap zone",
                ],
                escalation_threshold="Fire perimeter within 500m of structure → escalate to IC immediately",
            )

        if "WEATHER_ALERT" in event_types:
            return C2ActionPlay(
                domain="wildfire", play_type="weather_hold",
                target_role="Air Operations Branch Director",
                urgency=urgency,
                recommended_action=f"Evaluate ground stop for all aerial assets near {entity_id}",
                rationale="Wind events above minimums create collision risk for aerial assets",
                follow_on_steps=[
                    "Pull current ASOS/AWOS data for operational area",
                    "Notify all aerial supervisors of weather advisory",
                    "Issue RTB if winds exceed UAS flight envelope",
                ],
            )

        return C2ActionPlay(
            domain="wildfire", play_type="situational_awareness",
            target_role="Watch Officer",
            urgency="monitor",
            recommended_action=f"Maintain monitoring for {entity_id}",
            rationale=f"Signal activity score {composite_score} warrants elevated awareness",
            follow_on_steps=["Log in ICS-214", "Set 10-minute reassessment"],
        )


# ---------------------------------------------------------------------------
# Urban Search and Rescue
# ---------------------------------------------------------------------------

class UrbanSARontology(C2DomainOntology):
    DOMAIN_ID          = "urban_sar"
    DOMAIN_NAME        = "Urban Search and Rescue"
    DOMAIN_DESCRIPTION = "UAS and ground team coordination for urban SAR operations"

    RELEVANT_EVENT_TYPES = [
        "ASSET_OFFLINE", "SENSOR_LOSS", "COMMS_DEGRADED", "COMMS_RESTORED",
        "THREAT_IDENTIFIED", "GEOFENCE_BREACH", "MISSION_STARTED",
        "MISSION_COMPLETED", "MISSION_ABORTED", "NODE_FAILED",
        "AUTHORITY_DELEGATED", "HANDOFF_INITIATED", "HANDOFF_COMPLETE",
        "BATTERY_CRITICAL", "BATTERY_LOW", "WEATHER_ALERT",
    ]

    CHAINS = [
        DomainChain(
            name="Sensor Loss → Search Gap",
            domain="urban_sar",
            trigger_types=["SENSOR_LOSS", "ASSET_OFFLINE"],
            predictions=[
                {"event": "Uncleared search segment", "minutes": (0, 3), "confidence": 0.90,
                 "action": "Re-task adjacent UAS or redirect ground team to cover cleared zone"},
            ],
        ),
        DomainChain(
            name="Comms Degraded → Team Safety Risk",
            domain="urban_sar",
            trigger_types=["COMMS_DEGRADED"],
            predictions=[
                {"event": "Ground team out of contact", "minutes": (2, 8), "confidence": 0.75,
                 "action": "Issue check-in request and activate emergency contact protocol"},
            ],
        ),
    ]

    @classmethod
    def format_brief_headline(cls, entity_id, top_event, score):
        urgency = "CRITICAL" if score >= 80 else "HIGH" if score >= 55 else "ELEVATED"
        return f"[{urgency}] {entity_id} — {top_event.replace('_', ' ').title()}"

    @classmethod
    def get_action_play(cls, event_types, entity_id, composite_score):
        urgency = "immediate" if composite_score >= 80 else "90s" if composite_score >= 60 else "monitor"

        if "SENSOR_LOSS" in event_types or "ASSET_OFFLINE" in event_types:
            return C2ActionPlay(
                domain="urban_sar", play_type="search_coverage",
                target_role="Operations Section Chief",
                urgency=urgency,
                recommended_action=f"Re-assign search coverage for {entity_id} sector immediately",
                rationale="Search gap risks missing viable survivors — every minute counts",
                follow_on_steps=[
                    "Identify nearest available sensor or ground team",
                    "Issue re-tasking command via Operations Branch",
                    "Mark affected sector as uncleared on ICS-204",
                    "Notify Search Team Leaders of coverage change",
                ],
                escalation_threshold="Sector contains known victim location → escalate to IC",
            )

        if "COMMS_DEGRADED" in event_types:
            return C2ActionPlay(
                domain="urban_sar", play_type="comms_recovery",
                target_role="Communications Unit Leader",
                urgency=urgency,
                recommended_action=f"Restore comms with {entity_id} via alternate channel",
                rationale="Ground team out of contact is a life-safety risk in structural collapse environment",
                follow_on_steps=[
                    "Attempt contact via backup radio frequency",
                    "Dispatch liaison to last known team position",
                    "Activate LASSO or personal locator beacon check",
                ],
                escalation_threshold="No contact after 3 attempts → activate emergency localization protocol",
            )

        return C2ActionPlay(
            domain="urban_sar", play_type="situational_awareness",
            target_role="Watch Officer",
            urgency="monitor",
            recommended_action=f"Monitor {entity_id} — score {composite_score}",
            rationale="Elevated activity warrants watchlist status",
            follow_on_steps=["Log in ICS-214", "Set 5-minute reassessment"],
        )


# ---------------------------------------------------------------------------
# Military ACE (Agile Combat Employment) — the CANVAS TA2 context
# ---------------------------------------------------------------------------

class MilitaryACEOntology(C2DomainOntology):
    DOMAIN_ID          = "military_ace"
    DOMAIN_NAME        = "Agile Combat Employment (ACE)"
    DOMAIN_DESCRIPTION = "Distributed C2 for expeditionary operations in contested/degraded environments"

    RELEVANT_EVENT_TYPES = [
        "COMMS_DEGRADED", "COMMS_RESTORED", "LINK_LOST", "LINK_DEGRADED",
        "THREAT_IDENTIFIED", "THREAT_NEUTRALIZED",
        "AUTHORITY_DELEGATED", "AUTHORITY_REVOKED",
        "MISSION_STARTED", "MISSION_COMPLETED", "MISSION_ABORTED",
        "NODE_DEGRADED", "NODE_FAILED", "NODE_RECOVERED",
        "ENGAGEMENT_AUTHORIZED", "ENGAGEMENT_DENIED", "ENGAGEMENT_COMPLETE",
        "HANDOFF_INITIATED", "HANDOFF_COMPLETE",
        "ASSET_OFFLINE", "ASSET_ONLINE",
        "GEOFENCE_BREACH", "SENSOR_LOSS",
        "PEER_OBSERVATION",
    ]

    CHAINS = [
        DomainChain(
            name="Link Degraded → Authority Pre-delegation",
            domain="military_ace",
            trigger_types=["LINK_DEGRADED", "COMMS_DEGRADED"],
            predictions=[
                {"event": "Link loss within communication window", "minutes": (1, 5), "confidence": 0.80,
                 "action": "Pre-delegate authority to next echelon before link fails"},
                {"event": "Autonomous fallback activation", "minutes": (2, 8), "confidence": 0.75,
                 "action": "Verify last-transmitted orders are ROE-compliant for autonomous execution"},
            ],
        ),
        DomainChain(
            name="Multiple Node Failure → C2 Resilience Test",
            domain="military_ace",
            trigger_types=["NODE_FAILED", "COMMS_DEGRADED"],
            min_signals=2,
            predictions=[
                {"event": "Adversarial jamming suspected", "minutes": (0, 5), "confidence": 0.65,
                 "action": "Activate PACE alternate comms and notify higher for threat assessment"},
                {"event": "Distributed authority convergence required", "minutes": (2, 10), "confidence": 0.80,
                 "action": "Execute CRDT mesh resync — verify surviving nodes have current world state"},
            ],
        ),
        DomainChain(
            name="Threat Identified → Engagement Decision",
            domain="military_ace",
            trigger_types=["THREAT_IDENTIFIED"],
            predictions=[
                {"event": "Positive Identification (PID) required", "minutes": (1, 4), "confidence": 0.85,
                 "action": "Request PID confirmation from ISR asset before engagement authority granted"},
                {"event": "Engagement window closes", "minutes": (2, 8), "confidence": 0.70,
                 "action": "Target may maneuver or exploit — decision delay has tactical cost"},
            ],
        ),
    ]

    @classmethod
    def format_brief_headline(cls, entity_id, top_event, score):
        urgency = "FLASH" if score >= 90 else "IMMEDIATE" if score >= 75 else "PRIORITY" if score >= 55 else "ROUTINE"
        return f"[{urgency}] {entity_id} — {top_event.replace('_', ' ').title()}"

    @classmethod
    def get_action_play(cls, event_types, entity_id, composite_score):
        urgency = "immediate" if composite_score >= 75 else "90s" if composite_score >= 55 else "monitor"

        if "LINK_LOST" in event_types or ("COMMS_DEGRADED" in event_types and composite_score >= 70):
            return C2ActionPlay(
                domain="military_ace", play_type="degraded_comms_protocol",
                target_role="Mission Commander",
                urgency="immediate",
                recommended_action=f"Execute PACE plan for {entity_id} — activate alternate comms",
                rationale="Link loss in ACE activates pre-authorized autonomous execution — verify ROE compliance",
                follow_on_steps=[
                    "Attempt contact via Primary, Alternate, Contingency channels",
                    "Verify last transmitted orders are within current ROE",
                    "Issue pre-delegation to next authority tier if link unrecoverable",
                    "Notify higher of C2 status via out-of-band method",
                ],
                escalation_threshold="No contact for 5 minutes → execute emergency action message (EAM)",
            )

        if "THREAT_IDENTIFIED" in event_types:
            return C2ActionPlay(
                domain="military_ace", play_type="engagement_decision",
                target_role="Mission Commander / JTAC",
                urgency=urgency,
                recommended_action=f"Initiate PID process for threat near {entity_id}",
                rationale="OODA decision clock starts at identification — delay risks target maneuver",
                follow_on_steps=[
                    "Request ISR asset confirmation of PID (LOAC compliance)",
                    "Verify target is within pre-authorized engagement criteria",
                    "Coordinate deconfliction with adjacent units",
                    "Issue engagement authority if PID confirmed and criteria met",
                ],
                escalation_threshold="PID not achievable → do not engage, request higher authority",
            )

        if "NODE_FAILED" in event_types or "NODE_DEGRADED" in event_types:
            return C2ActionPlay(
                domain="military_ace", play_type="c2_resilience",
                target_role="C2 Node Operator",
                urgency=urgency,
                recommended_action=f"Redistribute authority from failed node {entity_id}",
                rationale="ACE depends on distributed authority — single node failure must not halt operations",
                follow_on_steps=[
                    "Identify surviving nodes with capacity to absorb tasks",
                    "Issue authority delegation messages via mesh protocol",
                    "Verify CRDT world state convergence across surviving nodes",
                    "Update PACE plan to reflect changed topology",
                ],
            )

        return C2ActionPlay(
            domain="military_ace", play_type="force_tracking",
            target_role="Watch Officer",
            urgency="monitor",
            recommended_action=f"Maintain situational awareness for {entity_id}",
            rationale=f"Signal activity score {composite_score} warrants monitoring",
            follow_on_steps=["Update Common Operating Picture", "Set 2-minute reassessment"],
        )


# ---------------------------------------------------------------------------
# Disaster Response
# ---------------------------------------------------------------------------

class DisasterResponseOntology(C2DomainOntology):
    DOMAIN_ID          = "disaster_response"
    DOMAIN_NAME        = "Disaster Response"
    DOMAIN_DESCRIPTION = "Multi-agency incident coordination for natural disaster and mass casualty events"

    RELEVANT_EVENT_TYPES = [
        "COMMS_DEGRADED", "COMMS_RESTORED", "ASSET_OFFLINE", "ASSET_ONLINE",
        "SENSOR_LOSS", "SENSOR_RESTORED", "MISSION_STARTED", "MISSION_ABORTED",
        "NODE_FAILED", "AUTHORITY_DELEGATED", "HANDOFF_INITIATED",
        "WEATHER_ALERT", "GEOFENCE_BREACH", "BATTERY_CRITICAL",
    ]

    @classmethod
    def get_action_play(cls, event_types, entity_id, composite_score):
        urgency = "immediate" if composite_score >= 80 else "5min" if composite_score >= 55 else "monitor"

        if "COMMS_DEGRADED" in event_types:
            return C2ActionPlay(
                domain="disaster_response", play_type="comms_continuity",
                target_role="Communications Unit Leader (COML)",
                urgency=urgency,
                recommended_action=f"Activate backup comms for {entity_id} EOC link",
                rationale="EOC communications loss degrades multi-agency coordination",
                follow_on_steps=[
                    "Activate SHARES or GETS backup network",
                    "Notify Agency Representatives of comms status",
                    "Establish runner protocol if all electronic comms fail",
                ],
                escalation_threshold="All comms pathways fail → physical courier to jurisdiction EOC",
            )

        return C2ActionPlay(
            domain="disaster_response", play_type="incident_monitoring",
            target_role="Watch Officer",
            urgency="monitor",
            recommended_action=f"Log and monitor {entity_id} — score {composite_score}",
            rationale="Activity above baseline warrants tracking",
            follow_on_steps=["Log in WebEOC", "Set 15-minute reassessment"],
        )


# ---------------------------------------------------------------------------
# Domain registry
# ---------------------------------------------------------------------------

_REGISTRY: Dict[str, type] = {
    "wildfire":         WildfireOntology,
    "urban_sar":        UrbanSARontology,
    "military_ace":     MilitaryACEOntology,
    "disaster_response": DisasterResponseOntology,
}


def get_ontology(domain_id: str = "disaster_response") -> type:
    """Return the ontology class for a given domain."""
    return _REGISTRY.get(domain_id, DisasterResponseOntology)


def list_domains() -> List[Dict]:
    return [
        {
            "id":           cls.DOMAIN_ID,
            "name":         cls.DOMAIN_NAME,
            "description":  cls.DOMAIN_DESCRIPTION,
            "event_types":  len(cls.RELEVANT_EVENT_TYPES),
            "chains":       len(cls.CHAINS),
        }
        for cls in _REGISTRY.values()
    ]


def register_ontology(domain_id: str, ontology_class: type):
    """Register a custom domain ontology at runtime."""
    _REGISTRY[domain_id] = ontology_class
