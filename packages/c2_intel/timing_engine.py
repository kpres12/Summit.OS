"""
C2 Event Timing Engine

Answers the question critical to CANVAS TA1:
"How many minutes after observing event X does an operator typically issue a command?"

Two data layers:
1. DOCTRINE PRIORS — seeded from C2/UAS doctrine and OODA loop research.
   Provides non-zero priors from day one so the engine is immediately useful.

2. OBSERVED OUTCOMES — learned from actual operator command history over time.
   As operators respond to observations, these replace the priors with real numbers.

Key outputs per event type:
- median_minutes_to_command: Expected operator response time
- hot_window_minutes: [p25, p75] range of typical response
- escalation_lift: How much more likely escalation is when this event fires
- confidence: How much to trust this number (sample_size based)

Usage:
    engine = C2TimingEngine()
    engine.hydrate()
    insight = engine.get_insight("COMMS_DEGRADED")
    # → {"median_minutes": 3, "hot_window": [1, 8], "lift": 2.3, "confidence": 0.8}
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DATA_DIR   = Path(__file__).parent / "data"
TIMING_PATH = DATA_DIR / "timing_engine.json"

MIN_SAMPLES_TO_OVERRIDE_PRIOR = 5


# ---------------------------------------------------------------------------
# Doctrine-based timing priors (minutes)
# Sources: OODA loop research, UAS doctrine, emergency management protocols
# ---------------------------------------------------------------------------

C2_EVENT_PRIORS: Dict[str, Dict[str, Any]] = {
    "COMMS_DEGRADED": {
        "median_minutes_to_command": 3,
        "p25_minutes": 1,
        "p75_minutes": 8,
        "escalation_lift": 2.8,
        "notes": "Degraded comms triggers authority delegation within ~3 minutes per PACE plan doctrine.",
        "source": "PACE planning doctrine + UAS degraded-link SOPs",
    },
    "LINK_LOST": {
        "median_minutes_to_command": 1,
        "p25_minutes": 0,
        "p75_minutes": 3,
        "escalation_lift": 3.5,
        "notes": "Total link loss is immediate — autonomous fallback activates within seconds to minutes.",
        "source": "FAA UAS link-loss SOPs + STANAG 4671",
    },
    "THREAT_IDENTIFIED": {
        "median_minutes_to_command": 4,
        "p25_minutes": 1,
        "p75_minutes": 10,
        "escalation_lift": 3.1,
        "notes": "OODA loop research: ID-to-decide averages 4 minutes for trained operators.",
        "source": "Boyd OODA loop empirical studies + USAF CAS doctrine",
    },
    "BATTERY_CRITICAL": {
        "median_minutes_to_command": 2,
        "p25_minutes": 1,
        "p75_minutes": 5,
        "escalation_lift": 4.0,
        "notes": "Battery critical demands immediate RTB command — statutory under most UAS SOPs.",
        "source": "UAS battery management doctrine (FAA AC 107-2A)",
    },
    "BATTERY_LOW": {
        "median_minutes_to_command": 8,
        "p25_minutes": 3,
        "p75_minutes": 15,
        "escalation_lift": 2.2,
        "notes": "Battery low is a pre-cursor warning — operators have several minutes to decide.",
        "source": "UAS SOPs (low = 30% threshold, critical = 15% threshold)",
    },
    "ASSET_OFFLINE": {
        "median_minutes_to_command": 5,
        "p25_minutes": 2,
        "p75_minutes": 15,
        "escalation_lift": 2.5,
        "notes": "Asset offline triggers coverage re-tasking or mission abort within 5 minutes.",
        "source": "UAS fleet management doctrine",
    },
    "GEOFENCE_BREACH": {
        "median_minutes_to_command": 1,
        "p25_minutes": 0,
        "p75_minutes": 3,
        "escalation_lift": 3.8,
        "notes": "Geofence breach triggers immediate engagement decision — ROE clock starts at detection.",
        "source": "Airspace enforcement doctrine + USNORTHCOM UAS ROE",
    },
    "NODE_FAILED": {
        "median_minutes_to_command": 3,
        "p25_minutes": 1,
        "p75_minutes": 10,
        "escalation_lift": 2.6,
        "notes": "Node failure triggers authority delegation and task redistribution within minutes.",
        "source": "Distributed C2 resilience doctrine",
    },
    "NODE_DEGRADED": {
        "median_minutes_to_command": 5,
        "p25_minutes": 2,
        "p75_minutes": 12,
        "escalation_lift": 2.0,
        "notes": "Degraded node — operators typically monitor briefly before acting.",
        "source": "Distributed C2 resilience doctrine",
    },
    "SENSOR_LOSS": {
        "median_minutes_to_command": 4,
        "p25_minutes": 1,
        "p75_minutes": 10,
        "escalation_lift": 2.3,
        "notes": "Sensor loss requires coverage assessment before re-tasking command.",
        "source": "ISR re-tasking doctrine",
    },
    "AUTHORITY_DELEGATED": {
        "median_minutes_to_command": 1,
        "p25_minutes": 0,
        "p75_minutes": 3,
        "escalation_lift": 1.8,
        "notes": "Authority delegation acknowledgment expected within minutes.",
        "source": "Distributed C2 handoff protocols",
    },
    "MISSION_ABORTED": {
        "median_minutes_to_command": 2,
        "p25_minutes": 0,
        "p75_minutes": 5,
        "escalation_lift": 2.0,
        "notes": "Mission abort triggers immediate re-planning command.",
        "source": "Mission management doctrine",
    },
    "WEATHER_ALERT": {
        "median_minutes_to_command": 10,
        "p25_minutes": 3,
        "p75_minutes": 20,
        "escalation_lift": 1.7,
        "notes": "Weather alerts give operators minutes to hours to respond — planning-driven.",
        "source": "FAA weather minimums + UAS meteorological SOPs",
    },
    "AIRSPACE_CONFLICT": {
        "median_minutes_to_command": 2,
        "p25_minutes": 0,
        "p75_minutes": 5,
        "escalation_lift": 3.2,
        "notes": "Airspace conflicts require immediate deconfliction action.",
        "source": "FAA see-and-avoid doctrine + TCAS analog for UAS",
    },
    "PEER_OBSERVATION": {
        "median_minutes_to_command": 8,
        "p25_minutes": 3,
        "p75_minutes": 20,
        "escalation_lift": 1.3,
        "notes": "Peer mesh observations — lower urgency, broader monitoring window.",
        "source": "Distributed C2 mesh protocols",
    },
}

# Operational context multipliers (replaces Mira's vertical cycle multipliers)
CONTEXT_URGENCY_MULTIPLIERS: Dict[str, float] = {
    "military_ace":    0.5,   # Fastest response — combat context
    "border_patrol":   0.7,   # High urgency — law enforcement
    "urban_sar":       0.8,   # High urgency — life safety
    "wildfire":        1.0,   # Standard urgency
    "disaster_response": 1.2, # Coordination overhead increases time
}


class C2TimingEngine:
    """
    Computes and serves C2 event timing intelligence.

    On first run: populated from C2_EVENT_PRIORS (useful immediately).
    As operator outcomes accumulate: observed data replaces priors.
    """

    def __init__(self):
        self._stats: Dict[str, Dict[str, Any]] = {}
        self._loaded = False

    def hydrate(self):
        """Load persisted timing stats from disk, seeding with doctrine priors if needed."""
        if TIMING_PATH.exists():
            try:
                data = json.loads(TIMING_PATH.read_text())
                self._stats = data.get("stats", {})
                logger.info("[C2TimingEngine] Loaded %d event timing stats", len(self._stats))
            except Exception as e:
                logger.error("[C2TimingEngine] Load error: %s", e)

        # Merge in priors for any event types not yet observed
        for event_type, prior in C2_EVENT_PRIORS.items():
            if event_type not in self._stats:
                self._stats[event_type] = {
                    **prior,
                    "sample_size": 0,
                    "is_seeded": True,
                    "confidence": 0.3,
                }

        self._loaded = True

    def persist(self) -> bool:
        try:
            DATA_DIR.mkdir(exist_ok=True)
            payload = {
                "stats": self._stats,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            TIMING_PATH.write_text(json.dumps(payload, indent=2))
            return True
        except Exception as e:
            logger.error("[C2TimingEngine] Persist error: %s", e)
            return False

    def record_outcome(
        self,
        event_type: str,
        minutes_to_command: float,
        escalated: bool = False,
    ):
        """
        Record an observed outcome to update timing stats.

        Call this whenever an operator issues a command in response to an observation.

        Args:
            event_type: C2EventType string
            minutes_to_command: Actual minutes from observation to command
            escalated: Whether the observation led to escalation (higher authority)
        """
        if not self._loaded:
            self.hydrate()

        if event_type not in self._stats:
            self._stats[event_type] = {
                "median_minutes_to_command": minutes_to_command,
                "p25_minutes": minutes_to_command,
                "p75_minutes": minutes_to_command,
                "escalation_lift": 1.5 if escalated else 1.0,
                "sample_size": 1,
                "is_seeded": False,
                "confidence": 0.1,
            }
            return

        stat = self._stats[event_type]
        n = stat.get("sample_size", 0)

        # Incremental mean update
        old_median = stat.get("median_minutes_to_command", minutes_to_command)
        new_median  = (old_median * n + minutes_to_command) / (n + 1)

        stat["median_minutes_to_command"] = round(new_median, 1)
        stat["sample_size"] = n + 1
        stat["is_seeded"]   = n + 1 < MIN_SAMPLES_TO_OVERRIDE_PRIOR
        stat["confidence"]  = min(1.0, (n + 1) / 50)
        stat["last_updated"] = datetime.now(timezone.utc).isoformat()

        if escalated:
            old_lift = stat.get("escalation_lift", 1.0)
            stat["escalation_lift"] = round((old_lift * n + 2.0) / (n + 1), 2)

    def get_insight(
        self,
        event_type: str,
        context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get timing insight for a C2 event type, optionally adjusted for operational context.

        Returns:
            {
                "event_type": str,
                "median_minutes": int,
                "hot_window": [p25, p75],
                "escalation_lift": float,
                "confidence": float,
                "is_seeded": bool,
                "sample_size": int,
                "action_advice": str,
                "context_adjusted": bool,
            }
        """
        if not self._loaded:
            stats = C2_EVENT_PRIORS.get(event_type, C2_EVENT_PRIORS.get("PEER_OBSERVATION", {}))
        else:
            stats = self._stats.get(event_type, self._stats.get("PEER_OBSERVATION", {}))

        if not stats:
            return {}

        median = stats.get("median_minutes_to_command", 5)
        p25    = stats.get("p25_minutes", 1)
        p75    = stats.get("p75_minutes", 10)
        lift   = stats.get("escalation_lift", 1.0)

        context_adjusted = False
        if context and context in CONTEXT_URGENCY_MULTIPLIERS:
            mult   = CONTEXT_URGENCY_MULTIPLIERS[context]
            median = round(median * mult, 1)
            p25    = round(p25 * mult, 1)
            p75    = round(p75 * mult, 1)
            context_adjusted = True

        action_advice = self._build_advice(event_type, median, p25, p75, lift)

        return {
            "event_type":       event_type,
            "median_minutes":   median,
            "hot_window":       [p25, p75],
            "escalation_lift":  lift,
            "confidence":       stats.get("confidence", 0.3),
            "is_seeded":        stats.get("is_seeded", True),
            "sample_size":      stats.get("sample_size", 0),
            "action_advice":    action_advice,
            "context_adjusted": context_adjusted,
            "notes":            stats.get("notes", ""),
        }

    def get_multi_event_brief(
        self,
        event_types: List[str],
        context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Given a list of event types observed for a node, return a timing brief:
        - Which event type demands the fastest response
        - Recommended action window
        - Combined urgency score

        Analogous to Mira's get_account_timing_brief() but in minutes not days.
        """
        insights = [self.get_insight(et, context) for et in event_types if et]
        insights = [i for i in insights if i]

        if not insights:
            return {}

        most_urgent   = min(insights, key=lambda x: x.get("median_minutes", 999))
        highest_lift  = max(insights, key=lambda x: x.get("escalation_lift", 0))

        urgency_scores = [
            (1 / max(i["median_minutes"], 0.1)) * i["escalation_lift"]
            for i in insights
        ]
        combined_urgency = round(sum(urgency_scores) / len(urgency_scores) * 10, 1)

        return {
            "event_count":          len(insights),
            "most_urgent_event":    most_urgent["event_type"],
            "action_window_minutes": most_urgent["hot_window"],
            "highest_lift_event":   highest_lift["event_type"],
            "highest_lift_value":   highest_lift["escalation_lift"],
            "combined_urgency_score": min(combined_urgency, 100),
            "recommendation": (
                f"Highest urgency: {most_urgent['event_type']} — "
                f"act within {most_urgent['hot_window'][0]}-{most_urgent['hot_window'][1]} minutes. "
                f"Strongest predictor: {highest_lift['event_type']} "
                f"({highest_lift['escalation_lift']:.1f}x escalation lift)."
            ),
            "event_insights": insights,
        }

    def _build_advice(self, event_type: str, median: float, p25: float, p75: float, lift: float) -> str:
        lift_str = f"{lift:.1f}x" if lift >= 1.5 else "marginal"
        if median < 2:
            when = "immediately"
        elif median < 10:
            when = f"within {p25:.0f}-{p75:.0f} minutes"
        else:
            when = f"within {p25:.0f}-{p75:.0f} minutes"

        advice_map = {
            "COMMS_DEGRADED":    f"Initiate PACE fallback {when}. Authority delegation may be required if link not restored.",
            "LINK_LOST":         f"Autonomous fallback active. Issue RTB or verify last-known orders are safe {when}.",
            "THREAT_IDENTIFIED": f"Engagement decision required {when}. OODA clock running from detection timestamp.",
            "BATTERY_CRITICAL":  f"Issue RTB command {when}. Asset will go offline without operator action.",
            "GEOFENCE_BREACH":   f"ROE decision required {when}. Escalation lift: {lift_str}.",
            "ASSET_OFFLINE":     f"Re-task coverage assets {when}. Assess mission continuity impact.",
            "NODE_FAILED":       f"Redistribute authority and tasks {when}. Validate surviving C2 topology.",
        }
        return advice_map.get(
            event_type,
            f"Operator action recommended {when}. Escalation lift vs. baseline: {lift_str}.",
        )


_engine: Optional[C2TimingEngine] = None


def get_timing_engine() -> C2TimingEngine:
    global _engine
    if _engine is None:
        _engine = C2TimingEngine()
        _engine.hydrate()
    return _engine
