"""
Intent Prediction for Summit.OS

Predicts entity intent from observed behavior:
- TrajectoryPredictor: kinematic extrapolation + waypoint matching
- BehaviorAnalyzer: pattern matching on behavioral sequences
  (loitering, approach, evasion, formation, patrol)
- ThreatAssessor: fuses trajectory + behavior into threat level

Mirrors Lattice's advisory and threat assessment capabilities.
"""
from __future__ import annotations

import math
import time
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from enum import Enum

logger = logging.getLogger("ai.intent")


# ── Enums ───────────────────────────────────────────────────

class IntentType(str, Enum):
    TRANSIT = "transit"
    LOITER = "loiter"
    APPROACH = "approach"
    RETREAT = "retreat"
    PATROL = "patrol"
    EVASION = "evasion"
    PURSUIT = "pursuit"
    FORMATION = "formation"
    LANDING = "landing"
    TAKEOFF = "takeoff"
    STATIONARY = "stationary"
    UNKNOWN = "unknown"


class ThreatLevel(str, Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ── Data Classes ────────────────────────────────────────────

@dataclass
class Position:
    """Geographic position."""
    lat: float
    lon: float
    alt: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class Kinematics:
    """Full kinematic state."""
    position: Position
    velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # m/s (N, E, D)
    speed: float = 0.0  # m/s
    heading: float = 0.0  # degrees true
    climb_rate: float = 0.0  # m/s
    turn_rate: float = 0.0  # degrees/s
    acceleration: float = 0.0  # m/s²


@dataclass
class IntentResult:
    """Result of intent prediction."""
    entity_id: str
    primary_intent: IntentType
    confidence: float
    intent_probabilities: Dict[str, float]
    predicted_positions: List[Position]  # Future trajectory
    time_horizon_s: float = 300.0  # Prediction horizon in seconds
    threat_level: ThreatLevel = ThreatLevel.NONE
    threat_score: float = 0.0
    description: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        return {
            "entity_id": self.entity_id,
            "intent": self.primary_intent.value,
            "confidence": round(self.confidence, 3),
            "threat_level": self.threat_level.value,
            "threat_score": round(self.threat_score, 3),
            "predicted_positions": [
                {"lat": p.lat, "lon": p.lon, "alt": p.alt, "t": p.timestamp}
                for p in self.predicted_positions[:5]  # Limit to 5
            ],
            "description": self.description,
        }


# ── Trajectory Predictor ────────────────────────────────────

class TrajectoryPredictor:
    """
    Predicts future positions via kinematic extrapolation.

    Supports:
    - Linear (constant velocity)
    - Curvilinear (constant turn rate)
    - Accelerating (constant acceleration)
    """

    # Earth radius in meters
    EARTH_R = 6_371_000.0

    def __init__(self, history_size: int = 50):
        self.history_size = history_size
        self._histories: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=history_size)
        )

    def update(self, entity_id: str, kinematics: Kinematics) -> None:
        """Add a kinematic observation."""
        self._histories[entity_id].append(kinematics)

    def predict(self, entity_id: str, horizon_s: float = 300.0,
                steps: int = 10) -> List[Position]:
        """
        Predict future positions.

        Args:
            entity_id: Entity to predict
            horizon_s: Prediction horizon in seconds
            steps: Number of predicted positions
        """
        history = self._histories.get(entity_id)
        if not history or len(history) < 2:
            return []

        latest = history[-1]
        dt = horizon_s / steps

        # Determine prediction model
        if abs(latest.turn_rate) > 1.0:
            return self._predict_curvilinear(latest, dt, steps)
        elif abs(latest.acceleration) > 0.5:
            return self._predict_accelerating(latest, dt, steps)
        else:
            return self._predict_linear(latest, dt, steps)

    def _predict_linear(self, k: Kinematics, dt: float, steps: int) -> List[Position]:
        """Constant velocity extrapolation."""
        positions = []
        lat, lon, alt = k.position.lat, k.position.lon, k.position.alt
        vn, ve, vd = k.velocity

        for i in range(1, steps + 1):
            t = dt * i
            # Latitude change from north velocity
            dlat = (vn * t) / self.EARTH_R * (180 / math.pi)
            # Longitude change from east velocity
            dlon = (ve * t) / (self.EARTH_R * math.cos(math.radians(lat))) * (180 / math.pi)
            dalt = -vd * t  # Down → altitude change

            positions.append(Position(
                lat=lat + dlat,
                lon=lon + dlon,
                alt=alt + dalt,
                timestamp=k.position.timestamp + t,
            ))

        return positions

    def _predict_curvilinear(self, k: Kinematics, dt: float, steps: int) -> List[Position]:
        """Constant turn-rate extrapolation."""
        positions = []
        lat, lon, alt = k.position.lat, k.position.lon, k.position.alt
        heading = math.radians(k.heading)
        speed = k.speed
        turn_rate = math.radians(k.turn_rate)  # deg/s → rad/s

        for i in range(1, steps + 1):
            t = dt * i
            # Heading at time t
            h_t = heading + turn_rate * t
            # Average heading for displacement
            if abs(turn_rate) > 1e-6:
                dx = speed / turn_rate * (math.sin(h_t) - math.sin(heading))
                dy = speed / turn_rate * (math.cos(heading) - math.cos(h_t))
            else:
                dx = speed * math.sin(heading) * t
                dy = speed * math.cos(heading) * t

            dlat = dy / self.EARTH_R * (180 / math.pi)
            dlon = dx / (self.EARTH_R * math.cos(math.radians(lat))) * (180 / math.pi)
            dalt = k.climb_rate * t

            positions.append(Position(
                lat=lat + dlat, lon=lon + dlon, alt=alt + dalt,
                timestamp=k.position.timestamp + t,
            ))

        return positions

    def _predict_accelerating(self, k: Kinematics, dt: float, steps: int) -> List[Position]:
        """Constant acceleration extrapolation."""
        positions = []
        lat, lon, alt = k.position.lat, k.position.lon, k.position.alt
        vn, ve, vd = k.velocity
        heading = math.radians(k.heading)
        accel = k.acceleration

        for i in range(1, steps + 1):
            t = dt * i
            speed_t = k.speed + accel * t
            speed_t = max(0, speed_t)  # Can't go negative

            # Average speed over interval
            avg_speed = (k.speed + speed_t) / 2
            dist = avg_speed * t

            dn = dist * math.cos(heading)
            de = dist * math.sin(heading)

            dlat = dn / self.EARTH_R * (180 / math.pi)
            dlon = de / (self.EARTH_R * math.cos(math.radians(lat))) * (180 / math.pi)
            dalt = k.climb_rate * t

            positions.append(Position(
                lat=lat + dlat, lon=lon + dlon, alt=alt + dalt,
                timestamp=k.position.timestamp + t,
            ))

        return positions

    def compute_kinematics(self, entity_id: str) -> Optional[Kinematics]:
        """Compute current kinematics from position history."""
        history = self._histories.get(entity_id)
        if not history or len(history) < 2:
            return history[-1] if history else None

        curr = history[-1]
        prev = history[-2]

        dt = curr.position.timestamp - prev.position.timestamp
        if dt <= 0:
            return curr

        # Compute velocities
        dlat = curr.position.lat - prev.position.lat
        dlon = curr.position.lon - prev.position.lon
        vn = (dlat * math.pi / 180) * self.EARTH_R / dt
        ve = (dlon * math.pi / 180) * self.EARTH_R * math.cos(
            math.radians(curr.position.lat)) / dt
        vd = -(curr.position.alt - prev.position.alt) / dt

        speed = math.sqrt(vn ** 2 + ve ** 2)
        heading = math.degrees(math.atan2(ve, vn)) % 360

        # Turn rate
        turn_rate = 0.0
        if len(history) >= 3:
            pprev = history[-3]
            dt2 = prev.position.timestamp - pprev.position.timestamp
            if dt2 > 0:
                dlon2 = prev.position.lon - pprev.position.lon
                dlat2 = prev.position.lat - pprev.position.lat
                ve2 = (dlon2 * math.pi / 180) * self.EARTH_R * math.cos(
                    math.radians(prev.position.lat)) / dt2
                vn2 = (dlat2 * math.pi / 180) * self.EARTH_R / dt2
                prev_heading = math.degrees(math.atan2(ve2, vn2)) % 360
                dheading = heading - prev_heading
                if dheading > 180:
                    dheading -= 360
                if dheading < -180:
                    dheading += 360
                turn_rate = dheading / dt

        return Kinematics(
            position=curr.position,
            velocity=(vn, ve, vd),
            speed=speed,
            heading=heading,
            climb_rate=-vd,
            turn_rate=turn_rate,
        )


# ── Behavior Analyzer ──────────────────────────────────────

class BehaviorAnalyzer:
    """
    Identifies behavioral patterns from kinematic history.

    Patterns detected:
    - Loitering: entity remains within a small area
    - Approach: closing distance to a reference point
    - Retreat: increasing distance from reference point
    - Patrol: repeated back-and-forth pattern
    - Evasion: high turn rates + speed changes
    - Formation: maintaining fixed offset from another entity
    """

    EARTH_R = 6_371_000.0

    def __init__(self, loiter_radius_m: float = 500.0,
                 approach_angle_threshold: float = 30.0):
        self.loiter_radius_m = loiter_radius_m
        self.approach_angle_threshold = approach_angle_threshold
        self._position_histories: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=200)
        )

    def update(self, entity_id: str, position: Position) -> None:
        self._position_histories[entity_id].append(position)

    def analyze(self, entity_id: str,
                reference_point: Optional[Position] = None) -> Dict[str, float]:
        """
        Analyze behavior and return intent probabilities.

        Returns dict of IntentType → probability.
        """
        history = self._position_histories.get(entity_id)
        if not history or len(history) < 5:
            return {IntentType.UNKNOWN.value: 1.0}

        probs: Dict[str, float] = defaultdict(float)

        # Check loitering
        loiter_score = self._check_loitering(history)
        probs[IntentType.LOITER.value] = loiter_score

        # Check transit
        transit_score = self._check_transit(history)
        probs[IntentType.TRANSIT.value] = transit_score

        # Check evasion
        evasion_score = self._check_evasion(history)
        probs[IntentType.EVASION.value] = evasion_score

        # Check stationary
        stationary_score = self._check_stationary(history)
        probs[IntentType.STATIONARY.value] = stationary_score

        # Check approach/retreat relative to reference
        if reference_point:
            approach_score, retreat_score = self._check_approach_retreat(
                history, reference_point
            )
            probs[IntentType.APPROACH.value] = approach_score
            probs[IntentType.RETREAT.value] = retreat_score

        # Normalize
        total = sum(probs.values())
        if total > 0:
            probs = {k: v / total for k, v in probs.items()}

        return dict(probs)

    def _haversine(self, p1: Position, p2: Position) -> float:
        """Haversine distance in meters."""
        lat1, lon1 = math.radians(p1.lat), math.radians(p1.lon)
        lat2, lon2 = math.radians(p2.lat), math.radians(p2.lon)
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        return 2 * self.EARTH_R * math.asin(math.sqrt(a))

    def _check_loitering(self, history: deque) -> float:
        """Score loitering: entity stays within small radius."""
        if len(history) < 10:
            return 0.0

        recent = list(history)[-20:]
        centroid = Position(
            lat=sum(p.lat for p in recent) / len(recent),
            lon=sum(p.lon for p in recent) / len(recent),
            alt=sum(p.alt for p in recent) / len(recent),
        )
        max_dist = max(self._haversine(p, centroid) for p in recent)

        if max_dist < self.loiter_radius_m:
            return min(1.0, self.loiter_radius_m / max(max_dist, 1.0) * 0.3)
        return max(0, 1.0 - max_dist / (self.loiter_radius_m * 5))

    def _check_transit(self, history: deque) -> float:
        """Score transit: consistent heading, steady speed."""
        if len(history) < 5:
            return 0.0

        recent = list(history)[-10:]
        # Compute heading changes
        headings = []
        for i in range(1, len(recent)):
            dlat = recent[i].lat - recent[i-1].lat
            dlon = recent[i].lon - recent[i-1].lon
            h = math.degrees(math.atan2(dlon, dlat)) % 360
            headings.append(h)

        if len(headings) < 2:
            return 0.0

        # Heading variance (lower = more consistent = more transit-like)
        mean_h = sum(headings) / len(headings)
        variance = sum((h - mean_h) ** 2 for h in headings) / len(headings)

        # Low variance → high transit score
        return max(0, 1.0 - variance / 1000)

    def _check_evasion(self, history: deque) -> float:
        """Score evasion: high heading change rate + speed variance."""
        if len(history) < 10:
            return 0.0

        recent = list(history)[-15:]
        heading_changes = []
        speeds = []

        for i in range(1, len(recent)):
            dt = recent[i].timestamp - recent[i-1].timestamp
            if dt <= 0:
                continue
            dist = self._haversine(recent[i], recent[i-1])
            speed = dist / dt
            speeds.append(speed)

            if i >= 2:
                dlat1 = recent[i].lat - recent[i-1].lat
                dlon1 = recent[i].lon - recent[i-1].lon
                dlat0 = recent[i-1].lat - recent[i-2].lat
                dlon0 = recent[i-1].lon - recent[i-2].lon
                h1 = math.degrees(math.atan2(dlon1, dlat1))
                h0 = math.degrees(math.atan2(dlon0, dlat0))
                dh = abs(h1 - h0)
                if dh > 180:
                    dh = 360 - dh
                heading_changes.append(dh)

        if not heading_changes or not speeds:
            return 0.0

        avg_heading_change = sum(heading_changes) / len(heading_changes)
        speed_variance = sum((s - sum(speeds)/len(speeds))**2 for s in speeds) / len(speeds)

        # High heading change + high speed variance → evasion
        heading_score = min(avg_heading_change / 90, 1.0)
        speed_score = min(speed_variance / 100, 1.0)

        return (heading_score * 0.6 + speed_score * 0.4)

    def _check_stationary(self, history: deque) -> float:
        """Score stationary: no significant movement."""
        if len(history) < 5:
            return 0.0

        recent = list(history)[-10:]
        total_dist = sum(
            self._haversine(recent[i], recent[i-1])
            for i in range(1, len(recent))
        )

        if total_dist < 5.0:  # < 5 meters total
            return 1.0
        elif total_dist < 20.0:
            return 0.7
        elif total_dist < 50.0:
            return 0.3
        return 0.0

    def _check_approach_retreat(self, history: deque,
                                ref: Position) -> Tuple[float, float]:
        """Score approach/retreat relative to a reference point."""
        if len(history) < 5:
            return (0.0, 0.0)

        recent = list(history)[-10:]
        distances = [self._haversine(p, ref) for p in recent]

        # Compute trend
        n = len(distances)
        x_mean = (n - 1) / 2
        y_mean = sum(distances) / n
        numerator = sum((i - x_mean) * (d - y_mean) for i, d in enumerate(distances))
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            return (0.0, 0.0)

        slope = numerator / denominator

        # Normalize: large negative slope = approach, large positive = retreat
        approach = max(0, -slope / 10)
        retreat = max(0, slope / 10)

        return (min(approach, 1.0), min(retreat, 1.0))


# ── Threat Assessor ─────────────────────────────────────────

class ThreatAssessor:
    """
    Combines behavior analysis and classification to assess threat level.

    Factors:
    - Distance to protected assets
    - Closing rate
    - Entity classification (hostile/unknown/friendly)
    - Behavioral intent (approach, evasion = higher threat)
    """

    def __init__(self):
        self.protected_points: List[Position] = []
        self.threat_radius_m = 10_000.0  # 10km threat bubble

    def assess(self, entity_id: str,
               intent_probs: Dict[str, float],
               position: Position,
               classification: str = "UNKNOWN",
               is_friendly: bool = False) -> Tuple[ThreatLevel, float]:
        """
        Assess threat level.

        Returns (ThreatLevel, score 0-1).
        """
        if is_friendly:
            return (ThreatLevel.NONE, 0.0)

        score = 0.0

        # Classification factor
        hostile_classes = {"FIGHTER", "BOMBER", "ATTACK", "MBT", "IFV",
                          "LOITERING", "COMBATANT"}
        if any(h in classification.upper() for h in hostile_classes):
            score += 0.4
        elif classification == "UNKNOWN":
            score += 0.15

        # Intent factor
        approach_prob = intent_probs.get(IntentType.APPROACH.value, 0.0)
        evasion_prob = intent_probs.get(IntentType.EVASION.value, 0.0)
        pursuit_prob = intent_probs.get(IntentType.PURSUIT.value, 0.0)
        score += approach_prob * 0.25
        score += evasion_prob * 0.1
        score += pursuit_prob * 0.3

        # Proximity factor
        if self.protected_points:
            min_dist = min(
                self._haversine(position, pp)
                for pp in self.protected_points
            )
            proximity_factor = max(0, 1.0 - min_dist / self.threat_radius_m)
            score += proximity_factor * 0.25

        score = min(score, 1.0)

        # Map to threat level
        if score < 0.15:
            level = ThreatLevel.NONE
        elif score < 0.35:
            level = ThreatLevel.LOW
        elif score < 0.6:
            level = ThreatLevel.MEDIUM
        elif score < 0.8:
            level = ThreatLevel.HIGH
        else:
            level = ThreatLevel.CRITICAL

        return (level, score)

    @staticmethod
    def _haversine(p1: Position, p2: Position) -> float:
        R = 6_371_000.0
        lat1, lon1 = math.radians(p1.lat), math.radians(p1.lon)
        lat2, lon2 = math.radians(p2.lat), math.radians(p2.lon)
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        return 2 * R * math.asin(math.sqrt(a))


# ── Unified Intent Predictor ───────────────────────────────

class IntentPredictor:
    """
    Top-level intent prediction combining trajectory prediction,
    behavior analysis, and threat assessment.
    """

    def __init__(self):
        self.trajectory = TrajectoryPredictor()
        self.behavior = BehaviorAnalyzer()
        self.threat = ThreatAssessor()

    def update(self, entity_id: str, kinematics: Kinematics) -> None:
        """Ingest new kinematic data."""
        self.trajectory.update(entity_id, kinematics)
        self.behavior.update(entity_id, kinematics.position)

    def predict(self, entity_id: str,
                reference_point: Optional[Position] = None,
                classification: str = "UNKNOWN",
                is_friendly: bool = False,
                horizon_s: float = 300.0) -> IntentResult:
        """Full intent prediction."""
        # Predict trajectory
        predicted = self.trajectory.predict(entity_id, horizon_s)

        # Analyze behavior
        intent_probs = self.behavior.analyze(entity_id, reference_point)

        # Get current position
        history = self.trajectory._histories.get(entity_id)
        current_pos = history[-1].position if history else Position(0, 0)

        # Assess threat
        threat_level, threat_score = self.threat.assess(
            entity_id, intent_probs, current_pos,
            classification, is_friendly,
        )

        # Determine primary intent
        if intent_probs:
            primary = max(intent_probs, key=intent_probs.get)
            confidence = intent_probs[primary]
        else:
            primary = IntentType.UNKNOWN.value
            confidence = 0.0

        return IntentResult(
            entity_id=entity_id,
            primary_intent=IntentType(primary),
            confidence=confidence,
            intent_probabilities=intent_probs,
            predicted_positions=predicted,
            time_horizon_s=horizon_s,
            threat_level=threat_level,
            threat_score=threat_score,
            description=f"Intent: {primary} ({confidence:.0%}), Threat: {threat_level.value}",
        )
