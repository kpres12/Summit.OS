"""
Entity Classification for Summit.OS

Lattice-style Automatic Target Recognition (ATR) pipeline:
- BayesianClassifier: updates entity class probabilities as new
  evidence (detections, radar signatures, behavioral features) arrives.
- RuleBasedClassifier: deterministic rules for quick classification
  (e.g., ADS-B → civilian aircraft, transponder codes).

Both classifiers produce ClassificationResult which can be fused.
"""
from __future__ import annotations

import math
import time
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from abc import ABC, abstractmethod
from collections import defaultdict

logger = logging.getLogger("ai.classification")


# ── Domain Taxonomy ─────────────────────────────────────────

class EntityTaxonomy:
    """
    Hierarchical entity class taxonomy matching Lattice entity model.

    Level 0: Domain     (AIR, GROUND, SURFACE, SUBSURFACE, SPACE, CYBER)
    Level 1: Category   (FIXED_WING, ROTARY_WING, UAV, ...)
    Level 2: Type       (FIGHTER, BOMBER, TRANSPORT, ...)
    Level 3: Specific   (F-16, C-130, MQ-9, ...)
    """

    TAXONOMY: Dict[str, Dict[str, Dict[str, List[str]]]] = {
        "AIR": {
            "FIXED_WING": {
                "FIGHTER": ["F-16", "F-35", "SU-27", "MIG-29"],
                "BOMBER": ["B-52", "B-2", "TU-95"],
                "TRANSPORT": ["C-130", "C-17", "IL-76"],
                "CIVILIAN": ["B737", "A320", "CESSNA-172"],
            },
            "ROTARY_WING": {
                "ATTACK": ["AH-64", "KA-52", "MI-24"],
                "UTILITY": ["UH-60", "CH-47", "MI-8"],
                "CIVILIAN": ["R44", "EC135", "BELL-206"],
            },
            "UAV": {
                "MALE": ["MQ-9", "BAYRAKTAR-TB2", "WING-LOONG"],
                "SMALL": ["RQ-11", "SWITCHBLADE", "DJI-MATRICE"],
                "LOITERING": ["SHAHED-136", "HERO-30", "LANCET"],
            },
        },
        "GROUND": {
            "ARMORED": {
                "MBT": ["M1A2", "T-72", "LEOPARD-2"],
                "IFV": ["M2A3", "BMP-3", "CV90"],
                "APC": ["M113", "BTR-80", "STRYKER"],
            },
            "WHEELED": {
                "TRUCK": ["HEMTT", "KAMAZ", "URAL"],
                "CIVILIAN": ["SEDAN", "SUV", "PICKUP"],
            },
            "PERSON": {
                "MILITARY": ["SOLDIER", "SENTRY"],
                "CIVILIAN": ["PEDESTRIAN"],
            },
        },
        "SURFACE": {
            "COMBATANT": {
                "CARRIER": ["NIMITZ", "FORD"],
                "DESTROYER": ["ARLEIGH-BURKE", "TYPE-055"],
                "FRIGATE": ["TYPE-26", "FREMM"],
            },
            "CIVILIAN": {
                "CARGO": ["CONTAINER-SHIP", "TANKER"],
                "FISHING": ["TRAWLER", "LONGLINER"],
            },
        },
    }

    @classmethod
    def get_all_classes(cls) -> List[str]:
        """Flatten taxonomy to all leaf class names."""
        classes = []
        for domain in cls.TAXONOMY.values():
            for category in domain.values():
                for type_name in category.values():
                    classes.extend(type_name)
        return classes

    @classmethod
    def get_domain_classes(cls, domain: str) -> List[str]:
        """Get all classes under a domain."""
        classes = []
        for category in cls.TAXONOMY.get(domain, {}).values():
            for type_name in category.values():
                classes.extend(type_name)
        return classes


# ── Data Classes ────────────────────────────────────────────

@dataclass
class ClassificationResult:
    """Result of classifying an entity."""
    entity_id: str
    top_class: str
    top_confidence: float
    class_probabilities: Dict[str, float]
    domain: str = ""
    category: str = ""
    type_name: str = ""
    # Metadata
    classifier_name: str = ""
    evidence_count: int = 0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        return {
            "entity_id": self.entity_id,
            "top_class": self.top_class,
            "top_confidence": self.top_confidence,
            "class_probabilities": dict(sorted(
                self.class_probabilities.items(),
                key=lambda x: x[1], reverse=True
            )[:10]),  # Top 10
            "domain": self.domain,
            "category": self.category,
            "classifier": self.classifier_name,
            "evidence_count": self.evidence_count,
        }


@dataclass
class Evidence:
    """A piece of evidence for classification."""
    source: str  # "visual", "radar", "adsb", "behavioral", "acoustic"
    feature_name: str
    value: Any
    confidence: float = 1.0
    timestamp: float = field(default_factory=time.time)


# ── Abstract Classifier ────────────────────────────────────

class EntityClassifier(ABC):
    """Base class for entity classifiers."""

    @abstractmethod
    def classify(self, entity_id: str, evidence: List[Evidence]) -> ClassificationResult:
        ...

    @abstractmethod
    def update(self, entity_id: str, evidence: Evidence) -> ClassificationResult:
        """Incrementally update classification with new evidence."""
        ...

    @abstractmethod
    def reset(self, entity_id: str) -> None:
        """Reset classification state for an entity."""
        ...


# ── Bayesian Classifier ────────────────────────────────────

class BayesianClassifier(EntityClassifier):
    """
    Bayesian entity classifier that updates class probabilities
    as evidence arrives, matching Lattice's incremental ATR approach.

    P(class | evidence) ∝ P(evidence | class) * P(class)

    Likelihood models are defined per (source, feature_name) pair.
    """

    def __init__(self, classes: Optional[List[str]] = None):
        self.classes = classes or EntityTaxonomy.get_all_classes()
        # Per-entity state: entity_id → {class_name: log_probability}
        self._posteriors: Dict[str, Dict[str, float]] = {}
        self._evidence_counts: Dict[str, int] = defaultdict(int)
        # Likelihood tables: (source, feature) → {value → {class → probability}}
        self._likelihood_tables: Dict[Tuple[str, str], Dict[Any, Dict[str, float]]] = {}
        self._build_default_likelihoods()

    def _build_default_likelihoods(self):
        """Build default likelihood tables for common evidence types."""
        # Visual detection class → entity class mapping
        visual_map: Dict[str, Dict[str, float]] = {
            "person": {"SOLDIER": 0.4, "SENTRY": 0.2, "PEDESTRIAN": 0.4},
            "vehicle": {"SEDAN": 0.3, "SUV": 0.2, "PICKUP": 0.2, "STRYKER": 0.1, "HEMTT": 0.1, "KAMAZ": 0.1},
            "aircraft": {"CESSNA-172": 0.3, "B737": 0.2, "A320": 0.2, "C-130": 0.1, "F-16": 0.1, "F-35": 0.1},
            "vessel": {"CONTAINER-SHIP": 0.3, "TANKER": 0.2, "TRAWLER": 0.2, "LONGLINER": 0.1,
                       "ARLEIGH-BURKE": 0.1, "TYPE-055": 0.1},
            "drone": {"DJI-MATRICE": 0.4, "RQ-11": 0.2, "SWITCHBLADE": 0.1, "MQ-9": 0.1,
                      "BAYRAKTAR-TB2": 0.1, "SHAHED-136": 0.1},
        }
        self._likelihood_tables[("visual", "class_name")] = visual_map

        # Speed ranges → entity classes
        speed_map: Dict[str, Dict[str, float]] = {
            "stationary": {"SENTRY": 0.3, "PEDESTRIAN": 0.3, "TRAWLER": 0.2, "CONTAINER-SHIP": 0.1, "M1A2": 0.1},
            "slow": {"PEDESTRIAN": 0.3, "TRAWLER": 0.2, "SEDAN": 0.2, "SOLDIER": 0.2, "DJI-MATRICE": 0.1},
            "medium": {"SEDAN": 0.2, "SUV": 0.2, "HEMTT": 0.15, "STRYKER": 0.15, "M1A2": 0.15, "BMP-3": 0.15},
            "fast": {"F-16": 0.2, "F-35": 0.2, "SU-27": 0.15, "MIG-29": 0.15, "B737": 0.15, "A320": 0.15},
            "very_fast": {"F-16": 0.25, "F-35": 0.25, "SU-27": 0.2, "MIG-29": 0.2, "B-2": 0.1},
        }
        self._likelihood_tables[("behavioral", "speed_category")] = speed_map

        # Radar cross-section ranges
        rcs_map: Dict[str, Dict[str, float]] = {
            "tiny": {"DJI-MATRICE": 0.3, "RQ-11": 0.3, "SWITCHBLADE": 0.2, "HERO-30": 0.2},
            "small": {"CESSNA-172": 0.2, "MQ-9": 0.2, "F-35": 0.2, "BAYRAKTAR-TB2": 0.2, "LANCET": 0.2},
            "medium": {"F-16": 0.2, "C-130": 0.15, "SU-27": 0.15, "MIG-29": 0.15, "AH-64": 0.15, "UH-60": 0.2},
            "large": {"B737": 0.2, "A320": 0.2, "C-17": 0.2, "B-52": 0.2, "IL-76": 0.2},
            "very_large": {"NIMITZ": 0.3, "FORD": 0.3, "CONTAINER-SHIP": 0.2, "TANKER": 0.2},
        }
        self._likelihood_tables[("radar", "rcs_category")] = rcs_map

    def _init_entity(self, entity_id: str) -> Dict[str, float]:
        """Initialize uniform prior for an entity."""
        n = len(self.classes)
        uniform = math.log(1.0 / n) if n > 0 else 0.0
        priors = {c: uniform for c in self.classes}
        self._posteriors[entity_id] = priors
        return priors

    def _get_likelihood(self, evidence: Evidence, class_name: str) -> float:
        """Get P(evidence | class) from likelihood tables."""
        key = (evidence.source, evidence.feature_name)
        table = self._likelihood_tables.get(key)

        if table is None:
            return 1.0  # Uninformative

        value_map = table.get(evidence.value)
        if value_map is None:
            return 1.0 / len(self.classes)  # Uniform if value unknown

        return value_map.get(class_name, 1e-6)  # Small probability for unseen

    def classify(self, entity_id: str, evidence: List[Evidence]) -> ClassificationResult:
        """Full classification from a batch of evidence."""
        self.reset(entity_id)
        for ev in evidence:
            self.update(entity_id, ev)
        return self._make_result(entity_id)

    def update(self, entity_id: str, evidence: Evidence) -> ClassificationResult:
        """Bayesian update: posterior ∝ likelihood × prior (in log space)."""
        if entity_id not in self._posteriors:
            self._init_entity(entity_id)

        posteriors = self._posteriors[entity_id]
        self._evidence_counts[entity_id] += 1

        for cls in self.classes:
            likelihood = self._get_likelihood(evidence, cls)
            if likelihood > 0:
                posteriors[cls] += math.log(likelihood) * evidence.confidence
            else:
                posteriors[cls] += math.log(1e-10)

        # Normalize in log space
        max_log = max(posteriors.values())
        log_sum = max_log + math.log(sum(
            math.exp(v - max_log) for v in posteriors.values()
        ))
        for cls in posteriors:
            posteriors[cls] -= log_sum

        return self._make_result(entity_id)

    def _make_result(self, entity_id: str) -> ClassificationResult:
        posteriors = self._posteriors.get(entity_id, {})
        if not posteriors:
            return ClassificationResult(
                entity_id=entity_id, top_class="UNKNOWN",
                top_confidence=0.0, class_probabilities={},
                classifier_name="bayesian",
            )

        # Convert from log space
        max_log = max(posteriors.values())
        probs = {cls: math.exp(v - max_log) for cls, v in posteriors.items()}
        total = sum(probs.values())
        probs = {cls: p / total for cls, p in probs.items()}

        top_class = max(probs, key=probs.get)
        return ClassificationResult(
            entity_id=entity_id,
            top_class=top_class,
            top_confidence=probs[top_class],
            class_probabilities=probs,
            classifier_name="bayesian",
            evidence_count=self._evidence_counts.get(entity_id, 0),
        )

    def reset(self, entity_id: str) -> None:
        self._posteriors.pop(entity_id, None)
        self._evidence_counts.pop(entity_id, None)

    def add_likelihood_table(self, source: str, feature: str,
                             table: Dict[Any, Dict[str, float]]) -> None:
        """Register a custom likelihood table."""
        self._likelihood_tables[(source, feature)] = table


# ── Rule-Based Classifier ──────────────────────────────────

class RuleBasedClassifier(EntityClassifier):
    """
    Deterministic rule-based classifier for fast, high-confidence
    classification from strong indicators (transponder codes, ADS-B, IFF).
    """

    def __init__(self):
        self._rules: List[Tuple[str, Any, str, float]] = []
        self._results: Dict[str, ClassificationResult] = {}
        self._build_default_rules()

    def _build_default_rules(self):
        """Default classification rules."""
        # (feature_name, value_or_predicate, class_name, confidence)
        self._rules = [
            # ADS-B → civilian aircraft
            ("has_adsb", True, "CIVILIAN_AIRCRAFT", 0.95),
            # IFF Mode 4/5 → friendly military
            ("iff_mode", "mode_4", "FRIENDLY_MILITARY", 0.9),
            ("iff_mode", "mode_5", "FRIENDLY_MILITARY", 0.95),
            # Transponder squawk codes
            ("squawk", "7700", "EMERGENCY_AIRCRAFT", 0.99),
            ("squawk", "7600", "COMMS_FAILURE_AIRCRAFT", 0.99),
            ("squawk", "7500", "HIJACKED_AIRCRAFT", 0.99),
            # AIS → surface vessel
            ("has_ais", True, "CIVILIAN_VESSEL", 0.9),
            # Altitude rules
            ("altitude_category", "space", "SPACE_OBJECT", 0.85),
            ("altitude_category", "very_high", "HIGH_ALT_AIRCRAFT", 0.7),
        ]

    def classify(self, entity_id: str, evidence: List[Evidence]) -> ClassificationResult:
        """Apply rules to evidence batch."""
        best_class = "UNKNOWN"
        best_confidence = 0.0
        probs: Dict[str, float] = defaultdict(float)

        for ev in evidence:
            for feat_name, feat_value, cls, conf in self._rules:
                if ev.feature_name == feat_name:
                    match = False
                    if callable(feat_value):
                        match = feat_value(ev.value)
                    else:
                        match = ev.value == feat_value

                    if match:
                        combined = conf * ev.confidence
                        probs[cls] = max(probs[cls], combined)
                        if combined > best_confidence:
                            best_confidence = combined
                            best_class = cls

        result = ClassificationResult(
            entity_id=entity_id,
            top_class=best_class,
            top_confidence=best_confidence,
            class_probabilities=dict(probs),
            classifier_name="rule_based",
            evidence_count=len(evidence),
        )
        self._results[entity_id] = result
        return result

    def update(self, entity_id: str, evidence: Evidence) -> ClassificationResult:
        prev = self._results.get(entity_id)
        prev_evidence: List[Evidence] = []
        if prev:
            # Re-classify with combined evidence
            pass
        return self.classify(entity_id, prev_evidence + [evidence])

    def reset(self, entity_id: str) -> None:
        self._results.pop(entity_id, None)

    def add_rule(self, feature_name: str, value: Any,
                 class_name: str, confidence: float) -> None:
        """Add a custom classification rule."""
        self._rules.append((feature_name, value, class_name, confidence))


# ── Fusion Classifier ──────────────────────────────────────

class FusionClassifier(EntityClassifier):
    """
    Fuses results from Bayesian and Rule-Based classifiers.

    Rule-based results take priority when confidence is high (>0.85),
    otherwise Bayesian posterior is used. This mirrors how Lattice
    combines deterministic identifiers (IFF, ADS-B) with probabilistic
    ATR outputs.
    """

    def __init__(self, classes: Optional[List[str]] = None):
        self.bayesian = BayesianClassifier(classes)
        self.rule_based = RuleBasedClassifier()
        self._rule_priority_threshold = 0.85

    def classify(self, entity_id: str, evidence: List[Evidence]) -> ClassificationResult:
        rule_result = self.rule_based.classify(entity_id, evidence)
        bayes_result = self.bayesian.classify(entity_id, evidence)

        if rule_result.top_confidence >= self._rule_priority_threshold:
            rule_result.classifier_name = "fusion(rule)"
            return rule_result

        # Merge probabilities: weighted combination
        merged: Dict[str, float] = {}
        all_classes = set(list(rule_result.class_probabilities.keys()) +
                         list(bayes_result.class_probabilities.keys()))
        for cls in all_classes:
            r = rule_result.class_probabilities.get(cls, 0.0)
            b = bayes_result.class_probabilities.get(cls, 0.0)
            merged[cls] = 0.6 * b + 0.4 * r  # Weight Bayes higher

        if not merged:
            return bayes_result

        total = sum(merged.values())
        if total > 0:
            merged = {k: v / total for k, v in merged.items()}

        top_class = max(merged, key=merged.get)
        return ClassificationResult(
            entity_id=entity_id,
            top_class=top_class,
            top_confidence=merged[top_class],
            class_probabilities=merged,
            classifier_name="fusion(bayes+rule)",
            evidence_count=len(evidence),
        )

    def update(self, entity_id: str, evidence: Evidence) -> ClassificationResult:
        self.rule_based.update(entity_id, evidence)
        return self.bayesian.update(entity_id, evidence)

    def reset(self, entity_id: str) -> None:
        self.bayesian.reset(entity_id)
        self.rule_based.reset(entity_id)
