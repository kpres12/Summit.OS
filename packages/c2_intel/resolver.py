"""
C2 Entity Resolver

Ported from Mira Signals' entity_resolution_v2.py.
Domain remapped: company name matching → C2 entity/track deconfliction.

Solves the fundamental multi-source fusion problem:
- Is "opensky-abc123" the same physical aircraft as "radar-track-042"?
- Is "HELI-SIM-01" the same entity as callsign "NOVEMBER-7"?
- Does "MAV-04" match "mavlink-heli-sim-04"?

Algorithms unchanged from Mira. Dictionaries remapped to C2 domain.
"""

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
import difflib


# =============================================================================
# CALLSIGN / DESIGNATOR DATABASE
# =============================================================================

# Known callsign → canonical entity ID mappings
# Format: short_designator → [canonical_id, alias1, alias2, ...]
ENTITY_CALLSIGNS: Dict[str, List[str]] = {
    # Simulated / test vehicles
    "sim-01": ["heli-sim-01", "mavlink-heli-sim-01"],
    "sim-02": ["heli-sim-02", "mavlink-heli-sim-02"],

    # Standard ICAO mode-S prefixes (examples)
    "N": ["faa-registered"],       # US civil
    "G": ["uk-civil"],             # UK civil

    # Common military-style callsign patterns
    "alpha": ["alpha-flight", "alpha-element"],
    "bravo": ["bravo-flight", "bravo-element"],
    "charlie": ["charlie-flight", "charlie-element"],
    "reaper": ["mq-9-reaper"],
    "predator": ["mq-1-predator"],
    "shadow": ["rq-7-shadow"],
    "raven": ["rq-11-raven"],
    "grey eagle": ["mq-1c-grey-eagle"],
}

# Reverse mapping
CALLSIGN_TO_CANONICAL: Dict[str, str] = {}
for cs, aliases in ENTITY_CALLSIGNS.items():
    canonical = aliases[0]
    CALLSIGN_TO_CANONICAL[cs] = canonical
    for alias in aliases:
        CALLSIGN_TO_CANONICAL[alias.lower().replace(" ", "-")] = canonical


# Known entity aliases across sensor sources
ENTITY_ALIASES: Dict[str, Set[str]] = {
    # Same physical entity reported by different sensor systems
    "heli-sim-01": {"mavlink-heli-sim-01", "sim01", "sim-01"},
    "heli-sim-02": {"mavlink-heli-sim-02", "sim02", "sim-02"},
}


# =============================================================================
# ID NORMALIZATION
# =============================================================================

# Prefixes added by sensor adapters — strip for canonical comparison
ENTITY_PREFIXES = [
    r'^opensky-',
    r'^mavlink-',
    r'^radar-track-',
    r'^iff-',
    r'^ads-?b-',
    r'^eo-ir-',
    r'^mesh-',
]

# Suffixes that don't affect identity
ENTITY_SUFFIXES = [
    r'-track$', r'-contact$', r'-fused$',
]


def normalize_entity_id(entity_id: str) -> str:
    """
    Normalize an entity ID for comparison across sensor sources.

    "opensky-abc123"    → "abc123"
    "mavlink-HELI-SIM-01" → "heli-sim-01"
    "Radar-Track-042"   → "track-042"
    """
    if not entity_id:
        return ""

    normalized = entity_id.lower().strip()

    # Strip sensor-adapter prefixes
    for prefix in ENTITY_PREFIXES:
        normalized = re.sub(prefix, "", normalized)

    # Strip trailing suffixes
    for suffix in ENTITY_SUFFIXES:
        normalized = re.sub(suffix, "", normalized)

    # Normalize separators
    normalized = re.sub(r"[-_\s]+", "-", normalized)
    normalized = normalized.strip("-")

    return normalized


def tokenize_entity_id(entity_id: str) -> Set[str]:
    """Split an entity ID into meaningful tokens for Jaccard comparison."""
    normalized = normalize_entity_id(entity_id)
    tokens = re.split(r"[-_\s]+", normalized)
    # Filter noise tokens
    noise = {"track", "contact", "fused", "unknown", "n", "a"}
    return {t for t in tokens if len(t) > 1 and t not in noise}


# =============================================================================
# SIMILARITY SCORING (algorithms unchanged from Mira)
# =============================================================================

def levenshtein_similarity(s1: str, s2: str) -> float:
    if not s1 or not s2:
        return 0.0
    return difflib.SequenceMatcher(None, s1, s2).ratio()


def token_similarity(id1: str, id2: str) -> float:
    tokens1 = tokenize_entity_id(id1)
    tokens2 = tokenize_entity_id(id2)
    if not tokens1 or not tokens2:
        return 0.0
    intersection = tokens1 & tokens2
    union = tokens1 | tokens2
    return len(intersection) / len(union) if union else 0.0


def callsign_match(id1: str, id2: str) -> float:
    n1 = id1.lower().strip().replace(" ", "-")
    n2 = id2.lower().strip().replace(" ", "-")
    c1 = CALLSIGN_TO_CANONICAL.get(n1)
    c2 = CALLSIGN_TO_CANONICAL.get(n2)
    if c1 and c2 and c1 == c2:
        return 1.0
    return 0.0


def alias_match(id1: str, id2: str) -> float:
    n1 = normalize_entity_id(id1)
    n2 = normalize_entity_id(id2)
    for canonical, aliases in ENTITY_ALIASES.items():
        all_ids = {canonical} | {normalize_entity_id(a) for a in aliases}
        if n1 in all_ids and n2 in all_ids:
            return 1.0
    return 0.0


# =============================================================================
# MAIN RESOLVER
# =============================================================================

@dataclass
class EntityMatch:
    """Result of entity matching."""
    is_match: bool
    similarity: float
    confidence: float
    match_type: str   # "exact", "alias", "callsign", "fuzzy", "token", "partial"
    canonical_id: str
    explanation: str


class C2EntityResolver:
    """
    Multi-source entity deconfliction for C2 environments.

    Determines whether tracks/contacts from different sensor sources
    refer to the same physical entity.

    Usage:
        resolver = C2EntityResolver()
        result = resolver.match("opensky-abc123", "radar-track-042")
        if result.is_match:
            print(result.canonical_id)  # merged entity ID

        # Register known sensor cross-references at runtime
        resolver.add_alias("heli-sim-01", "sensor-A-track-7")
    """

    def __init__(self, fuzzy_threshold: float = 0.85, token_threshold: float = 0.7):
        self.fuzzy_threshold = fuzzy_threshold
        self.token_threshold = token_threshold
        self.custom_aliases: Dict[str, Set[str]] = defaultdict(set)
        self._cache: Dict[str, str] = {}

    def add_alias(self, canonical: str, alias: str):
        cn = normalize_entity_id(canonical)
        an = normalize_entity_id(alias)
        self.custom_aliases[cn].add(an)
        self._cache.clear()

    def resolve(self, entity_id: str) -> str:
        """Resolve an entity ID to its canonical form."""
        if not entity_id:
            return ""

        cache_key = entity_id
        if cache_key in self._cache:
            return self._cache[cache_key]

        id_lower = entity_id.lower().strip().replace(" ", "-")

        # Callsign database
        if id_lower in CALLSIGN_TO_CANONICAL:
            canonical = CALLSIGN_TO_CANONICAL[id_lower]
            self._cache[cache_key] = canonical
            return canonical

        normalized = normalize_entity_id(entity_id)

        # Custom aliases
        for canonical, aliases in self.custom_aliases.items():
            if normalized in aliases or normalized == canonical:
                self._cache[cache_key] = canonical
                return canonical

        # Built-in aliases
        for canonical, aliases in ENTITY_ALIASES.items():
            if normalized == canonical or normalized in {normalize_entity_id(a) for a in aliases}:
                self._cache[cache_key] = canonical
                return canonical

        self._cache[cache_key] = normalized
        return normalized

    def match(self, id1: str, id2: str) -> EntityMatch:
        """Determine if two entity IDs refer to the same physical entity."""
        if not id1 or not id2:
            return EntityMatch(False, 0.0, 0.0, "none", "", "One or both IDs are empty")

        n1 = normalize_entity_id(id1)
        n2 = normalize_entity_id(id2)
        c1 = self.resolve(id1)
        c2 = self.resolve(id2)

        # 1. Exact after normalization
        if n1 == n2:
            return EntityMatch(True, 1.0, 1.0, "exact", c1, "Exact match after normalization")

        # 2. Canonical match
        if c1 == c2:
            return EntityMatch(True, 1.0, 0.95, "canonical", c1, "Match via canonical resolution")

        # 3. Callsign database
        if callsign_match(id1, id2) == 1.0:
            return EntityMatch(True, 1.0, 0.95, "callsign", c1, "Match via callsign database")

        # 4. Alias database
        if alias_match(id1, id2) == 1.0:
            return EntityMatch(True, 1.0, 0.95, "alias", c1, "Match via known alias")

        # 5. Fuzzy string match
        fuzzy_sim = levenshtein_similarity(n1, n2)
        if fuzzy_sim >= self.fuzzy_threshold:
            return EntityMatch(True, fuzzy_sim, fuzzy_sim * 0.9, "fuzzy", c1,
                               f"Fuzzy match ({fuzzy_sim:.2f})")

        # 6. Token match
        token_sim = token_similarity(id1, id2)
        if token_sim >= self.token_threshold:
            return EntityMatch(True, token_sim, token_sim * 0.85, "token", c1,
                               f"Token match ({token_sim:.2f} overlap)")

        # 7. Partial containment
        if n1 in n2 or n2 in n1:
            longer = n1 if len(n1) > len(n2) else n2
            shorter = n2 if len(n1) > len(n2) else n1
            overlap = len(shorter) / len(longer)
            if overlap >= 0.5:
                return EntityMatch(True, overlap, overlap * 0.75, "partial", shorter,
                                   f"Partial match ({shorter} in {longer})")

        max_sim = max(fuzzy_sim, token_sim)
        return EntityMatch(False, max_sim, 0.0, "none", c1,
                           f"No match (best similarity: {max_sim:.2f})")

    def are_same_entity(self, id1: str, id2: str) -> bool:
        return self.match(id1, id2).is_match

    def find_best_match(
        self, entity_id: str, candidates: List[str], min_similarity: float = 0.7
    ) -> Optional[Tuple[str, EntityMatch]]:
        best_candidate = None
        best_result = None
        best_sim = 0.0
        for candidate in candidates:
            result = self.match(entity_id, candidate)
            if result.similarity > best_sim and result.similarity >= min_similarity:
                best_sim = result.similarity
                best_candidate = candidate
                best_result = result
        return (best_candidate, best_result) if best_candidate else None


# =============================================================================
# DOMAIN DISAMBIGUATION (aerial / ground / maritime)
# =============================================================================

DOMAIN_MODIFIERS = {"aerial", "air", "ground", "surface", "subsurface", "maritime", "space"}


def extract_domain_modifier(entity_id: str) -> Optional[str]:
    id_lower = entity_id.lower()
    for mod in DOMAIN_MODIFIERS:
        if f"-{mod}" in id_lower or f"_{mod}" in id_lower:
            return mod
    return None


__all__ = [
    "C2EntityResolver",
    "EntityMatch",
    "normalize_entity_id",
    "tokenize_entity_id",
    "levenshtein_similarity",
    "token_similarity",
    "callsign_match",
    "alias_match",
    "ENTITY_CALLSIGNS",
    "ENTITY_ALIASES",
]
