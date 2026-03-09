"""
mission_patterns.py — Mission pattern library for Summit.OS CyberSynetic engine.

Builds a reusable library from completed missions. When a new mission is being
planned, suggests the most relevant successful patterns.

Pattern matching is deliberately simple: tag overlap + asset type overlap,
ranked by success_rate × log(use_count). No ML needed — the pattern library
is useful from the very first mission and becomes more valuable with each one.
"""
from __future__ import annotations

import json
import logging
import math
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    insert,
    select,
    update,
)
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

logger = logging.getLogger("learning.mission_patterns")

UTC = timezone.utc


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class MissionPattern:
    pattern_id:    str
    name:          str
    description:   str
    asset_types:   list[str]   # what kinds of assets were used
    asset_count:   int
    avg_duration_s: float
    success_rate:  float
    use_count:     int
    domain_tags:   list[str]   # e.g. ["wildfire", "search_rescue", "surveillance"]
    template:      dict        # parameterized mission template
    last_used:     datetime

    def to_dict(self) -> dict:
        d = asdict(self)
        d["last_used"] = self.last_used.isoformat()
        return d


# ---------------------------------------------------------------------------
# SQLAlchemy table
# ---------------------------------------------------------------------------

_metadata = MetaData()

patterns_table = Table(
    "mission_patterns",
    _metadata,
    Column("id",             Integer, primary_key=True, autoincrement=True),
    Column("pattern_id",     String(64), nullable=False, unique=True),
    Column("name",           String(256), nullable=False),
    Column("description",    Text, nullable=False, default=""),
    Column("asset_types_json", Text, nullable=False, default="[]"),
    Column("asset_count",    Integer, nullable=False, default=1),
    Column("avg_duration_s", Float, nullable=False, default=0.0),
    Column("success_rate",   Float, nullable=False, default=0.0),
    Column("use_count",      Integer, nullable=False, default=1),
    Column("domain_tags_json", Text, nullable=False, default="[]"),
    Column("template_json",  Text, nullable=False, default="{}"),
    Column("last_used",      DateTime(timezone=True), nullable=False),
    Index("ix_mp_pattern_id", "pattern_id"),
    Index("ix_mp_success_rate", "success_rate"),
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_asset_types(mission: dict) -> list[str]:
    """Pull asset type identifiers out of a mission dict."""
    types: list[str] = []
    assets = mission.get("assets", []) or mission.get("asset_ids", []) or []
    if isinstance(assets, list):
        for a in assets:
            if isinstance(a, dict):
                t = a.get("type") or a.get("entity_type") or "unknown"
                types.append(str(t).lower())
            elif isinstance(a, str):
                types.append("unknown")
    asset_type = mission.get("asset_type") or mission.get("entity_type")
    if asset_type:
        types.append(str(asset_type).lower())
    return list(set(types)) or ["unknown"]


def _extract_domain_tags(mission: dict) -> list[str]:
    """Extract domain/context tags from a mission dict."""
    tags: list[str] = []
    for field in ("domain_tags", "tags", "domain", "context_tags", "category"):
        val = mission.get(field)
        if isinstance(val, list):
            tags.extend(str(t).lower() for t in val)
        elif isinstance(val, str) and val:
            tags.append(val.lower())
    return list(set(tags))


def _mission_fingerprint(mission: dict) -> str:
    """
    Create a stable fingerprint for a mission based on asset types and domain tags.
    Missions with the same fingerprint are candidates for the same pattern.
    """
    asset_types = sorted(_extract_asset_types(mission))
    domain_tags = sorted(_extract_domain_tags(mission))
    return f"assets:{','.join(asset_types)}|tags:{','.join(domain_tags)}"


def _tag_overlap(a: list[str], b: list[str]) -> float:
    """Jaccard similarity between two tag sets."""
    set_a, set_b = set(a), set(b)
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    intersection = set_a & set_b
    return len(intersection) / len(union)


def _asset_overlap(a: list[str], b: list[str]) -> float:
    """Overlap coefficient for asset type sets."""
    set_a, set_b = set(a), set(b)
    if not set_a and not set_b:
        return 1.0
    return len(set_a & set_b) / max(len(set_a), len(set_b))


# ---------------------------------------------------------------------------
# Library
# ---------------------------------------------------------------------------

class MissionPatternLibrary:
    """
    Records completed missions and identifies reusable patterns.

    Pattern creation logic:
    1. Each completed mission is checked against existing patterns using
       asset type + domain tag overlap.
    2. If overlap ≥ 0.6, the mission is folded into the existing pattern
       (updating avg_duration, success_rate, use_count).
    3. Otherwise, a new pattern is created.
    4. Suggestions are ranked by: success_rate × log1p(use_count),
       filtered by context relevance.
    """

    def __init__(self, database_url: str) -> None:
        self._db_url = database_url
        self._engine: Optional[AsyncEngine] = None

    async def initialize(self) -> None:
        self._engine = create_async_engine(self._db_url, echo=False, future=True)
        async with self._engine.begin() as conn:
            await conn.run_sync(_metadata.create_all)
        logger.info("MissionPatternLibrary initialized")

    def _ensure_engine(self) -> AsyncEngine:
        if self._engine is None:
            raise RuntimeError("MissionPatternLibrary.initialize() has not been called")
        return self._engine

    def _row_to_pattern(self, row: Any) -> MissionPattern:
        return MissionPattern(
            pattern_id=row.pattern_id,
            name=row.name,
            description=row.description,
            asset_types=json.loads(row.asset_types_json or "[]"),
            asset_count=row.asset_count,
            avg_duration_s=row.avg_duration_s,
            success_rate=row.success_rate,
            use_count=row.use_count,
            domain_tags=json.loads(row.domain_tags_json or "[]"),
            template=json.loads(row.template_json or "{}"),
            last_used=row.last_used,
        )

    async def record_mission(
        self,
        mission:    dict,
        outcome:    str,    # 'completed' | 'failed' | 'aborted'
        duration_s: float,
    ) -> None:
        """
        Record a completed (or failed) mission for pattern learning.

        If a sufficiently similar pattern exists, update it.
        Otherwise, seed a new pattern.
        """
        if outcome == "aborted":
            # Aborted missions don't contribute to patterns
            return

        success = outcome == "completed"
        asset_types = _extract_asset_types(mission)
        domain_tags = _extract_domain_tags(mission)
        asset_count = max(1, len(mission.get("assets", []) or mission.get("asset_ids", [])) or 1)

        # Load all existing patterns to find the best match
        all_patterns = await self.get_all_patterns()
        best_match: Optional[MissionPattern] = None
        best_score  = 0.0
        MERGE_THRESHOLD = 0.6

        for p in all_patterns:
            tag_sim   = _tag_overlap(domain_tags, p.domain_tags)
            asset_sim = _asset_overlap(asset_types, p.asset_types)
            combined  = 0.5 * tag_sim + 0.5 * asset_sim
            if combined >= MERGE_THRESHOLD and combined > best_score:
                best_score  = combined
                best_match  = p

        now = datetime.now(UTC)

        if best_match:
            # Fold this mission into the existing pattern
            n = best_match.use_count
            new_n = n + 1
            # Running average for duration
            new_avg_duration = (best_match.avg_duration_s * n + duration_s) / new_n
            # Running average for success rate
            new_success_rate = (best_match.success_rate * n + (1.0 if success else 0.0)) / new_n
            # Merge asset types and domain tags (union)
            merged_types = list(set(best_match.asset_types) | set(asset_types))
            merged_tags  = list(set(best_match.domain_tags) | set(domain_tags))

            engine = self._ensure_engine()
            async with engine.begin() as conn:
                await conn.execute(
                    update(patterns_table)
                    .where(patterns_table.c.pattern_id == best_match.pattern_id)
                    .values(
                        use_count=new_n,
                        avg_duration_s=round(new_avg_duration, 2),
                        success_rate=round(new_success_rate, 4),
                        asset_types_json=json.dumps(sorted(merged_types)),
                        domain_tags_json=json.dumps(sorted(merged_tags)),
                        last_used=now,
                    )
                )
            logger.debug(
                "Folded mission into pattern '%s' (use_count=%d success_rate=%.2f)",
                best_match.pattern_id, new_n, new_success_rate,
            )

        else:
            # Create a new pattern from this mission
            pattern_id  = str(uuid.uuid4())
            name        = _build_pattern_name(asset_types, domain_tags)
            description = _build_pattern_description(mission, asset_types, domain_tags, outcome)
            template    = _build_template(mission, asset_types, domain_tags)

            engine = self._ensure_engine()
            async with engine.begin() as conn:
                await conn.execute(
                    insert(patterns_table).values(
                        pattern_id=pattern_id,
                        name=name,
                        description=description,
                        asset_types_json=json.dumps(sorted(asset_types)),
                        asset_count=asset_count,
                        avg_duration_s=round(duration_s, 2),
                        success_rate=1.0 if success else 0.0,
                        use_count=1,
                        domain_tags_json=json.dumps(sorted(domain_tags)),
                        template_json=json.dumps(template),
                        last_used=now,
                    )
                )
            logger.debug(
                "Created new mission pattern '%s' (%s)", pattern_id, name
            )

    async def suggest_templates(
        self,
        asset_types:  list[str],
        context_tags: list[str] = [],
        limit:        int = 5,
    ) -> list[MissionPattern]:
        """
        Return mission patterns relevant to the current situation.

        Scored by: relevance × success_rate × log1p(use_count)
        relevance = 0.5 × asset_overlap + 0.5 × tag_overlap
        """
        all_patterns = await self.get_all_patterns()
        if not all_patterns:
            return []

        scored: list[tuple[float, MissionPattern]] = []
        for p in all_patterns:
            if p.success_rate <= 0.0:
                continue  # never suggest a pattern that has only failed
            tag_sim   = _tag_overlap(context_tags, p.domain_tags)
            asset_sim = _asset_overlap(asset_types, p.asset_types)
            relevance = 0.5 * tag_sim + 0.5 * asset_sim
            if relevance <= 0.0:
                continue
            score = relevance * p.success_rate * math.log1p(p.use_count)
            scored.append((score, p))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [p for _, p in scored[:limit]]

    async def get_all_patterns(self) -> list[MissionPattern]:
        engine = self._ensure_engine()
        async with engine.connect() as conn:
            result = await conn.execute(
                select(patterns_table).order_by(
                    patterns_table.c.success_rate.desc(),
                    patterns_table.c.use_count.desc(),
                )
            )
            rows = result.fetchall()
        return [self._row_to_pattern(r) for r in rows]

    async def get_pattern(self, pattern_id: str) -> Optional[MissionPattern]:
        engine = self._ensure_engine()
        async with engine.connect() as conn:
            result = await conn.execute(
                select(patterns_table).where(
                    patterns_table.c.pattern_id == pattern_id
                )
            )
            row = result.first()
        return self._row_to_pattern(row) if row else None


# ---------------------------------------------------------------------------
# Template / name helpers
# ---------------------------------------------------------------------------

def _build_pattern_name(asset_types: list[str], domain_tags: list[str]) -> str:
    asset_str = "+".join(sorted(asset_types)[:2]) or "asset"
    tag_str   = "/".join(sorted(domain_tags)[:2]) or "general"
    return f"{asset_str.title()} — {tag_str.replace('_', ' ').title()}"


def _build_pattern_description(
    mission: dict,
    asset_types: list[str],
    domain_tags: list[str],
    outcome: str,
) -> str:
    parts = []
    if domain_tags:
        parts.append(f"Domain: {', '.join(domain_tags)}")
    if asset_types:
        parts.append(f"Assets: {', '.join(asset_types)}")
    objective = mission.get("objective") or mission.get("description") or ""
    if objective:
        parts.append(f"Objective: {objective[:120]}")
    parts.append(f"First observed outcome: {outcome}")
    return " | ".join(parts)


def _build_template(
    mission: dict,
    asset_types: list[str],
    domain_tags: list[str],
) -> dict:
    """
    Extract a parameterized template from a completed mission.

    Strips instance-specific IDs while preserving structural intent.
    """
    return {
        "asset_types":  sorted(asset_types),
        "domain_tags":  sorted(domain_tags),
        "objective":    mission.get("objective") or mission.get("description") or "",
        "priority":     mission.get("priority") or "MEDIUM",
        "waypoints":    [],   # stripped — location is instance-specific
        "rules_of_engagement": mission.get("rules_of_engagement") or {},
        "parameters":   {
            k: v
            for k, v in mission.items()
            if k not in (
                "id", "mission_id", "asset_id", "asset_ids", "assets",
                "created_at", "updated_at", "ts_iso", "waypoints",
                "location", "lat", "lon",
            )
            and not isinstance(v, (list, dict))
        },
    }
