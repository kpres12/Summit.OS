"""
Data Classification for Heli.OS

Implements data classification labels matching defense standards:
- UNCLASSIFIED (U)
- CONFIDENTIAL (C)
- SECRET (S)
- TOP SECRET (TS)
- TOP SECRET / SCI (TS/SCI)

Enforces classification policies:
- Write-up: data can be classified higher but not lower
- Need-to-know: users can only access data at or below their clearance
- Marking: all data objects carry classification labels
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
from enum import IntEnum

logger = logging.getLogger("security.classification")


class ClassificationLevel(IntEnum):
    """Classification levels in ascending order of sensitivity."""

    UNCLASSIFIED = 0
    CUI = 1  # Controlled Unclassified Information
    CONFIDENTIAL = 2
    SECRET = 3
    TOP_SECRET = 4
    TOP_SECRET_SCI = 5

    @classmethod
    def from_string(cls, s: str) -> "ClassificationLevel":
        mapping = {
            "U": cls.UNCLASSIFIED,
            "UNCLASSIFIED": cls.UNCLASSIFIED,
            "CUI": cls.CUI,
            "C": cls.CONFIDENTIAL,
            "CONFIDENTIAL": cls.CONFIDENTIAL,
            "S": cls.SECRET,
            "SECRET": cls.SECRET,
            "TS": cls.TOP_SECRET,
            "TOP_SECRET": cls.TOP_SECRET,
            "TS/SCI": cls.TOP_SECRET_SCI,
            "TOP_SECRET_SCI": cls.TOP_SECRET_SCI,
        }
        return mapping.get(s.upper(), cls.UNCLASSIFIED)

    def to_banner(self) -> str:
        """Return classification banner string."""
        banners = {
            0: "UNCLASSIFIED",
            1: "CUI",
            2: "CONFIDENTIAL",
            3: "SECRET",
            4: "TOP SECRET",
            5: "TOP SECRET // SCI",
        }
        return banners.get(self.value, "UNCLASSIFIED")

    def to_short(self) -> str:
        shorts = {0: "U", 1: "CUI", 2: "C", 3: "S", 4: "TS", 5: "TS/SCI"}
        return shorts.get(self.value, "U")


@dataclass
class DataClassification:
    """Classification label for a data object."""

    level: ClassificationLevel = ClassificationLevel.UNCLASSIFIED
    caveats: List[str] = field(default_factory=list)  # e.g., ["NOFORN", "REL TO USA"]
    compartments: List[str] = field(default_factory=list)  # SCI compartments
    handling_caveats: List[str] = field(default_factory=list)  # e.g., ["ORCON"]
    originator: str = ""
    classification_reason: str = ""
    declassify_on: str = ""  # e.g., "20350101" or "25X1"
    created_at: float = field(default_factory=time.time)

    @property
    def banner(self) -> str:
        """Full classification banner."""
        parts = [self.level.to_banner()]
        if self.compartments:
            parts.append("// " + "/".join(self.compartments))
        if self.caveats:
            parts.append("// " + "/".join(self.caveats))
        return " ".join(parts)

    @property
    def portion_mark(self) -> str:
        """Short portion marking for inline use."""
        mark = f"({self.level.to_short()}"
        if self.compartments:
            mark += "/" + "/".join(self.compartments)
        mark += ")"
        return mark

    def to_dict(self) -> Dict:
        return {
            "level": self.level.to_banner(),
            "level_short": self.level.to_short(),
            "caveats": self.caveats,
            "compartments": self.compartments,
            "handling": self.handling_caveats,
            "banner": self.banner,
            "portion_mark": self.portion_mark,
        }


class ClassificationPolicy:
    """
    Enforces classification policies.

    Key rules:
    1. Data cannot be declassified (lowered) without explicit authority
    2. Users can only access data at or below their clearance
    3. Aggregation: combining data may produce higher classification
    4. All mutations are audit-logged
    """

    def __init__(self):
        self._object_labels: Dict[str, DataClassification] = {}
        self._audit_log: List[Dict] = []

    def label(self, object_id: str, classification: DataClassification) -> None:
        """Apply a classification label to an object."""
        existing = self._object_labels.get(object_id)

        if existing and classification.level < existing.level:
            logger.warning(
                f"Attempted to declassify {object_id} from "
                f"{existing.level.to_banner()} to {classification.level.to_banner()} — denied"
            )
            self._log("declassify_denied", object_id, classification)
            raise PermissionError(
                f"Cannot declassify from {existing.level.to_banner()} "
                f"to {classification.level.to_banner()}"
            )

        self._object_labels[object_id] = classification
        self._log("labeled", object_id, classification)

    def get_label(self, object_id: str) -> DataClassification:
        """Get the classification label for an object."""
        return self._object_labels.get(
            object_id,
            DataClassification(level=ClassificationLevel.UNCLASSIFIED),
        )

    def check_access(self, user_clearance: str, object_id: str) -> bool:
        """
        Check if a user with given clearance can access an object.

        Returns True if user's clearance >= object's classification.
        """
        user_level = ClassificationLevel.from_string(user_clearance)
        obj_label = self.get_label(object_id)

        granted = user_level >= obj_label.level
        self._log(
            "access_check",
            object_id,
            obj_label,
            extra={
                "user_clearance": user_clearance,
                "granted": granted,
            },
        )

        return granted

    def compute_aggregate_level(self, object_ids: List[str]) -> ClassificationLevel:
        """
        Compute the classification for an aggregation of objects.

        The aggregate takes the HIGHEST classification of any component.
        """
        max_level = ClassificationLevel.UNCLASSIFIED
        for oid in object_ids:
            label = self.get_label(oid)
            if label.level > max_level:
                max_level = label.level
        return max_level

    def upgrade(
        self, object_id: str, new_level: ClassificationLevel, reason: str = ""
    ) -> DataClassification:
        """
        Upgrade (reclassify higher) an object's classification.

        This is always allowed (write-up principle).
        """
        current = self.get_label(object_id)
        if new_level <= current.level:
            return current

        upgraded = DataClassification(
            level=new_level,
            caveats=current.caveats,
            compartments=current.compartments,
            handling_caveats=current.handling_caveats,
            originator=current.originator,
            classification_reason=reason or current.classification_reason,
            declassify_on=current.declassify_on,
        )
        self._object_labels[object_id] = upgraded
        self._log("upgraded", object_id, upgraded, extra={"reason": reason})
        return upgraded

    def _log(
        self,
        event: str,
        object_id: str,
        classification: DataClassification,
        extra: Optional[Dict] = None,
    ) -> None:
        entry = {
            "timestamp": time.time(),
            "event": event,
            "object_id": object_id,
            "level": classification.level.to_banner(),
            **(extra or {}),
        }
        self._audit_log.append(entry)
        if len(self._audit_log) > 10000:
            self._audit_log = self._audit_log[-10000:]

    def get_audit_log(self, limit: int = 100) -> List[Dict]:
        return self._audit_log[-limit:]
