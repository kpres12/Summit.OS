"""
Summit.OS Ontology Action Runner

The ONLY way to modify ontology state is through ActionRunner.execute().

Flow:
  1. Look up ActionTypeDef in registry
  2. Validate inputs against input_properties schema
  3. Run all validator functions (business rules)
  4. Load (or create) the target ObjectInstance
  5. Apply the mutation (merge inputs into properties)
  6. Persist via ObjectStore._upsert()
  7. Run side_effect hooks (publish events, trigger downstream)
  8. Write an AuditEntry — always, even on rejection

The audit trail is append-only and never deleted.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .registry import get_registry
from .store import get_store
from .types import ActionResult, AuditEntry, ObjectInstance

logger = logging.getLogger("ontology.actions")


class ActionRunner:

    def execute(
        self,
        action_name: str,
        object_id: str,
        inputs: Dict[str, Any],
        actor_id: str = "system",
    ) -> ActionResult:
        """
        Execute a governed action against a target object.

        action_name: ActionTypeDef.name
        object_id:   the target ObjectInstance.object_id (or "" to create new)
        inputs:      key-value action inputs matching input_properties schema
        actor_id:    operator id or service name (for audit trail)
        """
        registry = get_registry()
        store = get_store()

        action = registry.get_action(action_name)
        if action is None:
            return ActionResult(
                success=False,
                error=f"Unknown action '{action_name}'",
            )

        # ── 1. Input schema validation ─────────────────────────────────────────
        schema_error = self._validate_schema(action.input_properties, inputs)
        if schema_error:
            audit = self._write_audit(
                action_name,
                action.target_type,
                object_id,
                actor_id,
                inputs,
                "rejected",
                schema_error,
            )
            return ActionResult(success=False, error=schema_error, audit_entry=audit)

        # ── 2. Load or create target instance ─────────────────────────────────
        instance = store.get(action.target_type, object_id)
        if instance is None and object_id:
            # Auto-create a stub instance for new objects
            instance = ObjectInstance(
                object_type=action.target_type,
                object_id=object_id,
                properties={"id": object_id},
            )
        elif instance is None and not object_id:
            new_id = str(uuid.uuid4())
            instance = ObjectInstance(
                object_type=action.target_type,
                object_id=new_id,
                properties={"id": new_id},
            )

        # ── 3. Business-rule validators ────────────────────────────────────────
        for validator in action.validators:
            try:
                error = validator(inputs, instance, store)
            except Exception as exc:
                error = f"Validator exception: {exc}"
            if error:
                audit = self._write_audit(
                    action_name,
                    action.target_type,
                    instance.object_id,
                    actor_id,
                    inputs,
                    "rejected",
                    error,
                )
                return ActionResult(success=False, error=error, audit_entry=audit)

        # ── 4. Apply mutation ──────────────────────────────────────────────────
        instance.properties.update(
            {k: v for k, v in inputs.items() if not k.startswith("_")}
        )
        instance = store._upsert(instance)

        # ── 5. Side effects ────────────────────────────────────────────────────
        side_effect_log: List[str] = []
        for fx in action.side_effects:
            try:
                result = fx(inputs, instance, store)
                if result:
                    side_effect_log.append(str(result))
            except Exception as exc:
                logger.warning("Side effect error in '%s': %s", action_name, exc)
                side_effect_log.append(f"side_effect_error: {exc}")

        # ── 6. Audit ───────────────────────────────────────────────────────────
        audit = self._write_audit(
            action_name,
            action.target_type,
            instance.object_id,
            actor_id,
            inputs,
            "success",
            "",
        )

        logger.info(
            "Action '%s' on %s/%s by %s — success",
            action_name,
            action.target_type,
            instance.object_id,
            actor_id,
        )

        return ActionResult(
            success=True,
            object_instance=instance,
            audit_entry=audit,
            side_effect_log=side_effect_log,
        )

    # ── internal helpers ──────────────────────────────────────────────────────

    def _validate_schema(self, property_defs, inputs: Dict[str, Any]) -> Optional[str]:
        """Validate required fields and basic type coercion."""
        from .types import PropertyKind

        for prop in property_defs:
            if prop.required and prop.name not in inputs:
                return f"Required input '{prop.name}' is missing"
            if prop.name in inputs:
                val = inputs[prop.name]
                if prop.kind == PropertyKind.ENUM and val not in prop.enum_values:
                    return (
                        f"'{prop.name}' must be one of {prop.enum_values}, got '{val}'"
                    )
                if prop.kind == PropertyKind.FLOAT:
                    try:
                        inputs[prop.name] = float(val)
                    except (TypeError, ValueError):
                        return f"'{prop.name}' must be a float, got '{val}'"
                if prop.kind == PropertyKind.INTEGER:
                    try:
                        inputs[prop.name] = int(val)
                    except (TypeError, ValueError):
                        return f"'{prop.name}' must be an integer, got '{val}'"
        return None

    def _write_audit(
        self,
        action_name: str,
        target_type: str,
        object_id: str,
        actor_id: str,
        inputs: Dict[str, Any],
        outcome: str,
        reason: str,
    ) -> AuditEntry:
        entry = AuditEntry(
            action_name=action_name,
            target_type=target_type,
            object_id=object_id,
            actor_id=actor_id,
            inputs=inputs,
            outcome=outcome,
            rejection_reason=reason,
        )
        get_audit_log().append(entry)
        return entry


# ── in-memory audit log ────────────────────────────────────────────────────────

_audit_log: List[AuditEntry] = []


def get_audit_log() -> List[AuditEntry]:
    return _audit_log


def recent_audit(limit: int = 100, actor_id: Optional[str] = None) -> List[AuditEntry]:
    entries = list(reversed(_audit_log))
    if actor_id:
        entries = [e for e in entries if e.actor_id == actor_id]
    return entries[:limit]


# ── singleton ──────────────────────────────────────────────────────────────────

_runner: ActionRunner | None = None


def get_action_runner() -> ActionRunner:
    global _runner
    if _runner is None:
        _runner = ActionRunner()
    return _runner
