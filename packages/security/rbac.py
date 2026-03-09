"""
Role-Based Access Control (RBAC) for Summit.OS

Provides:
- Role/Permission definitions for Summit.OS access control
- Hierarchical role inheritance
- Resource-level permission checks
- Audit logging

Roles mirror defense/operator structure:
  VIEWER → OPERATOR → MISSION_COMMANDER → ADMIN → SUPER_ADMIN
"""
from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from enum import Enum

logger = logging.getLogger("security.rbac")


class Action(str, Enum):
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    DELETE = "delete"
    ADMIN = "admin"


class Resource(str, Enum):
    ENTITIES = "entities"
    TRACKS = "tracks"
    TASKS = "tasks"
    MISSIONS = "missions"
    SENSORS = "sensors"
    VEHICLES = "vehicles"
    MESH = "mesh"
    ANALYTICS = "analytics"
    USERS = "users"
    SYSTEM = "system"
    CLASSIFICATION = "classification"
    AUDIT = "audit"


@dataclass
class Permission:
    """A permission grants an action on a resource."""
    action: Action
    resource: Resource
    conditions: Dict[str, Any] = field(default_factory=dict)

    def __hash__(self):
        return hash((self.action, self.resource))

    def __eq__(self, other):
        if not isinstance(other, Permission):
            return False
        return self.action == other.action and self.resource == other.resource

    def matches(self, action: Action, resource: Resource) -> bool:
        """Check if this permission grants the requested access."""
        if self.action == Action.ADMIN:
            return self.resource == resource
        return self.action == action and self.resource == resource


@dataclass
class Role:
    """A role with a set of permissions and optional parent roles."""
    name: str
    description: str = ""
    permissions: Set[Permission] = field(default_factory=set)
    parent_roles: List[str] = field(default_factory=list)
    max_classification: str = "UNCLASSIFIED"


class RBACEngine:
    """
    RBAC enforcement engine.

    Supports:
    - Hierarchical roles (child inherits parent permissions)
    - User-to-role assignment
    - Permission checking with resource scoping
    - Audit trail
    """

    def __init__(self):
        self._roles: Dict[str, Role] = {}
        self._user_roles: Dict[str, Set[str]] = {}
        self._audit_log: List[Dict] = []
        self._init_default_roles()

    def _init_default_roles(self):
        """Initialize default role hierarchy."""
        # VIEWER: read-only access to tracks and entities
        self.add_role(Role(
            name="VIEWER",
            description="Read-only access to operational picture",
            permissions={
                Permission(Action.READ, Resource.ENTITIES),
                Permission(Action.READ, Resource.TRACKS),
                Permission(Action.READ, Resource.ANALYTICS),
            },
            max_classification="UNCLASSIFIED",
        ))

        # OPERATOR: can interact with tasks and sensors
        self.add_role(Role(
            name="OPERATOR",
            description="Standard operator with task and sensor access",
            permissions={
                Permission(Action.READ, Resource.TASKS),
                Permission(Action.WRITE, Resource.TASKS),
                Permission(Action.READ, Resource.SENSORS),
                Permission(Action.EXECUTE, Resource.SENSORS),
                Permission(Action.READ, Resource.VEHICLES),
                Permission(Action.READ, Resource.MESH),
            },
            parent_roles=["VIEWER"],
            max_classification="CONFIDENTIAL",
        ))

        # MISSION_COMMANDER: can create/manage missions
        self.add_role(Role(
            name="MISSION_COMMANDER",
            description="Mission planning and command authority",
            permissions={
                Permission(Action.READ, Resource.MISSIONS),
                Permission(Action.WRITE, Resource.MISSIONS),
                Permission(Action.EXECUTE, Resource.MISSIONS),
                Permission(Action.WRITE, Resource.VEHICLES),
                Permission(Action.EXECUTE, Resource.VEHICLES),
                Permission(Action.DELETE, Resource.TASKS),
            },
            parent_roles=["OPERATOR"],
            max_classification="SECRET",
        ))

        # ADMIN: system administration
        self.add_role(Role(
            name="ADMIN",
            description="System administrator",
            permissions={
                Permission(Action.ADMIN, Resource.USERS),
                Permission(Action.ADMIN, Resource.SYSTEM),
                Permission(Action.ADMIN, Resource.MESH),
                Permission(Action.READ, Resource.AUDIT),
                Permission(Action.READ, Resource.CLASSIFICATION),
                Permission(Action.WRITE, Resource.CLASSIFICATION),
            },
            parent_roles=["MISSION_COMMANDER"],
            max_classification="TOP_SECRET",
        ))

        # SUPER_ADMIN: unrestricted access
        self.add_role(Role(
            name="SUPER_ADMIN",
            description="Unrestricted system access",
            permissions={
                Permission(action, resource)
                for action in Action
                for resource in Resource
            },
            max_classification="TOP_SECRET_SCI",
        ))

    def add_role(self, role: Role) -> None:
        """Register a role."""
        self._roles[role.name] = role

    def assign_role(self, user_id: str, role_name: str) -> bool:
        """Assign a role to a user."""
        if role_name not in self._roles:
            logger.error(f"Role not found: {role_name}")
            return False

        if user_id not in self._user_roles:
            self._user_roles[user_id] = set()

        self._user_roles[user_id].add(role_name)
        self._log_audit(user_id, "role_assigned", {"role": role_name})
        return True

    def remove_role(self, user_id: str, role_name: str) -> bool:
        """Remove a role from a user."""
        if user_id in self._user_roles:
            self._user_roles[user_id].discard(role_name)
            self._log_audit(user_id, "role_removed", {"role": role_name})
            return True
        return False

    def check_permission(self, user_id: str, action: Action,
                         resource: Resource) -> bool:
        """
        Check if a user has permission for an action on a resource.

        Resolves full role hierarchy.
        """
        effective_perms = self.get_effective_permissions(user_id)
        granted = any(p.matches(action, resource) for p in effective_perms)

        self._log_audit(user_id, "permission_check", {
            "action": action.value,
            "resource": resource.value,
            "granted": granted,
        })

        return granted

    def get_effective_permissions(self, user_id: str) -> Set[Permission]:
        """Get all permissions for a user including inherited ones."""
        roles = self._user_roles.get(user_id, set())
        permissions: Set[Permission] = set()

        visited: Set[str] = set()
        to_visit = list(roles)

        while to_visit:
            role_name = to_visit.pop()
            if role_name in visited:
                continue
            visited.add(role_name)

            role = self._roles.get(role_name)
            if role:
                permissions.update(role.permissions)
                to_visit.extend(role.parent_roles)

        return permissions

    def get_max_classification(self, user_id: str) -> str:
        """Get the highest classification level a user can access."""
        roles = self._user_roles.get(user_id, set())
        levels = ["UNCLASSIFIED", "CONFIDENTIAL", "SECRET",
                  "TOP_SECRET", "TOP_SECRET_SCI"]

        max_level = "UNCLASSIFIED"
        for role_name in roles:
            role = self._roles.get(role_name)
            if role and role.max_classification in levels:
                if levels.index(role.max_classification) > levels.index(max_level):
                    max_level = role.max_classification

        return max_level

    def get_user_roles(self, user_id: str) -> List[str]:
        """Get all roles assigned to a user."""
        return list(self._user_roles.get(user_id, set()))

    def _log_audit(self, user_id: str, event: str, details: Dict) -> None:
        """Log an audit event."""
        entry = {
            "timestamp": time.time(),
            "user_id": user_id,
            "event": event,
            **details,
        }
        self._audit_log.append(entry)
        # Keep last 10000 entries in memory
        if len(self._audit_log) > 10000:
            self._audit_log = self._audit_log[-10000:]

    def get_audit_log(self, user_id: Optional[str] = None,
                      limit: int = 100) -> List[Dict]:
        """Get audit log entries."""
        log = self._audit_log
        if user_id:
            log = [e for e in log if e.get("user_id") == user_id]
        return log[-limit:]
