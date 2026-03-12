/**
 * RBAC — role-based access control helpers.
 *
 * Keycloak realm roles map to Summit.OS capability tiers:
 *
 *   VIEWER           → read-only OPS (no dispatch, no missions)
 *   OPERATOR         → full OPS view
 *   MISSION_COMMANDER → OPS + COMMAND views
 *   ADMIN            → all views
 *   SUPER_ADMIN      → all views + admin actions
 *
 * The user.roles array comes from the id_token via /api/auth/me.
 * Checks are fail-closed — if roles are missing/empty, no access.
 */

export type SummitRole =
  | 'VIEWER'
  | 'OPERATOR'
  | 'MISSION_COMMANDER'
  | 'ADMIN'
  | 'SUPER_ADMIN';

export type AppView = 'ops' | 'command' | 'dev';

// Ordered by privilege level — higher index = more privileged
const ROLE_RANK: Record<SummitRole, number> = {
  VIEWER:            0,
  OPERATOR:          1,
  MISSION_COMMANDER: 2,
  ADMIN:             3,
  SUPER_ADMIN:       4,
};

/** Returns the highest-ranked SummitRole from a user's roles array. */
export function highestRole(roles: string[]): SummitRole | null {
  let best: SummitRole | null = null;
  let bestRank = -1;
  for (const r of roles) {
    const rank = ROLE_RANK[r as SummitRole];
    if (rank !== undefined && rank > bestRank) {
      best    = r as SummitRole;
      bestRank = rank;
    }
  }
  return best;
}

/** Which views this user is allowed to enter. */
export function allowedViews(roles: string[]): AppView[] {
  const top = highestRole(roles);
  if (!top) return [];
  switch (top) {
    case 'VIEWER':            return ['ops'];
    case 'OPERATOR':          return ['ops'];
    case 'MISSION_COMMANDER': return ['ops', 'command'];
    case 'ADMIN':
    case 'SUPER_ADMIN':       return ['ops', 'command', 'dev'];
  }
}

/** Can this user dispatch assets? */
export function canDispatch(roles: string[]): boolean {
  const top = highestRole(roles);
  if (!top) return false;
  return ROLE_RANK[top] >= ROLE_RANK['OPERATOR'];
}

/** Can this user approve/create missions? */
export function canManageMissions(roles: string[]): boolean {
  const top = highestRole(roles);
  if (!top) return false;
  return ROLE_RANK[top] >= ROLE_RANK['MISSION_COMMANDER'];
}

/** Can this user access admin/security settings? */
export function canAdmin(roles: string[]): boolean {
  const top = highestRole(roles);
  if (!top) return false;
  return ROLE_RANK[top] >= ROLE_RANK['ADMIN'];
}

/** Display label for a role. */
export function roleLabel(role: SummitRole): string {
  const labels: Record<SummitRole, string> = {
    VIEWER:            'Viewer',
    OPERATOR:          'Operator',
    MISSION_COMMANDER: 'Operations Lead',
    ADMIN:             'Admin',
    SUPER_ADMIN:       'Super Admin',
  };
  return labels[role];
}
