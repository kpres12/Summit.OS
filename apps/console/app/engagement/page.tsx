'use client';

/**
 * /engagement — operator surface for human-in-the-loop kinetic
 * authorization. Mounted as a top-level route alongside /command, /ops,
 * /dev. Restricted to MISSION_COMMANDER+ roles via ProtectedRoute.
 */

import dynamic from 'next/dynamic';
import { useAuth } from '@/components/AuthProvider';
import ProtectedRoute from '@/components/ProtectedRoute';

const EngagementQueue = dynamic(
  () => import('@/components/ops/EngagementQueue'),
  { ssr: false },
);

export default function EngagementPage() {
  return (
    <ProtectedRoute>
      <EngagementQueueGuarded />
    </ProtectedRoute>
  );
}

function EngagementQueueGuarded() {
  const { user } = useAuth();
  if (!user) {
    return <div style={{ padding: 24, color: '#FF3B3B' }}>Not authenticated</div>;
  }

  // Heuristic: treat the user's effective role as the doctrine role used by
  // the engagement gate. ADMIN and SUPER_ADMIN map to mission_commander+
  // for purposes of authorization; the gate's RBAC check enforces the
  // actual minimum role required by the engagement class.
  const role = (user.roles ?? []).includes('MISSION_COMMANDER')
    ? 'mission_commander'
    : (user.roles ?? []).includes('ADMIN') || (user.roles ?? []).includes('SUPER_ADMIN')
    ? 'joint_force_commander'
    : 'operator';

  return (
    <div style={{ height: '100vh' }}>
      <EngagementQueue
        operatorId={user.id || user.email || 'unknown-operator'}
        operatorRole={role}
      />
    </div>
  );
}
