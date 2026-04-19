'use client';
import ProtectedRoute from '@/components/ProtectedRoute';
import dynamic from 'next/dynamic';
import { useEffect } from 'react';
import { useRole, Role } from '@/hooks/useRole';
import { useAuth } from '@/components/AuthProvider';
import { useTier } from '@/hooks/useTier';
import { allowedViews, highestRole } from '@/lib/rbac';

const OpsLayout     = dynamic(() => import('@/components/ops/OpsLayout'),        { ssr: false });
const CommandLayout = dynamic(() => import('@/components/command/CommandLayout'), { ssr: false });
const DevLayout     = dynamic(() => import('@/components/dev/DevLayout'),         { ssr: false });

export default function Home() {
  return (
    <ProtectedRoute>
      <RoleRouter />
    </ProtectedRoute>
  );
}

/** Maps a user's highest auth role to their default view. */
function defaultViewForRoles(roles: string[]): Role {
  const top = highestRole(roles);
  switch (top) {
    case 'ADMIN':
    case 'SUPER_ADMIN':       return 'ops';
    case 'MISSION_COMMANDER': return 'command';
    default:                  return 'ops';
  }
}

function RoleRouter() {
  const { role, setRole, loaded } = useRole();
  const { user } = useAuth();
  const { can, loading: tierLoading } = useTier();
  const allowed = allowedViews(user?.roles ?? []);

  // On first load (no stored role), derive from auth roles
  useEffect(() => {
    if (loaded && !role && user) {
      setRole(defaultViewForRoles(user.roles));
    }
  }, [loaded, role, user, setRole]);

  // If stored role is no longer permitted (role downgraded), reset to default
  useEffect(() => {
    if (loaded && role && user && !allowed.includes(role)) {
      setRole(defaultViewForRoles(user.roles));
    }
  }, [loaded, role, user, allowed, setRole]);

  if (!loaded || !role) return null;

  // Cycle through allowed views on switch — no picker
  const handleSwitchRole = () => {
    const idx  = allowed.indexOf(role);
    const next = allowed[(idx + 1) % allowed.length];
    setRole(next);
  };

  // Tier gates — fall back to ops if tier insufficient
  if (role === 'command' && !tierLoading && !can('command_view')) {
    setRole('ops');
    return null;
  }
  if (role === 'dev' && !tierLoading && !can('dev_view')) {
    setRole('ops');
    return null;
  }

  if (role === 'ops')     return <OpsLayout     onSwitchRole={handleSwitchRole} />;
  if (role === 'command') return <CommandLayout  onSwitchRole={handleSwitchRole} />;
  if (role === 'dev')     return <DevLayout      onSwitchRole={handleSwitchRole} />;
  return null;
}
