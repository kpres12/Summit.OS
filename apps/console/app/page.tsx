'use client';
import ProtectedRoute from '@/components/ProtectedRoute';
import UpgradePrompt from '@/components/UpgradePrompt';
import dynamic from 'next/dynamic';
import { useRole } from '@/hooks/useRole';
import { useAuth } from '@/components/AuthProvider';
import { useTier } from '@/hooks/useTier';
import { allowedViews } from '@/lib/rbac';
import RolePicker from '@/components/RolePicker';
const OpsLayout     = dynamic(() => import('@/components/ops/OpsLayout'),         { ssr: false });
const CommandLayout = dynamic(() => import('@/components/command/CommandLayout'),  { ssr: false });
const DevLayout     = dynamic(() => import('@/components/dev/DevLayout'),          { ssr: false });

export default function Home() {
  return (
    <ProtectedRoute>
      <RoleRouter />
    </ProtectedRoute>
  );
}

function RoleRouter() {
  const { role, setRole, clearRole } = useRole();
  const { user } = useAuth();
  const { can, requiredTier, loading: tierLoading } = useTier();
  const allowed = allowedViews(user?.roles ?? []);

  // If the stored role is no longer permitted (role was downgraded), clear it
  if (role && !allowed.includes(role)) {
    clearRole();
    return null;
  }

  if (!role) return <RolePicker onSelect={setRole} />;

  // Tier gates — role access (RBAC) is already satisfied above; check billing tier
  if (role === 'command' && !tierLoading && !can('command_view')) {
    return (
      <div style={{ width: '100vw', height: '100vh', background: '#080C0A', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <UpgradePrompt
          feature="Command View"
          requiredTier={requiredTier('command_view')!}
          description="Situation feed, 3-column layout, and handoff brief generator for mission commanders."
        />
      </div>
    );
  }

  if (role === 'dev' && !tierLoading && !can('dev_view')) {
    return (
      <div style={{ width: '100vw', height: '100vh', background: '#080C0A', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <UpgradePrompt
          feature="Developer View"
          requiredTier={requiredTier('dev_view')!}
          description="Entity explorer, adapter registry, message inspector, and inference dashboard."
        />
      </div>
    );
  }

  if (role === 'ops')     return <OpsLayout     onSwitchRole={clearRole} />;
  if (role === 'command') return <CommandLayout  onSwitchRole={clearRole} />;
  if (role === 'dev')     return <DevLayout      onSwitchRole={clearRole} />;
  return null;
}
