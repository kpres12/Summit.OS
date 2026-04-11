'use client';
import ProtectedRoute from '@/components/ProtectedRoute';
import UpgradePrompt from '@/components/UpgradePrompt';
import dynamic from 'next/dynamic';
import { useState } from 'react';
import { useRole, Role } from '@/hooks/useRole';
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
  const { role, setRole, clearRole, loaded } = useRole();
  const { user } = useAuth();
  const { can, requiredTier, loading: tierLoading } = useTier();
  const allowed = allowedViews(user?.roles ?? []);

  // Track the role active before opening the picker, so user can cancel back to it
  const [previousRole, setPreviousRole] = useState<Role | null>(null);

  // Don't render until localStorage is read — prevents flash of role picker
  if (!loaded) return null;

  // If stored role is no longer permitted (role downgraded), fall back to ops
  if (role && !allowed.includes(role)) {
    setRole('ops');
    return null;
  }

  const handleSwitchRole = () => {
    setPreviousRole(role);
    clearRole();
  };

  const handleBack = previousRole ? () => {
    setRole(previousRole);
    setPreviousRole(null);
  } : undefined;

  if (!role) {
    return (
      <RolePicker
        onSelect={(r) => { setPreviousRole(null); setRole(r); }}
        currentRole={previousRole}
        onBack={handleBack}
      />
    );
  }

  // Tier gates
  if (role === 'command' && !tierLoading && !can('command_view')) {
    return (
      <div style={{ width: '100vw', height: '100vh', background: '#080C0A', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: '16px' }}>
        <UpgradePrompt
          feature="Command View"
          requiredTier={requiredTier('command_view')!}
          description="Situation feed, 3-column layout, and handoff brief generator for mission commanders."
        />
        <button
          onClick={() => setRole('ops')}
          style={{
            fontFamily: 'var(--font-ibm-plex-mono), monospace',
            fontSize: '10px',
            letterSpacing: '0.15em',
            color: 'var(--text-dim)',
            background: 'transparent',
            border: '1px solid var(--border)',
            padding: '6px 16px',
            cursor: 'pointer',
          }}
        >
          ← BACK TO OPS
        </button>
      </div>
    );
  }

  if (role === 'dev' && !tierLoading && !can('dev_view')) {
    return (
      <div style={{ width: '100vw', height: '100vh', background: '#080C0A', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: '16px' }}>
        <UpgradePrompt
          feature="Developer View"
          requiredTier={requiredTier('dev_view')!}
          description="Entity explorer, adapter registry, message inspector, and inference dashboard."
        />
        <button
          onClick={() => setRole('ops')}
          style={{
            fontFamily: 'var(--font-ibm-plex-mono), monospace',
            fontSize: '10px',
            letterSpacing: '0.15em',
            color: 'var(--text-dim)',
            background: 'transparent',
            border: '1px solid var(--border)',
            padding: '6px 16px',
            cursor: 'pointer',
          }}
        >
          ← BACK TO OPS
        </button>
      </div>
    );
  }

  if (role === 'ops')     return <OpsLayout     onSwitchRole={handleSwitchRole} />;
  if (role === 'command') return <CommandLayout  onSwitchRole={handleSwitchRole} />;
  if (role === 'dev')     return <DevLayout      onSwitchRole={handleSwitchRole} />;
  return null;
}
