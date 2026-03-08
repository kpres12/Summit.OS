'use client';
import ProtectedRoute from '@/components/ProtectedRoute';
import dynamic from 'next/dynamic';
import { useRole } from '@/hooks/useRole';
import RolePicker from '@/components/RolePicker';
const OpsLayout = dynamic(() => import('@/components/ops/OpsLayout'), { ssr: false });
const CommandLayout = dynamic(() => import('@/components/command/CommandLayout'), { ssr: false });
const DevLayout = dynamic(() => import('@/components/dev/DevLayout'), { ssr: false });

export default function Home() {
  return (
    <ProtectedRoute>
      <RoleRouter />
    </ProtectedRoute>
  );
}

function RoleRouter() {
  const { role, setRole, clearRole } = useRole();
  if (!role) return <RolePicker onSelect={setRole} />;
  if (role === 'ops') return <OpsLayout onSwitchRole={clearRole} />;
  if (role === 'command') return <CommandLayout onSwitchRole={clearRole} />;
  if (role === 'dev') return <DevLayout onSwitchRole={clearRole} />;
  return null;
}
