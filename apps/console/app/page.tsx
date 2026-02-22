import TacticalLayoutClient from '@/components/tactical/TacticalLayoutClient';
import ProtectedRoute from '@/components/ProtectedRoute';

export default function Home() {
  return (
    <ProtectedRoute>
      <TacticalLayoutClient />
    </ProtectedRoute>
  );
}
