import dynamic from 'next/dynamic';

const TacticalLayout = dynamic(() => import('@/components/tactical/TacticalLayout'), {
  ssr: false,
});

export default function Home() {
  return <TacticalLayout />;
}
