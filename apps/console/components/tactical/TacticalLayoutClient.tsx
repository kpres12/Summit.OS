'use client';

import dynamic from 'next/dynamic';

const TacticalLayout = dynamic(() => import('./TacticalLayout'), { ssr: false });

export default function TacticalLayoutClient() {
  return <TacticalLayout />;
}
