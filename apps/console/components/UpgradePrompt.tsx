'use client';

import Link from 'next/link';
import type { Tier } from '@/hooks/useTier';

const TIER_LABELS: Record<Tier, string> = {
  free: 'Community',
  pro: 'Pro',
  org: 'Organization',
  enterprise: 'Enterprise',
};

const TIER_COLOR: Record<Tier, string> = {
  free: '#4FC3F7',
  pro: '#00E896',
  org: '#FFB300',
  enterprise: '#FF3B3B',
};

interface Props {
  feature: string;
  requiredTier: Tier;
  /** Optional short description of what the feature does */
  description?: string;
  /** Fill the parent container instead of just showing a badge */
  fill?: boolean;
}

/**
 * UpgradePrompt — drop-in paywall for tier-gated features.
 *
 * Usage:
 *   const { can, requiredTier } = useTier();
 *   if (!can('tasking')) {
 *     return <UpgradePrompt feature="Mission Tasking" requiredTier={requiredTier('tasking')!} fill />;
 *   }
 */
export default function UpgradePrompt({ feature, requiredTier, description, fill }: Props) {
  const color = TIER_COLOR[requiredTier];
  const label = TIER_LABELS[requiredTier];

  const inner = (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      gap: 12,
      padding: '32px 24px',
      textAlign: 'center',
    }}>
      {/* Lock icon */}
      <div style={{
        width: 40,
        height: 40,
        borderRadius: '50%',
        border: `1px solid ${color}30`,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        fontSize: 18,
        color,
      }}>
        ⬡
      </div>

      {/* Tier badge */}
      <div style={{
        background: `${color}15`,
        border: `1px solid ${color}40`,
        borderRadius: 4,
        padding: '3px 10px',
        fontSize: 9,
        fontFamily: 'var(--font-ibm-plex-mono), monospace',
        fontWeight: 700,
        letterSpacing: 2,
        color,
        textTransform: 'uppercase',
      }}>
        {label} feature
      </div>

      <div style={{ color: '#E8F5E9', fontSize: 13, fontWeight: 500 }}>
        {feature}
      </div>

      {description && (
        <div style={{ color: '#4A5A4E', fontSize: 11, maxWidth: 240 }}>
          {description}
        </div>
      )}

      <Link href="/billing" style={{
        background: color,
        color: '#080C0A',
        border: 'none',
        borderRadius: 4,
        padding: '8px 20px',
        fontSize: 11,
        fontFamily: 'var(--font-ibm-plex-mono), monospace',
        fontWeight: 700,
        letterSpacing: 1,
        cursor: 'pointer',
        textDecoration: 'none',
        display: 'inline-block',
        marginTop: 4,
      }}>
        Upgrade to {label}
      </Link>
    </div>
  );

  if (fill) {
    return (
      <div style={{
        width: '100%',
        height: '100%',
        minHeight: 200,
        background: '#080C0A',
        border: `1px solid ${color}15`,
        borderRadius: 8,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
      }}>
        {inner}
      </div>
    );
  }

  return inner;
}
