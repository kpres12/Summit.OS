/**
 * useTier — subscription tier context for feature gating.
 *
 * Usage:
 *   const { tier, can } = useTier();
 *   if (!can('tasking')) return <UpgradePrompt feature="Mission Tasking" requiredTier="pro" />;
 *
 * Tier hierarchy: free < pro < org < enterprise
 *
 * The subscription is fetched once from /v1/billing/subscription and cached
 * in memory for the session. No polling — refresh on page load is fine.
 */

import { useEffect, useState } from 'react';
import { apiFetch } from '@/lib/api';

export type Tier = 'free' | 'pro' | 'org' | 'enterprise';

const TIER_RANK: Record<Tier, number> = {
  free: 0,
  pro: 1,
  org: 2,
  enterprise: 3,
};

// Feature → minimum tier required
const FEATURE_GATES: Record<string, Tier> = {
  // OPS view features
  map: 'free',
  alerts: 'free',
  entities: 'free',
  detection_feed: 'free',

  // Pro features
  command_view: 'pro',
  tasking: 'pro',
  missions: 'pro',
  fusion: 'pro',
  webhooks: 'pro',

  // Org features
  dev_view: 'org',
  adapter_registry: 'org',
  rbac: 'org',
  audit_log: 'org',
  api_keys: 'org',

  // Enterprise features
  multi_tenant: 'enterprise',
  sso: 'enterprise',
  air_gapped: 'enterprise',
};

export interface TierState {
  tier: Tier;
  entityLimit: number;  // -1 = unlimited
  operatorLimit: number;
  subscriptionStatus: string;
  loading: boolean;
  /** Returns true if the current tier can use the named feature. */
  can: (feature: string) => boolean;
  /** Returns the minimum tier required for a feature, or null if it's free. */
  requiredTier: (feature: string) => Tier | null;
}

const DEFAULT_STATE: TierState = {
  tier: 'free',
  entityLimit: 10,
  operatorLimit: 1,
  subscriptionStatus: 'active',
  loading: true,
  can: () => false,
  requiredTier: () => null,
};

// Module-level cache — shared across hook instances in the same session
let _cached: { tier: Tier; entityLimit: number; operatorLimit: number; subscriptionStatus: string } | null = null;

export function useTier(): TierState {
  const [state, setState] = useState<Omit<TierState, 'can' | 'requiredTier'>>(() => ({
    tier: _cached?.tier ?? 'free',
    entityLimit: _cached?.entityLimit ?? 10,
    operatorLimit: _cached?.operatorLimit ?? 1,
    subscriptionStatus: _cached?.subscriptionStatus ?? 'active',
    loading: _cached === null,
  }));

  useEffect(() => {
    if (_cached !== null) return; // already have it

    apiFetch('/v1/billing/subscription')
      .then(r => (r.ok ? r.json() : null))
      .then((data: { tier: string; entity_limit: number; operator_limit: number; subscription_status: string } | null) => {
        const resolved = {
          tier: (data?.tier ?? 'free') as Tier,
          entityLimit: data?.entity_limit ?? 10,
          operatorLimit: data?.operator_limit ?? 1,
          subscriptionStatus: data?.subscription_status ?? 'active',
        };
        _cached = resolved;
        setState({ ...resolved, loading: false });
      })
      .catch(() => {
        // Network error — fall back to free tier (fail-safe)
        setState(s => ({ ...s, loading: false }));
      });
  }, []);

  function can(feature: string): boolean {
    const required = FEATURE_GATES[feature];
    if (!required) return true; // unlisted features are free
    return TIER_RANK[state.tier] >= TIER_RANK[required];
  }

  function requiredTier(feature: string): Tier | null {
    const required = FEATURE_GATES[feature];
    if (!required || TIER_RANK[state.tier] >= TIER_RANK[required]) return null;
    return required;
  }

  return { ...state, can, requiredTier };
}
