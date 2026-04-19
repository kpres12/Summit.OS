'use client';

import { useEffect, useState } from 'react';
import { apiFetch } from '@/lib/api';

// ── Types ─────────────────────────────────────────────────────────────────────

type Tier = 'free' | 'pro' | 'org' | 'enterprise';

interface Subscription {
  org_id: string;
  tier: Tier;
  subscription_status: string;
  entity_limit: number;   // -1 = unlimited
  operator_limit: number; // -1 = unlimited
}

const TIER_LABELS: Record<Tier, string> = {
  free: 'Community',
  pro: 'Pro',
  org: 'Organization',
  enterprise: 'Enterprise',
};

const TIER_COLOR: Record<Tier, string> = {
  free: '#4FC3F7',
  pro: '#00FF9C',
  org: '#FFB300',
  enterprise: '#FF3B3B',
};

const PLANS: {
  tier: Tier;
  price: string;
  period: string;
  headline: string;
  limits: string;
  features: string[];
  cta: string;
  highlighted: boolean;
}[] = [
  {
    tier: 'free',
    price: '$0',
    period: 'forever',
    headline: 'Community',
    limits: '10 entities · 1 operator',
    features: [
      'Real-time map (OPS view)',
      'Alert queue + entity detail',
      'Detection feed',
      'Open-source self-host',
      'Community support',
    ],
    cta: 'Current plan',
    highlighted: false,
  },
  {
    tier: 'pro',
    price: '$49',
    period: 'per month',
    headline: 'Pro',
    limits: '500 entities · 5 operators',
    features: [
      'Everything in Community',
      'COMMAND view',
      'Mission planning + tasking',
      'Sensor fusion pipeline',
      'Webhook integrations',
      'Email support',
    ],
    cta: 'Upgrade to Pro',
    highlighted: true,
  },
  {
    tier: 'org',
    price: '$199',
    period: 'per month',
    headline: 'Organization',
    limits: '5,000 entities · unlimited operators',
    features: [
      'Everything in Pro',
      'DEV view + adapter registry',
      'Multi-user RBAC',
      'Audit log (90-day retention)',
      'API key management',
      'Priority support',
    ],
    cta: 'Upgrade to Org',
    highlighted: false,
  },
  {
    tier: 'enterprise',
    price: 'Custom',
    period: '',
    headline: 'Enterprise',
    limits: 'Unlimited entities · unlimited operators',
    features: [
      'Everything in Organization',
      'Multi-tenant deployment',
      'SSO / SAML / OIDC',
      'On-prem or air-gapped deploy',
      'SLA + dedicated support',
      'Custom domain adapters',
    ],
    cta: 'Contact us',
    highlighted: false,
  },
];

// ── Helpers ───────────────────────────────────────────────────────────────────

function fmt(n: number) {
  return n === -1 ? '∞' : n.toLocaleString();
}

// ── Component ─────────────────────────────────────────────────────────────────

export default function BillingPage() {
  const [sub, setSub] = useState<Subscription | null>(null);
  const [loading, setLoading] = useState(true);
  const [checkoutLoading, setCheckoutLoading] = useState<Tier | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    apiFetch('/v1/billing/subscription')
      .then(r => (r.ok ? r.json() : null))
      .then((data: Subscription | null) => {
        if (data) setSub(data);
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, []);

  async function handleUpgrade(tier: Tier) {
    if (tier === 'enterprise') {
      window.location.href = 'mailto:sales@branca.ai?subject=Heli.OS%20Enterprise';
      return;
    }
    if (!sub) return;
    setCheckoutLoading(tier);
    setError(null);
    try {
      const res = await apiFetch('/v1/billing/checkout/polar', {
        method: 'POST',
        body: JSON.stringify({
          org_id: sub.org_id,
          tier,
          success_url: `${window.location.origin}/billing?upgraded=1`,
          cancel_url: `${window.location.origin}/billing`,
        }),
      });
      if (!res.ok) throw new Error('Checkout failed');
      const data = await res.json();
      window.location.href = data.checkout_url;
    } catch {
      setError('Could not start checkout. Please try again.');
    } finally {
      setCheckoutLoading(null);
    }
  }

  const currentTier = (sub?.tier ?? 'free') as Tier;

  return (
    <div style={{
      minHeight: '100vh',
      background: '#080C0A',
      color: '#E8F5E9',
      fontFamily: 'var(--font-ibm-plex-mono), monospace',
      padding: '48px 24px',
    }}>
      {/* Header */}
      <div style={{ maxWidth: 960, margin: '0 auto' }}>
        <div style={{ marginBottom: 8, color: '#4FC3F7', fontSize: 11, letterSpacing: 2, textTransform: 'uppercase' }}>
          Heli.OS
        </div>
        <h1 style={{
          fontFamily: 'var(--font-orbitron), sans-serif',
          fontSize: 28,
          fontWeight: 700,
          color: '#00FF9C',
          margin: '0 0 8px',
          letterSpacing: 1,
        }}>
          Plans &amp; Billing
        </h1>
        <p style={{ color: '#8A9A8E', fontSize: 13, margin: '0 0 40px' }}>
          Civilian UAV fleet ops · disaster response · wildfire coordination
        </p>

        {/* Current plan badge */}
        {!loading && sub && (
          <div style={{
            display: 'inline-flex',
            alignItems: 'center',
            gap: 12,
            background: '#0D1410',
            border: `1px solid ${TIER_COLOR[currentTier]}40`,
            borderRadius: 6,
            padding: '10px 16px',
            marginBottom: 40,
            fontSize: 12,
          }}>
            <span style={{ color: '#8A9A8E' }}>Current plan</span>
            <span style={{
              color: TIER_COLOR[currentTier],
              fontFamily: 'var(--font-orbitron), sans-serif',
              fontWeight: 700,
              fontSize: 11,
              letterSpacing: 1,
              textTransform: 'uppercase',
            }}>
              {TIER_LABELS[currentTier]}
            </span>
            <span style={{ color: '#4A5A4E', margin: '0 4px' }}>·</span>
            <span style={{ color: '#8A9A8E' }}>
              {fmt(sub.entity_limit)} entities
            </span>
            <span style={{ color: '#4A5A4E', margin: '0 4px' }}>·</span>
            <span style={{ color: '#8A9A8E' }}>
              {fmt(sub.operator_limit)} operators
            </span>
            <span style={{ color: '#4A5A4E', margin: '0 4px' }}>·</span>
            <span style={{
              color: sub.subscription_status === 'active' ? '#00FF9C' : '#FFB300',
              fontSize: 11,
            }}>
              {sub.subscription_status}
            </span>
          </div>
        )}

        {error && (
          <div style={{
            background: '#FF3B3B18',
            border: '1px solid #FF3B3B40',
            borderRadius: 6,
            padding: '10px 16px',
            marginBottom: 24,
            color: '#FF3B3B',
            fontSize: 12,
          }}>
            {error}
          </div>
        )}

        {/* Pricing grid */}
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(210px, 1fr))',
          gap: 16,
        }}>
          {PLANS.map(plan => {
            const isCurrent = plan.tier === currentTier;
            const isDowngrade = (
              ['free', 'pro', 'org', 'enterprise'].indexOf(plan.tier) <
              ['free', 'pro', 'org', 'enterprise'].indexOf(currentTier)
            );
            const accent = TIER_COLOR[plan.tier];

            return (
              <div key={plan.tier} style={{
                background: plan.highlighted ? '#0D1C13' : '#0A0F0C',
                border: `1px solid ${plan.highlighted ? '#00FF9C30' : '#1A2E1E'}`,
                borderRadius: 8,
                padding: '24px 20px',
                display: 'flex',
                flexDirection: 'column',
                gap: 16,
                position: 'relative',
                boxShadow: plan.highlighted ? '0 0 24px #00FF9C0A' : 'none',
              }}>
                {plan.highlighted && (
                  <div style={{
                    position: 'absolute',
                    top: -1,
                    left: 20,
                    background: '#00FF9C',
                    color: '#080C0A',
                    fontSize: 9,
                    fontWeight: 700,
                    letterSpacing: 2,
                    padding: '2px 8px',
                    borderRadius: '0 0 4px 4px',
                    fontFamily: 'var(--font-orbitron), sans-serif',
                  }}>
                    POPULAR
                  </div>
                )}

                {/* Tier name */}
                <div>
                  <div style={{
                    fontFamily: 'var(--font-orbitron), sans-serif',
                    fontSize: 13,
                    fontWeight: 700,
                    color: accent,
                    letterSpacing: 1,
                    textTransform: 'uppercase',
                    marginBottom: 4,
                  }}>
                    {plan.headline}
                  </div>
                  <div style={{ color: '#4A5A4E', fontSize: 10 }}>{plan.limits}</div>
                </div>

                {/* Price */}
                <div>
                  <span style={{ fontSize: 28, fontWeight: 700, color: '#E8F5E9' }}>
                    {plan.price}
                  </span>
                  {plan.period && (
                    <span style={{ fontSize: 11, color: '#4A5A4E', marginLeft: 4 }}>
                      /{plan.period}
                    </span>
                  )}
                </div>

                {/* Features */}
                <ul style={{ listStyle: 'none', padding: 0, margin: 0, display: 'flex', flexDirection: 'column', gap: 6, flex: 1 }}>
                  {plan.features.map(f => (
                    <li key={f} style={{ fontSize: 11, color: '#8A9A8E', display: 'flex', alignItems: 'flex-start', gap: 6 }}>
                      <span style={{ color: accent, marginTop: 1 }}>›</span>
                      {f}
                    </li>
                  ))}
                </ul>

                {/* CTA */}
                <button
                  disabled={isCurrent || isDowngrade || checkoutLoading === plan.tier}
                  onClick={() => handleUpgrade(plan.tier)}
                  style={{
                    background: isCurrent ? 'transparent' : plan.highlighted ? '#00FF9C' : '#0D1C13',
                    border: `1px solid ${isCurrent ? '#1A2E1E' : accent}`,
                    borderRadius: 4,
                    padding: '9px 0',
                    color: isCurrent ? '#4A5A4E' : plan.highlighted ? '#080C0A' : accent,
                    fontSize: 11,
                    fontFamily: 'var(--font-orbitron), sans-serif',
                    fontWeight: 700,
                    letterSpacing: 1,
                    cursor: isCurrent || isDowngrade ? 'default' : 'pointer',
                    opacity: isDowngrade ? 0.35 : 1,
                    transition: 'opacity 0.15s',
                    width: '100%',
                  }}
                >
                  {checkoutLoading === plan.tier
                    ? 'Redirecting...'
                    : isCurrent
                    ? 'Current plan'
                    : plan.cta}
                </button>
              </div>
            );
          })}
        </div>

        {/* Footer note */}
        <p style={{ textAlign: 'center', color: '#4A5A4E', fontSize: 11, marginTop: 32 }}>
          Payments processed by Polar.sh. Cancel anytime. No contracts.
          Enterprise plans billed annually with custom SLA.
        </p>
      </div>
    </div>
  );
}
