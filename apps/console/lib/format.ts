/**
 * Summit.OS — shared formatting utilities.
 *
 * Centralises helpers that were duplicated across 4+ component files.
 * All color helpers return CSS variable references so they respond to
 * the active DomainConfig palette automatically.
 */

// ── Relative-time formatting ────────────────────────────────────────────────

/** Compact relative-time string from a unix-epoch *seconds* timestamp. */
export function ageFromEpoch(epochSecs: number): string {
  const diff = Math.floor(Date.now() / 1000 - epochSecs);
  if (diff < 0) return 'now';
  if (diff < 60) return `${diff}s ago`;
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  return `${Math.floor(diff / 86400)}d ago`;
}

/** Compact relative-time string from an ISO-8601 date string. */
export function ageFromISO(isoString: string | null): string {
  if (!isoString) return '—';
  const ts = new Date(isoString).getTime();
  const diff = Math.floor((Date.now() - ts) / 1000);
  if (diff < 0) return 'now';
  if (diff < 60) return `${diff}s ago`;
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  return `${Math.floor(diff / 86400)}d ago`;
}

/** Terse relative-time (no "ago" suffix) for tight layouts. */
export function ageTerse(epochSecs: number): string {
  const diff = Math.floor(Date.now() / 1000 - epochSecs);
  if (diff < 0) return '0s';
  if (diff < 60) return `${diff}s`;
  if (diff < 3600) return `${Math.floor(diff / 60)}m`;
  return `${Math.floor(diff / 3600)}h`;
}

// ── Color helpers (return CSS variable references) ──────────────────────────

export function entityTypeColor(type: string): string {
  switch (type) {
    case 'active':
      return 'var(--accent)';
    case 'alert':
      return 'var(--critical)';
    case 'neutral':
      return 'var(--text-dim)';
    default:
      return 'var(--warning)';
  }
}

export function batteryColor(pct: number): string {
  if (pct > 40) return 'var(--accent)';
  if (pct > 20) return 'var(--warning)';
  return 'var(--critical)';
}

export function severityColor(severity: string): string {
  const s = severity.toUpperCase();
  if (s === 'CRITICAL') return 'var(--critical)';
  if (s === 'HIGH') return 'var(--critical-dim, rgba(255,59,59,0.8))';
  if (s === 'MED' || s === 'MEDIUM') return 'var(--warning)';
  return 'var(--text-dim)';
}

export function statusColor(status: string): string {
  switch (status.toUpperCase()) {
    case 'ACTIVE':
      return 'var(--accent)';
    case 'PLANNING':
      return 'var(--text-dim)';
    case 'FAILED':
      return 'var(--critical)';
    case 'COMPLETED':
      return 'var(--color-active)';
    default:
      return 'var(--text-dim)';
  }
}

// ── Domain tag abbreviation ─────────────────────────────────────────────────

export function domainTag(domain: string): string {
  switch (domain) {
    case 'aerial':
      return 'UAV';
    case 'ground':
      return 'GND';
    case 'maritime':
      return 'MAR';
    case 'fixed':
      return 'FIX';
    case 'sensor':
      return 'SEN';
    default:
      return domain.slice(0, 3).toUpperCase();
  }
}
