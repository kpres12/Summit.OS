/**
 * Tiny module-level event bus for broadcasting OSINT search results
 * from OpsIntel (WEB tab) into CommandLayout's SituationFeed.
 *
 * No external dependency — just a Set of listener callbacks.
 */

export interface OsintEvent {
  query: string;
  topSnippet: string;    // first result snippet, used as the feed description
  source: string;        // domain of first result
  resultCount: number;
}

type Listener = (ev: OsintEvent) => void;

const listeners = new Set<Listener>();

export function subscribeOsint(fn: Listener): () => void {
  listeners.add(fn);
  return () => listeners.delete(fn);
}

export function publishOsint(ev: OsintEvent) {
  listeners.forEach((fn) => fn(ev));
}
