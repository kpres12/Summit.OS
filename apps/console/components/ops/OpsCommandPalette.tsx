'use client';

/**
 * Command Palette — Summit.OS Ops Console
 *
 * Opens on '/' keypress. Fuzzy-searches available commands.
 * Commands: investigate alert, fly-to entity, dispatch mission,
 *           switch role, open layer, draw geofence, export handoff brief.
 */

import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';

interface Command {
  id: string;
  name: string;
  description: string;
  shortcut?: string;
}

const COMMANDS: Command[] = [
  {
    id: 'investigate',
    name: 'Investigate Alert',
    description: 'Fly to and select the highest-severity alert',
    shortcut: 'I',
  },
  {
    id: 'fly-to',
    name: 'Fly To Entity',
    description: 'Center map on a selected entity',
    shortcut: 'F',
  },
  {
    id: 'dispatch',
    name: 'Dispatch Mission',
    description: 'Dispatch the currently selected entity',
    shortcut: 'D',
  },
  {
    id: 'draw-geofence',
    name: 'Draw Geofence',
    description: 'Enter geofence drawing mode on the map',
  },
  {
    id: 'export-brief',
    name: 'Export Handoff Brief',
    description: 'Generate and download a command handoff brief',
  },
  {
    id: 'switch-ops',
    name: 'Switch to OPS View',
    description: 'Full-screen map operator interface',
    shortcut: '1',
  },
  {
    id: 'switch-command',
    name: 'Switch to COMMAND View',
    description: '3-column situation + map + resource layout',
    shortcut: '2',
  },
  {
    id: 'switch-dev',
    name: 'Switch to DEV View',
    description: 'Entity explorer, adapters, message inspector',
    shortcut: '3',
  },
];

interface OpsCommandPaletteProps {
  isOpen: boolean;
  onClose: () => void;
  onCommand: (cmd: string, params?: Record<string, unknown>) => void;
}

function fuzzyMatch(query: string, command: Command): boolean {
  if (!query) return true;
  const q = query.toLowerCase();
  const haystack = `${command.name} ${command.description}`.toLowerCase();
  // Simple substring fuzzy: all chars of q appear in order in haystack
  let qi = 0;
  for (let i = 0; i < haystack.length && qi < q.length; i++) {
    if (haystack[i] === q[qi]) qi++;
  }
  return qi === q.length;
}

export default function OpsCommandPalette({
  isOpen,
  onClose,
  onCommand,
}: OpsCommandPaletteProps): JSX.Element | null {
  const [query, setQuery] = useState('');
  const [activeIndex, setActiveIndex] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLUListElement>(null);

  const filtered = useMemo(
    () => COMMANDS.filter((c) => fuzzyMatch(query, c)),
    [query],
  );

  // Reset state when opening
  useEffect(() => {
    if (isOpen) {
      setQuery('');
      setActiveIndex(0);
      setTimeout(() => inputRef.current?.focus(), 0);
    }
  }, [isOpen]);

  // Clamp active index when filter changes
  useEffect(() => {
    setActiveIndex((prev) => Math.min(prev, Math.max(0, filtered.length - 1)));
  }, [filtered.length]);

  const execute = useCallback(
    (cmd: Command) => {
      onCommand(cmd.id);
      onClose();
    },
    [onCommand, onClose],
  );

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLInputElement>) => {
      if (e.key === 'Escape') {
        e.preventDefault();
        onClose();
      } else if (e.key === 'ArrowDown') {
        e.preventDefault();
        setActiveIndex((prev) => Math.min(prev + 1, filtered.length - 1));
      } else if (e.key === 'ArrowUp') {
        e.preventDefault();
        setActiveIndex((prev) => Math.max(prev - 1, 0));
      } else if (e.key === 'Enter') {
        e.preventDefault();
        if (filtered[activeIndex]) {
          execute(filtered[activeIndex]);
        }
      }
    },
    [filtered, activeIndex, execute, onClose],
  );

  if (!isOpen) return null;

  return (
    /* Full-screen overlay */
    <div
      role="dialog"
      aria-modal="true"
      aria-label="Command palette"
      className="fixed inset-0 flex items-start justify-center pt-[15vh]"
      style={{ background: 'rgba(0,0,0,0.72)', zIndex: 500 }}
      onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}
    >
      {/* Modal */}
      <div
        style={{
          width: '100%',
          maxWidth: '600px',
          background: 'var(--background-panel)',
          border: '1px solid var(--accent-30)',
          boxShadow: '0 16px 64px rgba(0,0,0,0.8)',
        }}
      >
        {/* Input row */}
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            padding: '10px 14px',
            borderBottom: '1px solid var(--accent-10)',
            gap: '10px',
          }}
        >
          <span
            style={{
              fontFamily: 'var(--font-ibm-plex-mono), monospace',
              fontSize: '12px',
              color: 'var(--accent-30)',
              flexShrink: 0,
            }}
          >
            &gt;
          </span>
          <input
            ref={inputRef}
            type="text"
            value={query}
            onChange={(e) => { setQuery(e.target.value); setActiveIndex(0); }}
            onKeyDown={handleKeyDown}
            placeholder="Type a command..."
            aria-label="Command search"
            style={{
              flex: 1,
              background: 'transparent',
              border: 'none',
              outline: 'none',
              fontFamily: 'var(--font-ibm-plex-mono), monospace',
              fontSize: '13px',
              color: 'var(--accent)',
              letterSpacing: '0.05em',
            }}
          />
          <span
            style={{
              fontFamily: 'var(--font-ibm-plex-mono), monospace',
              fontSize: '9px',
              color: 'var(--text-muted)',
              letterSpacing: '0.1em',
              flexShrink: 0,
            }}
          >
            ESC
          </span>
        </div>

        {/* Results list */}
        <ul
          ref={listRef}
          role="listbox"
          style={{
            listStyle: 'none',
            margin: 0,
            padding: '4px 0',
            maxHeight: '360px',
            overflowY: 'auto',
          }}
        >
          {filtered.length === 0 && (
            <li
              style={{
                padding: '12px 14px',
                fontFamily: 'var(--font-ibm-plex-mono), monospace',
                fontSize: '10px',
                color: 'var(--text-muted)',
                letterSpacing: '0.1em',
              }}
            >
              NO COMMANDS FOUND
            </li>
          )}
          {filtered.map((cmd, idx) => {
            const isActive = idx === activeIndex;
            return (
              <li
                key={cmd.id}
                role="option"
                aria-selected={isActive}
                onClick={() => execute(cmd)}
                onMouseEnter={() => setActiveIndex(idx)}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  padding: '9px 14px',
                  cursor: 'pointer',
                  background: isActive ? 'var(--accent-5)' : 'transparent',
                  borderLeft: isActive ? '2px solid var(--accent)' : '2px solid transparent',
                  transition: 'background 80ms',
                }}
              >
                <div>
                  <div
                    style={{
                      fontFamily: 'var(--font-ibm-plex-mono), monospace',
                      fontSize: '11px',
                      color: isActive ? 'var(--accent)' : 'var(--text-primary, rgba(200,230,201,0.9))',
                      letterSpacing: '0.05em',
                      fontWeight: isActive ? 700 : 400,
                    }}
                  >
                    {cmd.name}
                  </div>
                  <div
                    style={{
                      fontFamily: 'var(--font-ibm-plex-mono), monospace',
                      fontSize: '9px',
                      color: 'var(--text-dim)',
                      letterSpacing: '0.04em',
                      marginTop: '2px',
                    }}
                  >
                    {cmd.description}
                  </div>
                </div>
                {cmd.shortcut && (
                  <span
                    style={{
                      fontFamily: 'var(--font-ibm-plex-mono), monospace',
                      fontSize: '9px',
                      color: isActive ? 'var(--accent)' : 'var(--text-muted)',
                      background: isActive ? 'var(--accent-10)' : 'var(--background)',
                      border: `1px solid ${isActive ? 'var(--accent-30)' : 'var(--border)'}`,
                      padding: '1px 5px',
                      letterSpacing: '0.1em',
                      flexShrink: 0,
                    }}
                  >
                    {cmd.shortcut}
                  </span>
                )}
              </li>
            );
          })}
        </ul>

        {/* Footer hint */}
        <div
          style={{
            padding: '6px 14px',
            borderTop: '1px solid var(--accent-5)',
            display: 'flex',
            gap: '16px',
          }}
        >
          {[
            ['↑↓', 'navigate'],
            ['↵', 'execute'],
            ['ESC', 'close'],
          ].map(([key, label]) => (
            <span
              key={key}
              style={{
                fontFamily: 'var(--font-ibm-plex-mono), monospace',
                fontSize: '9px',
                color: 'var(--text-muted)',
                letterSpacing: '0.05em',
              }}
            >
              <span style={{ color: 'var(--text-dim)' }}>{key}</span>{' '}{label}
            </span>
          ))}
        </div>
      </div>
    </div>
  );
}
