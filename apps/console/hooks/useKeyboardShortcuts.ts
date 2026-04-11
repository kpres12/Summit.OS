'use client';

/**
 * Keyboard shortcuts for the ops console.
 *
 * Shortcuts:
 *   /           — open command palette
 *   Escape      — close command palette / deselect entity
 *   Space       — acknowledge focused alert
 *   F           — fly-to selected entity
 *   D           — dispatch (if entity selected)
 *   1/2/3       — switch to ops/command/dev role
 *   ?           — show shortcuts help
 */

import { useEffect } from 'react';

interface ShortcutHandlers {
  onOpenPalette?: () => void;
  onEscape?: () => void;
  onAck?: () => void;
  onFlyTo?: () => void;
  onDispatch?: () => void;
  onSwitchRole?: (role: 'ops' | 'command' | 'dev') => void;
  onHelp?: () => void;
}

function isInputFocused(): boolean {
  const tag = (document.activeElement?.tagName ?? '').toLowerCase();
  return tag === 'input' || tag === 'textarea' || tag === 'select';
}

export function useKeyboardShortcuts(handlers: ShortcutHandlers): void {
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Never intercept when typing in a form element
      if (isInputFocused()) return;

      const key = e.key;

      switch (key) {
        case '/':
          e.preventDefault();
          handlers.onOpenPalette?.();
          break;

        case 'Escape':
          handlers.onEscape?.();
          break;

        case ' ':
          e.preventDefault();
          handlers.onAck?.();
          break;

        case 'f':
        case 'F':
          handlers.onFlyTo?.();
          break;

        case 'd':
        case 'D':
          handlers.onDispatch?.();
          break;

        case '1':
          handlers.onSwitchRole?.('ops');
          break;

        case '2':
          handlers.onSwitchRole?.('command');
          break;

        case '3':
          handlers.onSwitchRole?.('dev');
          break;

        case '?':
          handlers.onHelp?.();
          break;

        default:
          break;
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => {
      document.removeEventListener('keydown', handleKeyDown);
    };
  }, [handlers]);
}
