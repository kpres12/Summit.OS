'use client';

import React, { Component, ErrorInfo, ReactNode } from 'react';

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

export default class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error('[Summit.OS] ErrorBoundary caught:', error, info.componentStack);
  }

  render() {
    if (this.state.hasError) {
      if (this.props.fallback) return this.props.fallback;
      return (
        <div
          style={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            height: '100%',
            minHeight: '200px',
            background: '#080C0A',
            gap: '12px',
            fontFamily: 'var(--font-ibm-plex-mono), monospace',
          }}
        >
          <span
            style={{
              fontFamily: 'var(--font-orbitron), Orbitron, sans-serif',
              color: '#FF3B3B',
              fontSize: '10px',
              letterSpacing: '0.2em',
            }}
          >
            COMPONENT ERROR
          </span>
          <span style={{ color: 'rgba(255,59,59,0.6)', fontSize: '9px', maxWidth: '320px', textAlign: 'center' }}>
            {this.state.error?.message ?? 'An unexpected error occurred in this panel.'}
          </span>
          <button
            onClick={() => this.setState({ hasError: false, error: null })}
            style={{
              padding: '6px 16px',
              background: 'transparent',
              border: '1px solid rgba(255,59,59,0.35)',
              color: '#FF3B3B',
              fontFamily: 'var(--font-ibm-plex-mono), monospace',
              fontSize: '9px',
              letterSpacing: '0.1em',
              cursor: 'pointer',
            }}
          >
            RETRY
          </button>
        </div>
      );
    }

    return this.props.children;
  }
}
