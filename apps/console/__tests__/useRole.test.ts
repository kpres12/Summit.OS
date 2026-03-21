import { describe, it, expect, beforeEach, vi } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { useRole } from '../hooks/useRole';

describe('useRole', () => {
  beforeEach(() => {
    localStorage.clear();
    vi.clearAllMocks();
  });

  it('defaults to null when no stored role', () => {
    const { result } = renderHook(() => useRole());
    // Initial state before useEffect fires
    expect(result.current.role).toBeNull();
  });

  it('loads stored role from localStorage', () => {
    localStorage.setItem('summit_role', 'ops');
    const { result } = renderHook(() => useRole());
    act(() => {});
    expect(result.current.role).toBe('ops');
  });

  it('setRole persists to localStorage and updates state', () => {
    const { result } = renderHook(() => useRole());
    act(() => {
      result.current.setRole('command');
    });
    expect(result.current.role).toBe('command');
    expect(localStorage.getItem('summit_role')).toBe('command');
  });

  it('clearRole removes from localStorage and sets null', () => {
    localStorage.setItem('summit_role', 'dev');
    const { result } = renderHook(() => useRole());
    act(() => {
      result.current.clearRole();
    });
    expect(result.current.role).toBeNull();
    expect(localStorage.getItem('summit_role')).toBeNull();
  });

  it('ignores invalid stored role values', () => {
    localStorage.setItem('summit_role', 'invalid_role');
    const { result } = renderHook(() => useRole());
    act(() => {});
    expect(result.current.role).toBeNull();
  });
});
