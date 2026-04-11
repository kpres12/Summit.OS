'use client';
import { useState, useEffect } from 'react';
export type Role = 'ops' | 'command' | 'dev';
export function useRole() {
  const [role, setRoleState] = useState<Role | null>(null);
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    const stored = localStorage.getItem('summit_role') as Role | null;
    if (stored && ['ops', 'command', 'dev'].includes(stored)) {
      setRoleState(stored);
    } else {
      // Default to OPS — no picker on first load
      setRoleState('ops');
    }
    setLoaded(true);
  }, []);

  const setRole = (r: Role) => {
    localStorage.setItem('summit_role', r);
    setRoleState(r);
  };
  const clearRole = () => {
    localStorage.removeItem('summit_role');
    setRoleState(null);
  };
  return { role, setRole, clearRole, loaded };
}
