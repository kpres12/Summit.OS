import '@testing-library/jest-dom';

// Vitest 1.x + jsdom 24 passes an invalid --localstorage-file arg that breaks
// the native localStorage implementation. Provide a clean in-memory shim.
const _store: Record<string, string> = {};
const localStorageMock: Storage = {
  getItem: (key: string) => _store[key] ?? null,
  setItem: (key: string, value: string) => { _store[key] = String(value); },
  removeItem: (key: string) => { delete _store[key]; },
  clear: () => { Object.keys(_store).forEach(k => delete _store[k]); },
  get length() { return Object.keys(_store).length; },
  key: (i: number) => Object.keys(_store)[i] ?? null,
};
Object.defineProperty(globalThis, 'localStorage', { value: localStorageMock, writable: true });
