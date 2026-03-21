import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { apiFetch } from '../lib/api';

describe('apiFetch', () => {
  beforeEach(() => {
    vi.stubGlobal('fetch', vi.fn());
    // Clear localStorage mock
    vi.stubGlobal('localStorage', {
      getItem: vi.fn().mockReturnValue(null),
      setItem: vi.fn(),
      removeItem: vi.fn(),
    });
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('returns response on 200', async () => {
    const mockRes = { ok: true, status: 200, json: async () => ({ data: 'ok' }) } as Response;
    vi.mocked(fetch).mockResolvedValueOnce(mockRes);

    const res = await apiFetch('/v1/health');
    expect(res.ok).toBe(true);
    expect(res.status).toBe(200);
  });

  it('dispatches policy-denied event on 403', async () => {
    const mockRes = {
      ok: false,
      status: 403,
      json: async () => ({ detail: { message: 'Policy denied', policy_violations: ['geofence'] } }),
    } as unknown as Response;
    vi.mocked(fetch).mockResolvedValueOnce(mockRes);

    const dispatchSpy = vi.spyOn(window, 'dispatchEvent');

    await apiFetch('/v1/tasks');

    expect(dispatchSpy).toHaveBeenCalledWith(
      expect.objectContaining({ type: 'policy-denied' })
    );
  });

  it('dispatches policy-denied event on 400', async () => {
    const mockRes = {
      ok: false,
      status: 400,
      json: async () => ({ detail: { message: 'Bad request' } }),
    } as unknown as Response;
    vi.mocked(fetch).mockResolvedValueOnce(mockRes);

    const dispatchSpy = vi.spyOn(window, 'dispatchEvent');

    await apiFetch('/v1/missions');

    expect(dispatchSpy).toHaveBeenCalledWith(
      expect.objectContaining({ type: 'policy-denied' })
    );
  });

  it('does not dispatch policy-denied on 404', async () => {
    const mockRes = {
      ok: false,
      status: 404,
      json: async () => ({ detail: 'Not found' }),
    } as unknown as Response;
    vi.mocked(fetch).mockResolvedValueOnce(mockRes);

    const dispatchSpy = vi.spyOn(window, 'dispatchEvent');

    await apiFetch('/v1/unknown');

    const policyEvents = dispatchSpy.mock.calls.filter(
      ([e]) => (e as CustomEvent).type === 'policy-denied'
    );
    expect(policyEvents).toHaveLength(0);
  });

  it('includes Authorization header when token in localStorage', async () => {
    vi.stubGlobal('localStorage', {
      getItem: vi.fn().mockReturnValue('test-token'),
      setItem: vi.fn(),
      removeItem: vi.fn(),
    });
    const mockRes = { ok: true, status: 200, json: async () => ({}) } as Response;
    vi.mocked(fetch).mockResolvedValueOnce(mockRes);

    await apiFetch('/v1/health');

    const [, init] = vi.mocked(fetch).mock.calls[0];
    const headers = init?.headers as Record<string, string>;
    expect(headers['Authorization']).toBe('Bearer test-token');
  });

  it('throws on network error', async () => {
    vi.mocked(fetch).mockRejectedValueOnce(new Error('Network error'));
    await expect(apiFetch('/v1/health')).rejects.toThrow('Network error');
  });
});
