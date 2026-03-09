import { afterEach, describe, expect, it, vi } from 'vitest';
import { POST } from '@/app/api/v1/resumes/improve/[[...path]]/route';

describe('improve proxy route', () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('forwards preview requests to the backend improve endpoint', async () => {
    const upstreamResponse = new Response(JSON.stringify({ ok: true }), {
      status: 200,
      headers: {
        'content-type': 'application/json',
        connection: 'keep-alive',
      },
    });
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockResolvedValue(upstreamResponse);
    const request = new Request('http://127.0.0.1:3000/api/v1/resumes/improve/preview', {
      method: 'POST',
      headers: {
        'content-type': 'application/json',
        connection: 'keep-alive',
      },
      body: JSON.stringify({ resume_id: 'resume-1', job_id: 'job-1' }),
    });

    const response = await POST(request, {
      params: Promise.resolve({ path: ['preview'] }),
    });

    expect(fetchMock).toHaveBeenCalledWith(
      'http://127.0.0.1:8000/api/v1/resumes/improve/preview',
      expect.objectContaining({
        method: 'POST',
        cache: 'no-store',
      })
    );
    const forwardedHeaders = fetchMock.mock.calls[0]?.[1]?.headers;
    expect(forwardedHeaders).toBeInstanceOf(Headers);
    expect((forwardedHeaders as Headers).get('content-type')).toBe('application/json');
    expect((forwardedHeaders as Headers).has('connection')).toBe(false);

    expect(response.status).toBe(200);
    expect(response.headers.get('content-type')).toBe('application/json');
    expect(response.headers.has('connection')).toBe(false);
    await expect(response.json()).resolves.toEqual({ ok: true });
  });
});
