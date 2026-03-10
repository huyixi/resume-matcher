import { afterEach, describe, expect, it, vi } from 'vitest';
import { ImproveRequestError, previewImproveResume } from '@/lib/api/resume';
import { getTailorErrorKey } from '@/lib/errors/tailor';

describe('tailor error handling', () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('parses structured improve errors from the API', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response(
        JSON.stringify({
          detail: 'The LLM provider rate limit was reached. Please try again shortly.',
          error_code: 'rate_limited',
        }),
        {
          status: 429,
          headers: {
            'content-type': 'application/json',
          },
        }
      )
    );

    await expect(previewImproveResume('resume-1', 'job-1')).rejects.toMatchObject({
      name: 'ImproveRequestError',
      status: 429,
      errorCode: 'rate_limited',
      message: 'The LLM provider rate limit was reached. Please try again shortly.',
    });
  });

  it('maps structured provider errors to stable translation keys', () => {
    const error = new ImproveRequestError(
      502,
      'The LLM provider returned an invalid response. Please try again or switch models.',
      'invalid_model_response'
    );

    expect(getTailorErrorKey(error)).toBe('tailor.errors.invalidModelResponse');
  });

  it('falls back to heuristic mapping for generic errors', () => {
    expect(getTailorErrorKey(new Error('request timed out with status 504'))).toBe(
      'tailor.errors.llmUnavailable'
    );
  });
});
