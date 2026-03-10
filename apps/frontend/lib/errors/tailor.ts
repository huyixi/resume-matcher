import { ImproveRequestError } from '@/lib/api/resume';

export function getTailorErrorKey(error: unknown): string {
  if (error instanceof ImproveRequestError) {
    switch (error.errorCode) {
      case 'api_key_missing':
      case 'auth_failed':
        return 'tailor.errors.apiKeyError';
      case 'rate_limited':
        return 'tailor.errors.rateLimit';
      case 'provider_unavailable':
      case 'provider_timeout':
        return 'tailor.errors.llmUnavailable';
      case 'invalid_model_response':
        return 'tailor.errors.invalidModelResponse';
      default:
        break;
    }

    if (error.status === 401) {
      return 'tailor.errors.apiKeyError';
    }
    if (error.status === 429) {
      return 'tailor.errors.rateLimit';
    }
    if (error.status === 502 || error.status === 503 || error.status === 504) {
      return 'tailor.errors.llmUnavailable';
    }
  }

  const message = error instanceof Error ? error.message.toLowerCase() : '';
  if (
    message.includes('api key') ||
    message.includes('unauthorized') ||
    message.includes('authentication') ||
    message.includes('401')
  ) {
    return 'tailor.errors.apiKeyError';
  }
  if (message.includes('rate limit') || message.includes('429')) {
    return 'tailor.errors.rateLimit';
  }
  if (
    message.includes('provider unavailable') ||
    message.includes('timed out') ||
    message.includes('timeout') ||
    message.includes('502') ||
    message.includes('503') ||
    message.includes('504')
  ) {
    return 'tailor.errors.llmUnavailable';
  }
  if (message.includes('invalid response') || message.includes('json')) {
    return 'tailor.errors.invalidModelResponse';
  }
  return 'tailor.errors.failedToPreview';
}
