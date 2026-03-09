const BACKEND_ORIGIN = process.env.BACKEND_ORIGIN || 'http://127.0.0.1:8000';

const HOP_BY_HOP_HEADERS = new Set([
  'connection',
  'content-length',
  'host',
  'keep-alive',
  'proxy-authenticate',
  'proxy-authorization',
  'te',
  'trailer',
  'transfer-encoding',
  'upgrade',
]);

export const dynamic = 'force-dynamic';
export const maxDuration = 120;
export const runtime = 'nodejs';

function filterHeaders(source: Headers): Headers {
  const headers = new Headers();

  for (const [key, value] of source.entries()) {
    if (!HOP_BY_HOP_HEADERS.has(key.toLowerCase())) {
      headers.set(key, value);
    }
  }

  return headers;
}

function buildBackendUrl(requestUrl: string, path?: string[]): string {
  const suffix = path && path.length > 0 ? `/${path.join('/')}` : '';
  const incomingUrl = new URL(requestUrl);
  return `${BACKEND_ORIGIN}/api/v1/resumes/improve${suffix}${incomingUrl.search}`;
}

async function proxyImproveRequest(request: Request, path?: string[]): Promise<Response> {
  const body =
    request.method === 'GET' || request.method === 'HEAD' ? undefined : await request.text();

  try {
    const upstream = await fetch(buildBackendUrl(request.url, path), {
      method: request.method,
      headers: filterHeaders(request.headers),
      body,
      cache: 'no-store',
    });

    return new Response(upstream.body, {
      status: upstream.status,
      statusText: upstream.statusText,
      headers: filterHeaders(upstream.headers),
    });
  } catch (error) {
    console.error('Failed to proxy improve request to backend:', error);
    return Response.json(
      { detail: 'Failed to reach resume backend. Please try again.' },
      { status: 502 }
    );
  }
}

type ImproveRouteContext = {
  params: Promise<{
    path?: string[];
  }>;
};

export async function POST(request: Request, { params }: ImproveRouteContext): Promise<Response> {
  const { path } = await params;
  return proxyImproveRequest(request, path);
}
