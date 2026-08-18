/**
 * Proxy to the Custodia service.
 *
 * The browser talks to the Next server, which forwards to Python. Keeps the
 * service address out of the client bundle and means the same build works in
 * development and inside compose.
 */

import { NextRequest } from "next/server";

const UPSTREAM = process.env.CUSTODIA_API_URL ?? "http://127.0.0.1:8080";

export const dynamic = "force-dynamic";

async function forward(req: NextRequest, path: string[]) {
  const url = new URL(`${UPSTREAM}/${path.join("/")}`);
  req.nextUrl.searchParams.forEach((value, key) => url.searchParams.set(key, value));

  const init: RequestInit = {
    method: req.method,
    headers: { "content-type": req.headers.get("content-type") ?? "application/json" },
    cache: "no-store",
  };
  if (req.method !== "GET" && req.method !== "HEAD") {
    init.body = await req.text();
  }

  try {
    const upstream = await fetch(url, init);
    const body = await upstream.text();
    return new Response(body, {
      status: upstream.status,
      headers: { "content-type": upstream.headers.get("content-type") ?? "application/json" },
    });
  } catch {
    return Response.json(
      { detail: `Custodia service unreachable at ${UPSTREAM}. Start it with: custodia serve` },
      { status: 503 },
    );
  }
}

type Ctx = { params: Promise<{ path: string[] }> };

export async function GET(req: NextRequest, ctx: Ctx) {
  return forward(req, (await ctx.params).path);
}
export async function POST(req: NextRequest, ctx: Ctx) {
  return forward(req, (await ctx.params).path);
}
