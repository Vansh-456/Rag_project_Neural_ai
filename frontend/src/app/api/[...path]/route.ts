// Server-side proxy: /api/<path> -> ${API_URL}/<path>. Adds API basic-auth credentials from
// server env and streams responses through (needed for Server-Sent Events).
import type { NextRequest } from "next/server";

export const dynamic = "force-dynamic";

const API_URL = process.env.API_URL ?? "http://localhost:8000";

function authHeader(): Record<string, string> {
  const user = process.env.API_USERNAME;
  const pass = process.env.API_PASSWORD;
  if (!user || !pass) return {};
  return { Authorization: `Basic ${Buffer.from(`${user}:${pass}`).toString("base64")}` };
}

async function proxy(req: NextRequest, { params }: { params: { path: string[] } }) {
  const target = new URL(`${API_URL.replace(/\/$/, "")}/${params.path.map(encodeURIComponent).join("/")}`);
  target.search = req.nextUrl.search;
  const headers: Record<string, string> = { ...authHeader() };
  const ct = req.headers.get("content-type");
  if (ct) headers["content-type"] = ct;
  const fwd = req.headers.get("x-forwarded-for");
  if (fwd) headers["x-forwarded-for"] = fwd;
  const init: RequestInit & { duplex?: string } = { method: req.method, headers, cache: "no-store" };
  if (req.method !== "GET" && req.method !== "HEAD") {
    init.body = req.body;
    init.duplex = "half";
  }
  try {
    const res = await fetch(target, init);
    const out = new Headers();
    for (const h of ["content-type", "cache-control", "retry-after"]) {
      const v = res.headers.get(h);
      if (v) out.set(h, v);
    }
    return new Response(res.body, { status: res.status, headers: out });
  } catch (err) {
    return Response.json({ detail: `API unreachable at ${API_URL}: ${String(err)}` }, { status: 502 });
  }
}

export { proxy as GET, proxy as POST };
