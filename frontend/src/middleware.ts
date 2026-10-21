// Optional HTTP basic auth for the whole UI (enabled when UI_BASIC_AUTH_USER is set).
import { NextResponse, type NextRequest } from "next/server";

export function middleware(req: NextRequest) {
  const user = process.env.UI_BASIC_AUTH_USER;
  const pass = process.env.UI_BASIC_AUTH_PASSWORD;
  if (!user || !pass) return NextResponse.next();
  const header = req.headers.get("authorization");
  if (header?.startsWith("Basic ")) {
    // Split at the first colon only: RFC 7617 allows colons in the password.
    const decoded = atob(header.slice(6));
    const sep = decoded.indexOf(":");
    if (sep >= 0 && decoded.slice(0, sep) === user && decoded.slice(sep + 1) === pass) {
      return NextResponse.next();
    }
  }
  return new NextResponse("Authentication required", {
    status: 401,
    headers: { "WWW-Authenticate": 'Basic realm="agvrag"' },
  });
}

export const config = { matcher: ["/((?!_next/static|_next/image|favicon.ico).*)"] };
