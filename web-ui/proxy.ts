/**
 * Proxy for Next.js (Next.js 16+)
 * For MVP: No authentication required
 */

import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";

// Single cross-cutting interception point for every API route under
// /api/* — the Next.js equivalent of a Spring @Around advice applied via
// one pointcut, instead of a log call duplicated into each route handler.
//
// Unlike Spring AOP, this proxy can't wrap the downstream Response the way
// @Around wraps a join point — NextResponse.next() hands control to the
// route handler and this function never sees what comes back. So this
// covers the request side (method, path, query, and any identifying PK in
// a JSON body) centrally; the response side (status, duration) is already
// logged for free by Next's own per-request dev server line ("METHOD path
// status in Xms"), which instrumentation.ts's console patch timestamps the
// same as everything else — no second interception point needed.
const JSON_CONTENT_TYPE = "application/json";
const BODY_METHODS = new Set(["POST", "PUT", "PATCH", "DELETE"]);
// Identifiers the app's routes actually key their data on — path-based ids
// (e.g. /api/chat/[id]/pdfs) and query-based ids (e.g. /api/chat?id=...)
// are already visible in pathname+search below; this only needs to cover
// ids that travel in a JSON body instead (e.g. POST /api/chat's { id },
// PATCH /api/vote's { chatId, messageId }).
const PK_KEYS = ["id", "chatId", "pdfId", "pdfIds", "messageId", "documentId"];

// Best-effort peek at the request body for those identifiers. Uses
// .clone() so the stream read here is independent of the one the actual
// route handler consumes downstream — reading request.json() directly
// would exhaust the body and break the handler's own await request.json().
async function extractBodyPk(
  request: NextRequest
): Promise<Record<string, unknown> | null> {
  if (!BODY_METHODS.has(request.method)) {
    return null;
  }
  if (!request.headers.get("content-type")?.includes(JSON_CONTENT_TYPE)) {
    return null;
  }

  try {
    const body: unknown = await request.clone().json();
    if (!body || typeof body !== "object") {
      return null;
    }

    const found: Record<string, unknown> = {};
    for (const key of PK_KEYS) {
      if (key in body) {
        found[key] = (body as Record<string, unknown>)[key];
      }
    }
    return Object.keys(found).length > 0 ? found : null;
  } catch {
    return null;
  }
}

export default async function proxy(request: NextRequest) {
  const { pathname, search } = request.nextUrl;

  if (pathname.startsWith("/api/")) {
    const pk = await extractBodyPk(request);
    console.log(`[API ->] ${request.method} ${pathname}${search}`, pk ?? "");
  }

  // For MVP: Allow all requests without authentication
  return NextResponse.next();
}

export const config = {
  matcher: [
    /*
     * Match all request paths except for the ones starting with:
     * - _next/static (static files)
     * - _next/image (image optimization files)
     * - favicon.ico (favicon file)
     */
    "/((?!_next/static|_next/image|favicon.ico).*)",
  ],
};
