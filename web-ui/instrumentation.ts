import { registerOTel } from "@vercel/otel";

export function register() {
  registerOTel({ serviceName: "pdf-chat-ollama" });

  // register() runs once per runtime — separately for the Node.js runtime
  // (route handlers) and the Edge runtime (proxy.ts, Next 16's renamed
  // middleware convention) — so this must be unconditional, not guarded to
  // one of them, or the other's console calls (e.g. every request line
  // proxy.ts logs) would never get timestamped.
  patchConsoleWithTimestamps();
}

// Node's console.log/error/warn print no time at all, so a server log line
// couldn't answer "when did this happen" without cross-referencing the
// terminal's own clock. Patching here (once, at server boot, in each
// runtime) stamps every existing and future call site across the whole
// app — including the many console.log calls in
// app/(chat)/api/chat/route.ts that must never be deleted (see CLAUDE.md),
// and every request line proxy.ts logs — without editing any call site
// individually.
let consolePatched = false;
function patchConsoleWithTimestamps() {
  if (consolePatched) {
    return;
  }
  consolePatched = true;

  const original = {
    log: console.log.bind(console),
    error: console.error.bind(console),
    warn: console.warn.bind(console),
  };

  for (const level of ["log", "error", "warn"] as const) {
    console[level] = (...args: unknown[]) => {
      original[level](`[${new Date().toISOString()}]`, ...args);
    };
  }
}
