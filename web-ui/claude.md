# CLAUDE.md (web-ui)

This file provides frontend-specific guidance to Claude Code. See the root `.claude/CLAUDE.md` for the overall project and backend architecture.

## Origin

This app is built on the Vercel AI Chatbot open-source template (Next.js App Router, NextAuth, Drizzle, `ai` SDK, artifacts system for code/text/sheet/image). It has been adapted to talk to this repo's local FastAPI + Ollama RAG backend instead of a hosted chat model for actual PDF Q&A. Template-provenance leftovers (Postgres-flavored Drizzle migrations, Vercel Blob/Redis env vars) still show up in a few places — see Known Issues below.

## Architecture

- **Chat/PDF Q&A path**: `lib/ai/provider.ts` (`ollamaChat`, `uploadPDF`, `listPDFs`, `deletePDF`) calls the FastAPI backend directly via `fetch` against `NEXT_PUBLIC_API_URL` (default `http://localhost:8001`), hitting `/api/v1/query` and `/api/v1/pdfs*`. This bypasses the `ai` SDK's own model layer for the actual RAG answer.
- **Title/artifact generation path**: `lib/ai/providers.ts` uses Vercel AI Gateway (`@ai-sdk/gateway`) for auxiliary generation (chat titles via `anthropic/claude-haiku-4.5`, and the artifacts feature for code/text/sheet/image). In test environment (`isTestEnvironment` from `lib/constants.ts`), `lib/ai/models.mock.ts` is used instead.
- **Auth**: NextAuth v5 (beta), config split between `app/(auth)/auth.config.ts` (edge-safe, no providers registered there) and `app/(auth)/auth.ts` (adds credentials provider, requires Node.js for bcrypt).
- **Database**: Drizzle ORM over SQLite (`better-sqlite3`), schema in `lib/db/schema.ts`, queries in `lib/db/queries.ts`. Tables: `user, chat, message, chat_pdf, document, suggestion, vote, stream`. `chat_pdf` associates chats with specific PDFs from the FastAPI backend.
- **Route groups**: `app/(auth)/` for login/register, `app/(chat)/` for the main chat UI.
- **Artifacts system** (`artifacts/{code,image,sheet,text}/`, `components/artifact*.tsx`, `lib/artifacts/server.ts`): the template's side-panel document editor (code execution, image generation, spreadsheet, text) — largely independent of the PDF RAG flow.
- **PDF-specific UI**: `components/pdf-upload.tsx`, `components/sidebar-pdfs.tsx`, `components/sources-panel.tsx` — upload, list, and cite PDFs from the FastAPI backend.

## Commands

```bash
pnpm dev              # dev server on :3000 (Turbopack)
pnpm lint             # npx ultracite@latest check (Biome-based, not ESLint)
pnpm format           # npx ultracite@latest fix
pnpm test             # Playwright e2e (sets PLAYWRIGHT=True env var first)
pnpm db:studio        # Drizzle Studio GUI
pnpm db:push          # push schema changes directly (no migration file)
```

`pnpm db:migrate` (and `pnpm build`, which runs it first) now work on a clean/empty SQLite DB. `npx tsx lib/db/init-db.ts` remains a valid alternative for local dev.

## Known Issues / Template Leftovers

- `lib/db/migrations/` was squashed to a single `0000_high_namora.sql` with `"dialect": "sqlite"` in `meta/_journal.json`, matching `drizzle.config.ts` and the runtime DB (`better-sqlite3`). This fixed an earlier `dialect: "postgresql"` mismatch that used to break `db:migrate` on a clean DB — `pnpm db:migrate` and `pnpm build` now work from empty.
- `.env.example` still lists Vercel-specific vars (`AI_GATEWAY_API_KEY`, `BLOB_READ_WRITE_TOKEN`, `POSTGRES_URL`, `REDIS_URL`) from the upstream template; only a subset is actually required for local dev against the FastAPI backend (`NEXT_PUBLIC_API_URL`, `AUTH_SECRET`, DB path). `REDIS_URL` is optional — `getStreamContext()` in `app/(chat)/api/chat/route.ts` degrades gracefully (resumable-stream support just turns off) if it's unset or unusable.
- `.cursor/rules/ultracite.mdc` — Ultracite/Biome lint rules apply here; prefer `pnpm lint` / `pnpm format` over introducing ESLint config.
- (2026-07-25) `lib/api/` — an empty placeholder directory for an in-progress `/api/v1/analyze/*` OCR/translation feature — was removed. That feature is now fully implemented via the existing `/api/v1/query` + `/api/v1/pdfs/*` FastAPI endpoints instead (see root CLAUDE.md).
