# database.md

이 프로젝트는 서로 독립적인 **DB 3개**를 사용한다. 각각 역할·기술·위치가 다르고, 서로 동기화되지 않는다(과거 채팅 기록만 예외적으로 이원화되어 있었으나 2026-07-23 정리됨 — 아래 "api.db" 섹션 참조).

| DB | 기술 | 위치 | 역할 |
|---|---|---|---|
| ChromaDB (벡터 DB) | ChromaDB (PersistentClient) | `data/vectors/` | PDF 텍스트 청크 + 임베딩 |
| API DB | SQLite + SQLAlchemy | `data/api.db` | PDF 메타데이터 |
| 프론트엔드 DB | SQLite + Drizzle ORM (`better-sqlite3`) | `web-ui/data/chat.db` | 채팅 기록 (유일한 채팅 저장소) |

---

## 1. ChromaDB (벡터 DB)

**위치**: `data/vectors/` (설정: `src/api/config.py`의 `Settings.VECTOR_DB_DIR`)
**클라이언트**: `chromadb.PersistentClient(path='data/vectors')`
**관리 코드**: `src/core/embeddings.py`(`VectorStore`)

### 구조
- **PDF 1개당 컬렉션 1개** — 컬렉션명은 업로드 시 생성되어 `api.db`의 `pdfs.collection_name`에 저장됨 (예: `pdf_1884475783623626966`)
- 컬렉션 안의 각 문서(document)는 **청크 1개** — OCR 처리된 PDF는 페이지 단위(`doc_count == page_count`), 네이티브 텍스트 PDF는 7,500자/겹침 100자 단위(`DocumentProcessor(chunk_size=7500, chunk_overlap=100)`)

### 청크 메타데이터 (실측, OCR 경로)
```json
{
  "pdf_id": "pdf_393662820633708541",
  "pdf_name": "대바늘_포포토끼.pdf",
  "source_file": "대바늘_포포토끼.pdf",
  "chunk_index": 0,
  "source_page": 1,
  "language_detected": "zh-cn",
  "ocr_confidence": 84.66457680250784,
  "text_boxes_count": 319
}
```
네이티브(비-OCR) 경로는 `source_page` 대신 `chunk_index`만으로 순서를 판단(`docs.sort()` 시 `source_page` 없으면 `chunk_index`로 폴백).

### 알려진 제약
- ⚠️ **업로드 시점의 OCR 결과가 박제됨** — 질의 시점 재-OCR 결과(번역 요청 시 `RAGService._reocr_pdf_chunks()`)는 여기 반영되지 않고 그 질의에만 사용됨. 컬렉션 자체를 갱신하려면 `PDFService.refresh_ocr()`(`POST /api/v1/pdfs/{pdf_id}/refresh-ocr`) 사용 — 원본 파일을 재-OCR해 같은 `collection_name`으로 컬렉션을 통째로 교체(`pdf_id` 불변)
- `data/vectors/` 삭제 시 전체 임베딩 초기화 — PDF 재업로드 필요
- 고아 컬렉션(대응하는 `api.db` 행이 없는 컬렉션)이 쌓일 수 있음 — `scripts/cleanup_orphans.py`로 dry-run/정리 가능

### 직접 조회
```python
import chromadb
client = chromadb.PersistentClient(path='data/vectors')

# 컬렉션 목록
for c in client.list_collections():
    col = client.get_collection(c.name)
    print(f'{c.name}: {col.count()}개 청크')

# 특정 컬렉션 내용 (collection_name은 api.db의 pdfs 테이블에서 조회)
col = client.get_collection('pdf_1884475783623626966')
result = col.get(include=['documents', 'metadatas'])
```

---

## 2. API DB (`data/api.db`)

**기술**: SQLite + SQLAlchemy (`src/api/database.py`)
**연결**: `DATABASE_URL = "sqlite:///data/api.db"`, `Base.metadata.create_all(bind=engine)`로 앱 시작 시 테이블 자동 생성 (마이그레이션 도구 없음 — 스키마 변경 시 기존 DB 파일과 어긋날 수 있음)

### 현재 사용 중인 모델 (1개)

**`PDFMetadata`** (테이블: `pdfs`)

| 컬럼 | 타입 | 설명 |
|---|---|---|
| `pdf_id` | String, PK | PDF 고유 ID |
| `name` | String | 원본 파일명 |
| `collection_name` | String, UNIQUE | ChromaDB 컬렉션명 |
| `upload_timestamp` | DateTime | 업로드 시각 |
| `doc_count` | Integer | 청크 수 |
| `page_count` | Integer | 원본 페이지 수 |
| `is_sample` | Boolean | 샘플 PDF 여부 |
| `file_path` | String | 원본 파일 경로 (`data/pdfs/uploads/{pdf_id}_{파일명}`) — 재-OCR이 읽는 원본 |

`doc_count == page_count`이면 OCR로 처리된 문서(페이지 단위 청크)라는 뜻 — 재-OCR/새로고침 관련 로직이 이 값으로 원본 판별에 사용(`_reocr_pdf_chunks()`, `refresh_ocr()`).

⚠️ **(2026-07-25 발견, 의도적으로 미해결) 소유자 컬럼이 아예 없음 — PDF 라이브러리는 전 사용자 공유 상태.** 프론트(`web-ui`) 쪽 `chat`/`message`/`vote`/`document`는 실제 로그인 계정(게스트 아닌 회원가입 계정 2개로 직접 검증)으로도 소유권 검증이 정확히 동작하는데, PDF는 `PDFMetadata`에 `user_id`가 없고 `web-ui/components/sidebar-pdfs.tsx`가 NextAuth 세션을 거치지 않고 브라우저에서 `http://localhost:8001/api/v1/pdfs*`를 직접 호출하므로, 로그인 여부·계정과 무관하게 **누구든 모든 PDF를 보고 삭제·재-OCR 가능**(쿠키 없는 curl로도 그대로 조회됨). 고치려면 이 테이블에 `user_id` 컬럼 추가+마이그레이션, PDF 관련 프론트 호출을 Next.js API 라우트 경유로 변경(서버측에서만 신뢰 가능한 로그인 사용자 id 전달), 기존 업로드분의 소유자 결정이 필요 — 범위가 커서 사용자 판단으로 이번엔 구현하지 않고 현황만 기록(`.claude/CLAUDE.md`의 "알려진 문제점 및 개선 계획" 5번 항목 참조).

### 잔재 테이블 (코드에서 미사용, DB 파일에만 남아있음)

⚠️ 아래 테이블은 `src/api/database.py`에 SQLAlchemy 모델이 없고, 현재 코드 어디에서도 읽거나 쓰지 않는다. 물리적으로 `api.db` 안에 남아있을 뿐이다.

| 테이블 | 유래 | 상태 |
|---|---|---|
| `analysis_results` | 미완성 OCR/번역 "analyze" 기능의 잔재 | ✅ 2026-07-23 `scripts/cleanup_orphans.py --apply`로 DROP됨 — 더 이상 존재하지 않음 |
| `chat_sessions` | 2026-07-23 제거된 백엔드 채팅 기록 (`session_id`, `created_at`, `last_active`) | 아직 남아있음 — 스크립트 대상 아님(추가 가능) |
| `messages` | 위와 동일 (`message_id`, `session_id`, `role`, `content`, `sources`(JSON), `timestamp`) | 아직 남아있음 — 스크립트 대상 아님(추가 가능) |

`chat_sessions`/`messages`가 왜 만들어졌고 왜 제거됐는지(세션이 답변 생성에 전혀 반영되지 않는 완전 무상태 구조였고, 프론트가 매 질의마다 `session_id: null`을 보내 세션 경계 자체가 깨져 있었으며, 조회 엔드포인트를 리포지토리 어디서도 호출하지 않았다는 조사 결과)는 `.claude/CLAUDE.md`의 "알려진 문제점 및 개선 계획" 6번 항목 참조.

`scripts/cleanup_orphans.py`는 고아 ChromaDB 컬렉션과 `analysis_results`를 dry-run/정리 대상으로 다룬다(2026-07-23 `--apply` 실행 완료 — 고아 컬렉션 3개 삭제, `analysis_results` DROP). `chat_sessions`/`messages`는 아직 스크립트 대상에 없음.

---

## 3. 프론트엔드 DB (`web-ui/data/chat.db`)

**기술**: Drizzle ORM + `better-sqlite3` (Vercel AI Chatbot 템플릿에서 그대로 가져옴)
**스키마**: `web-ui/lib/db/schema.ts`
**쿼리**: `web-ui/lib/db/queries.ts`
**마이그레이션**: `web-ui/lib/db/migrations/`(현재 `0000_high_namora.sql` 1개로 스쿼시됨, `dialect: "sqlite"`)

이 DB가 **이 프로젝트의 유일한 채팅 기록**이다 (백엔드 쪽 `chat_sessions`/`messages`는 위 참조, 2026-07-23 제거됨). 사이드바 대화 목록, 대화 재개, 삭제, 제목 생성이 전부 이 DB를 실시간으로 읽고 쓴다.

### 연결 설정 (`lib/db/queries.ts`)
```typescript
const sqlite = new Database(dbPath);
sqlite.pragma("journal_mode = WAL");
sqlite.pragma("busy_timeout = 5000");
```
(2026-07-25 추가) 기본 rollback-journal 모드는 읽기가 쓰기를 막아서, 사이드바의 잦은 폴링(`/api/history`, `/api/vote`, `/api/auth/session`)이 삭제 같은 쓰기와 겹치면 SQLite가 즉시 "database is locked"로 실패했다 — 실측: `DELETE /api/chat`가 처리되지 않은 예외로 500이 나면서 채팅이 실제로는 안 지워졌는데도 프론트가 성공으로 오인하는 버그의 근본 원인이었음(아래 `chat`/`user` 알려진 이슈 참조). WAL은 읽기/쓰기를 동시에 허용하고, `busy_timeout`은 그래도 남는 락 경합을 즉시 실패 대신 재시도하게 한다.

### 테이블

| 테이블 | 주요 컬럼 | 설명 |
|---|---|---|
| `chat` | `id`(PK), `created_at`, `title`, `visibility`, `user_id`(FK→user) | 대화(세션) 1개 = 행 1개. `user_id`는 생성 시점(`route.ts` POST 핸들러)에 항상 `session.user.id`(로그인된 실제 user id)로 명시적으로 채워짐 — 스키마 기본값 `"local-user"`는 이 값을 지정하지 않고 직접 insert할 때만 쓰이는 폴백이고, 정상 흐름에서는 실제로 쓰이지 않음 |
| `message` | `id`(PK), `chat_id`(FK→chat), `role`, `content`(레거시 호환용), `parts`(AI SDK 메시지 파츠, JSON 문자열), `created_at` | 채팅 메시지 1개 = 행 1개 |
| `chat_pdf` | `chat_id`(FK→chat), `pdf_id`, `added_at` | 이 프로젝트 전용 테이블 — 특정 대화에 선택된 PDF(백엔드 `pdf_id`)를 연결. PK 없음(복합키 없이 단순 로그성 테이블) |
| `user` | `id`(PK), `email`, `password`, `created_at` | Vercel AI Chatbot 템플릿에서 가져온 테이블이지만, **템플릿 잔재가 아니라 실제로 쓰인다** — NextAuth(`app/(auth)/auth.ts`)가 매 요청 세션을 이 테이블의 실제 행과 연결한다. 아래 "게스트 사용자 영속성" 참조 |
| `document` | `id`(PK), `title`, `content`, `kind`, `user_id`(FK→user), `created_at` | 템플릿의 아티팩트(코드/텍스트/시트/이미지 사이드패널) 기능용 — PDF RAG 흐름과 무관 |
| `suggestion` | `id`(PK), `document_id`(FK→document), `document_created_at`, `content`, `user_id`(FK→user), `created_at` | 아티팩트 기능용, 템플릿 잔재 |
| `vote` | `chat_id`(FK→chat), `message_id`, `is_upvoted` | 메시지 좋아요/싫어요, 템플릿 잔재 |
| `stream` | `id`(PK), `chat_id`(FK→chat), `content`, `created_at` | 스트리밍 재개용, 템플릿 잔재 |

`document`/`suggestion`/`vote`/`stream`은 Vercel AI Chatbot 템플릿에서 그대로 가져와 이 프로젝트의 핵심 흐름(PDF RAG 채팅)엔 안 쓰이지만, `user`는 인증에 실제로 쓰인다는 점에 주의.

### 게스트 사용자 영속성 (`user` 테이블, 2026-07-25 추가)

이 앱은 회원가입 없이 쓰는 로컬 툴이라 대부분 NextAuth의 **guest** Credentials 프로바이더(`app/(auth)/auth.ts`)로 자동 로그인된다 — 로그인 세션이 없을 때(`app/(chat)/page.tsx`가 `redirect("/api/auth/guest")`) `createGuestUser()`가 `user` 테이블에 새 행(`email: guest-<timestamp>`)을 만들고, 이 행의 `id`가 그 세션의 `session.user.id`가 되어 `chat.user_id`로 저장된다.

- ⚠️ **(2026-07-25 수정 전 버그)** 게스트 로그인 세션(NextAuth JWT 쿠키)이 만료되거나 지워지면, `createGuestUser()`가 매번 **완전히 새로운** `user` 행 + 새 id를 만들었다 — 그러면 이전 세션에서 만든 채팅의 `chat.user_id`는 다시는 로그인될 수 없는 "죽은" 게스트 id로 영구히 고정되어, `DELETE /api/chat`의 소유자 검사(`chat.userId !== session.user.id`)를 그 무엇으로도 통과할 수 없게 되고(항상 403), 삭제가 영구히 불가능해짐. 실측: `data/chat.db`에 `chat.user_id`가 `user` 테이블 어디에도 없는 고아 채팅이 발견됨
- ✅ **수정**: `auth.ts`의 guest `authorize()`가 이제 NextAuth 세션 쿠키와는 별도의 장기 쿠키(`guest-user-id`, httpOnly, `sameSite: lax`, 약 400일)를 확인해서, 있으면 기존 게스트 `user` 행을 재사용하고 없을 때만 새로 만든다 — 같은 브라우저는 인증 세션이 끊겨도 동일한 `user.id`를 유지하므로 예전 채팅이 계속 삭제 가능
- `getUserById(id)`(`lib/db/queries.ts`) — 이 영속성 로직을 위해 추가된 조회 함수(기존엔 `getUser(email)`만 있었음)

### 명령어
```bash
cd web-ui
pnpm db:generate   # 마이그레이션 파일 생성 (schema.ts 변경 후)
pnpm db:migrate    # 마이그레이션 적용 — 빈 DB에서도 동작
pnpm db:push       # 스키마 직접 반영 (마이그레이션 파일 없이, 로컬 빠른 반복용)
pnpm db:studio     # Drizzle Studio GUI로 직접 조회
```
대안: `npx tsx web-ui/lib/db/init-db.ts` (빈 DB 초기화), `web-ui/init-db.sh` (⚠️ `data/chat.db`와 `lib/db/migrations` 전체 삭제 후 재생성 — 파괴적)

### 알려진 이슈
- 템플릿 잔재 테이블(`document`/`suggestion`/`vote`/`stream`)이 이 프로젝트에서 쓰이지 않는 기능(아티팩트, 투표)을 위한 것이라 스키마가 실제 사용 범위보다 큼 — 정리 여부는 별도 판단 필요
- `.env.example`에 Vercel 전용 변수(`AI_GATEWAY_API_KEY`, `BLOB_READ_WRITE_TOKEN`, `POSTGRES_URL`, `REDIS_URL`)가 템플릿에서 그대로 남아있음 — 로컬 개발엔 `NEXT_PUBLIC_API_URL`, `AUTH_SECRET`, DB 경로만 필요
- `route.ts`의 `DELETE`/`lib/db/queries.ts`의 `deleteChatById`/`getChatById`에 로깅 추가됨(2026-07-25) — 삭제 실패 시 요청 id·인증/소유자 검사 결과·원본 DB 에러가 서버 콘솔에 남는다. 삭제가 안 될 때는 먼저 이 로그로 어느 단계(미인증/404/403/DB 에러)에서 막혔는지 확인할 것
