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

### 잔재 테이블 (코드에서 미사용, DB 파일에만 남아있음)

⚠️ 아래 3개 테이블은 `src/api/database.py`에 SQLAlchemy 모델이 없고, 현재 코드 어디에서도 읽거나 쓰지 않는다. 물리적으로 `api.db` 안에 남아있을 뿐이다.

| 테이블 | 유래 | 비고 |
|---|---|---|
| `analysis_results` | 미완성 OCR/번역 "analyze" 기능의 잔재 | `filename`, `original_text`, `translated_text`, `source_language`, `ocr_confidence`, `analyzed_at` 컬럼 |
| `chat_sessions` | 2026-07-23 제거된 백엔드 채팅 기록 | `session_id`, `created_at`, `last_active` |
| `messages` | 위와 동일 | `message_id`, `session_id`, `role`, `content`, `sources`(JSON), `timestamp` |

`chat_sessions`/`messages`가 왜 만들어졌고 왜 제거됐는지(세션이 답변 생성에 전혀 반영되지 않는 완전 무상태 구조였고, 프론트가 매 질의마다 `session_id: null`을 보내 세션 경계 자체가 깨져 있었으며, 조회 엔드포인트를 리포지토리 어디서도 호출하지 않았다는 조사 결과)는 `.claude/CLAUDE.md`의 "알려진 문제점 및 개선 계획" 6번 항목 참조.

`scripts/cleanup_orphans.py`는 현재 `analysis_results`와 고아 ChromaDB 컬렉션만 dry-run/정리 대상으로 다룬다 — `chat_sessions`/`messages`는 아직 스크립트 대상에 없음(추가 가능).

---

## 3. 프론트엔드 DB (`web-ui/data/chat.db`)

**기술**: Drizzle ORM + `better-sqlite3` (Vercel AI Chatbot 템플릿에서 그대로 가져옴)
**스키마**: `web-ui/lib/db/schema.ts`
**쿼리**: `web-ui/lib/db/queries.ts`
**마이그레이션**: `web-ui/lib/db/migrations/`(현재 `0000_high_namora.sql` 1개로 스쿼시됨, `dialect: "sqlite"`)

이 DB가 **이 프로젝트의 유일한 채팅 기록**이다 (백엔드 쪽 `chat_sessions`/`messages`는 위 참조, 2026-07-23 제거됨). 사이드바 대화 목록, 대화 재개, 삭제, 제목 생성이 전부 이 DB를 실시간으로 읽고 쓴다.

### 테이블

| 테이블 | 주요 컬럼 | 설명 |
|---|---|---|
| `chat` | `id`(PK), `created_at`, `title`, `visibility`, `user_id` | 대화(세션) 1개 = 행 1개. `user_id` 기본값 `"local-user"`(인증 없는 로컬 개발 가정) |
| `message` | `id`(PK), `chat_id`(FK→chat), `role`, `content`(레거시 호환용), `parts`(AI SDK 메시지 파츠, JSON 문자열), `created_at` | 채팅 메시지 1개 = 행 1개 |
| `chat_pdf` | `chat_id`(FK→chat), `pdf_id`, `added_at` | 이 프로젝트 전용 테이블 — 특정 대화에 선택된 PDF(백엔드 `pdf_id`)를 연결. PK 없음(복합키 없이 단순 로그성 테이블) |
| `user` | `id`(PK), `email`, `password`, `created_at` | 템플릿 잔재 — Vercel AI Chatbot 템플릿 호환용 스텁, 이 프로젝트는 인증 없이 `local-user` 고정 사용 |
| `document` | `id`(PK), `title`, `content`, `kind`, `user_id`(FK→user), `created_at` | 템플릿의 아티팩트(코드/텍스트/시트/이미지 사이드패널) 기능용 — PDF RAG 흐름과 무관 |
| `suggestion` | `id`(PK), `document_id`(FK→document), `document_created_at`, `content`, `user_id`(FK→user), `created_at` | 아티팩트 기능용, 템플릿 잔재 |
| `vote` | `chat_id`(FK→chat), `message_id`, `is_upvoted` | 메시지 좋아요/싫어요, 템플릿 잔재 |
| `stream` | `id`(PK), `chat_id`(FK→chat), `content`, `created_at` | 스트리밍 재개용, 템플릿 잔재 |

`user`/`document`/`suggestion`/`vote`/`stream`은 Vercel AI Chatbot 템플릿에서 그대로 가져온 것으로, 이 프로젝트의 핵심 흐름(PDF RAG 채팅)에는 `chat`/`message`/`chat_pdf`만 실질적으로 쓰인다.

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
- 템플릿 잔재 테이블(`user`~`stream`)이 이 프로젝트에서 쓰이지 않는 기능(아티팩트, 투표, 인증)을 위한 것이라 스키마가 실제 사용 범위보다 큼 — 정리 여부는 별도 판단 필요
- `.env.example`에 Vercel 전용 변수(`AI_GATEWAY_API_KEY`, `BLOB_READ_WRITE_TOKEN`, `POSTGRES_URL`, `REDIS_URL`)가 템플릿에서 그대로 남아있음 — 로컬 개발엔 `NEXT_PUBLIC_API_URL`, `AUTH_SECRET`, DB 경로만 필요
