# CLAUDE.md

이 문서는 Claude Code(claude.ai/code)가 이 저장소에서 작업할 때 참고하는 가이드입니다.

## 절대규칙
- 클로드는 환경변수 파일(.env*)을 건들지 않는다
- 답변은 한국어로만 한다
- Frontend: @web-ui/claude.md
- DB (ChromaDB / api.db / chat.db) 스키마·구조: @.claude/database.md

## 프로젝트 개요

Ollama와 LangChain을 사용해 PDF 문서와 대화하는 완전 로컬 RAG(검색 증강 생성) 애플리케이션.
- **Python 백엔드**: FastAPI REST API — RAG 파이프라인, PDF/OCR 처리, ChromaDB 벡터 저장
- **Next.js 프론트엔드**: 채팅 영속화, PDF 관리, PDF 페이지 뷰어, 모델 선택 UI

주 사용 사례: 스캔된 중국어 뜨개질 도안 PDF를 OCR로 읽어 한국어로 번역/질의.

## 아키텍처

### 백엔드 (Python)

**핵심 모듈 (`src/core/`)**:
- `document.py`: PDF 텍스트/OCR 로딩과 청크화. `DocumentProcessor.load_pdf()`가 `(documents, used_ocr)`를 반환 — `detect_if_image_based()`(텍스트 길이, CJK 인식 깨짐 비율, 단일문자 단어 비율)로 스캔 문서를 자동 감지하고, 네이티브 추출(`UnstructuredPDFLoader`)이 부실하면 OCR로 폴백
- `text_extractor.py`: `pdf2image`로 페이지를 이미지로 변환 후 **`deepseek-ocr:3b`(vision-LLM, Ollama 경유)**로 OCR (2026-07-24, `pytesseract`에서 전환 — 전처리 없이 원본 이미지 그대로 전달; 실측상 pytesseract가 자주 오인식하던 CJK 문자·워터마크·사진 속 캡션까지 정확히 인식). 언어 파라미터 불필요(모델이 알아서 스크립트를 인식). 후처리로 vision-LLM 특유의 마크다운 포맷(헤더 `#`, 굵게 `**`)과 사진이 포함된 페이지에서 가끔 나오는 base64 이미지 데이터(여러 줄로 래핑될 수 있음, `_clean_deepseek_ocr_output()`)를 제거. **반복 워터마크/캡션 줄 제거**(`_strip_recurring_watermark_lines` — 페이지 상단 3줄만 후보로, 퍼지 클러스터링으로 전체 페이지 60% 이상에서 반복되는 줄만 제거)는 엔진과 무관하게 그대로 유지
- `image_analysis.py`: `ImageAnalyzer` — `analyze_image_quality()`(블러/밝기/대비), `detect_language()`는 여전히 실제 경로에서 사용됨. `extract_text_with_ocr()`/`extract_text_boxes()`(pytesseract 기반)는 이제 안 쓰임(레거시, 테스트에서만 참조)
- `image_handler.py`: `ImageHandler`(회전/노이즈 제거/그레이스케일/워터마크 제거 전처리) — 실제 OCR 경로에서는 안 쓰임(레거시, 테스트에서만 참조). `text_extractor.py`가 인스턴스는 만들지만 실제로 호출하지는 않음
- `embeddings.py`: `VectorStore` — OllamaEmbeddings(`nomic-embed-text`) + ChromaDB 저장

**API 계층 (`src/api/`)**:
- `main.py`: FastAPI 앱 설정. CORS는 `http://localhost:3000`(Next.js)만 허용. `pdfs`, `query`, `models`, `health` 라우터 등록
- `database.py`: SQLite(`data/api.db`) + SQLAlchemy 모델. 상세 스키마·잔재 테이블은 @.claude/database.md 참조
- `routers/`: `pdfs`(업로드/목록/삭제), `query`(RAG 질의 + 세션 메시지 조회), `models`(Ollama 모델 목록), `health`
- `services/`: `pdf_service.py`(업로드→OCR→청크→임베딩), `rag_service.py`(질의 오케스트레이션 — 이 파일이 사실상 RAG의 전부)
- `config.py`: Ollama 연결 설정 (기본 http://localhost:11434)

**질의 처리 흐름** (`RAGService.query_multi_pdf`):
1. PDF 메타데이터 조회 → **서버가 직접 답할 수 있는 질문이면 LLM 호출 없이 즉시 반환** (아래 "LLM 우회 단축 경로" 참조)
2. 번역 의도 감지 시 → 원본 PDF를 좁힌 언어팩으로 재-OCR (페이지 범위 지정 시 해당 페이지만)
3. 전체 청크 40개 이하면 전체 문서 컨텍스트, 초과 시 MultiQueryRetriever 유사도 검색
4. 페이지 범위가 질문에 있으면 `source_page` 기준으로 청크 필터링
5. `data/context/*.json` 우선참조 컨텍스트를 프롬프트에 주입
6. 다중 페이지 번역이면 페이지별 루프, 아니면 단일 호출 → 답변 + 출처 반환

### 데이터 저장

3개 DB(ChromaDB, API DB, 프론트엔드 DB)의 스키마·구조·알려진 이슈는 @.claude/database.md 참조. 요약:
- **벡터 DB**: ChromaDB (`data/vectors/`) — PDF 1개당 컬렉션 1개. ⚠️ 업로드 시점 OCR 결과가 박제됨(재-OCR은 그 질의에만 반영, 컬렉션 자체를 갱신하려면 `refresh_ocr`)
- **원본 PDF**: `data/pdfs/uploads/` (`{pdf_id}_{파일명}`) — 재-OCR이 읽는 원본
- **API DB**: SQLite `data/api.db` — PDF 메타데이터(`PDFMetadata`)만 관리. 채팅 기록은 여기 없음 — 백엔드는 무상태 질의응답 엔진이고, 프론트 `chat.db`가 유일한 채팅 기록(2026-07-23 백엔드의 `ChatSession`/`ChatMessage` 저장 경로 전부 제거)
- **프론트엔드 DB**: `web-ui/data/chat.db` (Drizzle ORM + better-sqlite3) — **유일한 채팅 기록**

## 개발 명령어

### Python 백엔드

```bash
pip install -r requirements.txt

# FastAPI 서버 (포트 8001)
python run_api.py

# 테스트
python -m pytest tests/ -v
python -m pytest tests/ --cov=src
python -m pytest tests/test_ocr_pipeline.py -v          # 단일 파일
python -m pytest tests/test_ocr_pipeline.py::test_specific_case  # 단일 테스트

# pre-commit (pytest + pylint 실행 — 2026-07-23부터 CI와 동일하게 pytest 사용)
pre-commit install
pre-commit run --all-files
```

루트에 `pytest.ini`/`pyproject.toml` 없음 — pytest는 기본값으로 동작. CI(`.github/workflows/tests.yml`)는 Python 3.10~3.12 매트릭스로 `pytest --cov` 실행.

### 프론트엔드 (web-ui)

```bash
cd web-ui
pnpm install

pnpm dev           # 개발 서버 (포트 3000, Turbopack)
pnpm build         # 프로덕션 빌드 (lib/db/migrate 선실행)
pnpm lint          # Ultracite/Biome 검사
pnpm format        # 자동 수정

pnpm db:generate   # 마이그레이션 파일 생성
pnpm db:migrate    # 마이그레이션 적용 (빈 DB에서도 동작)
pnpm db:push       # 스키마 직접 반영 (마이그레이션 파일 없이, 로컬 빠른 반복용)
pnpm db:studio     # Drizzle Studio

pnpm test          # Playwright e2e
```

TS/TSX 코드는 `web-ui/.cursor/rules/ultracite.mdc`(Ultracite/Biome 규칙)의 적용을 받음.

### 전체 실행

```bash
./start_all.sh   # FastAPI(8001) + Next.js(3000) 동시 기동
# 또는 개별 실행:
python run_api.py            # 터미널 1
cd web-ui && pnpm dev        # 터미널 2
```

## 아키텍처 결정 및 핵심 패턴

### RAG 파이프라인
- 청크: 7,500자 / 겹침 100자 (`DocumentProcessor(chunk_size=7500, chunk_overlap=100)` — `pdf_service.py`)
- 전체 청크 40개 이하(`FULL_CONTEXT_CHUNK_LIMIT`)면 유사도 검색을 건너뛰고 **문서 전체를 컨텍스트로 사용** — 소규모 문서에서 "전부 추출해줘" 류 요청이 검색 누락으로 깨지는 것을 방지
- 대규모일 때만 MultiQueryRetriever(질의 변형 2개 생성) + top-k 검색
- 모든 처리는 로컬 — 데이터가 기기를 떠나지 않음

### LLM 우회 단축 경로 (서버 직접 응답)
RAG 프롬프트에는 청크 **텍스트**만 들어가고 메타데이터는 주입되지 않으므로, 시스템이 이미 알고 있는 정보는 LLM에게 물어도 절대 맞출 수 없음. 그래서 아래 질문은 LLM 호출 없이 즉시 응답:
- **메타데이터 질문** (`_METADATA_SHORT_CIRCUITS` — 페이지 수/파일명/업로드 날짜/청크 수, `_wants_page_count`/`_wants_filename`/`_wants_upload_date`/`_wants_chunk_count`) → `PDFMetadata`에서 직접 답변. `(키워드 감지 함수, 표시 라벨, PDF당 답변 줄 포맷터)` 튜플 리스트를 순서대로 검사해 첫 매치로 처리 — 새 메타데이터 질문 유형을 추가하려면 이 리스트에 튜플 하나만 추가하면 됨
- **페이지 원문 조회** (`_detect_page_range` + `_wants_raw_page_content`, 예: "1~2페이지 내용 알려줘" — 번역/분석 키워드 없을 때) → 저장된 페이지 텍스트를 그대로 반환

### 페이지 범위 지정
`_detect_page_range()`가 "1~2페이지", "1페이지부터 3페이지까지", "pages 2-4", "page 5" 등을 정규식으로 감지. 감지되면 (전체 문서 모드에 한해) `source_page` 기준으로 청크를 필터링하고, 재-OCR도 해당 페이지만 수행. 번역/분석 요청도 좁혀진 범위만 LLM에 전달.

### 답변 생성: 잘림 방지 + 페이지별 번역 루프
- 모든 생성 호출은 `_invoke_with_continuation()`(ChatOllama) 또는 `_invoke_ollama_chat_with_continuation()`(raw ollama, 씽킹 모델)을 거침 — Ollama의 `done_reason != "stop"`이면 최대 `MAX_CONTINUATION_ATTEMPTS`(2)회 이어쓰기 재시도, 그래도 미완이면 답변 텍스트에 ⚠️ 경고를 명시 (잘린 답변을 조용히 성공 처리하지 않음). `query_multi_pdf()`는 이 상태를 4번째 반환값 `truncated`(bool)로도 노출하고, API 응답의 `metadata.truncated`로 그대로 전달됨 — UI/외부 클라이언트가 텍스트 파싱 없이 판단 가능
- `_wants_verbatim_or_translation()`(verbatim 키워드 또는 번역 키워드)이 참이면 전체 문서 컨텍스트 모드에서 **페이지 수와 무관하게(1페이지여도)** `_translate_pages()`를 탐 — **페이지당 LLM 1회 호출**로 나눠 순서대로 이어붙임 (단일 대형 호출의 출력 한도 초과로 인한 잘림을 구조적으로 방지). 씽킹 모드는 페이지별 호출에서 비활성화(기계적 전사에 사고과정은 출력 예산 낭비). ⚠️ 예전엔 페이지가 정확히 1개일 때만 이 루프를 건너뛰고 검증이 전혀 없는 단일 호출 경로로 빠졌음 — 1페이지 번역 요청(`"11페이지만 번역해줘"`)이 형식 검증 없이 원문 그대로(번역 0%) 반환돼도 "성공"으로 처리되는 버그였음(2026-07-24 수정, `use_page_loop` 조건에서 `len(docs) > 1` → `>= 1`)
- `_translate_pages()`는 페이지 단위로 격리됨: 하드 실패(Ollama 크래시 등)는 `MAX_PAGE_RETRY_ATTEMPTS`(2)회 재시도 후에도 실패하면 그 페이지만 실패 표시(`failed_pages`)하고 계속 진행 — 완료된 페이지는 절대 유실되지 않음
- 번역 페이지는 생성 후 구조 검증: `_looks_correctly_structured()`("# 원문:" 섹션 다음에 "# 결과:" 섹션이 오고, 결과 섹션에 한글 번역 내용이 있는지 확인 — 2026-07-25부터 이 두 섹션 구조가 정답, 예전의 원문·번역 줄 단위 교차 형식은 폐기됨, 아래 "우선참조 컨텍스트" 절 위 문단 참조), `_looks_duplicated()`(중복 생성 감지 — 인접 블록 반복을 크기 1줄부터 스캔, **한글 줄이 포함된 중복만 플래그** — 번역 불가 OCR 잡음 줄이 원문/번역 자리에 똑같이 복사되는 정상 케이스는 재시도 낭비 없이 통과), `_looks_untranslated_output()`(타깃이 한국어인데 **페이지 전체** 한글 비율 15% 미만이면 오역/미번역으로 간주). 실패 시 교정 메시지와 함께 재시도 — 언어 미번역(`_looks_untranslated_output`) 실패는 다른 형식 문제와 섞이지 않는 전용의 더 강한 재시도 메시지 사용(2026-07-24 추가, 일반 교정 메시지만으로는 재시도 2회 모두 영어로 실패하는 사례가 실측됨)
- **번역 출력 형식(2026-07-25 변경)**: 원문 줄과 번역 줄을 교차 배치하던 기존 형식에서, `# 원문:` 섹션(모든 원문 줄) 다음에 `# 결과:` 섹션(같은 순서의 모든 번역 줄)이 오는 2블록 구조로 전환(`TRANSLATION_LINE_INSTRUCTIONS`). 실사용 확인 결과 이 새 형식으로도 R1~R9 같은 실제 뜨개 지시문 줄이 `# 결과:` 안에서 번역 안 된 채 남는 사례가 있었음(제목/헤딩 줄만 번역되고 본문 줄은 원문 그대로) — 이는 아래 "알려진 문제점" 3번(줄 단위 언어 이탈을 페이지 평균 검증이 못 잡는 문제)과 근본적으로 같은 한계이고, 사용자에게 보고했으나 추가 조치 여부는 아직 결정되지 않음(미해결)
- 재시도를 다 소진하고도 형식 검증에 실패한 페이지는 **결과는 유지하되 답변 본문에 인라인으로 경고 표시**(`⚠️ [페이지 X 번역 검증 실패 — ...]`)하고, 별도 `format_issue_pages` 리스트로도 반환됨 — `query_multi_pdf()`의 `truncated` 계산에도 포함됨(`bool(failed_pages) or bool(format_issue_pages) or truncated`). ⚠️ 예전엔 이 경우 `reasoning_steps`에만 경고가 남고 답변 텍스트·`truncated`엔 아무 표시가 없어서, 사용자 입장에선 실패가 성공과 구분되지 않았음(2026-07-24 수정)
- 번역 지시문(`_line_instructions_for`, 타깃이 한국어일 때)에는 "모든 번역 줄은 반드시 한글로", "마지막 페이지라고 요약/영어 전환 금지", "OCR 잡음 줄은 억지로 영어 해석을 지어내지 말고 원문 그대로 둘 것"을 명시적으로 포함(2026-07-24 추가)
- 번역 응답에는 `_normalize_korean_counts()` 후처리를 항상 적용 — "12개의 코" → "12코" (모델이 지시문을 따르지 않아도 결정적으로 보정)

### OCR과 워터마크 제거 (스캔 PDF)
- 업로드: `UnstructuredPDFLoader(strategy="fast")` → `detect_if_image_based()` 판정 → OCR 폴백 (`pdf2image` 300 DPI, 전처리 없이 원본 이미지 그대로 → `deepseek-ocr:3b`로 페이지별 OCR → 마크다운/base64 아티팩트 제거(`_clean_deepseek_ocr_output`) → CJK 공백 정리 → 반복 워터마크 줄 제거 → 페이지 단위 Document)
- `PDFService.upload_and_process()`는 `used_ocr=True`면 재청크를 건너뜀 (OCR 경로가 이미 페이지 청크를 반환)
- 튜닝 지점: `text_extractor.DEFAULT_DPI`, `document.MIN_TEXT_LENGTH` 등 감지 임계값, `text_extractor._WATERMARK_*` 상수, `DEEPSEEK_OCR_PROMPT`(OCR 지시 프롬프트)
- Ollama에 `deepseek-ocr:3b` 모델 pull 필요(`ollama pull deepseek-ocr:3b`) — 시스템 `tesseract` 바이너리는 더 이상 필요 없음(레거시 `image_analysis.py`/`image_handler.py`가 여전히 optional import하지만 실제 경로에서 안 씀)

### 질의 시점 재-OCR
- 번역 의도가 감지되면 원본 파일을 다시 OCR해서 그 질의에만 사용(`_reocr_pdf_chunks()`, 페이지 범위 지정 가능) — **결과는 ChromaDB에 저장 안 됨**. 전체 문서 모드 + `doc_count == page_count`(OCR로 처리된 문서)일 때만
- `_detect_ocr_language_override()`/`CJK_OCR_LANGUAGE`는 pytesseract 시절 언어팩 선택 로직의 흔적 — `ocr_language` 파라미터는 시그니처 호환을 위해 계속 받지만 `deepseek-ocr:3b`는 이를 사용하지 않음(스크립트를 자체 인식). 재-OCR 자체는 여전히 유효(오탈자 없는 새 판독을 다시 시도한다는 의미에서)하지만, 언어를 "좁힌다"는 원래 목적은 더 이상 실질적 효과가 없음

### 우선참조 컨텍스트 (`data/context/`)
- 모델이 자체 지식보다 **먼저** 참조해야 하는 모든 `*.json` 파일의 범용 저장소 — 용어집(`chi_knitting.json`), 규칙, 도메인 지식 등
- 파일 형태 2가지 (`RAGService._load_priority_context()`): 평면 `{"키": "값"}` 문자열 딕셔너리는 `키 → 값` 목록으로, 그 외 JSON은 파일명 헤더 아래 pretty-print
- **매 질의마다 전체 파일을 다시 로드** — 파일 추가/수정 후 서버 재시작 불필요
- ⚠️ **프롬프트 예시(few-shot)가 용어집을 이길 수 있음**: `TRANSLATION_LINE_INSTRUCTIONS`의 형식 예시에 실제 도메인 용어를 쓰면 모델이 용어집 대신 예시 표현을 따라감 (실제 발생했던 버그 — "下针→아래뜨기" 예시가 용어집의 "下针→K"를 눌렀음). 예시는 반드시 `[source line A]` 같은 플레이스홀더만 사용
- 복합어와 단독 글자 매핑이 겹칠 때(예: `下针→K`와 `针→코`)는 지시문이 "각 매핑을 해당 원문에만 문자 그대로 적용"하도록 명시되어 있음 (`[12针]` → `[12코]`, `[12개의 K]` 아님)

## 주요 파일

**백엔드**:
- `src/api/services/rag_service.py` — **RAG의 핵심.** 질의 오케스트레이션, LLM 우회 단축 경로, 재-OCR, 페이지별 번역 루프, 잘림/형식 안전망, 우선참조 컨텍스트 주입 전부 여기
- `src/core/document.py`, `text_extractor.py`, `image_handler.py`, `image_analysis.py` — PDF 텍스트 추출, OCR 폴백, 워터마크 제거
- `src/api/routers/query.py`, `pdfs.py` — 질의/PDF 엔드포인트 (`pdfs.py`에 `POST /{pdf_id}/refresh-ocr` — 컬렉션 재-OCR 갱신)
- `src/api/services/pdf_service.py` — 업로드/삭제/재-OCR 갱신(`refresh_ocr`)
- `src/api/database.py` — SQLAlchemy 모델
- `data/context/*.json` — 우선참조 컨텍스트

**프론트엔드**:
- `web-ui/lib/ai/provider.ts` — FastAPI 백엔드 직접 호출 (`ollamaChat` 등). `/api/v1/query`에는 30분 undici Agent 타임아웃 적용 (전체 문서 번역은 수십 분 걸릴 수 있음 — 기본 5분 타임아웃이면 `UND_ERR_HEADERS_TIMEOUT` 발생)
- `web-ui/app/(chat)/api/chat/route.ts` — 채팅 API 라우트 (`maxDuration = 1800`)
- `web-ui/components/elements/response.tsx` — 메시지 마크다운 렌더러. `remark-breaks` 적용됨 (단일 `\n` 줄바꿈 유지 — 백엔드가 보내는 줄 단위 번역/원문에 필수)

**설정**:
- `requirements.txt` — LangChain 1.0 스택, OCR 의존성(`pdf2image`/`ollama`; `pytesseract`/`opencv-python-headless`는 레거시 경로용으로 남아있음), `langdetect`
- `run_api.py` — uvicorn 엔트리포인트 (reload=True)

## 개발 워크플로

1. **API 엔드포인트 추가**: `src/api/routers/`에 라우터 생성 → `main.py`에 등록 → 필요 시 CORS 갱신 → `http://localhost:8001/docs`에서 확인
2. **RAG 동작 수정**: `rag_service.py`가 유일한 실경로. 프롬프트/지시문 상수도 이 파일 상단에 모여 있음
3. **프론트 DB 스키마 변경**: `pnpm db:generate` → `pnpm db:migrate` (빈 DB에서도 동작), 빠른 반복은 `pnpm db:push`
4. **우선참조 컨텍스트 추가**: `data/context/`에 JSON 파일 추가/수정 — 재시작 불필요
5. **OCR/RAG 변경 검증**: 유닛테스트만으로 불충분 — **반드시 실제 백엔드 + 실제 PDF + 실제 Ollama 모델로 end-to-end 확인** (샘플: `pdf_id=pdf_393662820633708541`, `data/pdfs/uploads/pdf_393662820633708541_대바늘_포포토끼.pdf`, 11페이지 — `GET /api/v1/pdfs`로 현재 등록된 pdf_id 재확인 후 사용할 것, 업로드마다 새 ID가 발급되므로 이 값은 바뀔 수 있음). `ollama pull deepseek-ocr:3b`가 먼저 돼 있어야 OCR 경로가 동작함. qwen3:14b 생성은 페이지당 수 분 걸리므로 백그라운드로 실행할 것. 테스트 업로드는 검증 후 반드시 삭제 (사용자가 웹 UI를 병행 사용 중)

## 알려진 제약

- **포트**: FastAPI 8001, Next.js 3000
- **ChromaDB**: `data/vectors/` 삭제 시 임베딩 초기화 (재업로드 필요)
- **동시성**: dev 모드는 `reload=True` — 소스 수정 시 서버 재시작됨 (진행 중 요청 유실 주의). Ollama는 요청을 순차 처리(`-np 1`)
- **OCR 의존성**: Ollama에 `deepseek-ocr:3b` pull 필요 (`ollama pull deepseek-ocr:3b`)

## 디버깅 팁

- **API 연결**: 백엔드가 `http://localhost:8001`인지, CORS가 요청 origin을 허용하는지 확인
- **벡터 DB 손상**: `data/vectors/` 삭제 후 PDF 재업로드
- **프론트 DB 오류**: `pnpm db:migrate`는 빈 `web-ui/data/chat.db`에서도 동작. 대안: `npx tsx web-ui/lib/db/init-db.ts`
- **중국어 OCR 깨짐/서식 잔재**: `ollama ps`로 `deepseek-ocr:3b`가 정상 응답하는지 확인. 마크다운(`#`/`**`)이나 base64 이미지 데이터가 저장된 텍스트에 남아있으면 `_clean_deepseek_ocr_output()`(`text_extractor.py`) 후처리 정규식이 그 출력 형태를 못 잡은 것 — 실제 원인이었던 사례: base64 페이로드가 여러 줄로 래핑되면 한 줄만 매칭하는 정규식은 이어지는 줄을 못 지움
- **ChromaDB 내용 직접 조회**: 조회 명령어는 @.claude/database.md 참조. 여기 저장된 텍스트는 **업로드 시점** OCR 결과임에 주의

## 알려진 문제점 및 개선 계획

실제 검증(대바늘_포포토끼.pdf, 11페이지)에서 확인된 미해결 이슈. 해결된 항목은 이 목록에서 제거하고 위 아키텍처 섹션에 현재 동작으로 반영함 — 히스토리가 필요하면 git log 참조.

1. ✅ **(해결됨, 2026-07-25) Streamlit 앱 및 관련 레거시 코드 제거** — Streamlit 방향(유지/제거) 결정: 제거로 확정. `src/app/`(Streamlit 앱 전체), `run.py`, `core/rag.py`(`RAGPipeline`), `core/llm.py`(`LLMManager`), `tests/test_rag.py`, `tests/test_models.py`(Streamlit 전용 유틸 테스트), `st_app_ui.png` 삭제. `start_all.sh`/`requirements.txt`(`streamlit`, `pdfplumber`, `pdfminer.six`)/README.md/`.claude/CLAUDE.md`에서 관련 참조 제거. `image_handler.py`/`image_analysis.py`의 pytesseract 레거시 코드는 별개 사안(OCR 엔진 전환 관련, Streamlit과 무관)이라 이번엔 그대로 둠 — 필요하면 별도로 논의
2. 🔶 **(2026-07-24 발견)** `data/pdfs/uploads/`에 `api.db`의 `pdfs` 테이블과 매칭되는 레코드가 없는 원본 PDF 파일이 최소 1개 존재(`pdf_1326292550554632241_대바늘_포포토끼.pdf` — 현재 유효한 건 `pdf_393662820633708541_...`뿐) — 이전 세션에서 삭제된 PDF의 원본 파일이 안 지워지고 남은 것으로 보임. `scripts/cleanup_orphans.py`는 현재 ChromaDB 컬렉션만 다루고 `data/pdfs/uploads/` 파일 시스템은 검사하지 않음 — 스크립트 대상에 추가하면 좋을 후보
3. ⚠️ **(2026-07-24 발견) 번역 형식 검증이 페이지 단위 평균이라 줄 단위 언어 이탈을 못 잡음** — `_looks_untranslated_output()`은 **페이지 전체**의 한글 비율(15% 기준)만 봄. 페이지 대부분이 정상적으로 한국어로 번역되면, 그 안의 일부 줄(특히 워터마크/캡션처럼 짧고 의미상 독립된 문장, 또는 "Long-tail Cast On 69 sts"처럼 특정 기술 용어 줄)이 영어로 새어도 검증을 통과함 — 실측: 핵심 뜨개 지시문(R1~R41)은 한국어로 정상 번역되면서도 저작권 고지/계정 안내 캡션 줄만 반복적으로 영어로 출력됨. 지시문에 "모든 번역 줄은 한글로" 규칙을 추가해도(위 아키텍처 섹션 참조) 이 특정 패턴은 계속 뚫림 — 프롬프트만으로는 한계가 있어 보이고, 근본 해결은 페이지가 아니라 **줄 단위 언어 검증**으로 바꿔야 할 것으로 보임(아직 미구현 — 범위가 커서 보류 중)
4. ⚠️ **(2026-07-24 발견/확인) qwen3:14b의 반복 생성(중복) 실패가 페이지마다 비결정적으로 발생** — "全下针"처럼 짧고 실제로 여러 번 반복되는 구조적 지시문 페이지에서, 모델이 이미 생성한 블록을 한 번 더 통째로 반복해버리는 자기회귀 반복 루프 현상이 관찰됨(`_looks_duplicated()`가 감지, 재시도 2회로도 못 고치는 경우 있음). **실행마다 실패하는 페이지가 달라짐**(동일 문서, 동일 코드로 3회 연속 전체 번역 시 실패 페이지가 매번 다름) — 코드 버그가 아니라 모델 자체의 샘플링 확률성(기본 temperature)에 기인하는 것으로 보임
   - ❌ **시도했으나 되돌림**: `repeat_penalty`를 1.1(Ollama 기본)→1.3으로 올려 페이지 루프 호출에 적용해봄 — 해당 실행에서 중복 생성은 0건으로 줄었으나, **더 심각한 새 오류**가 발생함(실측: 정상적으로 반복돼야 할 "코"/"针" 토큰을 모델이 회피하다가 극단적으로 낮은 확률의 토큰을 끌어써서 번역 중간에 **러시아어 단어가 삽입**되고 "[69针织]"처럼 단위 표기가 깨짐). 중복 생성은 이미 `_looks_duplicated()` 재시도 + 실패 시 인라인 경고 표시(위 아키텍처 섹션 참조)로 안전하게(눈에 띄게) 처리되고 있어서, 더 위험한 오류를 감수할 가치가 없다고 판단해 되돌림(`TRANSLATION_REPEAT_PENALTY` 상수/사용 코드 제거, `ChatOllama` 기본값으로 복귀). **다시 시도할 경우 1.3보다 훨씬 보수적인 값(예: 1.15)부터 점진적으로 테스트할 것, 그리고 반드시 전체 문서 라이브 재번역으로 부작용(단위 표기 깨짐·언어 혼입)까지 확인할 것**
5. ⚠️ **(2026-07-25 발견, 의도적으로 미해결 — 사용자 판단으로 현황만 기록)** PDF 라이브러리(업로드/목록/삭제/재-OCR)에 사용자별 격리가 전혀 없음. 실제 회원가입 계정 2개(guest 아님)로 직접 검증한 결과, `chat`/`message`/`vote`/`document`(아티팩트) 소유권 검증은 정확히 동작함(A 계정이 만든 비공개 채팅은 B 계정의 `/api/history`엔 안 보이고, B가 삭제/투표/조회 시도하면 전부 403) — 이 부분은 게스트 전용 버그가 아니라 실제 로그인 계정에서도 이미 올바르게 동작하는 것으로 확인됨. 반면 PDF는 (1) `PDFMetadata`(`api.db`, `.claude/database.md` 참조)에 소유자 컬럼 자체가 없고, (2) `web-ui/components/sidebar-pdfs.tsx`/`lib/ai/provider.ts`가 NextAuth 세션을 거치지 않고 브라우저에서 `http://localhost:8001/api/v1/pdfs*`를 직접 호출하므로, 로그인 여부·계정과 무관하게 **누구든 업로드된 모든 PDF를 보고 삭제·재-OCR할 수 있음**(쿠키 없이 curl로도 그대로 조회됨, 실측 확인). 고치려면 (a) `api.db`에 `user_id` 컬럼 추가+마이그레이션, (b) PDF 관련 호출을 브라우저→FastAPI 직접 호출 대신 Next.js API 라우트 경유로 변경(서버측에서만 신뢰 가능한 실제 로그인 사용자 id를 실어 보내기 위해), (c) 기존에 이미 업로드된 PDF의 소유자를 누구로 할지 결정 필요 — 범위가 커서 이번엔 구현하지 않고 현황만 기록하기로 함(사용자 확인, 2026-07-25)
