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
- **Next.js 프론트엔드**: 채팅 영속화, PDF 관리, 모델 선택 UI
- **Streamlit 앱**: 실험용 보조 UI (레거시, 독립 동작)

주 사용 사례: 스캔된 중국어 뜨개질 도안 PDF를 OCR로 읽어 한국어로 번역/질의.

## 아키텍처

### 백엔드 (Python)

**핵심 모듈 (`src/core/`)**:
- `document.py`: PDF 텍스트/OCR 로딩과 청크화. `DocumentProcessor.load_pdf()`가 `(documents, used_ocr)`를 반환 — `detect_if_image_based()`(텍스트 길이, CJK 인식 깨짐 비율, 단일문자 단어 비율)로 스캔 문서를 자동 감지하고, 네이티브 추출(`UnstructuredPDFLoader`)이 부실하면 OCR로 폴백
- `text_extractor.py`: `pytesseract` + `pdf2image` 기반 OCR (300 DPI, 기본 언어 `eng+chi_sim+chi_tra+kor`, `start_page`/`end_page`로 페이지 범위 지정 가능). CJK 문자 사이 불필요한 공백 제거, **반복 워터마크/캡션 줄 제거**(`_strip_recurring_watermark_lines` — 페이지 상단 3줄만 후보로, 퍼지 클러스터링으로 전체 페이지 60% 이상에서 반복되는 줄만 제거)
- `image_handler.py`: OCR 전처리 — 자동 회전, 노이즈 제거, 그레이스케일, **이미지 수준 워터마크 제거**(`remove_watermark()`, OpenCV Otsu 이진화로 연한 회색 워터마크 제거)
- `image_analysis.py`: `pytesseract` 래퍼(텍스트 + 텍스트 박스), 이미지 품질 지표(블러/밝기/대비), 언어 감지(`langdetect`)
- `embeddings.py`: `VectorStore` — OllamaEmbeddings(`nomic-embed-text`) + ChromaDB 저장. API와 Streamlit 양쪽에서 사용
- ⚠️ `rag.py`(`RAGPipeline`)와 `llm.py`(`LLMManager`)는 **레거시** — 실제 서빙 경로(FastAPI/Streamlit) 어디서도 사용하지 않고 `tests/test_rag.py`만 참조. 실제 RAG 로직은 `src/api/services/rag_service.py`에 인라인으로 구현되어 있음

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
- **API DB**: SQLite `data/api.db` — PDF 메타데이터(`PDFMetadata`)만 관리. 채팅 기록은 여기 없음(2026-07-23 제거, 아래 "알려진 문제점" 6번 참조)
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
python -m pytest tests/test_rag.py::test_specific_case  # 단일 테스트

# pre-commit (주의: pytest가 아니라 `unittest discover tests` + pylint를 실행함)
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
./start_all.sh   # FastAPI(8001) + Next.js(3000) + Streamlit(8501) 동시 기동
# 또는 개별 실행:
python run_api.py            # 터미널 1
cd web-ui && pnpm dev        # 터미널 2
python run.py                # (선택) Streamlit
```

## 아키텍처 결정 및 핵심 패턴

### RAG 파이프라인
- 청크: 7,500자 / 겹침 100자 (`DocumentProcessor(chunk_size=7500, chunk_overlap=100)` — `pdf_service.py`)
- 전체 청크 40개 이하(`FULL_CONTEXT_CHUNK_LIMIT`)면 유사도 검색을 건너뛰고 **문서 전체를 컨텍스트로 사용** — 소규모 문서에서 "전부 추출해줘" 류 요청이 검색 누락으로 깨지는 것을 방지
- 대규모일 때만 MultiQueryRetriever(질의 변형 2개 생성) + top-k 검색
- 모든 처리는 로컬 — 데이터가 기기를 떠나지 않음

### LLM 우회 단축 경로 (서버 직접 응답)
RAG 프롬프트에는 청크 **텍스트**만 들어가고 메타데이터는 주입되지 않으므로, 시스템이 이미 알고 있는 정보는 LLM에게 물어도 절대 맞출 수 없음. 그래서 아래 질문은 LLM 호출 없이 즉시 응답:
- **페이지 수 질문** (`_wants_page_count`, "몇 페이지"/"page count" 등) → `PDFMetadata.page_count`에서 직접 답변
- **페이지 원문 조회** (`_detect_page_range` + `_wants_raw_page_content`, 예: "1~2페이지 내용 알려줘" — 번역/분석 키워드 없을 때) → 저장된 페이지 텍스트를 그대로 반환
- 파일명/업로드일/청크 수 등 다른 메타데이터 질문은 아직 미구현 (개선 계획 참조)

### 페이지 범위 지정
`_detect_page_range()`가 "1~2페이지", "1페이지부터 3페이지까지", "pages 2-4", "page 5" 등을 정규식으로 감지. 감지되면 (전체 문서 모드에 한해) `source_page` 기준으로 청크를 필터링하고, 재-OCR도 해당 페이지만 수행. 번역/분석 요청도 좁혀진 범위만 LLM에 전달.

### 답변 생성: 잘림 방지 + 페이지별 번역 루프
- 모든 생성 호출은 `_invoke_with_continuation()`(ChatOllama) 또는 `_invoke_ollama_chat_with_continuation()`(raw ollama, 씽킹 모델)을 거침 — Ollama의 `done_reason != "stop"`이면 최대 `MAX_CONTINUATION_ATTEMPTS`(2)회 이어쓰기 재시도, 그래도 미완이면 답변 텍스트에 ⚠️ 경고를 명시 (잘린 답변을 조용히 성공 처리하지 않음)
- `_wants_verbatim_or_translation()`(verbatim 키워드 또는 번역 키워드)이 참이면: (1) 씽킹 모드 비활성화(기계적 전사에 사고과정은 출력 예산 낭비), (2) 다중 페이지면 `_translate_pages()` — **페이지당 LLM 1회 호출**로 나눠 순서대로 이어붙임 (단일 대형 호출의 출력 한도 초과로 인한 잘림을 구조적으로 방지)
- `_translate_pages()`는 페이지 단위로 격리됨: 하드 실패(Ollama 크래시 등)는 `MAX_PAGE_RETRY_ATTEMPTS`(2)회 재시도 후에도 실패하면 그 페이지만 실패 표시하고 계속 진행 — 완료된 페이지는 절대 유실되지 않음
- 번역 페이지는 생성 후 구조 검증: `_looks_correctly_interleaved()`(원문/번역 블록 분리 또는 번역 누락 감지), `_looks_duplicated()`(중복 생성 감지 — 인접 블록 반복을 크기 1줄부터 스캔, **한글 줄이 포함된 중복만 플래그** — 번역 불가 OCR 잡음 줄이 원문/번역 자리에 똑같이 복사되는 정상 케이스는 재시도 낭비 없이 통과), `_looks_untranslated_output()`(타깃이 한국어인데 한글 비율 15% 미만이면 오역/미번역으로 간주 — 마지막 페이지 영어 드리프트 방지). 실패 시 교정 메시지와 함께 재시도, 소진 시 결과는 유지하고 플래그만
- 번역 응답에는 `_normalize_korean_counts()` 후처리를 항상 적용 — "12개의 코" → "12코" (모델이 지시문을 따르지 않아도 결정적으로 보정)

### OCR과 워터마크 제거 (스캔 PDF)
- 업로드: `UnstructuredPDFLoader(strategy="fast")` → `detect_if_image_based()` 판정 → OCR 폴백 (`pdf2image` 300 DPI → 전처리(Otsu 이진화 포함) → `pytesseract --psm 6` → CJK 공백 정리 → 반복 워터마크 줄 제거 → 페이지 단위 Document)
- `PDFService.upload_and_process()`는 `used_ocr=True`면 재청크를 건너뜀 (OCR 경로가 이미 페이지 청크를 반환)
- 튜닝 지점: `text_extractor.DEFAULT_OCR_LANGUAGE`, `DEFAULT_DPI`, `document.MIN_TEXT_LENGTH` 등 감지 임계값, `text_extractor._WATERMARK_*` 상수
- 시스템 `tesseract` 바이너리 + 언어팩(`chi_sim`/`chi_tra`/`kor`) 필수 — Python 패키지만으로 부족

### 질의 시점 OCR 언어 좁히기 (재-OCR)
- Tesseract 언어팩에 `eng`이 섞이면 CJK 인식률이 실측으로 하락 (예: `下针` → `FH, Get,` 오인식). 그래서 기본값에서 `eng`을 빼는 대신 **질의별로** 좁힘
- `_detect_ocr_language_override()`: **번역 의도만 있으면** 트리거 — 소스+타깃 언어를 둘 다 명시하면 그 조합("중국어→한국어" → `chi_sim+chi_tra+kor`), 언어명이 0~1개면 CJK 기본값(`_DEFAULT_TRANSLATION_OCR_LANGUAGE = chi_sim+chi_tra+kor`)으로 폴백
- `_reocr_pdf_chunks()`: 원본 파일을 좁힌 언어팩으로 재-OCR (페이지 범위 지정 가능). **결과는 그 질의에만 사용, ChromaDB에는 저장 안 함**. 전체 문서 모드 + `doc_count == page_count`(OCR로 처리된 문서)일 때만

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
- `requirements.txt` — LangChain 1.0 스택, OCR 의존성(`pytesseract`/`pdf2image`/`opencv-python-headless`/`langdetect`)
- `run_api.py` — uvicorn 엔트리포인트 (reload=True)

## 개발 워크플로

1. **API 엔드포인트 추가**: `src/api/routers/`에 라우터 생성 → `main.py`에 등록 → 필요 시 CORS 갱신 → `http://localhost:8001/docs`에서 확인
2. **RAG 동작 수정**: `rag_service.py`가 유일한 실경로 (core/rag.py 아님). 프롬프트/지시문 상수도 이 파일 상단에 모여 있음
3. **프론트 DB 스키마 변경**: `pnpm db:generate` → `pnpm db:migrate` (빈 DB에서도 동작), 빠른 반복은 `pnpm db:push`
4. **우선참조 컨텍스트 추가**: `data/context/`에 JSON 파일 추가/수정 — 재시작 불필요
5. **OCR/RAG 변경 검증**: 유닛테스트만으로 불충분 — **반드시 실제 백엔드 + 실제 PDF + 실제 Ollama 모델로 end-to-end 확인** (샘플: `data/pdfs/uploads/pdf_1326292550554632241_대바늘_포포토끼.pdf`). qwen3:14b 생성은 페이지당 수 분 걸리므로 백그라운드로 실행할 것. 테스트 업로드는 검증 후 반드시 삭제 (사용자가 웹 UI를 병행 사용 중)

## 알려진 제약

- **포트**: FastAPI 8001, Next.js 3000, Streamlit 8501
- **ChromaDB**: `data/vectors/` 삭제 시 임베딩 초기화 (재업로드 필요)
- **동시성**: dev 모드는 `reload=True` — 소스 수정 시 서버 재시작됨 (진행 중 요청 유실 주의). Ollama는 요청을 순차 처리(`-np 1`)
- **OCR 의존성**: 시스템 `tesseract` + `chi_sim`/`chi_tra`/`kor` 언어팩 필수

## 디버깅 팁

- **API 연결**: 백엔드가 `http://localhost:8001`인지, CORS가 요청 origin을 허용하는지 확인
- **벡터 DB 손상**: `data/vectors/` 삭제 후 PDF 재업로드
- **프론트 DB 오류**: `pnpm db:migrate`는 빈 `web-ui/data/chat.db`에서도 동작. 대안: `npx tsx web-ui/lib/db/init-db.ts`
- **중국어 OCR 깨짐**: `ImageHandler.remove_watermark()`가 워터마크를 실제로 제거하는지 전처리 이미지를 덤프해서 먼저 확인
- **ChromaDB 내용 직접 조회**: 조회 명령어는 @.claude/database.md 참조. 여기 저장된 텍스트는 **업로드 시점** OCR 결과임에 주의

## 알려진 문제점 및 개선 계획

실제 검증(2026-07-23, 대바늘_포포토끼.pdf)에서 확인된 미해결 이슈와 개선 방향. 우선순위순.

1. ✅ **(해결됨, 2026-07-23) 마지막 페이지 번역 언어 드리프트** — 11페이지 번역 시 마지막 페이지만 한국어 대신 영어로 출력된 사례.
   - 조치: `_expects_korean_output()`(번역 의도 + 타깃 언어를 명시적으로 다른 언어로 지정하지 않았으면 True) + `_looks_untranslated_output()`(한글 비율 15% 미만이면 True)을 `_translate_pages()`의 페이지별 형식 검증에 추가 — 형식 오류로 간주해 기존 재시도 루프를 태움. 재시도 프롬프트에도 "요청한 언어로 번역"을 명시적으로 재확인시킴
2. ✅ **(해결됨, 2026-07-23) 번역 불가 OCR 잡음 줄이 중복 감지 재시도를 낭비** — `V 2061 : 人 2` 같은 잡음 줄은 모델이 원문/번역 자리에 똑같이 복사해 `_looks_duplicated()`에 걸리고, 재시도 2회(페이지당 수 분)를 소모한 뒤에야 결과 유지로 넘어감
   - 조치: `_looks_duplicated()`가 중복된 블록에 **한글(번역) 줄이 최소 1개 포함된 경우에만** True를 반환하도록 변경 — 잡음 줄만 반복된 경우(번역 대상 자체가 없어 재시도로 못 고치는 경우)는 더 이상 걸리지 않음. 실제 번역 내용이 중복된 진짜 문제(전체 페이지 재생성, 헤더/워터마크+번역 동반 중복)는 여전히 감지됨
3. ✅ **(해결됨, 2026-07-23) 수량 표기 어색함** — `[12针]`이 `[12코]` 대신 `[12개의 코]`로 출력됨 (용어는 맞지만 "개의"가 불필요)
   - 조치: `TRANSLATION_LINE_INSTRUCTIONS`에 "숫자와 단위는 붙여서 표기(12코, 12개의 코 금지)" 규칙 추가 + `_normalize_korean_counts()` 결정적 후처리(정규식 `(\d+)\s*개의\s*코` → `\1코`, `query_multi_pdf`에서 번역 응답에 항상 적용)로 이중 안전망. ⚠️ 정규식에 트레일링 `\b`(단어 경계)를 넣으면 안 됨 — 한글 조사(와/를 등)가 코에 공백 없이 붙어 Python의 유니코드 인식 `\b`가 한글-한글 사이를 경계로 보지 않아 매칭이 조용히 실패함
4. ✅ **(해결됨, 2026-07-23) 박제된 구버전 OCR 임베딩** — OCR 언어 수정 이전에 업로드된 PDF는 ChromaDB에 깨진 텍스트가 남아 있고, 번역 외 일반 질의는 이 텍스트를 그대로 씀
   - 조치(B안): "재-OCR 결과로 컬렉션 갱신" 기능 추가. `PDFService.refresh_ocr(pdf_id, db, ocr_language=None)` — 원본 파일을 `CJK_OCR_LANGUAGE`(`text_extractor.py`, `chi_sim+chi_tra+kor`)로 재-OCR하고 기존 컬렉션을 삭제 후 **같은 `collection_name`으로 재생성** (`pdf_id`/`collection_name` 불변, `doc_count`/`page_count`만 갱신). `POST /api/v1/pdfs/{pdf_id}/refresh-ocr` 엔드포인트로 노출, 웹 UI 사이드바에 PDF별 새로고침 아이콘(호버 시 노출) 추가. `doc_count != page_count`(원래 OCR 문서가 아님) 또는 원본 파일 없음이면 400, PDF 없음이면 404. `VectorStore.delete_collection_by_name()` 헬퍼 신설(기존 `delete_pdf`의 raw Chroma 생성 코드도 이걸로 정리) — 임의의 컬렉션명을 인스턴스 상태와 무관하게 삭제 가능
5. 🔧 **(스크립트 준비됨, 2026-07-23) 저장소 잔재 정리** — ChromaDB 고아 컬렉션 3개(2026-07-23 기준: `pdf_632425175926842791`·`pdf_973033043484023512` 0청크, `pdf_7679892920309835191` 1청크 — 전부 `api.db`의 `pdfs` 테이블에 대응 행 없음), `api.db`의 미사용 `analysis_results` 테이블(2행, 코드 어디서도 참조 안 함), 레거시 `core/rag.py`·`core/llm.py`(+ `tests/test_rag.py`)
   - 조치: `scripts/cleanup_orphans.py` 작성 — `api.db`의 `pdfs.collection_name`과 대조해 대응 행 없는 ChromaDB 컬렉션을 찾고, `analysis_results` 테이블 존재/행수 확인. 기본은 **dry-run**(보고만), `--apply`로 실제 삭제/DROP. 사용자가 직접 실행 여부 결정하기로 하고 아직 미실행(`--apply` 안 돌림) — 재실행 시 orphan 목록이 그 시점 기준으로 다시 계산되므로 위 3개 이름은 참고용
   - 레거시 모듈(`core/rag.py`, `core/llm.py`, `tests/test_rag.py`)은 Streamlit 방향성 결정 전까지 보류 — 스크립트 범위 아님
6. ✅ **(해결됨, 2026-07-23) 채팅 기록 이중 저장** — 백엔드 `api.db`와 프론트 `chat.db`에 같은 대화가 따로 저장되어 동기화되지 않음
   - **조사 결과**: 두 시스템이 같은 역할을 두고 경쟁하는 게 아니었음 — `RAGService.query_multi_pdf()`는 `session_id`를 받지도 않아 세션 히스토리를 답변 생성에 전혀 반영하지 않았고(완전히 무상태·단일턴), 프론트는 `/api/v1/query` 호출 시 항상 `session_id: null`을 보내 백엔드가 **매 질의마다 새 UUID를 생성**(대화 1개 = 세션 여러 개로 쪼개짐, 세션 경계 자체가 깨져 있었음), `GET /sessions/{id}/messages`는 리포지토리 어디서도 호출된 적 없음(문서의 curl 예시가 유일한 참조). 즉 백엔드 쪽은 "제2의 채팅 기록"이 아니라 **아무도 안 읽는 write-only 로그**였음. 반대로 프론트 `chat.db`는 사이드바 히스토리·대화 재개·제목 생성·삭제까지 실제로 동작하는 유일한 진짜 채팅 기록. 두 DB는 같은 커밋(`782a296`, Next.js+FastAPI 동시 도입)에서 조율 없이 각자의 스캐폴딩(Vercel AI Chatbot 템플릿의 Drizzle 저장소 + 새 REST 레이어의 자체 로그)을 그대로 가져오면서 생긴 결과
   - **조치**: 백엔드의 `ChatSession`/`ChatMessage` 저장 경로를 전부 제거 — `src/api/database.py`(모델 클래스), `src/api/services/rag_service.py`(`save_message()`/`get_session_messages()`), `src/api/routers/query.py`(`GET /sessions/{id}/messages` 엔드포인트, 질의 전후 저장 호출), `src/api/models.py`(`QueryRequest.session_id`, `QueryResponse.session_id`/`message_id`), 프론트 `web-ui/lib/ai/provider.ts`(대응 필드), `docs/api/rest-api.md`. 백엔드는 이제 순수 질의응답 엔진, 프론트 `chat.db`가 유일한 채팅 기록. `api.db`의 물리 `chat_sessions`/`messages` 테이블과 기존 데이터는 `analysis_results`와 같은 성격의 잔재로 남겨둠(제거 안 함) — `scripts/cleanup_orphans.py` 대상에 나중에 포함 가능
7. **서버 직접 응답 범위 확장** — 현재 페이지 수/페이지 원문만 LLM 우회. 파일명·업로드일·청크 수 등도 같은 패턴으로 확장 가능
8. **`needsDocumentContext()` 키워드 분류기(route.ts)가 조잡** — "this", "explain" 등 광범위한 영어 키워드라 일반 대화도 문서 질문으로 오분류 가능
9. **pre-commit이 pytest 대신 unittest 실행** — CI(pytest)와 불일치. `.pre-commit-config.yaml`의 entry를 pytest로 교체 검토
10. **API 응답에 기계판독용 `truncated` 플래그 없음** — 현재는 답변 텍스트의 ⚠️ 문구로만 표시. `metadata` dict에 boolean 추가하면 UI가 활용 가능
