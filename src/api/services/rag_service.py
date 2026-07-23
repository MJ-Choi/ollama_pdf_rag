"""RAG query service."""
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from sqlalchemy.orm import Session
from datetime import datetime

from langchain_ollama import ChatOllama
import ollama
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

from ..database import PDFMetadata, ChatSession, ChatMessage
from ..config import settings
from ...core.text_extractor import TextExtractor

logger = logging.getLogger(__name__)

# Below this many total chunks across the selected PDF(s), skip similarity
# search entirely and feed the model the ENTIRE document instead. Retrieval
# (MultiQueryRetriever, top-k) is a lossy approximation meant for large
# corpora; for a small, deliberately-selected PDF (the common case here —
# users pick specific PDFs to chat with) it can easily miss whole pages for
# vague/exhaustive requests like "extract all the Chinese text", since
# similarity search has no notion of "return everything". Full-document
# mode sidesteps that class of bug for the sizes this app typically sees.
FULL_CONTEXT_CHUNK_LIMIT = 40

# Answer-generation output budget reserved on top of estimated input size
# when sizing Ollama's num_ctx (context window). Without this, Ollama falls
# back to its own default (often just 2048 tokens), which can silently
# truncate context far below what the model actually supports.
NUM_CTX_OUTPUT_BUDGET = 2048
NUM_CTX_FLOOR = 4096
NUM_CTX_CEILING = 32768


def _estimate_num_ctx(*texts: str) -> int:
    """Rough token-budget estimate for Ollama's num_ctx from prompt text length.

    Uses a conservative ~2 chars/token estimate (safe for CJK-heavy text,
    where 1 char is often ~1 token) plus a fixed output budget, clamped to
    a sane floor/ceiling.
    """
    estimated_input_tokens = sum(len(t) for t in texts) // 2
    return max(
        NUM_CTX_FLOOR,
        min(NUM_CTX_CEILING, estimated_input_tokens + NUM_CTX_OUTPUT_BUDGET),
    )


# Keywords signaling the user wants raw source text reproduced as-is
# (line-by-line transcription/extraction), not a synthesized/summarized
# answer. The default chain-of-thought prompt actively pushes the model to
# "synthesize a comprehensive answer", which is wrong for these requests —
# e.g. "extract the Chinese text" got turned into a summary glossary table
# instead of the original lines.
VERBATIM_INTENT_KEYWORDS = [
    "그대로", "원문", "나열해", "줄 단위", "요약하지 말고", "요약 없이", "요약하지말고",
    "verbatim", "raw text", "line by line", "line-by-line", "as-is", "word for word",
    "그대로 추출", "그대로 번역",
]


def _wants_verbatim(question: str) -> bool:
    """Detect requests for a verbatim line-by-line reproduction rather than a synthesized answer."""
    lowered = question.lower()
    return any(keyword.lower() in lowered for keyword in VERBATIM_INTENT_KEYWORDS)


# Explicit language names → tesseract OCR language codes. Used to narrow the
# OCR language pack for a single query when the question names a source/target
# translation pair (e.g. "중국어 도안을 한국어로 번역해줘"). "eng" is deliberately
# NOT in DEFAULT_OCR_LANGUAGE's replacement here unless English is one of the
# named languages — mixing in an unneeded English dictionary measurably
# degrades Tesseract's CJK recognition (e.g. "下针" misread as "FH, Get,").
_LANGUAGE_NAME_TO_OCR_CODE = {
    "중국어": "chi_sim+chi_tra",
    "한국어": "kor",
    "영어": "eng",
}
TRANSLATION_INTENT_KEYWORDS = ["번역"]


def _detect_ocr_language_override(question: str) -> Optional[str]:
    """Detect an explicit source+target language pair in a translation-style
    question and return a narrowed tesseract language string, e.g.
    "chi_sim+chi_tra+kor" for "중국어가 source, 한국어가 target". Returns None
    (keep DEFAULT_OCR_LANGUAGE, which still includes "eng") unless both a
    translation-intent keyword AND at least two distinct languages are named —
    a single language mention isn't enough to safely narrow the language pack.
    """
    if not any(keyword in question for keyword in TRANSLATION_INTENT_KEYWORDS):
        return None
    matched_names = [name for name in _LANGUAGE_NAME_TO_OCR_CODE if name in question]
    if len(matched_names) < 2:
        return None
    codes: List[str] = []
    for name in matched_names:
        for part in _LANGUAGE_NAME_TO_OCR_CODE[name].split("+"):
            if part not in codes:
                codes.append(part)
    return "+".join(codes)


class RAGService:
    """Service for RAG operations."""

    def __init__(self):
        """Initialize RAG service."""
        self.persist_directory = settings.VECTOR_DB_DIR
        self.priority_context = self._load_priority_context()

    def _load_priority_context(self) -> str:
        """Load priority reference material from data/context/*.json.

        `data/context/` holds any number of JSON files the user maintains
        as authoritative reference material for RAG answers — term
        glossaries (e.g. Chinese knitting-pattern terminology), style/rule
        sheets, domain facts, etc. Every file in the directory is picked up
        automatically and must be consulted BEFORE the model's own built-in
        knowledge whenever it's relevant to the question.

        Two shapes are supported per file:
        - A flat string-to-string dict (e.g. {"下针": "K (겉뜨기)"}) is
          rendered as a "term → value" lookup list.
        - Anything else (nested objects, lists, etc.) is pretty-printed
          as-is under a heading with the filename.

        Returns a formatted prompt section, or "" if no context files exist.
        """
        context_dir = Path(settings.PROJECT_ROOT) / settings.CONTEXT_DIR
        if not context_dir.is_dir():
            return ""

        sections = []
        for json_path in sorted(context_dir.glob("*.json")):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load context file {json_path}: {e}")
                continue

            if not data:
                continue

            if isinstance(data, dict) and all(isinstance(v, str) for v in data.values()):
                body = "\n".join(f"- {key} → {value}" for key, value in data.items())
            else:
                body = json.dumps(data, ensure_ascii=False, indent=2)

            sections.append(f"[{json_path.name}]\n{body}")

        if not sections:
            return ""

        return (
            "PRIORITY REFERENCE CONTEXT (from data/context/). Treat this as authoritative "
            "and consult it FIRST whenever it's relevant to the question — prefer it over "
            "your own built-in knowledge or assumptions (e.g. use these exact terms/values "
            "instead of your own translation or recollection):\n\n"
            + "\n\n".join(sections)
            + "\n\n"
        )

    def _reocr_pdf_chunks(
        self,
        pdf: PDFMetadata,
        ocr_language: str,
        reasoning_steps: List[str],
    ) -> Optional[List[Document]]:
        """Re-run OCR on `pdf`'s original file with a narrower language pack,
        for this query only — the stored vector DB collection is untouched.

        Returns None (caller falls back to the stored/embedded chunks) when
        the PDF wasn't originally OCR'd, the file is missing, or extraction
        yields nothing.
        """
        # doc_count == page_count is how the OCR path stores documents (one
        # chunk per page, see PDFService.upload_and_process); a native-text
        # PDF chunked at 7500 chars/page almost always produces more chunks
        # than pages. Imperfect, but avoids re-OCR'ing large native PDFs.
        if pdf.doc_count != pdf.page_count:
            return None
        if not pdf.file_path or not Path(pdf.file_path).exists():
            reasoning_steps.append(f"⚠️ 재-OCR 건너뜀 — 원본 파일을 찾을 수 없음: {pdf.name}")
            return None

        reasoning_steps.append(f"🔁 재-OCR 중 ({ocr_language}): {pdf.name}")
        extractor = TextExtractor()
        extraction_result = extractor.extract_text_from_scanned_pdf(
            Path(pdf.file_path), ocr_language=ocr_language
        )
        docs = extractor.create_document_chunks(extraction_result)
        if not docs:
            reasoning_steps.append(f"⚠️ 재-OCR 결과 없음 — 기존 저장된 청크로 대체: {pdf.name}")
            return None

        for i, doc in enumerate(docs):
            doc.metadata.update({
                "pdf_id": pdf.pdf_id,
                "pdf_name": pdf.name,
                "chunk_index": i,
            })
        reasoning_steps.append(f"✅ 재-OCR로 {len(docs)}개 청크 재추출: {pdf.name}")
        return docs

    def query_multi_pdf(
        self,
        question: str,
        model: str,
        pdf_ids: Optional[List[str]],
        db: Session
    ) -> Tuple[str, List[Dict], List[str]]:
        """Query across multiple PDFs with source attribution.

        Args:
            question: User question
            model: LLM model to use
            pdf_ids: List of PDF IDs to query (None = all PDFs)
            db: Database session

        Returns:
            Tuple of (answer, sources, reasoning_steps)
        """
        reasoning_steps = []

        # Get PDF metadata
        query = db.query(PDFMetadata)
        if pdf_ids:
            query = query.filter(PDFMetadata.pdf_id.in_(pdf_ids))
        pdfs = query.all()

        if not pdfs:
            return "No PDFs found to query.", [], []

        reasoning_steps.append(f"📚 Searching across {len(pdfs)} PDF(s): {', '.join([p.name for p in pdfs])}")

        # An explicit "중국어 도안을 한국어로 번역해줘"-style question narrows OCR to
        # just the named source+target languages for THIS query (re-OCR from the
        # original file, not persisted back to the stored vector DB/collection).
        ocr_language_override = _detect_ocr_language_override(question)
        if ocr_language_override:
            reasoning_steps.append(
                f"🔤 번역 요청 감지 — 이번 질의에 한해 OCR 언어를 '{ocr_language_override}'로 좁혀서 원본 PDF를 다시 읽습니다"
            )

        # Initialize LLM (used for query-generation only; the final answer
        # call gets its own instance sized to the actual context below)
        llm = ChatOllama(model=model)
        reasoning_steps.append(f"🤖 Using model: {model}")

        # Small, deliberately-selected PDF set → skip similarity search and
        # use the whole document(s) as context. See FULL_CONTEXT_CHUNK_LIMIT.
        total_chunks = sum(pdf.doc_count for pdf in pdfs)
        use_full_context = total_chunks <= FULL_CONTEXT_CHUNK_LIMIT

        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        all_docs = []

        if use_full_context:
            reasoning_steps.append(
                f"📖 {total_chunks} total chunk(s) across selected PDF(s) — using full-document context "
                f"(no similarity search, so nothing gets missed)"
            )

            for pdf in pdfs:
                docs = None
                if ocr_language_override:
                    docs = self._reocr_pdf_chunks(pdf, ocr_language_override, reasoning_steps)

                if docs is not None:
                    all_docs.extend(docs)
                    continue

                vector_db = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=embeddings,
                    collection_name=pdf.collection_name
                )
                try:
                    reasoning_steps.append(f"📄 Loading full document: {pdf.name}")
                    raw = vector_db.get(include=["documents", "metadatas"])
                    docs = [
                        Document(page_content=text, metadata=meta or {})
                        for text, meta in zip(raw["documents"], raw["metadatas"])
                    ]
                    for doc in docs:
                        doc.metadata.setdefault("pdf_name", pdf.name)
                        doc.metadata.setdefault("pdf_id", pdf.pdf_id)
                    # Preserve original page/chunk order for a coherent read
                    docs.sort(
                        key=lambda d: (
                            d.metadata.get("source_page", d.metadata.get("chunk_index", 0)),
                            d.metadata.get("chunk_index", 0),
                        )
                    )
                    all_docs.extend(docs)
                    reasoning_steps.append(f"✅ Loaded {len(docs)} chunk(s) from {pdf.name}")
                except Exception as e:
                    reasoning_steps.append(f"⚠️ Error loading {pdf.name}: {str(e)}")
                    print(f"Error loading {pdf.name}: {e}")
        else:
            # Query prompt for multi-query retriever
            QUERY_PROMPT = PromptTemplate(
                input_variables=["question"],
                template="""You are an AI language model assistant. Your task is to generate 2
                different versions of the given user question to retrieve relevant documents from
                a vector database. By generating multiple perspectives on the user question, your
                goal is to help the user overcome some of the limitations of the distance-based
                similarity search. Provide these alternative questions separated by newlines.
                Original question: {question}"""
            )

            reasoning_steps.append("🔍 Generating alternative search queries...")
            if ocr_language_override:
                reasoning_steps.append(
                    "ℹ️ 재-OCR 언어 좁히기는 전체-문서 모드에서만 지원됩니다 (선택한 PDF가 많아 "
                    "유사도 검색 모드로 전환됨) — 기존 저장된 청크를 그대로 사용합니다"
                )

            for pdf in pdfs:
                vector_db = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=embeddings,
                    collection_name=pdf.collection_name
                )

                retriever = MultiQueryRetriever.from_llm(
                    vector_db.as_retriever(search_kwargs={"k": 3}),
                    llm,
                    prompt=QUERY_PROMPT
                )

                try:
                    reasoning_steps.append(f"📄 Retrieving from: {pdf.name}")
                    # Use invoke instead of deprecated get_relevant_documents
                    docs = retriever.invoke(question)
                    # Ensure metadata is present
                    for doc in docs:
                        if "pdf_name" not in doc.metadata:
                            doc.metadata["pdf_name"] = pdf.name
                        if "pdf_id" not in doc.metadata:
                            doc.metadata["pdf_id"] = pdf.pdf_id
                    all_docs.extend(docs)
                    reasoning_steps.append(f"✅ Found {len(docs)} relevant chunks in {pdf.name}")
                except Exception as e:
                    reasoning_steps.append(f"⚠️ Error retrieving from {pdf.name}: {str(e)}")
                    print(f"Error retrieving from {pdf.name}: {e}")

        reasoning_steps.append(f"📊 Total chunks retrieved: {len(all_docs)}")

        # In full-document mode, use every chunk; otherwise keep the
        # existing top-10 cap for retrieval-based (large corpus) mode.
        context_limit = len(all_docs) if use_full_context else 10

        # Format context with source labels
        context_parts = []
        for doc in all_docs[:context_limit]:
            source = doc.metadata.get("pdf_name", "Unknown")
            context_parts.append(f"[Source: {source}]\n{doc.page_content}\n")

        formatted_context = "\n---\n".join(context_parts)
        reasoning_steps.append(f"🔗 Using {min(len(all_docs), context_limit)} chunk(s) for context")

        verbatim_mode = _wants_verbatim(question)
        reasoning_steps.append(
            f"📝 Answer mode: {'verbatim line-by-line reproduction' if verbatim_mode else 'synthesized answer'}"
        )

        if verbatim_mode:
            # Reproduce source lines as-is — explicitly forbid summarizing,
            # paraphrasing, or reorganizing into tables.
            template = """{priority_context}Reproduce the source text below EXACTLY as it appears, line by line, in its original order.
        Do NOT summarize, paraphrase, synthesize into a table, deduplicate, or reorganize by topic.
        Keep every line separate, in the same sequence as the context. If a translation is requested,
        place the translation immediately after each original line (still one line at a time) — never
        merge multiple lines into a single summarized statement.
        Only insert a source label (e.g. [Source: filename]) when switching between different source documents.

        Context:
        {context}

        Question: {question}

        Output the content line by line, in original order, exactly as written in the context:"""
        else:
            # RAG prompt template with chain-of-thought
            template = """{priority_context}Answer the question based ONLY on the following context from multiple PDF documents.
        Each section is marked with its source document.

        Use chain-of-thought reasoning:
        1. First, identify which parts of the context are relevant to the question
        2. Analyze the information from each source document
        3. Synthesize the information to form a comprehensive answer
        4. Ensure you cite the source document name for each piece of information
        5. If information comes from multiple sources, mention all relevant sources
        6. If sources contradict, note the discrepancy and cite both sources

        Context:
        {context}

        Question: {question}

        Think step-by-step and provide your answer with source citations:"""

        # Size the answer-generation model's context window to what we're
        # actually sending it, instead of relying on Ollama's default
        # (often 2048 tokens) which can silently truncate long contexts.
        num_ctx = _estimate_num_ctx(template, formatted_context, self.priority_context, question)
        reasoning_steps.append(f"🧮 Sizing model context window: num_ctx={num_ctx}")
        answer_llm = ChatOllama(model=model, num_ctx=num_ctx)

        prompt = ChatPromptTemplate.from_template(template)
        chain = (
            {
                "context": lambda x: formatted_context,
                "question": lambda x: x,
                "priority_context": lambda x: self.priority_context,
            }
            | prompt
            | answer_llm
            | StrOutputParser()
        )

        reasoning_steps.append("💭 Generating answer with source citations...")

        # Check if model supports thinking (e.g., qwen3, deepseek-r1)
        thinking_models = ['qwen3', 'deepseek-r1', 'qwen', 'deepseek']
        supports_thinking = any(tm in model.lower() for tm in thinking_models)

        if supports_thinking:
            reasoning_steps.append("🧠 Using thinking-enabled model with chain-of-thought reasoning...")
            try:
                # Enhanced system message for chain-of-thought reasoning
                if verbatim_mode:
                    cot_system_message = f"""You are an expert AI assistant reproducing source text exactly as written.

{self.priority_context}Reproduce the source text below EXACTLY as it appears, line by line, in its original
order. Do NOT summarize, paraphrase, synthesize into a table, deduplicate, or reorganize by topic.
Keep every line separate, in the same sequence as the context. If a translation is requested, place
the translation immediately after each original line (still one line at a time) — never merge
multiple lines into a single summarized statement. Only insert a source label (e.g. [Source: filename])
when switching between different source documents.

Context from PDF documents:
{formatted_context}

Think about the correct line order and any requested translation, then output the content line by
line, in original order, exactly as written in the context."""
                else:
                    cot_system_message = f"""You are an expert AI assistant that uses chain-of-thought reasoning.

{self.priority_context}Answer the question based ONLY on the provided context from PDF documents.

CHAIN-OF-THOUGHT PROCESS:
1. **Read and understand** the question carefully
2. **Scan the context** to identify all relevant information
3. **Break down** the information by source document
4. **Analyze** how each piece relates to the question
5. **Synthesize** a comprehensive answer
6. **Cite sources** explicitly for every claim

Context from PDF documents:
{formatted_context}

Think through each step carefully, showing your reasoning process."""

                # Use Ollama client directly for thinking-capable models
                ollama_response = ollama.chat(
                    model=model,
                    messages=[
                        {"role": "system", "content": cot_system_message},
                        {"role": "user", "content": f"Question: {question}\n\nThink step-by-step and provide a detailed answer with source citations."}
                    ],
                    think=True,
                    stream=False,
                    options={"num_ctx": num_ctx}
                )

                # Add thinking process to reasoning steps
                if hasattr(ollama_response.message, 'thinking') and ollama_response.message.thinking:
                    thinking_text = ollama_response.message.thinking
                    # Show more of the thinking process (500 chars instead of 200)
                    reasoning_steps.append(f"💡 Model's chain-of-thought:\n{thinking_text[:500]}{'...' if len(thinking_text) > 500 else ''}")

                response = ollama_response.message.content
            except Exception as e:
                print(f"Error using thinking mode, falling back to standard: {e}")
                response = chain.invoke(question)
        else:
            response = chain.invoke(question)

        # Extract source information
        sources = [
            {
                "pdf_name": doc.metadata.get("pdf_name"),
                "pdf_id": doc.metadata.get("pdf_id"),
                "chunk_index": doc.metadata.get("chunk_index", 0)
            }
            for doc in all_docs[:context_limit]
        ]

        reasoning_steps.append("✨ Answer generated successfully!")

        return response, sources, reasoning_steps

    def save_message(
        self,
        session_id: str,
        role: str,
        content: str,
        sources: Optional[List[Dict]],
        db: Session
    ) -> ChatMessage:
        """Save chat message to database.

        Args:
            session_id: Chat session identifier
            role: Message role (user or assistant)
            content: Message content
            sources: Source documents (for assistant messages)
            db: Database session

        Returns:
            Saved chat message
        """
        # Ensure session exists
        session = db.query(ChatSession).filter(ChatSession.session_id == session_id).first()
        if not session:
            session = ChatSession(
                session_id=session_id,
                created_at=datetime.now(),
                last_active=datetime.now()
            )
            db.add(session)
        else:
            session.last_active = datetime.now()

        # Save message
        message = ChatMessage(
            session_id=session_id,
            role=role,
            content=content,
            sources=sources,
            timestamp=datetime.now()
        )
        db.add(message)
        db.commit()
        db.refresh(message)

        return message

    def get_session_messages(self, session_id: str, db: Session) -> List[ChatMessage]:
        """Get all messages for a session.

        Args:
            session_id: Chat session identifier
            db: Database session

        Returns:
            List of chat messages
        """
        return db.query(ChatMessage).filter(
            ChatMessage.session_id == session_id
        ).order_by(ChatMessage.timestamp).all()
