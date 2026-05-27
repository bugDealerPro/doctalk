# DocTalk

Minimal PDF chat application: upload one or more PDFs in Chainlit, ask questions across the session corpus, and get answers grounded in retrieved document chunks stored in Postgres with pgvector.

## Problem statement

Reviewers often need to ask questions against a PDF without reading the entire document. DocTalk provides a small but realistic vertical slice: ingest a PDF, index its text, and answer questions with source previews.

## Architecture

```mermaid
flowchart LR
    User --> Chainlit
    Chainlit --> Services
    Services --> PDFLoader
    Services --> VectorStore
    Services --> LangGraph
    LangGraph --> VectorStore
    LangGraph --> LLM
    VectorStore --> Postgres
    PDFLoader --> TempFile
```

**Flow**

1. **Upload** — Chainlit receives PDF(s), `services.ingest_pdf_file` writes each to a temp file, extracts text, chunks it, embeds chunks, stores metadata + vectors, then deletes the temp file. New uploads are appended to the current session corpus.
2. **Chat** — User question enters a LangGraph workflow:
   - validate the session has indexed documents
   - retrieve top-k chunks across all session documents via cosine similarity
   - call an OpenAI-compatible LLM with retrieved context
   - return answer with focused source previews (`filename`, chunk index)
3. **Storage** — Multiple documents per Chainlit `session_id`. Retrieval is always scoped to the current session.

## How to run

### Prerequisites

- Docker and Docker Compose
- An OpenAI-compatible API key

### Quick start

```bash
cp .env.example .env
# Edit .env and set OPENAI_API_KEY

docker compose up --build
```

Open [http://localhost:8000](http://localhost:8000), upload one or more text-based PDFs, then ask questions.

### Local development (without Docker)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Start Postgres with pgvector separately, then:
cp .env.example .env
export $(grep -v '^#' .env | xargs)

pytest
chainlit run app/main.py
```

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | — | Required API key |
| `OPENAI_BASE_URL` | — | Optional base URL for compatible providers |
| `LLM_MODEL` | `gpt-4o-mini` | Chat model |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `EMBEDDING_DIMENSIONS` | `1536` | Vector dimension (must match model) |
| `DOCTALK_DATABASE_URL` | local Postgres URL | App database connection string |
| `MAX_PDF_SIZE_MB` | `20` | Upload size limit |
| `CHUNK_SIZE` | `800` | Characters per chunk |
| `CHUNK_OVERLAP` | `150` | Chunk overlap |
| `RETRIEVAL_TOP_K` | `4` | Chunks retrieved per question |
| `MAX_SOURCE_PREVIEWS` | `3` | Max source lines shown in answers |

## What is implemented

- Chainlit UI with multi-PDF upload and chat
- Multi-document session corpus with session-scoped retrieval
- PDF text extraction with `pypdf`
- Recursive character chunking
- OpenAI-compatible embeddings + chat completions
- Postgres + pgvector storage with idempotent schema init
- LangGraph QA workflow with validation, retrieval, generation, and source previews
- Temp PDF cleanup after ingestion
- File size checks and user-facing error messages
- Docker Compose stack (app + Postgres)
- Fast unit tests with mocked LLM/embedding paths

## Intentionally skipped

- Authentication and multi-tenant isolation beyond Chainlit session IDs
- OCR for scanned/image PDFs
- Background workers and job queues
- Object storage (S3, etc.)
- Reranking, hybrid search, or agent tooling
- Production observability, rate limiting, and horizontal scaling

## Known limitations

- **Text-only PDFs** — scanned documents without a text layer will fail gracefully.
- **Session corpus only** — retrieval never crosses Chainlit sessions.
- **Synchronous ingestion** — large PDFs block the request until indexing completes.
- **Session-scoped retrieval** — no cross-user document sharing.
- **Embedding dimension coupling** — changing embedding models requires matching `EMBEDDING_DIMENSIONS` and likely re-indexing.

## Production improvement plan

1. **Async ingestion** — queue PDF processing (Celery/RQ) and show progress in UI.
2. **Auth + tenancy** — user accounts, document ownership, and access control.
3. **Durable object storage** — optional raw PDF retention with lifecycle policies.
4. **Better retrieval** — reranking, metadata filters, hybrid BM25 + vector search.
5. **OCR pipeline** — handle scanned PDFs via Tesseract or a managed OCR service.
6. **Observability** — structured logging, tracing, metrics, and health endpoints.
7. **Migrations** — Alembic instead of idempotent `CREATE IF NOT EXISTS`.
8. **Testing** — integration tests against Testcontainers Postgres and contract tests for LLM providers.

## Project layout

```
app/
  main.py          # Chainlit handlers
  services.py      # Upload + chat orchestration
  graph.py         # LangGraph workflow
  pdf_loader.py    # Extraction + chunking
  vector_store.py  # Embeddings + retrieval
  db.py            # Connection + schema init
  config.py        # Environment config
  prompts.py       # Prompt templates
tests/
```

## Tests

```bash
pytest
```

Tests cover PDF failure handling, chunking, empty chunk rejection, temp-file cleanup, graph behavior when no document is uploaded, and multi-document session retrieval (Nythera vs Zorvessa fixtures). LLM and embedding calls are mocked in unit tests; retrieval integration tests use deterministic fake embeddings against Postgres when available.
