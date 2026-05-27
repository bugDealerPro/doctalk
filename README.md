# DocTalk

Lightweight PDF chat application built with **Chainlit** and **LangGraph**. Upload one or more PDFs in a chat session, index them in **Postgres + pgvector**, and ask questions grounded in retrieved chunks with focused source previews.

## Problem statement

Reviewers often need to ask questions across PDFs without reading every page. DocTalk is a small, production-minded vertical slice: ingest PDFs, chunk and embed text, retrieve relevant passages per session, and answer with citations.

## Features

- **Multi-PDF sessions** — upload several PDFs in one Chainlit chat; each upload is appended to the session corpus
- **Session-scoped retrieval** — vector search runs only on the current `session_id`, never across other chats
- **LangGraph QA workflow** — validate → retrieve → generate with explicit prompts
- **Grounded answers** — LLM answers from retrieved excerpts only; states when the document does not contain the requested information
- **Focused sources** — up to 3 previews in the form `filename.pdf (chunk N): …`, centered on query terms
- **Robust PDF ingestion** — size limits, invalid PDF handling, empty-text detection; temp files deleted after processing
- **OpenAI-compatible APIs** — works with OpenAI or any compatible gateway via `OPENAI_BASE_URL`
- **Docker Compose** — app + pgvector Postgres with one command
- **Makefile** — common dev commands (`make up`, `make test`, …)

## Architecture

```mermaid
flowchart TB
    subgraph ui [Chainlit UI]
        User([User])
        Upload[PDF upload]
        Chat[Chat message]
    end

    subgraph app [Application]
        Main[main.py]
        Services[services.py]
        PDF[pdf_loader.py]
        Graph[graph.py LangGraph]
        VS[vector_store.py]
    end

    subgraph data [Postgres + pgvector]
        Docs[(documents)]
        Chunks[(document_chunks)]
    end

    LLM[(OpenAI-compatible LLM)]

    User --> Upload
    User --> Chat
    Upload --> Main
    Chat --> Main
    Main --> Services
    Services --> PDF
    PDF -->|extract + chunk| Services
    Services -->|embed + store| VS
    VS --> Docs
    VS --> Chunks
    Services -->|question| Graph
    Graph -->|retrieve by session_id| VS
    Graph --> LLM
    Graph -->|answer + sources| Main
```

**Ingestion flow**

1. User uploads PDF(s) in Chainlit.
2. `services.ingest_pdf_file` writes a temp file, extracts text (`pypdf`), chunks text (`RecursiveCharacterTextSplitter`), embeds chunks, stores rows, deletes the temp file.
3. UI confirms: `Uploaded X.pdf. Current session has N document(s) indexed.`

**Q&A flow (LangGraph)**

1. **Validate** — session has at least one indexed document.
2. **Retrieve** — top-k chunks across all session documents (deduplicated).
3. **Generate** — LLM answer from excerpts; citations filtered to query-relevant chunks (max 3).

**Retrieval**

Retrieval uses cosine similarity search over pgvector embeddings scoped to the current Chainlit session.

## Validated against

- Small single-page PDFs
- 50-page PDFs
- 200-page PDFs (~800 chunks)
- Table-heavy PDFs
- Noisy retrieval datasets
- Multi-document conflicting corpora
- Empty PDFs
- Corrupted PDFs
- Image-only PDFs without OCR

## Data model

| Table | Purpose |
|-------|---------|
| `documents` | Per-upload metadata: `session_id`, `filename`, `uploaded_at`, `page_count`, `chunk_count` |
| `document_chunks` | Chunk text + embedding; denormalized `session_id` and `filename` for fast session retrieval |

Retrieval filters on `session_id`. `document_id` is kept for metadata and source labels.

## Commands

| Command | Description |
|---------|-------------|
| `make env` | Copy `.env.example` → `.env` (no overwrite) |
| `make up` | Build and start app + Postgres (`http://localhost:8000`) |
| `make down` | Stop Docker Compose stack |
| `make test` | Run pytest |
| `make dev` | Run Chainlit locally (`chainlit run app/main.py`) |

Equivalent without Make:

```bash
docker compose up --build    # production-like stack
docker compose down
pytest
chainlit run app/main.py
```

### Quick start (Docker)

```bash
make env
# Edit .env — set OPENAI_API_KEY

make up
```

Open [http://localhost:8000](http://localhost:8000), upload one or more text-based PDFs, then ask questions.

**Important:** Use `DOCTALK_DATABASE_URL` for the app database. Do not set `DATABASE_URL` in `.env` — Chainlit would try to enable its own persistence layer and break the UI.

### Local development (without Docker)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

make env
# set OPENAI_API_KEY in .env

# Postgres with pgvector must be running, e.g.:
docker compose up db -d

make test
make dev
```

For `make dev`, ensure `.env` has `DOCTALK_DATABASE_URL=postgresql://doctalk:doctalk@localhost:5432/doctalk`.

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | — | Required API key |
| `OPENAI_BASE_URL` | — | Optional base URL (Ollama, vLLM, etc.) |
| `LLM_MODEL` | `gpt-4o-mini` | Chat model |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `EMBEDDING_DIMENSIONS` | `1536` | Vector size (must match model) |
| `DOCTALK_DATABASE_URL` | `postgresql://doctalk:doctalk@localhost:5432/doctalk` | App Postgres URL |
| `MAX_PDF_SIZE_MB` | `20` | Max upload size per PDF |
| `CHUNK_SIZE` | `800` | Characters per chunk |
| `CHUNK_OVERLAP` | `150` | Chunk overlap |
| `RETRIEVAL_TOP_K` | `4` | Chunks retrieved per question |
| `MAX_SOURCE_PREVIEWS` | `3` | Max source lines in answers |

Docker Compose sets `DOCTALK_DATABASE_URL` for the app service to point at the `db` container.

## What is implemented

- Chainlit UI (port 8000) with multi-file PDF upload (up to 10 per message)
- Multi-document session corpus; uploads append, do not replace
- PDF text extraction, chunking, embedding, pgvector storage
- Idempotent DB init + migration from legacy `chunks` table
- LangGraph workflow with validation, session retrieval, generation
- Query-focused citation filtering and filename-based source previews
- Clear user-facing errors (bad PDF, empty text, API failures)
- Unit + integration tests (multi-doc Nythera/Zorvessa fixtures when Postgres is available)

## Intentionally skipped

- Authentication and multi-tenant isolation beyond Chainlit session IDs
- OCR for scanned/image-only PDFs
- Background workers and job queues
- Object storage (S3, etc.)
- Reranking, hybrid BM25 + vector search, agent tools
- Production observability, rate limiting, horizontal scaling

## Known limitations

- **Text-only PDFs** — no OCR; scanned pages without a text layer fail gracefully.
- **Session boundary** — no cross-session or cross-user document search.
- **Synchronous ingestion** — large PDFs block until indexing finishes.
- **Embedding/model coupling** — changing embedding models requires matching `EMBEDDING_DIMENSIONS` and re-indexing.
- **Vector-only retrieval** — rare entity IDs may rank below semantically similar but wrong passages on very small corpora.

## Production improvement plan

1. **Async ingestion** — background workers and upload progress in the UI.
2. **Auth + tenancy** — user accounts and document ownership.
3. **Hybrid retrieval** — BM25 + vectors, reranking, metadata filters.
4. **OCR** — scanned PDF support.
5. **Observability** — structured logs, tracing, health checks.
6. **Alembic migrations** — replace idempotent `CREATE IF NOT EXISTS` bootstrap.

## Project layout

```
app/
  main.py          # Chainlit handlers
  services.py      # Upload + chat orchestration
  graph.py         # LangGraph QA workflow
  pdf_loader.py    # Extraction + chunking
  vector_store.py  # Embeddings, storage, retrieval, citations
  db.py            # Connection + schema init
  config.py        # Environment settings
  prompts.py       # LLM prompts
tests/
  fixtures/        # Planet dossier text for multi-doc tests
Makefile
docker-compose.yml
```

## Tests

```bash
make test
```

Covers PDF failure handling, chunking, temp-file cleanup, graph validation, citation filtering, and multi-document session retrieval. LLM/embedding calls are mocked in unit tests; Postgres integration tests use deterministic fake embeddings when the database is reachable (`docker compose up db -d`).

Example multi-doc scenarios (see `tests/test_multi_document.py`):

- “Which planet has magnetic storms every 17 hours?” → Nythera
- “Which planet has magnetic storms every 29 hours?” → Zorvessa
- “Compare Nythera and Zorvessa moons.” → Etris/Vallun vs Pala/Rix/Ond
- Retrieval from another session must not leak into the current session
