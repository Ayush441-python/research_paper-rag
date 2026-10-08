# Production-Ready Research RAG System

An end-to-end Retrieval-Augmented Generation (RAG) platform for conversational question answering over academic research papers and PDF documents. It combines dense vector search, BM25 keyword retrieval, LLM-based query expansion, and strict grounding prompts, backed by a DeepEval evaluation suite and served through FastAPI and Streamlit.

Benchmarked on the paper *"Attention Is All You Need"*.

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Pipeline Details](#pipeline-details)
- [Evaluation Framework](#evaluation-framework)
- [Performance, Latency & Cost](#performance-latency--cost)
- [API Reference](#api-reference)
- [Getting Started](#getting-started)
- [Observability](#observability)
- [Deployment](#deployment)

---

## Features

- **Hybrid retrieval**: Multi-Query dense retrieval + BM25 sparse search, merged with Reciprocal Rank Fusion (RRF).
- **Query transformation**: An LLM rewrites each question into multiple perspectives to bridge the user-to-document vocabulary gap.
- **Grounded generation**: Zero-temperature Groq LLM with prompts that restrict answers to retrieved evidence, require citations, and decline out-of-scope questions.
- **Resilient storage**: Redis vector store in production, with automatic fallback to an in-memory store when Redis is unreachable.
- **Rigorous evaluation**: 100 curated golden test cases across 6 quality and safety dimensions using DeepEval.
- **Full-stack interface**: FastAPI backend plus a Streamlit chat UI with upload, health monitoring, and multi-turn history.
- **Observability**: LangSmith tracing for latency, token usage, and retrieval scores.

---

## Architecture

```
User PDF Upload ──► PyPDF Loader ──► Recursive Chunking (500 chars / 50 overlap)
                                              │
                     ┌────────────────────────┴────────────────────────┐
                     ▼                                                 ▼
          HuggingFace BGE Embeddings                        In-Memory / Redis Store
                     │                                                 │
                     ▼                                                 ▼
User Query ──► MultiQueryRetriever (Groq LLM) ──┐             Keyword Search (BM25)
                     │                          │                      │
                     └──────────────────────────┼──────────────────────┘
                                                ▼
                                   Reciprocal Rank Fusion (RRF, k=60)
                                                │
                                                ▼
                                    Top-5 Ranked Context Chunks
                                                │
                                                ▼
                                  Grounded LLM Prompt (Groq)
                                                │
                                                ▼
                                    Traceable Answer + Citations
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Groq via `ChatGroq` (e.g. `openai/gpt-oss-20b`, LLaMA models), temperature 0 |
| Embeddings | `BAAI/bge-small-en-v1.5` (384-dim) via `langchain-huggingface` |
| Vector store | Redis (RedisVL / `langchain-redis`) with in-memory fallback |
| Sparse retrieval | BM25 via `rank-bm25` |
| Orchestration | LangChain |
| PDF parsing | `pypdf` |
| Backend | FastAPI, Uvicorn / Gunicorn |
| Frontend | Streamlit |
| Evaluation | DeepEval (LLM-as-a-Judge: `openai/gpt-oss-120b` via Groq) |
| Tracing | LangSmith |
| Hosting | Render |

---

## Project Structure

```
.
├── main.py                  # FastAPI application
├── app.py                   # Streamlit UI
├── src/
│   ├── rag_pipeline.py      # Pipeline orchestration (LangSmith @traceable)
│   ├── vectorstore/
│   │   └── redis.py         # Redis store + in-memory fallback
│   ├── retriever/
│   │   ├── mqr.py           # Multi-Query Retriever
│   │   └── hybrid.py        # BM25 + dense fusion via RRF
│   └── llm/
│       └── prompt.py        # Grounded system prompts
├── goldens/                 # DeepEval benchmark datasets
│   ├── retriever_dataset.json
│   ├── faithfullness_dataset.json
│   ├── correctness_dataset.json
│   ├── scpoe_dataset.json
│   ├── leakage_dataset.json
│   └── toxicity_dataset.json
├── latency_evals.py         # Latency SLO checks
├── cost_evals.py            # Token cost checks
├── render.yaml              # Render infrastructure config
└── Procfile
```

> Dataset file names (`faithfullness_dataset.json`, `scpoe_dataset.json`) are kept exactly as they appear in the repository.

---

## Pipeline Details

### Ingestion and Storage

| Parameter | Value |
|---|---|
| Parser | `pypdf` |
| Splitter | LangChain `RecursiveCharacterTextSplitter` |
| Chunk size | 500 characters |
| Chunk overlap | 50 characters (10%) |
| Embedding model | `BAAI/bge-small-en-v1.5` (384 dimensions, cached with `lru_cache(maxsize=1)`) |
| Redis index name | `pdf_rag` |

### Retrieval

1. **Multi-Query Retriever** (`src/retriever/mqr.py`): the LLM rewrites the user question into several query variants. Dense similarity search returns the **top 20** candidates.
2. **BM25** (`rank-bm25`): lexical matching over document segments, useful for technical terms, mathematical notation, and entity names.
3. **Reciprocal Rank Fusion** (`src/retriever/hybrid.py`): merges both ranked lists.

```
Score(d) = Σ over m ∈ {MQR, BM25}  of  1 / (k_rrf + rank_m(d) + 1),    k_rrf = 60
```

4. The **top 5** chunks are passed to the generator.

### Generation

- Groq `ChatGroq` at temperature 0 for deterministic output.
- `src/llm/prompt.py` instructs the model to rely only on retrieved evidence, cite relevant chunks, and refuse out-of-scope questions.

---

## Evaluation Framework

All quality metrics use an acceptance threshold of **≥ 0.70**, scored by an LLM judge (`openai/gpt-oss-120b` via Groq). Toxicity is the exception: it must score **≤ 0.10**.

| Dimension | Dataset size | Test cases | Metric and target |
|---|---|---|---|
| Faithfulness | 58.8 KB | 30 | Factual consistency ≥ 0.70, no hallucinations |
| Retriever quality | 12.8 KB | 10 | Contextual Relevancy, Precision, Recall ≥ 0.70 (also Hit Rate) |
| Correctness | 8.5 KB | 15 | GEval / semantic similarity vs. ideal answers ≥ 0.70 |
| Scope enforcement | 5.4 KB | 15 | Out-of-domain rejection ≥ 0.70 |
| Data leakage | 4.9 KB | 15 | Prompt-injection and system-prompt leakage defense ≥ 0.70 |
| Toxicity and safety | 3.5 KB | 15 | Toxicity ≤ 0.10 |
| **Total** | **~94 KB** | **100** | Comprehensive RAG reliability benchmark |

---

## Performance, Latency & Cost

These are the **targets (SLOs and budgets)** enforced by `latency_evals.py` and `cost_evals.py`.

### Latency

| Metric | Target |
|---|---|
| End-to-end P95 latency | ≤ 3,000 ms |
| Time to first token (P95) | ≤ 1,200 ms |
| Retry policy | Exponential backoff, base 0.5 s (`0.5 × 2^attempt`), max 2 retries (3 attempts total) |

### Cost

| Item | Value |
|---|---|
| Per-query budget | ≤ $0.0015 (about ₹0.13) |
| Uncached input tokens | $0.15 / 1M |
| Cached input tokens | $0.075 / 1M (50% reduction) |
| Output tokens | $0.60 / 1M |
| Projected at 2,000 queries/day | ≤ $3.00/day, ≤ $90/month (about ₹7,920) |

---

## API Reference

| Endpoint | Description |
|---|---|
| `GET /api/health` | Health check for orchestration and uptime monitoring |
| `POST /api/upload` | Multipart PDF upload, chunking, and embedding ingestion |
| `POST /api/ingestion` | Raw text batch ingestion |
| `POST /api/retriever` | Standalone retrieval for debugging rankings and top-k chunks |
| `POST /api/chat` | Full RAG chat: returns answer, matched source passages, and metadata |

---

## Getting Started

### Prerequisites

- Python 3.10+
- A Groq API key
- (Optional) A running Redis instance with vector search support
- (Optional) A LangSmith API key for tracing

### Installation

```bash
git clone <your-repo-url>
cd <your-repo-name>
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Configuration

Create a `.env` file (variable names below are typical; match them to your code):

```env
GROQ_API_KEY=your_groq_key
REDIS_URL=redis://localhost:6379        # optional; falls back to in-memory if unreachable
LANGSMITH_API_KEY=your_langsmith_key    # optional
LANGSMITH_TRACING=true                  # optional
```

### Run

```bash
# Backend
uvicorn main:app --reload --port 8000

# Frontend (in a second terminal)
streamlit run app.py
```

### Run Evaluations

```bash
deepeval test run <path-to-your-test-file>.py
```

---

## Streamlit UI

- Live backend connection indicator with auto-reconnect
- PDF upload with on-the-fly indexing and loading spinners
- Chat interface with multi-turn history and session management

---

## Observability

Pipeline stages in `src/rag_pipeline.py` are instrumented with LangSmith `@traceable` decorators, giving step-by-step traces of latency, token consumption, and retrieval scores for each chain run.

---

## Deployment

- Declarative infrastructure in `render.yaml` and `Procfile` for **Render**
- Production serving with Uvicorn / Gunicorn
- `/api/health` is available for health checks

---

## Highlights

- Hybrid RAG pipeline (Multi-Query + BM25 fused with RRF, k = 60) returning the top-5 context passages from 500-character chunks with 10% overlap
- Latency SLOs of ≤ 1,200 ms TTFT and ≤ 3,000 ms P95 end-to-end, with exponential-backoff retries
- 100 golden test cases across 6 dimensions with a ≥ 0.70 quality gate (toxicity ≤ 0.10)
- Cost budget of ≤ $0.0015 per query using prompt caching and local BGE embeddings
- FastAPI + Redis + Streamlit stack with LangSmith tracing