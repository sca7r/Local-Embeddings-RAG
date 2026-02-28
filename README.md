# Local RAG System (PDF-based)

A **fully local Retrieval-Augmented Generation (RAG) system** that answers questions from PDF documents using a high-performance hybrid search pipeline, no cloud APIs, no data leaving your machine.

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-green.svg)
![Qdrant](https://img.shields.io/badge/Qdrant-vector--store-red.svg)
![Ollama](https://img.shields.io/badge/Ollama-phi3:mini-orange.svg)
![ONNX](https://img.shields.io/badge/Reranker-ONNX%20INT8-purple.svg)

---

##  What's New

The system has been completely re-architected from a CLI script into a production-grade web service:

- **FastAPI backend** with a browser-based chat UI
- **Hybrid retrieval** — dense vector search (Qdrant) + BM25 keyword search, fused with Reciprocal Rank Fusion (RRF)
- **ONNX INT8 quantized reranker** — 3–4× faster cross-encoder inference via `export_reranker.py`
- **Semantic cache** — near-duplicate queries are answered instantly, bypassing the full pipeline
- **Streaming responses** via Server-Sent Events (`/query_stream`)
- **Fully configurable** via environment variables

---

## How It Works

```
PDF Upload
    │
    ▼
Text Extraction (pdfplumber)
    │
    ▼
Overlapping Chunking
    │
    ├─── Dense Embeddings ──► Qdrant Vector Store
    └─── BM25 Index (in-memory)
              │
              ▼
     Hybrid Retrieval (Top-K each)
              │
              ▼
     Reciprocal Rank Fusion (RRF)
              │
              ▼
     ONNX INT8 Cross-Encoder Reranker
              │
              ▼
     Local LLM (Ollama / phi3:mini)
              │
              ▼
         Answer (streaming or batch)
```

---

## Project Structure

```
Local-Embeddings-RAG/
├── src/
│   └── app.py               # FastAPI application (main backend)
|    └── export_reranker.py       # One-time ONNX export & INT8 quantization script
├── templates/
│   └── index.html           # Browser-based chat UI 
├── requirements.txt
└── README.md
```

---

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai) installed and running
- [Qdrant](https://qdrant.tech/documentation/guides/installation/) running locally (Docker recommended)

### 1. Install Ollama & pull a model

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull phi3:mini
```

> The default LLM is `phi3:mini`. You can switch to any Ollama model via the `LLAMA_MODEL` environment variable (e.g. `llama3`, `mistral`).

### 2. Start Qdrant

```bash
docker run -d -p 6333:6333 qdrant/qdrant
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/sca7r/Local-Embeddings-RAG.git
cd Local-Embeddings-RAG

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Setup: Export the ONNX Reranker (Run Once)

Before starting the server for the first time, export and quantize the cross-encoder reranker to ONNX INT8 format. This provides a 3–4× speedup over the default PyTorch backend.

```bash
python src/export_reranker.py
```

This will create a `./reranker_onnx/` directory containing the quantized model. The server falls back to the PyTorch CrossEncoder automatically if this step is skipped.

---

## Usage

```bash
uvicorn src.app:app --host 0.0.0.0 --port 8000
```

Then open **http://localhost:8000** in your browser.

1. Upload a PDF using the UI
2. Wait for the document to be processed (chunked, embedded, indexed)
3. Ask questions — answers stream back in real time

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Browser-based chat UI |
| `GET` | `/health` | Health check |
| `POST` | `/load_pdf` | Upload and index a PDF |
| `POST` | `/query` | Ask a question (full response) |
| `POST` | `/query_stream` | Ask a question (streaming SSE) |

---

## Configuration

All parameters are tunable via environment variables — no code changes needed.

| Variable | Default | Description |
|----------|---------|-------------|
| `CHUNK_SIZE` | `1000` | Characters per chunk |
| `CHUNK_OVERLAP` | `150` | Overlap between chunks |
| `TOP_K_RETRIEVE` | `4` | Chunks fetched per retriever (dense + BM25) |
| `RERANK_TOP_N` | `2` | Contexts kept after reranking |
| `RRF_K` | `60` | RRF fusion constant |
| `EMBED_MODEL_NAME` | `all-MiniLM-L6-v2` | Bi-encoder model |
| `RERANK_MODEL_NAME` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Reranker model |
| `RERANKER_ONNX_DIR` | `./reranker_onnx` | Path to exported ONNX reranker |
| `LLAMA_MODEL` | `phi3:mini` | Ollama model name |
| `SEMANTIC_CACHE_THRESHOLD` | `0.92` | Cosine similarity threshold for cache hits |
| `QDRANT_HOST` | `localhost` | Qdrant host |
| `QDRANT_PORT` | `6333` | Qdrant port |
| `OLLAMA_HOST` | `localhost` | Ollama host |
| `OLLAMA_PORT` | `11434` | Ollama port |

Example — use a different LLM and lower the rerank threshold:

```bash
LLAMA_MODEL=llama3 RERANK_TOP_N=3 uvicorn src.app:app --host 0.0.0.0 --port 8000
```

---

## Core Concepts

**Chunking** — PDF text is split into fixed-size character chunks with overlap, so concepts are never cut off at boundaries.

**Bi-Encoder Retrieval (Dense)** — Chunks are embedded with `all-MiniLM-L6-v2` and stored in Qdrant. At query time, the question is embedded and the top-K most similar chunks are retrieved by cosine similarity.

**BM25 Retrieval (Sparse)** — A classic keyword-based index runs in parallel with the dense retriever, catching exact-match terms the semantic search may miss.

**Reciprocal Rank Fusion (RRF)** — Results from both retrievers are merged into a single ranked list using RRF, which combines rank positions without needing score normalization.

**ONNX INT8 Cross-Encoder Reranker** — The fused candidates are scored jointly with the query using a quantized cross-encoder, promoting the most relevant chunks to the top.

**Semantic Cache** — Each query embedding is compared to previously cached queries. If cosine similarity exceeds the threshold (default 0.92), the stored answer is returned immediately — no retrieval or LLM call needed.

**Local LLM Generation** — Answers are generated by a locally running model via Ollama, constrained strictly to the retrieved context to minimize hallucinations.

---

## Troubleshooting

**Qdrant connection refused**
Make sure Qdrant is running: `docker ps`. If not, start it with `docker run -d -p 6333:6333 qdrant/qdrant`.

**Ollama connection issues**
```bash
ollama list          # check available models
ollama pull phi3:mini
ollama run phi3:mini  # test manually
```

**Scanned / image PDF returns no results**
`pdfplumber` extracts text-layer PDFs only. Run the PDF through an OCR tool (e.g. `ocrmypdf`) before uploading.

**ONNX reranker not loading**
Re-run `python export_reranker.py`. If you see errors about multiple `.onnx` files, the script automatically clears the stale directory before re-exporting. Check that both `tokenizer_config.json` and `model_quantized.onnx` exist in `./reranker_onnx/`.

---

## Design Goals

- **Privacy-preserving** — everything runs locally, no data sent to external APIs
- **Fast** — ONNX INT8 reranker, semantic cache, async pipeline, and thread-pool executor
- **Transparent** — per-request telemetry (embed ms, retrieval ms, generation ms) logged on every query
- **Tunable** — every parameter exposed as an environment variable