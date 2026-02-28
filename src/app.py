# src/app.py
import os
import uuid
import json
import time
import hashlib
import logging
import tempfile
import asyncio
from typing import List, Tuple, Optional


from fastapi.templating import Jinja2Templates
from fastapi import Request

templates = Jinja2Templates(directory="templates")

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

import threading
import numpy as np
import pdfplumber
import httpx
from cachetools import TTLCache
from concurrent.futures import ThreadPoolExecutor
from rank_bm25 import BM25Okapi

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForSequenceClassification

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# ================= CONFIG =================
CHUNK_SIZE          = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP       = int(os.getenv("CHUNK_OVERLAP", 150))
TOP_K_RETRIEVE      = int(os.getenv("TOP_K_RETRIEVE", 4))    # vectors fetched per retriever
RERANK_TOP_N        = int(os.getenv("RERANK_TOP_N", 2))      # contexts kept after reranking
MAX_CHUNKS          = int(os.getenv("MAX_CHUNKS", 30))
RRF_K               = int(os.getenv("RRF_K", 60))            # RRF constant (higher = smoother fusion)

EMBED_MODEL_NAME    = os.getenv("EMBED_MODEL_NAME", "all-MiniLM-L6-v2")
RERANK_MODEL_NAME   = os.getenv("RERANK_MODEL_NAME",  "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANKER_ONNX_DIR   = os.getenv("RERANKER_ONNX_DIR", "./reranker_onnx")   # path to exported ONNX model
LLAMA_MODEL         = os.getenv("LLAMA_MODEL", "phi3:mini")

COLLECTION_NAME     = os.getenv("COLLECTION_NAME", "rag_collection")
QDRANT_HOST         = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT         = int(os.getenv("QDRANT_PORT", 6333))
OLLAMA_HOST         = os.getenv("OLLAMA_HOST", "localhost")
OLLAMA_PORT         = os.getenv("OLLAMA_PORT", "11434")

EMBED_CACHE_SIZE    = int(os.getenv("EMBED_CACHE_SIZE", 4096))
RESPONSE_CACHE_SIZE = int(os.getenv("RESPONSE_CACHE_SIZE", 2048))
CACHE_TTL_SECONDS   = int(os.getenv("CACHE_TTL_SECONDS", 300))
MAX_WORKERS         = int(os.getenv("MAX_WORKERS", 4))

# Semantic cache similarity threshold (0–1). Hits above this skip the full pipeline.
SEMANTIC_CACHE_THRESHOLD = float(os.getenv("SEMANTIC_CACHE_THRESHOLD", 0.92))

# ================= LOGGING =================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag")

# ================= INIT =================
app = FastAPI(title="Local RAG Service — optimized")

# Bi-encoder (embeddings)
embed_model = SentenceTransformer(EMBED_MODEL_NAME)

# ONNX + INT8 CrossEncoder reranker
# Run  python export_reranker.py  once before starting the server.
# If the directory is missing the server falls back to the original PyTorch
# CrossEncoder so you can still run without the export step.
def _onnx_dir_is_ready(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    has_tokenizer = os.path.exists(os.path.join(path, "tokenizer_config.json"))
    # Look for the quantized model specifically; fallback to any .onnx
    has_quantized = os.path.exists(os.path.join(path, "model_quantized.onnx"))
    has_any_onnx  = any(f.endswith(".onnx") for f in os.listdir(path))
    return has_tokenizer and (has_quantized or has_any_onnx)

_USE_ONNX = _onnx_dir_is_ready(RERANKER_ONNX_DIR)

if _USE_ONNX:
    logger.info("Loading ONNX INT8 reranker from %s", RERANKER_ONNX_DIR)
    reranker_tokenizer = AutoTokenizer.from_pretrained(RERANKER_ONNX_DIR)
    # Load quantized model if available, otherwise fall back to model.onnx
    _quantized_path = os.path.join(RERANKER_ONNX_DIR, "model_quantized.onnx")
    _onnx_file      = "model_quantized.onnx" if os.path.exists(_quantized_path) else "model.onnx"
    reranker_model  = ORTModelForSequenceClassification.from_pretrained(
        RERANKER_ONNX_DIR, file_name=_onnx_file
    )
    logger.info("ONNX reranker ready ✓ (file: %s)", _onnx_file)
else:
    logger.warning(
        "ONNX reranker not found at '%s'. "
        "Run `python export_reranker.py` once to enable the 3-4x speedup. "
        "Falling back to PyTorch CrossEncoder.",
        RERANKER_ONNX_DIR,
    )
    from sentence_transformers import CrossEncoder as _CrossEncoder
    _ce_fallback       = _CrossEncoder(RERANK_MODEL_NAME)
    reranker_tokenizer = None
    reranker_model     = None

# Qdrant
qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

# In-memory caches
embed_cache:    TTLCache = TTLCache(maxsize=EMBED_CACHE_SIZE,    ttl=CACHE_TTL_SECONDS)
response_cache: TTLCache = TTLCache(maxsize=RESPONSE_CACHE_SIZE, ttl=CACHE_TTL_SECONDS)

# Executor for blocking calls
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

# BM25 state (rebuilt on every PDF load)
bm25_index:  Optional[BM25Okapi] = None
bm25_corpus: List[str]           = []

# ================= SEMANTIC CACHE =================
class SemanticCache:
    """
    Caches (query_embedding, answer) pairs.
    On a hit (cosine sim >= threshold) returns the stored answer instantly,
    bypassing reranking and LLM generation entirely.
    """
    def __init__(self, threshold: float = SEMANTIC_CACHE_THRESHOLD):
        self.threshold = threshold
        self._store: List[Tuple[np.ndarray, str]] = []

    def get(self, query_emb: np.ndarray) -> Optional[str]:
        q = query_emb.flatten()
        for emb, answer in self._store:
            sim = float(np.dot(q, emb.flatten()))   # embeddings are already L2-normalised
            if sim >= self.threshold:
                logger.info("semantic_cache hit (sim=%.4f)", sim)
                return answer
        return None

    def set(self, query_emb: np.ndarray, answer: str):
        self._store.append((query_emb.copy(), answer))

    def clear(self):
        self._store.clear()

semantic_cache = SemanticCache()

# ================= SCHEMAS =================
class QueryRequest(BaseModel):
    question: str

# ================= UTILITIES =================
def _hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()

def extract_text_from_pdf(path: str) -> str:
    pages = []
    with pdfplumber.open(path) as pdf:
        for p in pdf.pages:
            pages.append(p.extract_text() or "")
    return "\n\n".join(pages)

def chunk_text(text: str) -> List[str]:
    chunks, start = [], 0
    while start < len(text) and len(chunks) < MAX_CHUNKS:
        end = start + CHUNK_SIZE
        chunks.append(text[start:end].strip())
        start = max(0, end - CHUNK_OVERLAP)
    return [c for c in chunks if c]

def normalize_embedding(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 1:
        return (arr / (np.linalg.norm(arr) + 1e-12)).astype("float32")
    return (arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12)).astype("float32")

async def run_in_executor(func, *args, **kwargs):
    loop = asyncio.get_running_loop()        # Fix2: get_event_loop() deprecated in Py3.10+
    return await loop.run_in_executor(executor, lambda: func(*args, **kwargs))

def make_contexts_hash(contexts: List[str]) -> str:
    return hashlib.md5("||".join(contexts).encode()).hexdigest()[:16]

# ================= COLLECTION SETUP =================
def setup_collection(vector_size: int):
    existing = [c.name for c in qdrant.get_collections().collections]
    if COLLECTION_NAME in existing:
        qdrant.delete_collection(collection_name=COLLECTION_NAME)
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )

def index_embeddings(embeddings: np.ndarray, chunks: List[str]):
    points = [
        PointStruct(id=str(uuid.uuid4()), vector=emb.tolist(), payload={"text": chunks[i]})
        for i, emb in enumerate(embeddings)
    ]
    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)

# ================= BM25 =================
def build_bm25(chunks: List[str]):
    global bm25_index, bm25_corpus
    bm25_corpus = chunks
    bm25_index  = BM25Okapi([c.lower().split() for c in chunks])
    logger.info("BM25 index built over %d chunks", len(chunks))

def _bm25_search(query: str, top_k: int) -> List[Tuple[str, float]]:
    if bm25_index is None:
        return []
    scores      = bm25_index.get_scores(query.lower().split())
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(bm25_corpus[i], float(scores[i])) for i in top_indices]

# ================= HYBRID RETRIEVAL =================
def reciprocal_rank_fusion(
    *result_lists: List[Tuple[str, float]],
    k: int = RRF_K
) -> List[str]:
    """
    Fuses any number of ranked result lists (text, score) using RRF.
    Returns texts sorted by fused score, deduplicated.
    """
    scores: dict[str, float] = {}
    for results in result_lists:
        for rank, (text, _) in enumerate(results):
            scores[text] = scores.get(text, 0.0) + 1.0 / (k + rank + 1)
    return [t for t, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]

# ================= EMBEDDING (cached, thread-safe) =================
_embed_cache_lock = threading.Lock()  # Fix4: threading.Lock is uvloop-safe; TTLCache is not thread-safe

async def embed_texts_async(texts: List[str]) -> np.ndarray:
    # Fix7: build results as a dict keyed by position so cached/computed
    #       entries are always placed at the correct index regardless of order.
    results: dict[int, np.ndarray] = {}
    to_compute: List[Tuple[int, str]] = []   # (position_in_texts, text)

    with _embed_cache_lock:
        for i, t in enumerate(texts):
            key = _hash(t)
            if key in embed_cache:
                results[i] = np.array(embed_cache[key])
            else:
                to_compute.append((i, t))

    if to_compute:
        # Fix5: unpack only raw_texts; positions are tracked via to_compute
        positions, raw_texts = zip(*to_compute)
        computed = await run_in_executor(embed_model.encode, list(raw_texts))
        computed = normalize_embedding(np.array(computed))
        with _embed_cache_lock:
            for pos_in_batch, (orig_pos, text) in enumerate(to_compute):
                embed_cache[_hash(text)] = computed[pos_in_batch].tolist()
                results[orig_pos] = computed[pos_in_batch]

    # Reconstruct ordered array
    return normalize_embedding(np.stack([results[i] for i in range(len(texts))]))

# ================= QDRANT SEARCH =================
async def search_embeddings_async(query_emb: np.ndarray, top_k: int) -> List[Tuple[str, float]]:
    def _search():
        res = qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=query_emb[0].tolist(),
            limit=top_k,
            with_payload=True,
        )
        out = []
        for p in getattr(res, "points", []):
            payload = getattr(p, "payload", {}) or {}
            text    = payload.get("text", "") if isinstance(payload, dict) else payload["text"]
            score   = getattr(p, "score", 0.0) or 0.0
            out.append((text, score))
        return out
    return await run_in_executor(_search)

# ================= ONNX CROSSENCODER RERANKER =================
def _reranker_predict(pairs: List[Tuple[str, str]]) -> np.ndarray:
    """
    Runs the ONNX INT8 CrossEncoder if available, otherwise falls back
    to the original PyTorch CrossEncoder.
    """
    if _USE_ONNX:
        inputs = reranker_tokenizer(
            [p[0] for p in pairs],
            [p[1] for p in pairs],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        outputs = reranker_model(**inputs)
        return outputs.logits.squeeze(-1).detach().numpy()
    else:
        return np.array(_ce_fallback.predict(pairs))

async def rerank_candidates_async(
    question: str, candidates: List[str], top_n: int
) -> List[str]:
    if not candidates:
        return []
    pairs  = [(question, c) for c in candidates]
    scores = await run_in_executor(_reranker_predict, pairs)
    return [c for c, _ in sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)[:top_n]]

# ================= PROMPT BUILDER =================
def build_prompt(question: str, contexts: List[str]) -> str:
    context_block = "\n---\n".join(contexts)
    return (
        "Answer the question based only on the context below. "
        "Be concise and factual.\n\n"
        f"{context_block}\n\nQuestion: {question}"
    )

# ================= ROUTES =================
@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/load_pdf")
async def load_pdf(file: UploadFile = File(...)):
    temp_path: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(await file.read())
            tmp.flush()          # Fix: ensure bytes are written before pdfplumber reads the file
            temp_path = tmp.name

        text   = await run_in_executor(extract_text_from_pdf, temp_path)
        chunks = await run_in_executor(chunk_text, text)
        if not chunks:
            raise HTTPException(status_code=422, detail="No extractable text found in PDF. Is it a scanned/image PDF?")

        embeddings = await embed_texts_async(chunks)

        await run_in_executor(setup_collection, embeddings.shape[1])
        await run_in_executor(index_embeddings, embeddings, chunks)
        await run_in_executor(build_bm25, chunks)

        # Clear caches since knowledge base changed
        response_cache.clear()
        semantic_cache.clear()

        return {"status": "PDF processed successfully", "chunks": len(chunks)}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("load_pdf failed: %s", str(e))
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")
    finally:
        # Fix3: always clean up the temp file
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


@app.post("/query")
async def query(request: QueryRequest):
    """Non-streaming endpoint. Returns the full answer in one shot."""
    start    = time.perf_counter()
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Empty question")

    # 1. Embed query first so we can check the semantic cache
    q_start = time.perf_counter()
    q_emb   = await embed_texts_async([question])
    t_embed = time.perf_counter() - q_start

    # 2. Semantic cache check (skips retrieval + rerank + generation)
    cached_answer = semantic_cache.get(q_emb)
    if cached_answer:
        return {
            "answer": cached_answer,
            "meta":   {"cache": "semantic_hit", "total_ms": round((time.perf_counter() - start) * 1000, 1)},
        }

    # 3. Hybrid retrieval + rerank
    r_start = time.perf_counter()
    dense_task = search_embeddings_async(q_emb, TOP_K_RETRIEVE)
    bm25_task  = run_in_executor(_bm25_search, question, TOP_K_RETRIEVE)
    dense_results, bm25_results = await asyncio.gather(dense_task, bm25_task)
    fused    = reciprocal_rank_fusion(dense_results, bm25_results)
    contexts = await rerank_candidates_async(question, fused, RERANK_TOP_N)
    t_retrieval_rerank = time.perf_counter() - r_start

    # 4. Exact response cache (same question + same contexts)
    cache_key = f"{_hash(question)}|{make_contexts_hash(contexts)}"
    if cache_key in response_cache:
        return {
            "answer": response_cache[cache_key],
            "meta":   {"cache": "response_hit", "total_ms": round((time.perf_counter() - start) * 1000, 1)},
        }

    # 5. LLM generation
    g_start = time.perf_counter()
    prompt  = build_prompt(question, contexts)
    url     = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/generate"
    async with httpx.AsyncClient(timeout=140.0) as client:
        resp = await client.post(url, json={"model": LLAMA_MODEL, "prompt": prompt, "stream": False})
        if resp.status_code != 200:
            raise HTTPException(status_code=500, detail=f"LLM error: {resp.text}")
        answer = resp.json().get("response", "")
    t_gen = time.perf_counter() - g_start

    response_cache[cache_key] = answer
    semantic_cache.set(q_emb, answer)

    total = time.perf_counter() - start
    telemetry = {
        "timings": {
            "embed_ms":             round((time.perf_counter() - start - t_retrieval_rerank - t_gen) * 1000, 1),
            "retrieval_rerank_ms":  round(t_retrieval_rerank * 1000, 1),
            "generation_ms":        round(t_gen * 1000, 1),
            "total_ms":             round(total * 1000, 1),
        },
        "contexts_used": len(contexts),
        "cache": "miss",
    }
    logger.info("query finished: %s", telemetry)
    return {"answer": answer, "meta": telemetry}


@app.post("/query_stream")
async def query_stream(request: QueryRequest):
    """
    Streaming endpoint — tokens are pushed to the client as they are generated.
    Uses Server-Sent Events (text/event-stream).
    The full pipeline (embed → hybrid retrieve → rerank) still runs before
    generation starts, but the user sees the first token within ~300ms
    of generation beginning instead of waiting for the complete response.
    """
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Empty question")

    q_emb = await embed_texts_async([question])

    # Semantic cache — stream the cached answer token-by-token so UX is consistent
    cached = semantic_cache.get(q_emb)
    if cached:
        async def cached_stream():
            for word in cached.split(" "):
                yield f"data: {json.dumps({'token': word + ' '})}\n\n"
                await asyncio.sleep(0)   # yield control
            yield "data: [DONE]\n\n"
        return StreamingResponse(cached_stream(), media_type="text/event-stream")

    # Hybrid retrieval + rerank
    dense_task = search_embeddings_async(q_emb, TOP_K_RETRIEVE)
    bm25_task  = run_in_executor(_bm25_search, question, TOP_K_RETRIEVE)
    dense_results, bm25_results = await asyncio.gather(dense_task, bm25_task)
    fused    = reciprocal_rank_fusion(dense_results, bm25_results)
    contexts = await rerank_candidates_async(question, fused, RERANK_TOP_N)
    prompt   = build_prompt(question, contexts)

    # Accumulate the full answer so we can cache it after streaming
    answer_parts: List[str] = []

    async def token_stream():
        url = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/generate"
        try:
            async with httpx.AsyncClient(timeout=140.0) as client:
                async with client.stream(
                    "POST", url,
                    json={"model": LLAMA_MODEL, "prompt": prompt, "stream": True}
                ) as resp:
                    async for line in resp.aiter_lines():
                        if not line:
                            continue
                        chunk = json.loads(line)
                        token = chunk.get("response", "")
                        if token:
                            answer_parts.append(token)
                            yield f"data: {json.dumps({'token': token})}\n\n"
                        if chunk.get("done"):
                            # Store in both caches once generation is complete
                            full_answer = "".join(answer_parts)
                            cache_key = f"{_hash(question)}|{make_contexts_hash(contexts)}"
                            response_cache[cache_key] = full_answer
                            semantic_cache.set(q_emb, full_answer)
                            yield "data: [DONE]\n\n"
                            break
        except Exception as e:
            logger.exception("token_stream error")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(token_stream(), media_type="text/event-stream")


# ================= UI =================
@app.get("/", response_class=HTMLResponse)
async def ui(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})