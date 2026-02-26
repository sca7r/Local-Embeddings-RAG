import os
import uuid
from typing import List

import numpy as np
import pdfplumber
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from fastapi.responses import HTMLResponse


# ================= CONFIG =================
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K = 2
MAX_CHUNKS = 30
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
LLAMA_MODEL = "phi3:mini"
COLLECTION_NAME = "rag_collection"

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "localhost")
OLLAMA_PORT = os.getenv("OLLAMA_PORT", "11434")

# ================= INIT =================
app = FastAPI(title="Local RAG Service")

model = SentenceTransformer(EMBED_MODEL_NAME)
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

# ================= SCHEMAS =================
class LoadPDFRequest(BaseModel):
    pdf_path: str

class QueryRequest(BaseModel):
    question: str

# ================= UTIL FUNCTIONS =================
def extract_text_from_pdf(path: str) -> str:
    texts = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            texts.append(page.extract_text() or "")
    return "\n\n".join(texts)

def chunk_text(text: str) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunks.append(text[start:end].strip())
        start = end - CHUNK_OVERLAP
    return [c for c in chunks if c][:MAX_CHUNKS]

def embed_texts(texts: List[str]) -> np.ndarray:
    embeddings = model.encode(texts)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings.astype("float32")

def setup_collection(vector_size: int):
    qdrant.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=vector_size,
            distance=Distance.COSINE,
        ),
    )

def index_embeddings(embeddings: np.ndarray, chunks: List[str]):
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=emb.tolist(),
            payload={"text": chunks[i]}
        )
        for i, emb in enumerate(embeddings)
    ]
    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)

def search_embeddings(query_emb: np.ndarray, top_k: int):
    results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_emb[0].tolist(),
        limit=top_k,
        with_payload=True
    )
    return [point.payload["text"] for point in results.points]

def generate_answer(question: str, contexts: List[str]) -> str:
    prompt = (
        "Answer the question based only on the context below.\n\n"
        + "\n---\n".join(contexts)
        + "\n\nQuestion: "
        + question
    )

    response = requests.post(
        f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/generate",
        json={
            "model": LLAMA_MODEL,
            "prompt": prompt,
            "stream": False
        }
    )

    if response.status_code != 200:
        raise HTTPException(status_code=500, detail=response.text)

    return response.json()["response"]

# ================= ROUTES =================
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/load_pdf")
def load_pdf(request: LoadPDFRequest):
    text = extract_text_from_pdf(request.pdf_path)
    chunks = chunk_text(text)
    embeddings = embed_texts(chunks)

    setup_collection(embeddings.shape[1])
    index_embeddings(embeddings, chunks)

    return {"message": f"Indexed {len(chunks)} chunks successfully"}

@app.post("/query")
def query(request: QueryRequest):
    q_emb = embed_texts([request.question])
    retrieved_chunks = search_embeddings(q_emb, TOP_K)

    pairs = [(request.question, chunk) for chunk in retrieved_chunks]
    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(retrieved_chunks, scores),
        key=lambda x: x[1],
        reverse=True
    )

    contexts = [chunk for chunk, _ in ranked[:TOP_K]]

    answer = generate_answer(request.question, contexts)

    return {"answer": answer}




@app.get("/", response_class=HTMLResponse)
def professional_ui():
    return """
<!DOCTYPE html>
<html>
<head>
    <title>RAG AI Assistant</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: #f5f7fa;
            margin: 0;
            padding: 0;
        }

        header {
            background: #1f2937;
            color: white;
            padding: 20px;
            text-align: center;
            font-size: 20px;
            font-weight: 600;
        }

        .container {
            max-width: 900px;
            margin: 40px auto;
            background: white;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 10px 25px rgba(0,0,0,0.05);
        }

        .section {
            margin-bottom: 30px;
        }

        input, textarea {
            width: 100%;
            padding: 12px;
            border-radius: 8px;
            border: 1px solid #d1d5db;
            font-size: 14px;
        }

        button {
            padding: 10px 18px;
            background: #2563eb;
            color: white;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-weight: 500;
        }

        button:hover {
            background: #1d4ed8;
        }

        .chat-box {
            max-height: 400px;
            overflow-y: auto;
            border: 1px solid #e5e7eb;
            padding: 15px;
            border-radius: 10px;
            background: #fafafa;
        }

        .message {
            margin-bottom: 15px;
        }

        .user {
            font-weight: 600;
            color: #111827;
        }

        .assistant {
            margin-top: 5px;
            color: #374151;
        }

        .loading {
            color: #6b7280;
            font-style: italic;
        }
    </style>
</head>
<body>

<header>RAG AI Assistant</header>

<div class="container">

    <div class="section">
        <h3>Load Document</h3>
        <input id="pdfPath" placeholder="Enter full PDF path..." />
        <br/><br/>
        <button onclick="loadPDF()">Load PDF</button>
        <p id="loadStatus"></p>
    </div>

    <div class="section">
        <h3>Ask Questions</h3>
        <div class="chat-box" id="chat"></div>
        <br/>
        <textarea id="question" placeholder="Type your question here..."></textarea>
        <br/><br/>
        <button onclick="askQuestion()">Ask</button>
    </div>

</div>

<script>

async function loadPDF() {
    const path = document.getElementById("pdfPath").value;
    const status = document.getElementById("loadStatus");

    status.innerText = "Loading document...";

    const response = await fetch("/load_pdf", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({pdf_path: path})
    });

    if (!response.ok) {
        status.innerText = "Error loading PDF.";
        return;
    }

    const data = await response.json();
    status.innerText = data.message;
}

async function askQuestion() {
    const question = document.getElementById("question").value;
    const chat = document.getElementById("chat");

    chat.innerHTML += `
        <div class="message">
            <div class="user">You:</div>
            <div>${question}</div>
        </div>
    `;

    chat.innerHTML += `
        <div class="message loading" id="loading">
            Assistant is thinking...
        </div>
    `;

    chat.scrollTop = chat.scrollHeight;

    const response = await fetch("/query", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({question: question})
    });

    document.getElementById("loading").remove();

    if (!response.ok) {
        chat.innerHTML += `
            <div class="message assistant">
                Error generating answer.
            </div>
        `;
        return;
    }

    const data = await response.json();

    chat.innerHTML += `
        <div class="message">
            <div class="assistant">Assistant:</div>
            <div>${data.answer}</div>
        </div>
    `;

    chat.scrollTop = chat.scrollHeight;
}

</script>

</body>
</html>
"""