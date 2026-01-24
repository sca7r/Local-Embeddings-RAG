import os
import sys
from typing import List


import numpy as np
import pdfplumber
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.neighbors import NearestNeighbors
import subprocess
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ================= CONFIG =================
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K = 4
MAX_CHUNKS = 80
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
LLAMA_MODEL = "llama3"


# ================= MODELS =================
model = SentenceTransformer(EMBED_MODEL_NAME)
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# ================= PDF ====================
def extract_text_from_pdf(path: str) -> str:
    texts = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            texts.append(page.extract_text() or "")
    return "\n\n".join(texts)


# ================= CHUNKING ===============
def chunk_text(text: str) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunks.append(text[start:end].strip())
        start = end - CHUNK_OVERLAP
    return [c for c in chunks if c][:MAX_CHUNKS]


# ================= EMBEDDING ==============
def embed_texts(texts: List[str]) -> np.ndarray:
    embeddings = model.encode(texts, show_progress_bar=True)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings.astype("float32")


# ================= RERANKING ==============
def rerank_chunks(question: str, chunk_idxs: List[int], all_chunks: List[str], top_k: int):
    """Returns reranked chunk indices in order of relevance."""
    pairs = [(question, all_chunks[i]) for i in chunk_idxs]
    scores = reranker.predict(pairs)


    ranked = sorted(
        zip(chunk_idxs, scores),
        key=lambda x: x[1],
        reverse=True
    )


    return [idx for idx, _ in ranked[:top_k]]




# ================= VECTOR INDEX ===========
class VectorIndex:
    def __init__(self, embeddings: np.ndarray):
        self.nn = NearestNeighbors(metric="cosine")
        self.nn.fit(embeddings)


    def search(self, query_emb: np.ndarray, top_k: int) -> List[int]:
        _, indices = self.nn.kneighbors(query_emb, n_neighbors=top_k)
        return indices[0]


# ================= LLM ====================
def generate_answer_llama(question: str, contexts: List[str]) -> str:
    prompt = (
                "You are an AI assistant that answers questions strictly based on the provided context chunks, which are excerpts extracted from one or more PDF documents.\n"
                "Core rules:\n"
                    "- Use ONLY the provided context to answer. Do not use outside knowledge, even if you are confident.\n"
                    "- If the context does not contain enough information to answer reliably, say:I am not able to answer this from the provided document context.\n"
                    "- Prefer quoting or paraphrasing the document over guessing.\n"
                    "- If multiple chunks conflict, prioritize:\n"
                        "1) the most recent version or highest-level section (e.g., later pages, summary sections),\n"
                        "2) explicitly stated rules over examples or anecdotes.\n"


                "When answering:\n"
                    "- Be concise and precise.\n"
                    "- Explicitly reference key phrases, definitions, formulas, or tables from the context when helpful.\n"
                    "- If the user asks \"where\" or \"which page,\" include page numbers or section titles when they are present in the chunk metadata.\n"
                    "- If the question is ambiguous, briefly list the reasonable interpretations and clearly answer each, or ask the user to clarify.\n"


                "Input format you will receive:\n"
                    "- A user question.\n"
                    "- A list of context chunks. Each chunk may include:\n"
                        "1) text: the text snippet from the PDF\n"
                        "2) metadata: {page_number, section_title, etc.}\n"


                "Output format:"
                    "- Provide the best possible answer in clear prose.\n"
                    "- If relevant, include a short bullet list or step-by-step explanation.\n\n"



        "CONTEXT:\n"
        + "\n---\n".join(contexts)
        + "\n\nQUESTION:\n"
        + question
    )


    result = subprocess.run(
        ["ollama", "run", LLAMA_MODEL],
        input=prompt,
        text=True,
        capture_output=True
    )


    return result.stdout.strip()


# ================= MAIN ===================
def main():
    if len(sys.argv) < 2:
        print("Usage: python3 local_rag.py data/sample.pdf (give actual path to PDF you like to use)")
        sys.exit(1)


    pdf_path = sys.argv[1]


    print("Extracting PDF...")
    text = extract_text_from_pdf(pdf_path)


    print("Chunking text...")
    chunks = chunk_text(text)


    print(f"Embedding {len(chunks)} chunks locally...")
    embeddings = embed_texts(chunks)


    index = VectorIndex(embeddings)


    print("\nAsk questions (To exit, type \'exit\'):\n")


    while True:
        q = input("> ").strip()
        if q.lower() == "exit":
            break


        q_emb = embed_texts([q])
        idxs = index.search(q_emb, TOP_K)


        # Rerank and get indices in reranked order
        reranked_idxs = rerank_chunks(q, idxs, chunks, TOP_K)
        reranked_contexts = [chunks[i] for i in reranked_idxs]


        answer = generate_answer_llama(q, reranked_contexts)


        print("\n=== Relevant Answer ===")
        print(answer)


        #print("\n=== Retrieved Context (Reranked) ===")
        #for i in reranked_idxs:
            #print(f"[Chunk {i}]\n{chunks[i][:300]}\n")


        print("-" * 60)



if __name__ == "__main__":
    main()