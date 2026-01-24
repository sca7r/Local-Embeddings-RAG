import os
import sys
import re
import subprocess
from typing import List


import numpy as np
import pdfplumber
from sentence_transformers import SentenceTransformer, util, CrossEncoder
from sklearn.neighbors import NearestNeighbors


os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ================= CONFIG =================
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K = 4
MAX_CHUNKS = 80


EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLAMA_MODEL = "llama3"


FAITHFULNESS_THRESHOLD = 0.4
# =========================================



# ================= MODELS =================
embed_model = SentenceTransformer(EMBED_MODEL_NAME)
reranker = CrossEncoder(RERANK_MODEL_NAME)



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
    emb = embed_model.encode(texts, normalize_embeddings=True)
    return emb.astype("float32")



# ================= VECTOR INDEX ===========
class VectorIndex:
    def __init__(self, embeddings: np.ndarray):
        self.nn = NearestNeighbors(metric="cosine")
        self.nn.fit(embeddings)


    def search(self, query_emb: np.ndarray, top_k: int):
        _, indices = self.nn.kneighbors(query_emb, n_neighbors=top_k)
        return indices[0]



# ================= RERANKING ==============
def rerank_chunks(question: str, chunk_idxs: List[int], all_chunks: List[str]):
    pairs = [(question, all_chunks[i]) for i in chunk_idxs]
    scores = reranker.predict(pairs)


    ranked = sorted(
        zip(chunk_idxs, scores),
        key=lambda x: x[1],
        reverse=True
    )


    return [idx for idx, _ in ranked]




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



# ================= METRICS =================
def recall_at_k(retrieved, relevant, k):
    """
    Standard Recall@k: fraction of all relevant items that appear in top-k.
    retrieved: list of retrieved chunk indices (top-k)
    relevant: list of all relevant chunk indices (ground truth)
    k: number of top results considered
    """
    retrieved_set = set(retrieved[:k])
    relevant_set = set(relevant)
    if len(relevant_set) == 0:
        return 0.0
    hits = len(retrieved_set & relevant_set)
    return hits / len(relevant_set)



def precision_at_k(retrieved, relevant, k):
    """
    Standard Precision@k: fraction of top-k results that are relevant.
    retrieved: list of retrieved chunk indices (top-k)
    relevant: list of all relevant chunk indices (ground truth)
    k: number of top results considered
    """
    retrieved_set = set(retrieved[:k])
    relevant_set = set(relevant)
    hits = len(retrieved_set & relevant_set)
    return hits / k



def reciprocal_rank(retrieved, relevant):
    """
    Reciprocal Rank: 1 / (rank of first relevant item), or 0 if none found.
    retrieved: ordered list of retrieved chunk indices
    relevant: list of all relevant chunk indices (ground truth)
    """
    relevant_set = set(relevant)
    for i, r in enumerate(retrieved):
        if r in relevant_set:
            return 1.0 / (i + 1)
    return 0.0



def split_sentences(text):
    return [
        s.strip()
        for s in re.split(r"[.!?]", text)
        if len(s.strip()) > 15
    ]



def faithfulness_score(answer, contexts):
    """
    Fraction of answer sentences with embedding similar to at least one context chunk.
    """
    sents = split_sentences(answer)
    if not sents:
        return 0.0


    ctx_emb = embed_model.encode(contexts, normalize_embeddings=True)
    sent_emb = embed_model.encode(sents, normalize_embeddings=True)


    supported = 0
    for emb in sent_emb:
        if util.cos_sim(emb, ctx_emb).max() >= FAITHFULNESS_THRESHOLD:
            supported += 1


    return supported / len(sents)



def answer_relevance(question, answer):
    """
    Cosine similarity between question and answer embeddings.
    """
    q_emb = embed_model.encode([question], normalize_embeddings=True)
    a_emb = embed_model.encode([answer], normalize_embeddings=True)
    return float(np.dot(q_emb, a_emb.T)[0][0])



# ================= EVAL DATA ===============
EVAL_QUERIES = [
    {
        "question": "question related to the document",
        "relevant_chunk_ids": ["chunk id in which answer is present"]
    },
    {
        "question": "question related to the document",
        "relevant_chunk_ids": ["chunk id in which answer is present"]
    },
    {
        "question": "question related to the document",
        "relevant_chunk_ids": ["chunk id in which answer is present"]
    }
]



# ================= EVALUATION =============
def evaluate_rag(chunks, index):
    recall, precision, mrr = [], [], []
    faithfulness, relevance = [], []


    for sample in EVAL_QUERIES:
        q = sample["question"]
        relevant = sample["relevant_chunk_ids"]


        # Retrieve top-k chunks
        q_emb = embed_texts([q])
        retrieved_idxs = index.search(q_emb, TOP_K)


        # Rerank retrieved chunks
        reranked_idxs = rerank_chunks(q, retrieved_idxs, chunks)
        reranked_contexts = [chunks[i] for i in reranked_idxs]


        # Generate answer using reranked contexts
        answer = generate_answer_llama(q, reranked_contexts)


        # Compute retrieval metrics on the RERANKED results
        # (Change to retrieved_idxs if you want to evaluate pre-rerank retrieval)
        recall.append(recall_at_k(reranked_idxs, relevant, TOP_K))
        precision.append(precision_at_k(reranked_idxs, relevant, TOP_K))
        mrr.append(reciprocal_rank(reranked_idxs, relevant))


        # Compute answer quality metrics
        faithfulness.append(faithfulness_score(answer, reranked_contexts))
        relevance.append(answer_relevance(q, answer))


        # Debug output
        print("\nQUESTION:", q)
        print("\nANSWER:", answer)
        print("\nRE-RANKED CONTEXT:")
        for c in reranked_contexts:
            print("----")
            print(c[:300])


    return {
        "Recall@K": np.mean(recall),
        "Precision@K": np.mean(precision),
        "MRR": np.mean(mrr),
        "Faithfulness": np.mean(faithfulness),
        "Answer Relevance": np.mean(relevance),
        "RAF Score": np.mean(recall) * np.mean(faithfulness)
    }



# ================= MAIN ===================
def main():
    if len(sys.argv) < 2:
        print("Usage: python3 eval.py path/to/document.pdf (give actual path to PDF you like to use)")
        sys.exit(1)


    pdf_path = sys.argv[1]


    print("Extracting PDF...")
    text = extract_text_from_pdf(pdf_path)


    print("Chunking...")
    chunks = chunk_text(text)


    print(f"Embedding {len(chunks)} chunks...")
    embeddings = embed_texts(chunks)


    index = VectorIndex(embeddings)


    print("\n=== Running Evaluation ===")
    metrics = evaluate_rag(chunks, index)


    for k, v in metrics.items():
        print(f"{k:20s}: {v:.3f}")


    print("\n=== Ask Questions (type \'exit\' to exit) ===")
    while True:
        q = input("> ").strip()
        if q.lower() == "exit":
            break


        q_emb = embed_texts([q])
        retrieved_idxs = index.search(q_emb, TOP_K)


        
        reranked_idxs = rerank_chunks(q, retrieved_idxs, chunks)
        reranked_contexts = [chunks[i] for i in reranked_idxs]


        answer = generate_answer_llama(q, reranked_contexts)
        print("\nAnswer:\n", answer)
        print("-" * 60)



if __name__ == "__main__":
    main()