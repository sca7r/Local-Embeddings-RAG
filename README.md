# Local RAG System (PDF-based)


This project implements a **fully local Retrieval-Augmented Generation (RAG) system** for answering questions from PDF documents, along with an **evaluation framework** to measure retrieval quality, ranking quality, and answer grounding.
This includes configurable parameters that can be tuned based on evaluation results to improve performance.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![SentenceTransformers](https://img.shields.io/badge/SentenceTransformers-latest-green.svg)
![Ollama](https://img.shields.io/badge/Ollama-llama3-orange.svg)


The system is designed to be:
- **Privacy-preserving** (no cloud APIs)
- **Explainable** (transparent retrieval & metrics)
- **Measurable** (Recall, Precision, MRR, Faithfulness, RAF)

---

## What This Project Does

1. Extracts text from a PDF document  
2. Splits the text into overlapping chunks  
3. Embeds chunks using a sentence-transformer  
4. Retrieves relevant chunks using vector similarity  
5. Re-ranks retrieved chunks using a cross-encoder  
6. Generates answers using a local LLM (via Ollama)  
7. Evaluates retrieval and answer quality quantitatively  

---


## Core Concepts Explained

### 1️ Chunking
- The PDF text is split into fixed-size **character chunks**.
- Overlap ensures sentences and concepts are not cut abruptly.

### 2️ Embeddings (Bi-Encoder Retrieval)
- Each chunk is converted into a dense vector using:

```bash
all-MiniLM-L6-v2
```
This allows semantic retrieval, meaning:
- Queries do not need to match exact keywords
- Conceptually similar text can still be retrieved
- Cosine similarity is used for nearest-neighbor search.

### 3️ Vector Search (Recall-Oriented)
- The system retrieves the Top-K most similar chunks.
- This step is optimized for high recall, because:
- If relevant information is not retrieved, the LLM cannot answer correctly.
- Low precision at this stage is acceptable.

### 4️ Cross-Encoder Re-Ranking
- A cross-encoder jointly scores (question, chunk) pairs.
- This improves ranking quality by:
     - Considering the query and chunk together. Promoting the most relevant chunk to a higher rank
- This significantly improves MRR (Mean Reciprocal Rank).

### 5️ Local LLM Generation
- Answers are generated using a locally running LLaMA model via Ollama.
- Key constraints in the prompt:
     - Use only the provided context
     - Do not use external knowledge
- This minimizes hallucinations and enforces grounding


## Installation

### Prerequisites

- Python 3.8 or higher
- Ollama installed and running ([Install Ollama](https://ollama.ai))
  To install Ollama, Run
```bash
  curl -fsSL https://ollama.com/install.sh | sh
 ```
- After the Llama3 model is downloaded, run :
```bash
ollama pull llama3
```

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/rag-evaluation.git
cd rag-evaluation

# Create virtual environment
python3 -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```
### Usage
Interactive question-answering with any PDF
```bash
python3 local_rag.py path/to/your/document.pdf
```
Once the script runs, enter your question in the terminal and hit enter!

## Troubleshooting

### Zero Retrieval Metrics

If you get `Recall@K: 0.000`, `Precision@K: 0.000`, `MRR: 0.000`:

1. **Wrong PDF**: Ensure you're testing the same PDF used to create `relevant_chunk_ids`
2. **Verify ground truth**: Print chunk contents to confirm correct indices:
   ```python
   print("Chunk 0:", chunks[:500])
    ```
3. **Chunking changed**: If you modified `CHUNK_SIZE` or `CHUNK_OVERLAP`, chunk IDs shifted

### Ollama Connection issues
 Check Ollama is running
```bash
ollama list
```
 Pull Llama3 model
```bash
ollama pull llama3
```
 Test manually
```bash
ollama run llama3
```
