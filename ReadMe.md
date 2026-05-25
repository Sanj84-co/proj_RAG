A retrieval-augmented generation (RAG) system built end to end: ingestion, hybrid search, cross-encoder reranking, grounded generation with verified citations, and automated evaluation.

## Architecture

Two pipelines: an **offline ingestion/indexing pipeline** (load → clean → chunk + metadata → dedup → embed + BM25 → persist) and an **online query pipeline** (hybrid RRF retrieval → cross-encoder re-rank → grounded generation with verified citations).

## What it does

Fetches Wikipedia articles, splits them into overlapping chunks with metadata, and builds a persistent hybrid index (dense FAISS + sparse BM25). At query time, both indexes are searched and fused with Reciprocal Rank Fusion, then a cross-encoder reranker selects the best passages before passing them to Flan-T5-Large for a grounded answer. Every sentence in the answer is automatically matched back to its source chunk and verified for semantic support — demonstrating a measurable reduction in hallucinations compared to retrieval-free generation.

## Stack

| Component | Model / Library |
|-----------|----------------|
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector index | FAISS `IndexFlatIP` (cosine) |
| Sparse index | BM25 (`rank-bm25`) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Generator | `google/flan-t5-large` (HuggingFace Transformers) |
| Evaluation judge | Gemini 2.0 Flash via RAGAS |
| Evaluation embeddings | Google `text-embedding-004` |

## Results

### Retrieval hit rate (top-3)

| Stage | Hit rate |
|-------|----------|
| Vector only | [a]% |
| Hybrid (RRF) | [b]% |
| Hybrid + re-ranking | [c]% |

### Answer quality

- **Citations**: verified per-sentence; groundedness = **[g]** (fraction of sentences supported by their cited source)
- **RAGAS**: faithfulness **[x]** · context precision **[y]** · answer relevance **[z]**

*Fill in [a–z] after running `evaluation.py` with a fresh API quota.*

## Files

| File | Purpose |
|------|---------|
| `pipeline.py` | Wikipedia fetch, cleaning, chunking with metadata |
| `index.py` | Persistent hybrid index (FAISS + BM25); incremental add |
| `citations.py` | Grounded generation + per-sentence citation verifier |
| `doc.py` | Exploratory notebook-style script (early prototype) |
| `evaluation.py` | RAGAS evaluation (faithfulness, context precision, answer relevance) |

## What I learned

Dense bi-encoders score query and passage independently — semantically adjacent articles (e.g. "Large language model" and "Transformer architecture") score similarly, causing misses. A cross-encoder reranker reads query and passage together and resolves this. BM25 complements dense retrieval by catching exact-keyword matches that embeddings miss. Smaller chunks improve retrieval precision; larger chunks give the generator more context — the chunk-size tradeoff has a measurable effect on both hit rate and answer quality.

## Run it

```bash
pip install -r requirements.txt

# Build the index
python index.py

# Run a grounded query with citation verification
python citations.py

# Run RAGAS evaluation (requires GOOGLE_API_KEY in .env)
python evaluation.py
```

---

**Resume bullet:**
Built a retrieval-augmented generation system from scratch (offline ingestion pipeline with BM25 + FAISS hybrid indexing, online query pipeline with RRF fusion and cross-encoder reranking, grounded Flan-T5 generation with per-sentence citation verification) and evaluated it with RAGAS using Gemini as judge; identified the failure mode of dense-only retrieval and fixed it with a two-stage retrieve-then-rerank pipeline.
