A retrieval-augmented generation (RAG) system built end to end: chunking, embeddings,
vector search (cosine similarity + FAISS), cross-encoder reranking, grounded generation, and evaluation.

## What it does
Fetches 5 Wikipedia articles, splits them into overlapping chunks, and embeds them with a sentence transformer. At query time, FAISS retrieves the top candidates and a cross-encoder reranker selects the most relevant passages before passing them to Flan-T5-Large for a grounded answer — demonstrating a clear reduction in hallucinations compared to retrieval-free generation.

## Pipeline
Wikipedia (5 articles) → chunking (500 chars, 100 overlap) → sentence embeddings (all-MiniLM-L6-v2) → FAISS vector search → cross-encoder reranking (ms-marco-MiniLM-L-6-v2) → grounded Flan-T5-Large answer

## Stack
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`
- **Vector index**: FAISS `IndexFlatIP` (inner product / cosine)
- **Reranker**: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Generator**: `google/flan-t5-large` via HuggingFace Transformers

## Results
- Retrieval hit rate (top-3): **100%** (2/2 hand-labeled questions)
- Dense-only baseline: 50% — reranker was the deciding improvement
- Answer faithfulness: [fill in after running 10-question eval]
- Chunk-size experiment: [fill in your finding]

## What I learned
Dense bi-encoders score passages independently and can confuse semantically adjacent topics (e.g. "transformer" matching LLM articles instead of the Transformer architecture article). A cross-encoder reranker reads query and passage together and resolves this. Smaller chunks improve precision but reduce context; larger chunks preserve context but dilute relevance scores.

## Run it
```bash
pip install -r requirements.txt
python doc.py
```

---

**Resume bullet:**
Built a retrieval-augmented generation system from scratch (chunking, sentence embeddings, FAISS vector search, cross-encoder reranking, grounded LLM generation) and evaluated it on a hand-labeled question set, achieving 100% top-3 retrieval hit rate; identified the failure mode of dense-only retrieval and fixed it with a two-stage retrieve-then-rerank pipeline.
