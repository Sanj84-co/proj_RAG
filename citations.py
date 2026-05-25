from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import numpy as np, re
import index

_tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-large')
_model = AutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-large')
def llm(prompt, max_new_tokens=220):
    inputs = _tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512)
    outputs = _model.generate(**inputs, max_new_tokens=max_new_tokens)
    return [{'generated_text': _tokenizer.decode(outputs[0], skip_special_tokens=True)}]

def build_prompt(q, retrieved):
    context = "\n\n".join(f"[{i}] {p['text']}" for i, (p, _) in enumerate(retrieved, start=1))
    return (
        "Answer the question using ONLY the context below. "
        "If the answer is not in the context, say you don't know.\n\n"
        f"Context:\n{context}\n\nQuestion: {q}\nAnswer:"
    )

def answer(q,keep =2):
    hits = index.retrieve(q,keep=keep)
    text = llm(build_prompt(q,hits),max_new_tokens=220)[0]['generated_text']
    legend = {i: {"source": p["source"], "chunk_id": p["chunk_id"]}
              for i, (p,_) in enumerate(hits,start=1)}
    return text, legend, hits

def verify_citations(answer_text, hits, threshold=0.25):
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', answer_text.strip()) if s.strip()]
    if not sentences or not hits:
        return []
    chunk_vecs = index.indexer.encoder.encode(
        [p['text'] for p, _ in hits], normalize_embeddings=True
    )
    report = []
    for sent in sentences:
        sv = index.indexer.encoder.encode([sent], normalize_embeddings=True)[0]
        sims = [float(sv @ chunk_vecs[i]) for i in range(len(hits))]
        best_idx = int(np.argmax(sims))
        best_score = sims[best_idx]
        status = 'supported' if best_score >= threshold else 'UNSUPPORTED'
        report.append({
            "sentence": sent,
            "cited": best_idx + 1,
            "support": round(best_score, 2),
            "status": status,
            "source": hits[best_idx][0]['source'],
        })
    return report
def groundedness(answer_text, hits):
    rep = verify_citations(answer_text,hits)
    if not rep: return 0.0
    return sum(1 for r in rep if r['status'] == 'supported')/ len(rep)
ans,legend ,hits = answer('What architecture are large language models built on?')
print(ans, "\n")
for r in verify_citations(ans,hits):
     print(f"  [{r['status']}] support={r['support']} source={r['source']}  {r['sentence'][:70]}")
print("\nGroundedness:", round(groundedness(ans, hits), 2))
print("Sources:", legend)
