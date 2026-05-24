import requests
from sentence_transformers import SentenceTransformer, CrossEncoder
import numpy as np
import faiss
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def fetch_wikipedia(title):
    resp = requests.get(
        "https://en.wikipedia.org/w/api.php",
        params={"action":"query","titles":title,"prop":"extracts","explaintext":True,"format":"json"},
        headers={"User-Agent":"proj-rag/1.0 (educational project)"},
        timeout=10
    )
    resp.raise_for_status()
    pages = resp.json()["query"]["pages"]
    page = next(iter(pages.values()))
    if "extract" not in page:
        raise ValueError(f"No extract for '{title}'")
    return page["extract"]

titles = ["Large language model", "Transformer (deep learning architecture)", "Retrieval-augmented generation", "Word embedding", "Neural network"]
documents = []
for title in titles:
    try:
        text = fetch_wikipedia(title)
        documents.append({'title':title,'text':text})
        print(f'Loaded: {title} ({len(text)} characters)')
    except Exception as e:
        print(f'Skipped {title}: {e}')
print('\nTotal documents: ', len(documents))
def chunk_text(text,chunk_size = 500, overlap = 100):
    chunks =[]
    start = 0
    while start < len(text):
        end = start+ chunk_size
        chunks.append(text[start:end])
        start+=chunk_size-overlap
    return chunks
passages = []
for doc in documents:
    for chunk in chunk_text(doc['text']):
        passages.append({'source':doc['title'],'text':chunk})
print('Total chunks: ', len(passages))
if passages:
    print('\nExample chunk: \n', passages[min(4, len(passages)-1)]['text'])
else:
    raise RuntimeError("No passages loaded — check that Wikipedia fetches succeeded above.")
embedder = SentenceTransformer('all-MiniLM-L6-v2')
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
chunk_texts = [f"{p['source']}: {p['text']}" for p in passages]
chunks_vectors = embedder.encode(chunk_texts,show_progress_bar=True,normalize_embeddings=True)
print('Each chunk is now a vector of len: ' , len(chunks_vectors[0]))
def retrieve(question,k=3):
    q_vec = embedder.encode([question],normalize_embeddings=True)[0]
    scores = chunks_vectors @ q_vec
    top_idx = np.argsort(scores)[::-1][:k]
    return [(passages[i],float(scores[i])) for i in top_idx]
question = 'What problem does retrieval-augemented generation solve?'
for passage,score in retrieve(question):
    print(f"[score{score:.2f}] from: {passage['source']}")
    print(passage['text'][:200], '...\n')
index = faiss.IndexFlatIP(chunks_vectors.shape[1])
index.add(chunks_vectors.astype('float32'))

def retrieve_faiss(question, k=3):
    q = embedder.encode([question],normalize_embeddings=True).astype('float32')
    scores,idx = index.search(q,k)
    return [(passages[i],float(s))for i ,s in zip(idx[0],scores[0])]

def retrieve_reranked(question, k=3, candidate_k=15):
    candidates = retrieve_faiss(question, k=candidate_k)
    pairs = [[question, p['text']] for p, _ in candidates]
    rerank_scores = reranker.predict(pairs)
    ranked = sorted(zip(candidates, rerank_scores), key=lambda x: x[1], reverse=True)
    return [(p, float(s)) for (p, _), s in ranked[:k]]

question = 'What problem does retrieval-augemented generation solve?'
for passage,score in retrieve_faiss(question):
    print(f"[score{score:.2f}] from: {passage['source']}")
    print(passage['text'][:200], '...\n')

_tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-large')
_model = AutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-large')
def llm(prompt, max_new_tokens=200):
    inputs = _tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512)
    outputs = _model.generate(**inputs, max_new_tokens=max_new_tokens)
    return [{'generated_text': _tokenizer.decode(outputs[0], skip_special_tokens=True)}]
def build_prompt(question,retrieved):
    context = "\n\n".join(f"[{p['source']}] {p['text']}" for p, _ in retrieved)
    return (
        'Answer the question usinng ONLY the context below. '
        'If the answer is not in the context, say you dont know.\n\n'
        f"Context: \n{context}\n\nQuestion: {question}\nAnswer" 
    )
def rag_answer(question, k =3):
    retrieved = retrieve_reranked(question,k)
    prompt = build_prompt(question,retrieved)
    answer = llm(prompt,max_new_tokens=200)[0]['generated_text']
    return answer, retrieved
answer, source = rag_answer('What problem does RAG solve?')
print('Answer:', answer)
print('\nGrounded in:', [s[0]['source'] for s in source])
def no_rag_answer(question):
    return llm(f"Question: {question}\nAnswer:",max_new_tokens=200)[0]['generated_text']
q ='According to the documents,how does RAG reduce hallucinations?'
print("WITHOUT retrieval (memory only):\n", no_rag_answer(q))
print("\nWITH retrieval (grounded):\n", rag_answer(q)[0])
eval_set = [
    {'q':"What is a word embedding?","should_come_from":"Word embedding"},
    {"q":"What is a transformer in deep learning?","should_come_from": "Transformer (deep learning architecture)"},

]

hits = 0
for item in eval_set:
    retieved= retrieve_reranked(item['q'],k=3)
    sources = [p['source']for p, _ in retieved]
    correct = item['should_come_from']in sources
    hits+= correct
    print(f"{'OK ' if correct else 'MISS'} | {item['q']} -> {sources}")

print(f"\nRetrieval hit rate (top-3): {hits}/{len(eval_set)} = {hits/len(eval_set):.0%}")