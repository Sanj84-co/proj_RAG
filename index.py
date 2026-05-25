from sentence_transformers import SentenceTransformer,CrossEncoder
from rank_bm25 import BM25Okapi
import faiss,numpy as np, json,pickle,os
from pipeline import build_chunks,documents,load_wikipedia
#indexes the chunks and saves them in disk.
class Indexer:
    def __init__(self,model_name = 'all-MiniLM-l6-v2'):
        self.model_name = model_name
        self.encoder = SentenceTransformer(model_name)
        self.passages = []
        self.vectors = None
        self.faiss_index = None
        self.bm25 = None 
        self._seen_chunk_ids = set()
    def add_documents(self,documents):
        new = [c for c in build_chunks(documents) if c['chunk_id'] not in self._seen_chunk_ids]
        if not new:
            print('Nothing new to index.'); return
        print(f"Indexing {len(new)} new chunks...")
        new_vecs = self.encoder.encode([c['text'] for c in new],show_progress_bar=True,normalize_embeddings=True).astype("float32")
        self.vectors = new_vecs if self.vectors is None else np.vstack([self.vectors,new_vecs])
        self.passages.extend(new)
        self._seen_chunk_ids.update(c['chunk_id'] for c in new)
        self.faiss_index = faiss.IndexFlatIP(self.vectors.shape[1]); self.faiss_index.add(self.vectors)
        self.bm25 = BM25Okapi([p['text'].lower().split() for p in self.passages])
        print(f"Index now holds {len(self.passages)} chunks.")
    def save(self,folder = "rag_index"):
        if self.vectors is None:
            print("Nothing indexed yet — skipping save."); return
        os.makedirs(folder,exist_ok=True)
        faiss.write_index(self.faiss_index,f"{folder}/vectors.faiss")
        np.save(f"{folder}/vectors.npy",self.vectors)
        with open(f"{folder}/passages.json", "w") as f:json.dump(self.passages,f)
        with open(f"{folder}/bm25.pkl", "wb") as f:pickle.dump(self.bm25,f)
        with open(f"{folder}/manifest.json", "w") as f:
            json.dump({"model_name": self.model_name,"num_chunks": len(self.passages),
                       "dim":int(self.vectors.shape[1])},f,indent =2)
        print("Saved index to" , folder)
    @classmethod
    def load(cls,folder = "rag_index"):
        manifest = json.load(open(f"{folder}/manifest.json"))
        ix = cls(manifest['model_name'])
        ix.faiss_index = faiss.read_index(f"{folder}/vectors.faiss")
        ix.vectors = np.load(f"{folder}/vectors.npy")
        ix.passages = json.load(open(f"{folder}/passages.json"))
        ix.bm25 = pickle.load(open(f"{folder}/bm25.pkl", "rb"))
        ix._seen_chunk_ids = { p['chunk_id'] for p in ix.passages}
        print(f"Loaded index: {len(ix.passages)} chunks, model {ix.model_name}")
        return ix

indexer = Indexer()
indexer.add_documents(documents)
indexer.save("rag_index")

# Prove incremental indexing works — add a new doc without rebuilding everything:
more = load_wikipedia(["Recommender system"])
indexer.add_documents(more)     # only the new chunks get embedded
indexer.save("rag_index")
ix = Indexer.load("rag_index")
passages = ix.passages
def vector_search(q,fetch = 20):
    v = ix.encoder.encode([q],normalize_embeddings=True).astype('float32')
    _,idx = ix.faiss_index.search(v,fetch)
    return list(idx[0])
def keyword_search(q,fetch =20):
    scores = ix.bm25.get_scores(q.lower().split())
    return list(np.argsort(scores)[::-1][:fetch])
def hybrid_search(q,fetch = 20, k_rrf = 60):
    fused = {}
    for rank,i in enumerate(vector_search(q,fetch)): fused[i] = fused.get(i,0) + 1/(k_rrf+rank)
    for rank,i in enumerate(keyword_search(q,fetch)): fused[i] = fused.get(i,0) + 1/(k_rrf+rank)
    top = sorted(fused,key=fused.get,reverse=True)[:fetch]
    return [passages[i] for i in top]
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
def retrieve(q,fetch=20,keep=4):
    canidates = hybrid_search(q,fetch)
    scores = cross_encoder.predict([(q,p['text']) for p in canidates])
    ranked = sorted(zip(canidates,scores),key = lambda x: x[1],reverse = True)
    return [(p, float(s)) for p, s in ranked[:keep]]
