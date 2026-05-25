import requests,hashlib
import re
def doc_id_for(text):
    return hashlib.md5(text.encode()).hexdigest()[:12]
def load_wikipedia(titles):
    docs = []
    for title in titles:
        try:
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
            text = page["extract"]
            docs.append({'doc_id': doc_id_for(text),"source":title,"text":text})
            print("Loaded:",title)
        except Exception as e:
            print("Skipped", title, e)
    return docs

documents = load_wikipedia(["Information retrieval", "Search engine","Vector database","Large language model","Transformer (deep learning architecture)", "Word embedding"])
def clean(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()
def chunk_text(text,size = 500,overlap=100):
    chunks,start = [],0
    while start < len(text):
        chunks.append(text[start:start+size])
        start += size-overlap
    return chunks
def build_chunks(documents):
    passages = []
    for doc in documents:
        for pos, ch in enumerate(chunk_text(clean(doc['text']))):
            passages.append({
                "chunk_id": f"{doc['doc_id']}_{pos}",
                "doc_id": doc['doc_id'],
                "source": doc["source"],
                "position": pos,
                "text": ch,
            })
    return passages