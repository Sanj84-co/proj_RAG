import os, sys, types, warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="google")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="ragas")
from dotenv import load_dotenv
load_dotenv()
if not os.environ.get('GOOGLE_API_KEY'):
    raise EnvironmentError("GOOGLE_API_KEY not found — add it to your .env file.")

# ragas 0.4.3 imports ChatVertexAI from a path removed in langchain-community 0.4.x;
# register a shim so the import resolves without downgrading the whole ecosystem.
import langchain_google_vertexai as _vx
_m = types.ModuleType("langchain_community.chat_models.vertexai")
_m.ChatVertexAI = _vx.ChatVertexAI
sys.modules["langchain_community.chat_models.vertexai"] = _m

from langchain_google_genai import ChatGoogleGenerativeAI
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import GoogleEmbeddings
from ragas import evaluate
from ragas import RunConfig
from ragas.metrics._faithfulness import faithfulness
from ragas.metrics._answer_relevance import answer_relevancy
from ragas.metrics._context_precision import context_precision
from datasets import Dataset
from index import passages, vector_search, hybrid_search, retrieve
from citations import answer
q = "What architecture are large language models built on?"
print("VECTOR:        ", [passages[i]["source"] for i in vector_search(q,4)])
print("HYBRID:        ", [p["source"] for p in hybrid_search(q,4)])
print("HYBRID+RERANK: ", [p["source"] for p,_ in retrieve(q,keep=4)])

eval_set = [
    {"q": "What is BM25 used for?", "source": "Okapi BM25",
     "ground_truth": "BM25 is a ranking function search engines use to score document relevance to a query."},
    {"q": "What architecture are large language models built on?", "source": "Transformer (deep learning architecture)",
     "ground_truth": "Large language models are built on the transformer architecture."},
    {"q": "How does a vector database retrieve similar items?", "source": "Vector database",
     "ground_truth": "It performs nearest-neighbor search over embedding vectors."},
    # add 4-6 more
]

def hit_rate(fn,label,keep =4):
    h = 0
    for it in eval_set:
        got = fn(it['q']); srcs = [p["source"] for p in got] if not isinstance(got[0],tuple) else [p['source'] for p,_ in got]
        h+= it["source"] in srcs[:keep]
    print(f"{label}: {h}/{len(eval_set)} = {h/len(eval_set):.0%}")

hit_rate(lambda q: [passages[i] for i in vector_search(q,4)], "Vector only")
hit_rate(lambda q: hybrid_search(q,4),                       "Hybrid")
hit_rate(lambda q: retrieve(q,keep=4),                       "Hybrid + rerank")

judge = LangchainLLMWrapper(ChatGoogleGenerativeAI(model='gemini-2.0-flash-lite', timeout=300, max_retries=3))
emb = GoogleEmbeddings(model='models/text-embedding-004')
rows = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
for it in eval_set:
    ans, legend, hits = answer(it["q"])
    rows["question"].append(it["q"]); rows["answer"].append(ans)
    rows["contexts"].append([p["text"] for p,_ in hits]); rows["ground_truth"].append(it["ground_truth"])

result = evaluate(Dataset.from_dict(rows),
                  metrics=[faithfulness, answer_relevancy, context_precision], llm=judge, embeddings=emb,
                  run_config=RunConfig(timeout=300, max_retries=5, max_wait=60),
                  batch_size=1)
print(result)