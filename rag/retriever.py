from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, Optional

import chromadb
from chromadb.utils import embedding_functions

CHROMA_DIR = Path(__file__).resolve().parent / "chroma_store"

_EMBED_FN = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return _client


def retrieve_policy(query: str, n_results: int = 3) -> List[Dict[str, Any]]:
    """Semantic search over policy docs. Returns [{text, source, score}]."""
    coll = _get_client().get_or_create_collection("policies", embedding_function=_EMBED_FN)
    if coll.count() == 0:
        return []
    result = coll.query(query_texts=[query], n_results=min(n_results, coll.count()))
    out = []
    for text, meta, dist in zip(result["documents"][0], result["metadatas"][0], result["distances"][0]):
        out.append({"text": text, "source": meta.get("source"), "score": 1 - dist})
    return out


def search_transcript(customer_id: str, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
    """Semantic search restricted to one customer's transcript turns.
    Catches paraphrases a literal substring search would miss (e.g. the
    agent searching for "policy number" also surfaces "it's P8921...
    actually maybe P8927")."""
    coll = _get_client().get_or_create_collection("transcripts", embedding_function=_EMBED_FN)
    if coll.count() == 0:
        return []
    result = coll.query(
        query_texts=[query],
        n_results=min(n_results, coll.count()),
        where={"customer_id": customer_id},
    )
    out = []
    if not result["documents"] or not result["documents"][0]:
        return out
    for text, meta, dist in zip(result["documents"][0], result["metadatas"][0], result["distances"][0]):
        out.append({
            "text": text,
            "turn_index": meta.get("turn_index"),
            "speaker": meta.get("speaker"),
            "score": 1 - dist,
        })
    # keep transcript order for readability
    out.sort(key=lambda x: x["turn_index"])
    return out


if __name__ == "__main__":
    print("Policy query test:")
    for r in retrieve_policy("can we cancel a policy if the customer changed their mind about the number"):
        print(" -", r["source"], round(r["score"], 3), "-", r["text"][:80].replace("\n", " "))
