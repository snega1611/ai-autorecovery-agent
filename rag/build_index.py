"""
Builds two Chroma collections:

  1. "policies"    -- one chunk per policy doc section, used by the
                       get_policy tool to answer "what does policy require?"
  2. "transcripts"  -- one chunk per transcript turn (customer turns only,
                       plus a little agent-turn context), used by the
                       search_transcript tool for semantic evidence search
                       (catches paraphrases the agent's literal keyword
                       search over the transcript text would miss).

Both use a small local sentence-transformer (all-MiniLM-L6-v2, ~80MB,
CPU-friendly) via Chroma's built-in embedding function, so nothing leaves
the machine and there's no API cost.

Run: python rag/build_index.py
"""
from __future__ import annotations
import json
import re
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
POLICY_DIR = DATA_DIR / "policies"
CHROMA_DIR = Path(__file__).resolve().parent / "chroma_store"

EMBED_FN = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)


def _chunk_policy_doc(text: str, doc_name: str):
    """Chunk by markdown section (## headers) or numbered list items --
    policy docs are short, so this stays simple rather than pulling in a
    generic recursive splitter."""
    sections = re.split(r"\n(?=#{1,2} )", text.strip())
    chunks = []
    for i, section in enumerate(sections):
        section = section.strip()
        if section:
            chunks.append({"id": f"{doc_name}_sec{i}", "text": section, "doc": doc_name})
    return chunks


def build_policy_collection(client: chromadb.Client):
    coll = client.get_or_create_collection("policies", embedding_function=EMBED_FN)
    # wipe and rebuild for reproducibility
    existing = coll.get()
    if existing["ids"]:
        coll.delete(ids=existing["ids"])

    ids, docs, metas = [], [], []
    for policy_file in sorted(POLICY_DIR.glob("*.md")):
        text = policy_file.read_text()
        for chunk in _chunk_policy_doc(text, policy_file.stem):
            ids.append(chunk["id"])
            docs.append(chunk["text"])
            metas.append({"source": policy_file.name})

    coll.add(ids=ids, documents=docs, metadatas=metas)
    print(f"Indexed {len(ids)} policy chunks from {len(list(POLICY_DIR.glob('*.md')))} docs")
    return coll


def build_transcript_collection(client: chromadb.Client):
    coll = client.get_or_create_collection("transcripts", embedding_function=EMBED_FN)
    existing = coll.get()
    if existing["ids"]:
        coll.delete(ids=existing["ids"])

    transcripts = json.loads((DATA_DIR / "transcripts.json").read_text())

    ids, docs, metas = [], [], []
    for tx in transcripts:
        for i, turn in enumerate(tx["turns"]):
            chunk_id = f"{tx['transcript_id']}_turn{i}"
            ids.append(chunk_id)
            docs.append(f"{turn['speaker']}: {turn['text']}")
            metas.append({
                "transcript_id": tx["transcript_id"],
                "customer_id": tx["customer_id"],
                "speaker": turn["speaker"],
                "turn_index": i,
            })

    if ids:
        coll.add(ids=ids, documents=docs, metadatas=metas)
    print(f"Indexed {len(ids)} transcript turns from {len(transcripts)} transcripts")
    return coll


def main():
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    build_policy_collection(client)
    build_transcript_collection(client)


if __name__ == "__main__":
    main()
