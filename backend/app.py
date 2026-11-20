
"""
AuroraStay QnA — RAG Backend (Flask)

Endpoints:
    GET  /health  -> {"status": "ok"}
    POST /query   -> body: {"question": "...", "top_k": 4}
                     returns: {"answer": "...", "sources": [...]}

Runtime files expected next to this script (produced by the notebook):
    chunks.json, chunk_embeddings.npy, faiss_index.bin, config.json

Environment variables:
    OPENAI_API_KEY  - required for answer generation
    EMBED_MODEL     - optional override of config.json embed_model
    LLM_MODEL       - optional override of config.json llm_model
"""

import os, json
import numpy as np
import faiss
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from openai import OpenAI

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------- Load configuration & artifacts ----------------
with open(os.path.join(BASE_DIR, "config.json")) as f:
    CONFIG = json.load(f)

EMBED_MODEL = os.environ.get("EMBED_MODEL", CONFIG["embed_model"])
LLM_MODEL   = os.environ.get("LLM_MODEL",   CONFIG["llm_model"])
DEFAULT_K   = int(CONFIG.get("top_k", 4))

# Corpus metadata
with open(os.path.join(BASE_DIR, "chunks.json")) as f:
    CHUNKS = json.load(f)                      # list of {text, document, page}

# Vectors + index (index.ntotal must equal len(CHUNKS))
EMBEDDINGS = np.load(os.path.join(BASE_DIR, "chunk_embeddings.npy"))
INDEX = faiss.read_index(os.path.join(BASE_DIR, "faiss_index.bin"))
assert INDEX.ntotal == len(CHUNKS), "Index/corpus size mismatch"

# Models (loaded once at startup — important for latency)
EMBEDDER = SentenceTransformer(EMBED_MODEL)
LLM = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

SYSTEM_PROMPT = """You are AuroraStay Hotels' internal guest-services assistant.
Answer the staff member's question using ONLY the provided policy context.
Rules:
1. Use only facts present in the CONTEXT. Never invent policies, prices, or times.
2. If the context does not contain enough information, say:
   "I could not find this in the policy documents" and suggest which document the staff member should check.
3. Be concise and specific (include times, fees, property codes like ASH-XXX when present).
4. End with a 'Sources:' line listing the document name(s) and page number(s) used."""

app = Flask(__name__)


def retrieve(question, top_k):
    """Embed question -> FAISS search -> list of chunk dicts with scores."""
    q = EMBEDDER.encode([question], normalize_embeddings=True,
                        convert_to_numpy=True).astype("float32")
    scores, ids = INDEX.search(q, min(top_k, INDEX.ntotal))
    hits = []
    for score, idx in zip(scores[0], ids[0]):
        if idx < 0:
            continue
        chunk = dict(CHUNKS[idx])
        chunk["score"] = float(score)
        hits.append(chunk)
    return hits


def build_prompt(question, hits):
    blocks = [
        f"[Context {i}] {h['document']} | page {h['page']}\n{h['text']}"
        for i, h in enumerate(hits, start=1)
    ]
    return ("CONTEXT:\n" + "\n\n".join(blocks) +
            f"\n\nQUESTION: {question}\n\nANSWER:")


@app.get("/health")
def health():
    return jsonify({"status": "ok", "chunks": len(CHUNKS)})


@app.post("/query")
def query():
    data = request.get_json(force=True, silent=True) or {}
    question = (data.get("question") or "").strip()
    top_k = int(data.get("top_k", DEFAULT_K))
    if not question:
        return jsonify({"error": "'question' is required"}), 400

    hits = retrieve(question, top_k)

    resp = LLM.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(question, hits)},
        ],
        temperature=0.0,
        max_tokens=400,
    )
    answer = resp.choices[0].message.content.strip()

    sources = [
        {"document": h["document"], "page": h["page"],
         "score": round(h["score"], 4), "text": h["text"][:600]}
        for h in hits
    ]
    return jsonify({"answer": answer, "sources": sources})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
