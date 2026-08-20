from __future__ import annotations

import glob
import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from blindspot import prompts

EMBED_MODEL = "together_ai/BAAI/bge-base-en-v1.5"
CHUNK_TOKENS = 256
CHUNK_OVERLAP = 40
MAX_CONTEXT_CHARS = 6500
MIN_CHUNK_CHARS = 40

@lru_cache(maxsize=1)
def _tokenizer(model: str = EMBED_MODEL):
    """the embedding model's own tokenizer."""
    from tokenizers import Tokenizer

    return Tokenizer.from_pretrained(model.split("/", 1)[-1] if "/" in model else model)


def tokenize(text: str, model: str = EMBED_MODEL) -> list[str]:
    """text as the embedding model's tokens."""
    return _tokenizer(model).encode(text, add_special_tokens=False).tokens


def token_count(text: str, model: str = EMBED_MODEL) -> int:
    return len(_tokenizer(model).encode(text, add_special_tokens=False).ids)


def chunk(text: str, size: int = CHUNK_TOKENS, overlap: int = CHUNK_OVERLAP,
          model: str = EMBED_MODEL) -> list[str]:
    """split text into overlapping windows of `size` tokens."""
    encoded = _tokenizer(model).encode(text, add_special_tokens=False)
    offsets = encoded.offsets
    if not offsets:
        return []

    step = max(1, size - overlap)
    pieces = []
    for start in range(0, len(offsets), step):
        window = offsets[start:start + size]
        if not window:
            break
        piece = text[window[0][0]:window[-1][1]].strip()
        if piece:
            pieces.append(piece)
        if start + size >= len(offsets):
            break
    return pieces


def _stringify(record: dict[str, Any]) -> str:
    lead = [f"{k}: {record[k].strip()}" for k in ("need", "domain", "axis", "group")
            if isinstance(record.get(k), str) and record[k].strip()]
    rest = [f"{k}: {v.strip()}" for k, v in record.items()
            if k not in ("need", "domain", "axis", "group") and isinstance(v, str) and v.strip()]
    return " | ".join(lead + rest).strip()


def extract_text(obj: Any) -> list[str]:
    """flatten a source JSON document into text blobs suitable for chunking."""
    if obj is None:
        return []
    if isinstance(obj, str):
        return [obj.strip()] if obj.strip() else []
    if isinstance(obj, list):
        return [b for item in obj for b in extract_text(item)]
    if isinstance(obj, dict):
        if isinstance(obj.get("needs"), list):
            out = []
            for item in obj["needs"]:
                if isinstance(item, dict):
                    s = _stringify(item)
                    if s:
                        out.append(s)
                else:
                    out.extend(extract_text(item))
            return out
        out = [f"{k}: {v.strip()}" for k, v in obj.items()
               if isinstance(v, str) and v.strip()]
        for v in obj.values():
            if isinstance(v, (dict, list)):
                out.extend(extract_text(v))
        return out
    return []


def embed(texts: list[str], model: str = EMBED_MODEL, batch_size: int = 96) -> np.ndarray:
    """L2-normalised embeddings, so cosine similarity is a dot product."""
    import litellm

    vectors: list[np.ndarray] = []
    for i in range(0, len(texts), batch_size):
        resp = litellm.embedding(model=model, input=texts[i : i + batch_size])
        vectors.append(np.array([d["embedding"] for d in resp.data], dtype=np.float32))
        if len(texts) > batch_size:
            print(f"  embedded {min(i + batch_size, len(texts))}/{len(texts)}")
    if not vectors:
        return np.zeros((0, 1), dtype=np.float32)
    mat = np.vstack(vectors)
    return mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12)


@dataclass
class Index:
    """A flat, in-memory vector index over every source document."""

    chunks: list[str]
    tags: list[str]
    embeddings: np.ndarray

    def __len__(self) -> int:
        return len(self.chunks)

    def search(self, query_vec: np.ndarray, top_k: int) -> list[tuple[float, str, int, str]]:
        if not len(self):
            return []
        sims = self.embeddings @ query_vec
        k = min(top_k, len(sims))
        best = np.argpartition(-sims, kth=k - 1)[:k]
        hits = [(float(sims[j]), self.tags[j], int(j), self.chunks[j]) for j in best]
        hits.sort(key=lambda h: -h[0])
        return hits

    def context_for(self, query_vec: np.ndarray, top_k: int) -> str:
        """render the top-k hits as a citable CONTEXT block."""
        parts, total = [], 0
        for _, tag, j, text in self.search(query_vec, top_k):
            block = f"[{tag}#c{j}] {text}".strip()
            if total + len(block) + 2 > MAX_CONTEXT_CHARS:
                break
            parts.append(block)
            total += len(block) + 2
        return f"{prompts.RAG_CONTEXT_HEADER}\n" + "\n\n".join(parts) if parts else ""


def _cache_key(paths: list[Path], model: str) -> str:
    sig = "|".join(f"{p}:{p.stat().st_mtime_ns}:{p.stat().st_size}" for p in paths)
    return hashlib.md5(f"{model}|{CHUNK_TOKENS}|{CHUNK_OVERLAP}|{sig}".encode()).hexdigest()[:16]


def build_index(
    source_dir: str | Path,
    model: str = EMBED_MODEL,
    cache_dir: str | Path = "rag_index_cache",
) -> Index:
    """
    build (or reload) the global index over every ``*.json`` under `source_dir`.

    embedding the corpus costs money, so the result is cached against the
    content signature of the source files and the chunking parameters.
    """
    source_dir = Path(source_dir)
    files = sorted(Path(p) for p in glob.glob(str(source_dir / "**" / "*.json"), recursive=True))
    if not files:
        raise FileNotFoundError(
            f"No source documents under {source_dir}. "
            f"These ship in the results archive — run `blindspot fetch`, or pass --source-dir."
        )

    cache_dir = Path(cache_dir)
    key = _cache_key(files, model)
    meta_path = cache_dir / f"{key}.json"
    emb_path = cache_dir / f"{key}.npy"

    if meta_path.exists() and emb_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        print(f"  reusing cached index ({len(meta['chunks'])} chunks) from {meta_path}")
        return Index(meta["chunks"], meta["tags"], np.load(emb_path))

    chunks: list[str] = []
    tags: list[str] = []
    for path in files:
        try:
            blobs = extract_text(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            print(f"  [skip] {path.name}: not valid JSON")
            continue
        for blob in blobs:
            for piece in chunk(blob):
                if len(piece) >= MIN_CHUNK_CHARS:
                    chunks.append(piece)
                    tags.append(path.name)

    if not chunks:
        raise ValueError(f"No usable text extracted from {len(files)} files under {source_dir}")

    print(f"  indexing {len(chunks)} chunks from {len(files)} source documents")
    embeddings = embed(chunks, model)

    cache_dir.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps({"chunks": chunks, "tags": tags}), encoding="utf-8")
    np.save(emb_path, embeddings)
    return Index(chunks, tags, embeddings)


def contexts_for(index: Index, queries: list[str], top_k: int,
                 model: str = EMBED_MODEL) -> list[str]:
    """CONTEXT blocks for a list of prompts. queries are the prompts alone."""
    if not queries:
        return []
    query_vecs = embed(queries, model)
    return [index.context_for(query_vecs[i], top_k) for i in range(len(queries))]
