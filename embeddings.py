"""Cached semantic embeddings; interchangeable Gemini/Sentence Transformer backends."""

import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time
from contextlib import closing

import numpy as np
from dotenv import load_dotenv
from normalize import ROOT

HERE = Path(__file__).resolve().parent


def unit_vectors(values, count):
    vectors = np.asarray(values, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] != count or vectors.shape[1] == 0:
        raise ValueError("embedding count or dimensions are invalid")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if not np.isfinite(vectors).all() or not np.isfinite(norms).all() or (norms == 0).any():
        raise ValueError("embeddings must be finite and nonzero")
    return vectors / norms


class BaseEmbedder:
    def __init__(self, backend=None, model=None, cache=None, *, input_format="bare-base-v1"):
        load_dotenv(ROOT / ".env", override=False)
        self.backend = backend or os.getenv("SCHEMA_EMBEDDING_BACKEND", "gemini")
        if self.backend not in {"gemini", "sentence-transformers"}:
            raise ValueError("unsupported embedding backend")
        self.model = model or os.getenv("SCHEMA_EMBEDDING_MODEL") or (
            "gemini-embedding-001" if self.backend == "gemini" else "BAAI/bge-m3")
        self.cache = Path(cache) if cache is not None else HERE / ".cache" / "bases.sqlite3"
        # Isolate cache entries across providers/models/input and task conventions.
        self.settings = dict(backend=self.backend, model=self.model,
                             input_format=input_format, task="SEMANTIC_SIMILARITY")
        self._client = None
        self.cache_hits = 0
        self.embedded_count = 0

    def _encode(self, texts):
        if self.backend == "gemini":
            from google import genai
            from google.genai import types
            if self._client is None:
                key = os.getenv("GEMINI_API_KEY")
                if not key:
                    raise ValueError("GEMINI_API_KEY is required for uncached Gemini embeddings")
                self._client = genai.Client(api_key=key,
                    http_options=types.HttpOptions(timeout=60000))
            from google.genai.errors import APIError
            for attempt in range(4):
                try:
                    response = self._client.models.embed_content(
                        model=self.model, contents=texts,
                        config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY"))
                    break
                except APIError as exc:
                    details = exc.details.get("error", {}).get("details", []) if isinstance(exc.details, dict) else []
                    delays = [float(d["retryDelay"].removesuffix("s")) for d in details if "retryDelay" in d]
                    if exc.code != 429 or attempt == 3 or not delays or not 0 <= max(delays) < 60:
                        raise
                    time.sleep(max(delays) + 1)
            return [e.values for e in response.embeddings or []]
        if self._client is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError("Install schema_matching/requirements-local.txt for local models") from exc
            self._client = SentenceTransformer(self.model, device="cpu",
                cache_folder=str(HERE / ".cache" / "models"), trust_remote_code=False)
        return self._client.encode(texts, batch_size=32, normalize_embeddings=True,
                                   show_progress_bar=False)

    def encode(self, bases):
        if not bases or any(not isinstance(b, str) or not b.strip() for b in bases):
            raise ValueError("provide nonempty base strings")
        self.cache.parent.mkdir(parents=True, exist_ok=True)
        unique = list(dict.fromkeys(bases))
        values = {}
        keys = {b: hashlib.sha256(json.dumps([self.settings, b], sort_keys=True).encode()).hexdigest()
                for b in unique}
        with closing(sqlite3.connect(self.cache)) as db, db:
            db.execute("CREATE TABLE IF NOT EXISTS embeddings (key TEXT PRIMARY KEY, vector TEXT NOT NULL)")
            for base in unique:
                row = db.execute("SELECT vector FROM embeddings WHERE key=?", (keys[base],)).fetchone()
                if row is not None:
                    values[base] = json.loads(row[0])
                    self.cache_hits += 1
            missing = [b for b in unique if b not in values]
            for start in range(0, len(missing), 64):
                batch = missing[start:start + 64]
                vectors = unit_vectors(self._encode(batch), len(batch))
                for base, vector in zip(batch, vectors):
                    values[base] = vector.tolist()
                    db.execute("INSERT OR REPLACE INTO embeddings VALUES (?, ?)",
                               (keys[base], json.dumps(values[base])))
                self.embedded_count += len(batch)
                db.commit()
        return unit_vectors([values[b] for b in bases], len(bases))
