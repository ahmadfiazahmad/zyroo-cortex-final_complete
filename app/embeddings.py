"""Embeddings — Weeks 3-4, upgraded to real neural embeddings in production.

Two providers, chosen by EMBEDDING_PROVIDER in .env:

* "fastembed" (production) — real neural embeddings from
  sentence-transformers/all-MiniLM-L6-v2 (384 dims) via ONNX, so
  synonyms and paraphrases actually match. The model (~90 MB) is
  downloaded once from Hugging Face on first use. No torch, no paid API.
* "hash" (default, tests/offline) — a deterministic hash-based embedding
  (also 384 dims, unit norm): words shared by query and chunk raise
  their cosine similarity. Fully offline and reproducible, but it is
  keyword overlap, not semantics — use it only for tests and local dev
  without a model download.

Both providers return 384-dim unit-norm lists, so they fit the same
pgvector Vector(384) column and the rest of the code only ever calls
``embed_text`` / ``embed_texts``.
"""
import hashlib
import logging
import time

from .config import settings

logger = logging.getLogger("cortex")

DIM = 384

# ---------------------------------------------------------------------------
# Hash fallback (offline / tests)
# ---------------------------------------------------------------------------


def _hash_embed(text: str) -> list:
    vector = [0.0] * DIM
    for word in (text or "").lower().split():
        word = word.strip(".,!?;:\"'()[]{}")
        if not word:
            continue
        digest = hashlib.sha256(word.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "big") % DIM
        vector[idx] += 1.0
    norm = sum(v * v for v in vector) ** 0.5
    if norm == 0:
        return vector
    return [v / norm for v in vector]


# ---------------------------------------------------------------------------
# FastEmbed (real neural embeddings, production)
# ---------------------------------------------------------------------------
_fastembed_model = None


def _load_fastembed_model():
    global _fastembed_model
    if _fastembed_model is None:
        import os

        from fastembed import TextEmbedding

        # Vercel's filesystem is read-only except /tmp, and /tmp is
        # per-function-instance (it survives warm invocations, but a cold
        # start re-downloads the ~90 MB model). Point the cache there on
        # Vercel; locally the default cache directory is used.
        # FASTEMBED_CACHE_DIR overrides this when set.
        cache_dir = os.environ.get("FASTEMBED_CACHE_DIR")
        if cache_dir is None and os.getenv("VERCEL") == "1":
            cache_dir = "/tmp/fastembed_cache"
        kwargs: dict = {"model_name": settings.embedding_model}
        if cache_dir:
            kwargs["cache_dir"] = cache_dir
        _fastembed_model = TextEmbedding(**kwargs)
    return _fastembed_model


def _fastembed_texts(texts: list) -> list:
    model = _load_fastembed_model()
    # FastEmbed returns normalised embeddings for this model already;
    # normalise again defensively so cosine distance behaves the same
    # whichever provider produced the vector.
    vectors = []
    for vector in model.embed(texts):
        values = [float(v) for v in vector]
        norm = sum(v * v for v in values) ** 0.5
        vectors.append([v / norm for v in values] if norm else values)
    return vectors


# ---------------------------------------------------------------------------
# Public API — used by documents, search and chat
# ---------------------------------------------------------------------------


def embed_text(text: str) -> list:
    """Embed one text (384 floats, unit norm)."""
    return embed_texts([text])[0]


def embed_texts(texts: list, max_retries: int = 3) -> list:
    """Embed a batch, with retry/backoff (the Week 3-4 requirement).

    With the FastEmbed provider, a failed model load (e.g. no network
    for the one-time model download) falls back to the hash embedding
    for that batch rather than failing the upload/search.
    """
    provider = (settings.embedding_provider or "hash").lower()
    last_error = None
    for attempt in range(max_retries):
        try:
            if provider == "fastembed":
                try:
                    return _fastembed_texts(texts)
                except Exception as e:
                    logger.warning(
                        "FastEmbed failed (%s); falling back to hash embeddings", e
                    )
                    return [_hash_embed(t) for t in texts]
            return [_hash_embed(t) for t in texts]
        except Exception as e:  # pragma: no cover - defensive
            last_error = e
            time.sleep(0.2 * (2 ** attempt))
    raise last_error  # pragma: no cover
