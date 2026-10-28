"""Chunking + similarity — Weeks 3-4.

Tokens are approximated as words (1 word ~= 1 token for this project).
Chunks are ~500 tokens with ~50-token overlap, exactly as the brief asks.
"""
import math


def count_tokens(text: str) -> int:
    """Approximate token count: number of whitespace-separated words."""
    return len((text or "").split())


def chunk_text(text: str, chunk_tokens: int = 500, overlap_tokens: int = 50) -> list:
    """Split text into overlapping chunks.

    Returns a list of {"text", "token_count"} dicts. A short text gives
    one chunk; longer text slides forward by (chunk - overlap) tokens.
    """
    words = (text or "").split()
    if not words:
        return []
    if len(words) <= chunk_tokens:
        joined = " ".join(words)
        return [{"text": joined, "token_count": len(words)}]
    step = chunk_tokens - overlap_tokens
    chunks = []
    for start in range(0, len(words), step):
        window = words[start : start + chunk_tokens]
        if not window:
            break
        chunks.append({"text": " ".join(window), "token_count": len(window)})
        if start + chunk_tokens >= len(words):
            break
    return chunks


def cosine_similarity(a: list, b: list) -> float:
    """Cosine similarity between two equal-length vectors (pure Python)."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def top_k_chunks(query_embedding: list, chunks: list, k: int = 5) -> list:
    """Rank chunks by cosine similarity to the query embedding."""
    scored = []
    for chunk in chunks:
        emb = chunk.get_embedding() if hasattr(chunk, "get_embedding") else chunk.get("embedding", [])
        score = cosine_similarity(query_embedding, emb)
        scored.append((score, chunk))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [{"score": round(score, 4), "chunk": chunk} for score, chunk in scored[:k]]
