"""LLM (Groq) + grounded fallback — Week 5.

Uses Groq only when GROQ_API_KEY is set. Without a key (tests, local
dev) it returns a grounded answer built from the retrieved chunks, so
the whole RAG flow works offline and tests never hit the network.
"""
import re

from .config import settings
from .security import build_rag_prompt

# Rough Groq price used only for the cost estimate in /api/admin/analytics
# (llama-3.1-8b-instant is about $0.05 / 1M input tokens, $0.08 / 1M output).
ESTIMATED_COST_PER_TOKEN_USD = 0.00000006


def estimate_tokens(text: str) -> int:
    return len((text or "").split())


def _fallback_answer(question: str, chunks: list) -> tuple:
    """Grounded answer from retrieved chunks — no LLM call at all."""
    words = {w.strip(".,!?").lower() for w in question.split() if len(w) > 3}
    picked = []
    for chunk in chunks:
        text = chunk["text"] if isinstance(chunk, dict) else chunk.text
        if words and any(w in text.lower() for w in words):
            picked.append(text)
    if not picked:
        picked = [
            (c["text"] if isinstance(c, dict) else c.text) for c in chunks[:2]
        ]
    if not picked:
        answer = "I could not find anything about that in your documents."
    else:
        # Keep the answer short: the sentence(s) most relevant to the
        # question, not just the first sentence of the chunk.
        sentences = []
        for text in picked:
            sentences.extend(re.split(r"(?<=[.!?])\s+", text.strip()))
        scored = []
        for s in sentences:
            low = s.lower()
            score = sum(1 for w in words if w in low)
            scored.append((score, s.strip()))
        scored.sort(key=lambda item: item[0], reverse=True)
        best = [s for score, s in scored[:2] if s]
        answer = "Based on your documents: " + " ".join(best)
    prompt_tokens = estimate_tokens(build_rag_prompt(question, chunks))
    completion_tokens = estimate_tokens(answer)
    return answer, prompt_tokens, completion_tokens


def generate_answer(question: str, chunks: list) -> tuple:
    """Return (answer, prompt_tokens, completion_tokens).

    Chunks are dicts {"text", "document_id", "chunk_index", "score"} or
    Chunk objects — both are accepted.
    """
    prompt = build_rag_prompt(question, chunks)
    if not settings.groq_api_key:
        return _fallback_answer(question, chunks)
    try:
        from groq import Groq

        client = Groq(api_key=settings.groq_api_key)
        completion = client.chat.completions.create(
            model=settings.groq_model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": question},
            ],
            temperature=0.2,
            max_tokens=1024,
        )
        answer = completion.choices[0].message.content or ""
        usage = getattr(completion, "usage", None)
        if usage:
            return answer, int(usage.prompt_tokens or 0), int(usage.completion_tokens or 0)
        return answer, estimate_tokens(prompt), estimate_tokens(answer)
    except Exception:
        # Never fail a chat request because the LLM is unreachable —
        # fall back to the grounded answer from the retrieved chunks.
        return _fallback_answer(question, chunks)


def estimate_cost_usd(total_tokens: int) -> float:
    return round(total_tokens * ESTIMATED_COST_PER_TOKEN_USD, 6)
