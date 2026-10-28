"""Middleware — security headers (helmet equivalent), request logging,
and in-memory sliding-window rate limiting (Weeks 1 + 7-8).
"""
import logging
import time
from collections import defaultdict, deque

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

logger = logging.getLogger("cortex")

SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "X-XSS-Protection": "1; mode=block",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    # 'unsafe-inline' is needed so Swagger UI (/docs) still renders.
    "Content-Security-Policy": (
        "default-src 'self'; style-src 'self' 'unsafe-inline'; "
        "script-src 'self' 'unsafe-inline'"
    ),
}

# requests per minute, by path prefix. Chat/search cost LLM tokens, so
# they are stricter than the rest of the API.
RATE_LIMITS = [
    ("/api/auth", 60),
    ("/api/chat", 60),
    ("/api/search", 60),
    ("", 300),
]


def _limit_for(path: str) -> int:
    for prefix, limit in RATE_LIMITS:
        if path.startswith(prefix):
            return limit
    return 300


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        for key, value in SECURITY_HEADERS.items():
            response.headers[key] = value
        return response


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        started = time.time()
        response = await call_next(request)
        elapsed_ms = (time.time() - started) * 1000
        logger.info(
            "%s %s -> %s (%.1f ms)",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """In-memory sliding-window limiter, keyed by client IP + path group."""

    def __init__(self, app, enabled: bool = True):
        super().__init__(app)
        self.enabled = enabled
        self.hits = defaultdict(deque)  # key -> timestamps

    def reset(self):
        self.hits.clear()

    async def dispatch(self, request, call_next):
        if not self.enabled:
            return await call_next(request)
        path = request.url.path
        if path in ("/api/health", "/docs", "/openapi.json", "/redoc"):
            return await call_next(request)
        limit = _limit_for(path)
        client = request.client.host if request.client else "unknown"
        key = f"{client}:{path.split('/')[2] if len(path.split('/')) > 2 else 'root'}"
        now = time.time()
        window = self.hits[key]
        while window and now - window[0] > 60:
            window.popleft()
        if len(window) >= limit:
            return JSONResponse(
                status_code=429,
                content={"detail": "Too many requests. Please slow down and try again."},
            )
        window.append(now)
        return await call_next(request)
