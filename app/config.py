"""Configuration — every secret comes from .env, nothing is hard-coded.

Copy .env.example to .env and paste your own keys there.
"""
import os

from pydantic_settings import BaseSettings, SettingsConfigDict


def is_vercel() -> bool:
    """True when running as a Vercel serverless function (Vercel sets
    VERCEL=1 automatically). Used to switch off anything that needs a
    continuously running process and to use serverless-friendly pooling."""
    return os.getenv("VERCEL") == "1"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "Cortex AI Backend API"
    # SQLite by default so the project runs anywhere with zero setup.
    # For production set e.g. postgresql://user:pass@host:5432/cortex
    database_url: str = "sqlite:///./cortex.db"

    jwt_secret: str = "change-me-in-production-use-a-long-random-secret"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60 * 24  # 1 day

    groq_api_key: str = ""
    groq_model: str = "llama-3.1-8b-instant"

    stripe_secret_key: str = ""
    stripe_webhook_secret: str = ""
    stripe_price_pro: str = ""
    stripe_price_team: str = ""
    app_base_url: str = "http://localhost:8000"

    # Emails (comma-separated) that are made admin on registration.
    # Example: ADMIN_EMAILS=ahmad@example.com,mentor@zyroo.org
    admin_emails: str = ""

    max_upload_mb: int = 10
    chat_max_chars: int = 2000

    # Secret Vercel Cron sends as `Authorization: Bearer <CRON_SECRET>`
    # when it calls GET /api/cron/usage-reset. Generate a long random value.
    cron_secret: str = ""

    @property
    def effective_max_upload_mb(self) -> int:
        """Vercel rejects request bodies over ~4.5 MB at the platform
        level, so on Vercel the upload cap is 4 MB even if MAX_UPLOAD_MB
        is set higher. Locally the configured value applies."""
        return min(self.max_upload_mb, 4) if is_vercel() else self.max_upload_mb

    # Embeddings: "hash" = deterministic offline fallback (default, used
    # by the tests), "fastembed" = real neural embeddings
    # (sentence-transformers/all-MiniLM-L6-v2, 384 dims, via ONNX).
    # Set EMBEDDING_PROVIDER=fastembed in production for true semantic search.
    embedding_provider: str = "hash"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    @property
    def admin_email_set(self) -> set:
        return {e.strip().lower() for e in self.admin_emails.split(",") if e.strip()}


settings = Settings()
