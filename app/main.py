"""Cortex AI Backend API — Weeks 1-8 (Python + FastAPI).

Run:  uvicorn main:app --reload
Docs: http://localhost:8000/docs
"""
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .database import init_db
from .middleware import (
    RateLimitMiddleware,
    RequestLoggingMiddleware,
    SecurityHeadersMiddleware,
)
from .plans import reset_due_usage
from .routers import admin, auth, billing, chat, cron, documents, search, webhooks, workspaces

logging.basicConfig(level=logging.INFO)

tags_metadata = [
    {"name": "Auth", "description": "Registration, login (JWT) and the current user."},
    {"name": "Documents", "description": "Upload, list, get and delete documents."},
    {"name": "Search", "description": "Semantic search over document chunks."},
    {"name": "Chat", "description": "RAG chat sessions and messages (Groq LLM)."},
    {"name": "Billing", "description": "Plans, Stripe Checkout, usage and cancellation."},
    {"name": "Workspaces", "description": "Team workspaces, members and shared resources."},
    {"name": "Admin", "description": "Admin-only users, workspaces and analytics."},
    {"name": "Cron", "description": "Scheduled jobs (Vercel Cron calls these)."},
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    scheduler = None
    # On Vercel there is no long-running process: a BackgroundScheduler
    # would die with the invocation, so the daily reset is done by
    # Vercel Cron calling GET /api/cron/usage-reset instead (see
    # vercel.json). Locally / on Render the scheduler still runs.
    if os.getenv("VERCEL") != "1":
        try:
            from apscheduler.schedulers.background import BackgroundScheduler

            from .database import SessionLocal

            def reset_job():
                db = SessionLocal()
                try:
                    reset_due_usage(db)
                finally:
                    db.close()

            scheduler = BackgroundScheduler()
            scheduler.add_job(reset_job, "interval", days=1, id="usage-reset")
            scheduler.start()
        except Exception:
            pass  # scheduler is a nice-to-have; lazy reset also guards requests
    yield
    if scheduler:
        scheduler.shutdown()


app = FastAPI(
    title="Cortex AI Backend API",
    description=(
        "Cortex AI — RAG-powered document chat with team workspaces, "
        "Stripe billing and admin analytics. Built for the ZYROO Cortex AI "
        "Backend Development Internship (Weeks 1-8) in Python + FastAPI.\n\n"
        "**Auth:** register / log in to get a JWT, then send it as "
        "`Authorization: Bearer <token>` on every protected endpoint."
    ),
    version="1.0.0",
    openapi_tags=tags_metadata,
    lifespan=lifespan,
)

# Week 1: CORS, logging, security headers, rate limiting, error handler.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(RateLimitMiddleware)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logging.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


@app.get("/api/health", tags=["Auth"], summary="Health check")
def health():
    """Confirm the server is running."""
    return {"status": "ok", "service": "Cortex AI Backend API", "version": "1.0.0"}


@app.get("/", summary="Service info")
def root():
    return {
        "service": "Cortex AI Backend API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/health",
    }


app.include_router(auth.router)
app.include_router(documents.router)
app.include_router(search.router)
app.include_router(chat.router)
app.include_router(billing.router)
app.include_router(webhooks.router)
app.include_router(workspaces.router)
app.include_router(admin.router)
app.include_router(cron.router)
