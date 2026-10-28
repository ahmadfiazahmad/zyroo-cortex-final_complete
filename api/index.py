"""Vercel entry point for the FastAPI app.

Vercel's Python runtime treats files under ``api/`` as serverless
functions and supports ASGI apps natively: exporting an ``app`` variable
that is a FastAPI instance is enough — no adapter (e.g. Mangum) needed.
The root ``main.py`` is NOT auto-detected by Vercel, which is why this
file exists. Local development still uses ``uvicorn main:app``.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.main import app  # noqa: E402

__all__ = ["app"]
