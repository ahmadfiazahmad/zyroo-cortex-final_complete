# Cortex AI Backend — Final Submission (Weeks 1–8)

ZYROO Cortex AI Backend Development Internship · Final Week (Weeks 7–8).
Stack: **Python 3.10+ · FastAPI · SQLAlchemy · Groq LLM · Stripe · Pytest**.

A complete, end-to-end RAG platform: users register, upload documents, search them by meaning, and chat with cited answers from a Groq LLM — with subscription billing, shared team workspaces, admin analytics, automated tests, security hardening, full API docs, and deployment configs.

## Architecture

```
Client / Swagger UI (/docs)
        │
        ▼
FastAPI app (app/main.py) — CORS, security headers, request logging,
rate limiting, global error handler, daily usage-reset scheduler
        │
        ├── Auth        app/routers/auth.py        JWT, bcrypt, /api/auth/me
        ├── Documents   app/routers/documents.py   upload → extract → chunk → embed
        ├── Search      app/routers/search.py      cosine similarity over chunks
        ├── Chat        app/routers/chat.py        RAG: retrieve → Groq → cited reply
        ├── Billing     app/routers/billing.py     plans, Stripe Checkout, cancel
        ├── Webhooks    app/routers/webhooks.py    verified Stripe webhooks
        ├── Workspaces  app/routers/workspaces.py  Team workspaces + members
        └── Admin       app/routers/admin.py       users, workspaces, analytics
        │
        ▼
SQLAlchemy models (app/models.py) — SQLite by default,
PostgreSQL + pgvector via DATABASE_URL (see "Vector storage" below)
        │
        ├── app/chunking.py    ~500-token chunks, ~50-token overlap
        ├── app/embeddings.py  FastEmbed neural embeddings (prod) / hash fallback (tests)
        ├── app/retrieval.py   pgvector top-k search (Postgres) / Python cosine (SQLite)
        ├── app/llm.py         Groq chat completions + grounded offline fallback
        ├── app/security.py    passwords, JWT, sanitization, prompt-injection guard
        └── app/plans.py       Free / Pro / Team limits + quota + usage reset
```

## What each week built

| Week | Feature |
|------|---------|
| 1 | FastAPI setup, CORS, request logging, global error handler, `.env` config, `GET /api/health` |
| 2 | User model, bcrypt passwords, JWT register/login/me, `get_current_user` |
| 3–4 | Document + Chunk models, PDF/DOCX/TXT upload, extraction, overlapping chunking, batch embeddings, `POST /api/search` (cosine similarity) |
| 5 | Chat sessions + messages, RAG pipeline with cited sources, SSE streaming |
| 6 | Free/Pro/Team plans, Stripe Checkout, verified webhooks, quota enforcement, scheduled usage reset |
| 7–8 | Team workspaces, admin + analytics, pytest suite, security hardening, OpenAPI docs, deployment |

### Plan limits

| Plan | Price | Documents | Messages / period | Storage |
|------|-------|-----------|-------------------|---------|
| Free | $0 | 10 | 100 | 100 MB |
| Pro | $19/mo | 500 | 5,000 | 10 GB |
| Team | $49/mo | 5,000 | 50,000 | 100 GB |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # then paste your keys into .env
uvicorn main:app --reload
```

Open **http://localhost:8000/docs** for the interactive API docs.

### Environment variables

| Variable | Required | What it is |
|----------|----------|------------|
| `DATABASE_URL` | No | SQLite default (`sqlite:///./cortex.db`); set a Postgres URL in production — the app then uses pgvector (see below) |
| `EMBEDDING_PROVIDER` | No | `hash` (default, offline/tests) or `fastembed` (real neural embeddings — set this in production) |
| `JWT_SECRET` | Yes (prod) | Long random secret for signing JWTs |
| `GROQ_API_KEY` | No | Groq key for live LLM answers. Without it, chat uses a grounded fallback built from your retrieved chunks, so everything still works offline |
| `GROQ_MODEL` | No | Default `llama-3.1-8b-instant` |
| `STRIPE_SECRET_KEY` | For billing | Stripe test/live secret key |
| `STRIPE_WEBHOOK_SECRET` | For billing | `whsec_...` from `stripe listen --forward-to localhost:8000/api/webhooks/stripe` |
| `STRIPE_PRICE_PRO` / `STRIPE_PRICE_TEAM` | For billing | Stripe Price IDs for the Pro ($19) and Team ($49) plans |
| `APP_BASE_URL` | No | Default `http://localhost:8000` (used for Stripe success/cancel URLs) |
| `ADMIN_EMAILS` | No | Comma-separated emails that become admin on registration |
| `EMBEDDING_PROVIDER` | No | `hash` (default, offline/tests) or `fastembed` (real neural embeddings; set on Vercel) |
| `CRON_SECRET` | Vercel only | Long random secret for Vercel Cron (`GET /api/cron/usage-reset`); invent it yourself and set the same value in Vercel |
| `MAX_UPLOAD_MB` | No | Upload cap in MB (default 10; forced to 4 on Vercel — see Deployment) |

Without Stripe keys, billing endpoints return a clear `503` telling you what to set; plans, usage and quotas all work without keys. Without a Groq key, chat answers are grounded in your documents by the fallback — no network call is made.

**Admin access:** set your email in `ADMIN_EMAILS` before registering, or promote a user in the database (`role='admin'`). Admins can list users/workspaces, suspend accounts, and view analytics.

## Try it

```bash
# 1. Health
curl http://localhost:8000/api/health

# 2. Register + log in (save the token)
curl -X POST http://localhost:8000/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{"name":"Ahmad","email":"ahmad@example.com","password":"password123"}'

# 3. Upload a document (counts toward the Free plan's 10-document quota)
curl -X POST http://localhost:8000/api/documents \
  -H "Authorization: Bearer <token>" \
  -F "file=@notes.txt;type=text/plain"

# 4. Semantic search
curl -X POST http://localhost:8000/api/search \
  -H "Authorization: Bearer <token>" -H "Content-Type: application/json" \
  -d '{"query":"what is this document about?"}'

# 5. RAG chat — create a session, then ask a question
curl -X POST http://localhost:8000/api/chat/sessions \
  -H "Authorization: Bearer <token>" -H "Content-Type: application/json" \
  -d '{"title":"My notes"}'
curl -X POST http://localhost:8000/api/chat/sessions/<id>/messages \
  -H "Authorization: Bearer <token>" -H "Content-Type: application/json" \
  -d '{"content":"What is the capital of France?"}'
# -> assistant reply with "sources" citing the document chunk(s) used

# 6. Billing
curl http://localhost:8000/api/billing/plans
curl http://localhost:8000/api/billing/me -H "Authorization: Bearer <token>"
```

## Running the tests

```bash
pytest -v
```

20 tests, all passing, no network needed: unit tests (chunking, quotas, usage reset, embeddings, prompt-injection guard) and integration tests (full RAG flow, quota blocking, workspace + admin flow, webhook signature rejection, security headers). Tests use an isolated SQLite database (`test_cortex.db`, deleted after the run) and never call Groq or Stripe live.

## API reference

Auth: send `Authorization: Bearer <token>` on every protected endpoint.

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/api/health` | — | Health check |
| POST | `/api/auth/register` | — | Register, returns JWT + user |
| POST | `/api/auth/login` | — | Log in, returns JWT + user |
| GET | `/api/auth/me` | JWT | Logged-in user |
| POST | `/api/documents` | JWT | Upload PDF/DOCX/TXT (max 10 MB) |
| GET | `/api/documents` | JWT | List my + shared documents |
| GET | `/api/documents/{id}` | JWT | One document |
| DELETE | `/api/documents/{id}` | JWT | Delete document + its chunks |
| POST | `/api/search` | JWT | Semantic search (top-k chunks) |
| POST | `/api/chat/sessions` | JWT | Create a chat session |
| GET | `/api/chat/sessions` | JWT | List my + shared sessions |
| DELETE | `/api/chat/sessions/{id}` | JWT | Delete a session |
| POST | `/api/chat/sessions/{id}/messages` | JWT | Send a message, get a cited RAG reply |
| GET | `/api/chat/sessions/{id}/messages` | JWT | Message history |
| POST | `/api/chat/sessions/{id}/messages/stream` | JWT | Same reply, streamed (SSE) |
| GET | `/api/billing/plans` | — | Free / Pro / Team plans + limits |
| GET | `/api/billing/me` | JWT | My plan, usage and limits |
| POST | `/api/billing/checkout` | JWT | Stripe Checkout session (Pro/Team) |
| POST | `/api/billing/cancel` | JWT | Cancel at end of period |
| POST | `/api/webhooks/stripe` | Stripe sig | Verified Stripe webhook |
| POST | `/api/workspaces` | JWT + Team | Create a workspace (creator = owner) |
| GET | `/api/workspaces` | JWT | List my workspaces |
| GET | `/api/workspaces/{id}` | JWT member | One workspace |
| POST | `/api/workspaces/{id}/invite` | JWT owner | Invite a member by email |
| DELETE | `/api/workspaces/{id}/members/{userId}` | JWT owner | Remove a member |
| GET | `/api/workspaces/{id}/documents` | JWT member | Shared documents |
| GET | `/api/workspaces/{id}/chat-sessions` | JWT member | Shared chat sessions |
| GET | `/api/admin/users` | JWT admin | List users (paginated, no password hashes) |
| GET | `/api/admin/workspaces` | JWT admin | List workspaces |
| PATCH | `/api/admin/users/{id}/suspend` | JWT admin | Suspend / reactivate a user |
| GET | `/api/admin/analytics` | JWT admin | Daily active users, messages/day, plan distribution, subscriptions, LLM token usage + estimated cost |

Full machine-readable docs: `openapi.json` and `postman_collection.json` in the repo root (also at `/openapi.json` and `/docs` when the server runs).

## Security

- **Headers:** `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `X-XSS-Protection`, `Referrer-Policy`, CSP (Swagger UI still works).
- **Rate limiting:** in-memory sliding window — 60/min on `/api/auth`, `/api/chat`, `/api/search` (LLM calls cost tokens), 300/min elsewhere.
- **Prompt injection:** retrieved chunks are wrapped in `<context_documents>...</context_documents>`, the system prompt says to treat them as passive reference data only, and override phrases ("Ignore previous instructions", "SYSTEM OVERRIDE", …) are stripped from retrieved context.
- **Input limits:** chat messages max 2,000 characters, uploads max 10 MB and only `.pdf` / `.docx` / `.txt`.
- **Sanitization:** control characters and null bytes are stripped from string inputs.
- **Secrets:** everything from `.env` only — no keys in code. Stripe webhooks verify the signature with `stripe.Webhook.construct_event` and reject missing/invalid signatures with 400.
- **Suspended users** are rejected with 403 on every protected endpoint, immediately.

## How the pieces fit

- **Upload:** file → text extraction (pypdf / python-docx / plain UTF-8) → ~500-token overlapping chunks → local embeddings → document `ready`. Usage counts against your plan, or against the workspace's shared Team quota when `workspace_id` is given.
- **Search / RAG:** query → embedding → top-k chunks you can access (yours + your workspaces') → delimited prompt → Groq (or the grounded fallback) → reply stored with its cited sources and token counts.

## Vector storage (pgvector)

On **PostgreSQL**, chunk embeddings live in a real pgvector column: `Chunk.embedding` is `Vector(384)`, the app enables the extension on startup (`CREATE EXTENSION IF NOT EXISTS vector` in `app/database.py`), and there is an HNSW index (`vector_cosine_ops`) for fast search. Retrieval runs in the database — `ORDER BY embedding <=> :query_vector LIMIT k` (`app/retrieval.py`) — so only the k best chunks leave the database.

On **SQLite** (local dev and tests) the same column is a Text field holding a JSON list, and ranking falls back to pure-Python cosine similarity, so the project still runs anywhere with zero setup and the tests need no Postgres server.

Embeddings come from **FastEmbed** (`sentence-transformers/all-MiniLM-L6-v2`, 384 dims, ONNX — no torch, no paid API) when `EMBEDDING_PROVIDER=fastembed`. The default `hash` provider is a deterministic offline fallback for tests; it is keyword overlap, not semantics. If FastEmbed can't load its model (e.g. no network for the one-time ~90 MB download), it falls back to hash rather than failing the request.

Migrating an existing Postgres database that was created by the old JSON version:

```bash
psql "$DATABASE_URL" -f migrations/001_pgvector.sql
EMBEDDING_PROVIDER=fastembed python scripts/reembed_chunks.py
```

The re-embed step matters: old vectors were produced by a different model, so they must be regenerated for similarity search to be meaningful.
- **Billing:** Checkout creates a Stripe session with `metadata={user_id, plan}`; the webhook activates the plan and starts a fresh period. `customer.subscription.updated/deleted` sync status and downgrade to Free. Quotas are enforced by a dependency (429 with a clear message) and counters reset daily — by APScheduler locally / on Render, by Vercel Cron on Vercel (plus a lazy reset on request).
- **Workspaces:** a Team-plan user creates a workspace and becomes owner; the owner invites members by email. Shared documents and chat sessions count against the workspace's Team quota, and only members can see them. Admins oversee everything via `/api/admin/*`.
- **Analytics:** daily active users (messages + uploads, last 30 days), messages per day, plan distribution, active vs. cancelled subscriptions, and total LLM tokens/requests with an estimated cost.

## Deployment

**Live API URL:** _(paste your deployed URL here after deploying)_

### Vercel + Neon (current setup)

1. **Neon:** create a project at neon.tech (pick the region closest to your Vercel region). In the Neon dashboard, open the SQL Editor and run `CREATE EXTENSION IF NOT EXISTS vector;` (the app also runs this on startup). Copy the **pooled** connection string — it looks like `postgresql://user:password@ep-xxx-pooler.us-east-2.aws.neon.tech/neondb?sslmode=require`. Use the pooled string in the app; the direct (non-pooler) string is for migrations only.
2. **Vercel:** import the GitHub repo into Vercel. Vercel's Python runtime serves the app from `api/index.py` (the root `main.py` is for local uvicorn only — Vercel does not auto-detect it). `vercel.json` routes all requests there, sets a 60-second / 1024 MB function limit, and schedules the daily usage-reset cron.
3. **Environment variables in Vercel** (Dashboard → Settings → Environment Variables) — see the table below for the full list and where each value comes from. At minimum: `DATABASE_URL` (Neon pooled string), `JWT_SECRET`, `GROQ_API_KEY`, `STRIPE_SECRET_KEY`, `STRIPE_WEBHOOK_SECRET`, `STRIPE_PRICE_PRO`, `STRIPE_PRICE_TEAM`, `APP_BASE_URL` (your `https://<app>.vercel.app` URL), `CRON_SECRET` (a long random value you invent — Vercel Cron sends it automatically), and `ADMIN_EMAILS` (your email, so you become admin on registration). `EMBEDDING_PROVIDER=fastembed` and `MAX_UPLOAD_MB=4` are already set in `vercel.json`.
4. **First deploy / migrations:** on a fresh Neon database the app creates the tables, the `Vector(384)` column and the HNSW index itself on first startup. If you are migrating an existing database that still has JSON embeddings, run the migration from your own machine (use the **direct** Neon string here, not the pooled one):
   ```bash
   psql "$DATABASE_URL" -f migrations/001_pgvector.sql
   DATABASE_URL="<direct-neon-url>" EMBEDDING_PROVIDER=fastembed python scripts/reembed_chunks.py
   ```
5. **Stripe webhook:** in the Stripe dashboard, add an endpoint at `https://<your-app>.vercel.app/api/webhooks/stripe` and copy its signing secret into `STRIPE_WEBHOOK_SECRET`.
6. **Check it works:** `curl https://<your-app>.vercel.app/api/health` should return `{"status":"ok",...}`, and `/docs` should render the Swagger UI.

**Vercel limits to know about (Hobby plan):**
- **Uploads are capped at 4 MB on Vercel** (the platform rejects request bodies over ~4.5 MB). The app enforces this automatically when `VERCEL=1`. Locally / on Render the cap is 10 MB.
- **FastEmbed cold start:** the ~90 MB embedding model is downloaded once per function instance into `/tmp` and cached there while the instance is warm. The first request after a cold start (or the first upload/search after a deploy) can take 30–90 seconds while the model downloads, and may time out on a cold function. Warm requests are fast. If that is a problem, set `EMBEDDING_PROVIDER=hash` in Vercel instead (retrieval still works, but by keyword overlap rather than true semantics), or use Vercel Pro / an external embeddings API.
- **Neon scale-to-zero:** after ~5 minutes idle, Neon suspends the compute; the next request wakes it (a few seconds) and the app discards the stale connection automatically (`pool_pre_ping` + `NullPool` on Vercel).
- **Usage reset:** there is no long-running process on Vercel, so the APScheduler job is disabled there. Vercel Cron calls `GET /api/cron/usage-reset` once a day at midnight UTC instead (Hobby allows one cron run per day, with timing guaranteed only within the hour). The reset is idempotent, and usage is also reset lazily on each request, so a missed cron run cannot over-count anyone.
- **Function limits:** `vercel.json` sets `maxDuration: 60` seconds and 1024 MB memory (the Hobby-safe values). Long Groq chat replies and large uploads must finish inside that window.

### Render (still available)

1. Push this repo to GitHub.
2. Render → New → Blueprint → pick the repo (`render.yaml` creates the web service + a Postgres database).
3. In the Render dashboard, set `GROQ_API_KEY`, `STRIPE_SECRET_KEY`, `STRIPE_WEBHOOK_SECRET`, `STRIPE_PRICE_PRO`, `STRIPE_PRICE_TEAM`, and `APP_BASE_URL` to your live URL.
4. Set the Stripe webhook endpoint to `https://<your-app>.onrender.com/api/webhooks/stripe`.

### Railway / Docker

```bash
docker build -t cortex-ai-backend .
docker run -p 8000:8000 --env-file .env cortex-ai-backend
```

`Procfile` is included for Railway/Heroku-style deploys: `web: uvicorn main:app --host 0.0.0.0 --port $PORT`.

## Submission

Submit only your GitHub repository link via the Zyroo portal/form (no screenshots or separate files unless your mentor asks). Join the official Zyroo WhatsApp Channel and Community for task updates and deadlines — see the task PDFs.
