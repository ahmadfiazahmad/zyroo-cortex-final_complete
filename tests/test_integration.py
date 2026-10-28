"""Integration tests — full RAG flow, quota blocking, workspace + admin flow."""
import io

from app.models import User
from tests.conftest import auth_headers, make_admin, make_user, register

DOC_TEXT = (
    "Cortex AI is a retrieval-augmented generation chatbot. "
    "It answers questions using documents uploaded by the user. "
    "The capital of France is Paris, and the Eiffel Tower is in Paris."
)


def upload_txt(client, headers, text=DOC_TEXT, filename="notes.txt", workspace_id=None):
    files = {"file": (filename, io.BytesIO(text.encode()), "text/plain")}
    data = {}
    if workspace_id:
        data["workspace_id"] = workspace_id
    return client.post("/api/documents", files=files, data=data, headers=headers)


def test_health_and_openapi(client):
    assert client.get("/api/health").json()["status"] == "ok"
    spec = client.get("/openapi.json").json()
    assert spec["info"]["title"] == "Cortex AI Backend API"
    assert "/api/auth/register" in spec["paths"]
    assert "/api/workspaces" in spec["paths"]
    assert "/api/admin/analytics" in spec["paths"]


def test_auth_register_login_me_and_duplicate(client, db):
    user = register(client)
    assert user["user"]["email"] == "ahmad@example.com"
    assert "password" not in str(user["user"])
    headers = auth_headers(client)
    me = client.get("/api/auth/me", headers=headers)
    assert me.status_code == 200
    assert me.json()["email"] == "ahmad@example.com"
    # Duplicate email is rejected.
    assert client.post(
        "/api/auth/register",
        json={"name": "Again", "email": "ahmad@example.com", "password": "password123"},
    ).status_code == 400
    # Wrong password: generic 401.
    assert client.post(
        "/api/auth/login", json={"email": "ahmad@example.com", "password": "wrongpass"}
    ).status_code == 401
    # No token: 401.
    assert client.get("/api/auth/me").status_code == 401


def test_full_rag_flow(client, db):
    """Register -> login -> upload -> search -> chat session -> grounded answer."""
    register(client)
    headers = auth_headers(client)

    upload = upload_txt(client, headers)
    assert upload.status_code == 200, upload.text
    doc = upload.json()
    assert doc["status"] == "ready"
    doc_id = doc["id"]

    docs = client.get("/api/documents", headers=headers).json()
    assert len(docs) == 1 and docs[0]["id"] == doc_id

    search = client.post(
        "/api/search", json={"query": "capital of France", "top_k": 3}, headers=headers
    )
    assert search.status_code == 200, search.text
    assert search.json()["results"], "search should find the uploaded chunk"
    assert "Paris" in search.json()["results"][0]["text"]

    session = client.post(
        "/api/chat/sessions",
        json={"title": "France chat", "document_ids": [doc_id]},
        headers=headers,
    )
    assert session.status_code == 200, session.text
    session_id = session.json()["id"]

    reply = client.post(
        f"/api/chat/sessions/{session_id}/messages",
        json={"content": "What is the capital of France?"},
        headers=headers,
    )
    assert reply.status_code == 200, reply.text
    body = reply.json()
    assert "Paris" in body["content"], body
    assert body["sources"], "assistant reply must cite sources"
    assert body["sources"][0]["document_id"] == doc_id

    history = client.get(f"/api/chat/sessions/{session_id}/messages", headers=headers)
    assert len(history.json()) == 2  # user + assistant

    # Chat message that is too long is rejected by validation (422).
    long_text = "x" * 2001
    assert client.post(
        f"/api/chat/sessions/{session_id}/messages",
        json={"content": long_text},
        headers=headers,
    ).status_code == 422


def test_chat_scoped_documents_authorization(client, db):
    """A user cannot scope a chat session to another user's documents
    (private or workspace they are not a member of) and read them via RAG."""
    from app.models import User

    register(client)  # Ahmad = user A
    headers_a = auth_headers(client)
    secret = "The secret code is BLUEBIRD-123 and only Ahmad should see it."
    upload = upload_txt(client, headers_a, text=secret)
    assert upload.status_code == 200, upload.text
    doc_id = upload.json()["id"]

    make_user(client, db, "other@example.com", name="Other")
    headers_b = auth_headers(client, "other@example.com")

    # B cannot scope a session to A's private document.
    scoped = client.post(
        "/api/chat/sessions",
        json={"title": "sneaky", "document_ids": [doc_id]},
        headers=headers_b,
    )
    assert scoped.status_code == 403
    assert "BLUEBIRD" not in scoped.text

    # Unknown document ID is a 404, not a leak.
    assert client.post(
        "/api/chat/sessions",
        json={"title": "ghost", "document_ids": ["00000000-0000-0000-0000-000000000000"]},
        headers=headers_b,
    ).status_code == 404

    # A can still scope to their own document and get a grounded answer.
    session_a = client.post(
        "/api/chat/sessions",
        json={"title": "mine", "document_ids": [doc_id]},
        headers=headers_a,
    )
    assert session_a.status_code == 200, session_a.text
    reply = client.post(
        f"/api/chat/sessions/{session_a.json()['id']}/messages",
        json={"content": "What is the secret code?"},
        headers=headers_a,
    )
    assert reply.status_code == 200, reply.text
    assert "BLUEBIRD" in reply.json()["content"]

    # Workspace document: B cannot scope to it until invited.
    user_a = db.query(User).filter(User.email == "ahmad@example.com").first()
    user_a.plan_tier = "team"
    db.commit()
    ws = client.post("/api/workspaces", json={"name": "Secret WS"}, headers=headers_a)
    assert ws.status_code == 200, ws.text
    ws_id = ws.json()["id"]
    ws_upload = upload_txt(client, headers_a, text=secret, workspace_id=ws_id)
    assert ws_upload.status_code == 200, ws_upload.text
    ws_doc_id = ws_upload.json()["id"]
    assert client.post(
        "/api/chat/sessions",
        json={"title": "ws sneaky", "document_ids": [ws_doc_id]},
        headers=headers_b,
    ).status_code == 403
    # After being invited, B can scope to the shared document.
    assert client.post(
        f"/api/workspaces/{ws_id}/invite",
        json={"email": "other@example.com"},
        headers=headers_a,
    ).status_code == 200
    assert client.post(
        "/api/chat/sessions",
        json={"title": "ws member", "document_ids": [ws_doc_id]},
        headers=headers_b,
    ).status_code == 200


def test_upload_rejects_bad_type_and_delete_cascades(client, db):
    register(client)
    headers = auth_headers(client)
    bad = client.post(
        "/api/documents",
        files={"file": ("evil.exe", io.BytesIO(b"nope"), "application/octet-stream")},
        headers=headers,
    )
    assert bad.status_code == 400
    upload = upload_txt(client, headers)
    doc_id = upload.json()["id"]
    assert client.delete(f"/api/documents/{doc_id}", headers=headers).status_code == 200
    assert client.get(f"/api/documents/{doc_id}", headers=headers).status_code == 404


def test_quota_blocking_flow(client, db):
    """Free-plan user at the document limit is blocked with 429."""
    register(client)
    headers = auth_headers(client)
    user = db.query(User).filter(User.email == "ahmad@example.com").first()
    user.documents_uploaded_count = 10  # Free plan max_documents
    db.commit()
    blocked = upload_txt(client, headers)
    assert blocked.status_code == 429
    assert "limit reached" in blocked.json()["detail"]

    # Message quota: at the limit, chat is blocked too.
    user.documents_uploaded_count = 0
    user.messages_sent_count = 100  # Free plan max_messages
    db.commit()
    session = client.post("/api/chat/sessions", json={"title": "t"}, headers=headers).json()
    reply = client.post(
        f"/api/chat/sessions/{session['id']}/messages",
        json={"content": "hello"},
        headers=headers,
    )
    assert reply.status_code == 429
    assert "limit reached" in reply.json()["detail"]


def test_workspace_and_admin_flow(client, db):
    """Create workspace -> invite -> member access -> non-member 403 ->
    admin analytics + suspend."""
    owner = make_user(client, db, "owner@example.com", name="Owner", plan="team")
    member = make_user(client, db, "member@example.com", name="Member")
    outsider = make_user(client, db, "outsider@example.com", name="Outsider")
    admin = make_admin(client, db)

    owner_h = auth_headers(client, "owner@example.com")
    member_h = auth_headers(client, "member@example.com")
    outsider_h = auth_headers(client, "outsider@example.com")
    admin_h = auth_headers(client, "admin@example.com")

    # Free user cannot create a workspace (402).
    assert client.post(
        "/api/workspaces", json={"name": "Nope"}, headers=member_h
    ).status_code == 402

    created = client.post("/api/workspaces", json={"name": "Team Alpha"}, headers=owner_h)
    assert created.status_code == 200, created.text
    ws_id = created.json()["id"]

    # Invite by email; duplicate invite is rejected.
    invited = client.post(
        f"/api/workspaces/{ws_id}/invite", json={"email": "member@example.com"}, headers=owner_h
    )
    assert invited.status_code == 200, invited.text
    assert client.post(
        f"/api/workspaces/{ws_id}/invite", json={"email": "member@example.com"}, headers=owner_h
    ).status_code == 400
    # Non-owner cannot invite.
    assert client.post(
        f"/api/workspaces/{ws_id}/invite", json={"email": "outsider@example.com"}, headers=member_h
    ).status_code == 403

    # Member uploads a shared doc; member + owner can see it, outsider cannot.
    upload = upload_txt(client, owner_h, workspace_id=ws_id)
    assert upload.status_code == 200, upload.text
    assert client.get(f"/api/workspaces/{ws_id}/documents", headers=member_h).status_code == 200
    assert len(client.get(f"/api/workspaces/{ws_id}/documents", headers=member_h).json()) == 1
    assert client.get(f"/api/workspaces/{ws_id}/documents", headers=outsider_h).status_code == 403
    assert client.get(f"/api/workspaces/{ws_id}/chat-sessions", headers=member_h).status_code == 200

    # Owner cannot remove themselves; owner can remove the member.
    member_id = member["id"]
    assert client.delete(
        f"/api/workspaces/{ws_id}/members/{owner['id']}", headers=owner_h
    ).status_code == 400
    assert client.delete(
        f"/api/workspaces/{ws_id}/members/{member_id}", headers=owner_h
    ).status_code == 200
    assert client.get(f"/api/workspaces/{ws_id}/documents", headers=member_h).status_code == 403

    # Admin endpoints: non-admin is forbidden, admin sees analytics.
    assert client.get("/api/admin/analytics", headers=owner_h).status_code == 403
    analytics = client.get("/api/admin/analytics", headers=admin_h)
    assert analytics.status_code == 200, analytics.text
    body = analytics.json()
    assert body["plan_distribution"]["Team"] >= 1
    assert "llm_usage_monitoring" in body
    assert "daily_active_users" in body

    users = client.get("/api/admin/users", headers=admin_h).json()
    assert len(users) >= 4
    assert "password_hash" not in str(users)

    # Suspend the outsider: their token is immediately rejected with 403.
    suspended = client.patch(
        f"/api/admin/users/{outsider['id']}/suspend",
        json={"is_suspended": True},
        headers=admin_h,
    )
    assert suspended.status_code == 200
    assert suspended.json()["is_suspended"] is True
    assert client.get("/api/auth/me", headers=outsider_h).status_code == 403
    # Reactivate.
    client.patch(
        f"/api/admin/users/{outsider['id']}/suspend",
        json={"is_suspended": False},
        headers=admin_h,
    )
    assert client.get("/api/auth/me", headers=outsider_h).status_code == 200


def test_stripe_webhook_rejects_bad_signature(client):
    r = client.post(
        "/api/webhooks/stripe",
        content=b'{"type":"checkout.session.completed"}',
        headers={"stripe-signature": "bad"},
    )
    # No Stripe keys in tests -> 503 (not configured); with keys it would be 400.
    assert r.status_code in (400, 503)


def test_cron_usage_reset_requires_secret(client, db, monkeypatch):
    """GET /api/cron/usage-reset: 503 without CRON_SECRET, 401 with a wrong
    one, 200 with the right one. This is the Vercel Cron replacement for
    the APScheduler job, which cannot run on serverless."""
    import app.config as config_module
    import app.routers.cron as cron_module

    assert client.get("/api/cron/usage-reset").status_code in (401, 503)
    monkeypatch.setattr(config_module.settings, "cron_secret", "test-cron-secret")
    assert client.get(
        "/api/cron/usage-reset", headers={"Authorization": "Bearer wrong"}
    ).status_code == 401
    ok = client.get(
        "/api/cron/usage-reset", headers={"Authorization": "Bearer test-cron-secret"}
    )
    assert ok.status_code == 200, ok.text
    assert ok.json()["reset_users"] == 0


def test_vercel_entry_point_exports_fastapi_app(client):
    """api/index.py (the Vercel entry point) must export the same FastAPI
    app as main.py, so Vercel's Python runtime can serve it."""
    import importlib.util
    import os

    spec = importlib.util.spec_from_file_location(
        "vercel_entry", os.path.join("api", "index.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from app.main import app as main_app

    assert module.app is main_app
    assert module.app.openapi()["info"]["title"] == "Cortex AI Backend API"


def test_security_headers_present(client):
    r = client.get("/api/health")
    assert r.headers["X-Content-Type-Options"] == "nosniff"
    assert r.headers["X-Frame-Options"] == "DENY"
    assert "Content-Security-Policy" in r.headers
