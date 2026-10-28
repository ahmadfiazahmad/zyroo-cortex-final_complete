"""Test setup: isolated SQLite DB, no live Groq/Stripe calls.

Environment is set BEFORE the app is imported, so the app's engine
points at the test database.
"""
import os
import sys

os.environ["DATABASE_URL"] = "sqlite:///./test_cortex.db"
os.environ["JWT_SECRET"] = "test-secret-for-pytest-only-long-enough-123456"
os.environ["GROQ_API_KEY"] = ""
os.environ["STRIPE_SECRET_KEY"] = ""
os.environ["STRIPE_WEBHOOK_SECRET"] = ""
os.environ["ADMIN_EMAILS"] = ""

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from fastapi.testclient import TestClient

from app.database import Base, SessionLocal, engine
from app.main import app
from app.models import User


@pytest.fixture(scope="session", autouse=True)
def _clean_db_file():
    if os.path.exists("test_cortex.db"):
        os.remove("test_cortex.db")
    yield
    if os.path.exists("test_cortex.db"):
        os.remove("test_cortex.db")


@pytest.fixture()
def db():
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    session = SessionLocal()
    yield session
    session.close()
    Base.metadata.drop_all(bind=engine)


@pytest.fixture()
def client(db):
    # Reset the in-memory rate limiter between tests.
    for mw in app.user_middleware:
        pass
    with TestClient(app) as c:
        yield c


def register(client, name="Ahmad", email="ahmad@example.com", password="password123"):
    r = client.post(
        "/api/auth/register", json={"name": name, "email": email, "password": password}
    )
    assert r.status_code == 200, r.text
    return r.json()


def auth_headers(client, email="ahmad@example.com", password="password123"):
    r = client.post("/api/auth/login", json={"email": email, "password": password})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


def make_user(client, db, email, name="User", plan="free", role="user"):
    """Register a user, then set plan/role directly in the DB (test-only)."""
    data = register(client, name=name, email=email)
    user = db.query(User).filter(User.id == data["user"]["id"]).first()
    user.plan_tier = plan
    user.role = role
    db.commit()
    return data["user"]


def make_admin(client, db, email="admin@example.com"):
    return make_user(client, db, email, name="Admin", role="admin")
