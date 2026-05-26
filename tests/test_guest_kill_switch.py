"""GUEST_MODE_ENABLED=false turns the app invite-only (Clerk required)."""
import pytest
from fastapi.testclient import TestClient

import main as main_module
from src import auth, config


@pytest.fixture
def guest_client():
    c = TestClient(main_module.app)
    c.headers.update({
        "X-Guest-Mode": "1",
        "X-Guest-Session-Id": auth.sign_guest_session_id("pytest-session"),
    })
    return c


def test_guest_allowed_when_enabled(guest_client, monkeypatch):
    monkeypatch.setattr(config, "GUEST_MODE_ENABLED", True)
    r = guest_client.get("/api/dashboard")
    assert r.status_code != 401


def test_guest_rejected_when_disabled(guest_client, monkeypatch):
    monkeypatch.setattr(config, "GUEST_MODE_ENABLED", False)
    r = guest_client.get("/api/dashboard")
    assert r.status_code == 401
