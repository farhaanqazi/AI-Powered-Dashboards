"""Top-level smoke tests."""


def test_root_serves_spa(client):
    """GET / returns the built index.html (or 404 if dist not built yet)."""
    response = client.get("/")
    assert response.status_code in (200, 404)
    if response.status_code == 200:
        assert "text/html" in response.headers.get("content-type", "")


def test_unknown_api_returns_404(client):
    response = client.get("/api/does-not-exist")
    assert response.status_code == 404
