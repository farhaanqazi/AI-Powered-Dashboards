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


def test_main_module_has_no_duplicate_route_names():
    """Detect duplicate route handler function names in main.py.

    Currently `serve_dynamic_assets` is defined twice. Both definitions are
    overwritten in the FastAPI routing table, but the dead one is a bug magnet.
    """
    import main as main_module
    seen = {}
    for route in main_module.app.routes:
        endpoint = getattr(route, "endpoint", None)
        if endpoint is None:
            continue
        name = endpoint.__qualname__
        path = getattr(route, "path", "?")
        seen.setdefault(name, []).append(path)
    duplicates = {n: paths for n, paths in seen.items() if len(paths) > 1}
    assert not duplicates, f"Duplicate endpoint functions: {duplicates}"


def test_main_imports_re_only_once():
    src = (
        __import__("pathlib").Path(__file__).resolve().parents[1] / "main.py"
    ).read_text(encoding="utf-8")
    occurrences = src.count("\nimport re\n") + (1 if src.startswith("import re\n") else 0)
    assert occurrences == 1, f"`import re` appears {occurrences} times in main.py"
