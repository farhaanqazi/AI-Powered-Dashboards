"""Config robustness — a present-but-blank env var must fall back to default.

Regression: `.env` shipped `JOB_SPOOL_DIR=` (blank), so `os.makedirs("")` raised
FileNotFoundError [WinError 3] on every upload. Blank means "unset".
"""
from __future__ import annotations

from src.config import _env_str


def test_env_str_unset_uses_default(monkeypatch):
    monkeypatch.delenv("X_SPOOL_TEST", raising=False)
    assert _env_str("X_SPOOL_TEST", "/tmp/default") == "/tmp/default"


def test_env_str_blank_uses_default(monkeypatch):
    monkeypatch.setenv("X_SPOOL_TEST", "")
    assert _env_str("X_SPOOL_TEST", "/tmp/default") == "/tmp/default"


def test_env_str_whitespace_uses_default(monkeypatch):
    monkeypatch.setenv("X_SPOOL_TEST", "   ")
    assert _env_str("X_SPOOL_TEST", "/tmp/default") == "/tmp/default"


def test_env_str_real_value_is_used(monkeypatch):
    monkeypatch.setenv("X_SPOOL_TEST", "/data/spool")
    assert _env_str("X_SPOOL_TEST", "/tmp/default") == "/data/spool"


def test_critical_path_configs_never_blank(monkeypatch):
    # Even with the offending blank value present, the resolved config is usable.
    monkeypatch.setenv("JOB_SPOOL_DIR", "")
    monkeypatch.setenv("CLEANED_DF_DURABLE_DIR", "")
    monkeypatch.setenv("DATABASE_URL", "")
    import importlib
    import src.config as cfg
    importlib.reload(cfg)
    try:
        assert cfg.JOB_SPOOL_DIR.strip() != ""
        assert cfg.CLEANED_DF_DURABLE_DIR.strip() != ""
        assert cfg.DATABASE_URL.strip() != ""
    finally:
        # Restore module to env-free defaults for any later test importing it.
        monkeypatch.undo()
        importlib.reload(cfg)
