"""Clerk JWT verification for FastAPI."""
from __future__ import annotations  # PEP 604 (str | None) is eval'd at def-time

import base64
import hashlib
import hmac
import os
import secrets
from functools import lru_cache

import jwt
from fastapi import Depends, HTTPException, Request, Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jwt import PyJWKClient

from src import config

_bearer = HTTPBearer(auto_error=False)

GUEST_HEADER = "X-Guest-Mode"
GUEST_SID_HEADER = "X-Guest-Session-Id"


def _sign_guest_sid(sid: str) -> str:
    """Return ``<sid>.<hmac>`` so the server can later prove it minted ``sid``."""
    sig = hmac.new(
        config.GUEST_SESSION_SECRET.encode(), sid.encode(), hashlib.sha256
    ).hexdigest()
    return f"{sid}.{sig}"


def sign_guest_session_id(raw: str) -> str:
    """Public: turn a raw guest id into the server-signed token the client
    must present (used by the frontend handshake and by tests)."""
    return _sign_guest_sid(raw)


def _verified_guest_sid(token: str | None) -> str | None:
    """Return the sid only if ``token`` carries a valid server signature.

    A guest cannot forge another guest's id (and thus read their history)
    without the server secret — unsigned/foreign values are rejected here and
    the caller mints a fresh, empty session instead (fail-closed)."""
    if not token or "." not in token:
        return None
    sid, _, sig = token.rpartition(".")
    if not sid or not sig:
        return None
    expected = hmac.new(
        config.GUEST_SESSION_SECRET.encode(), sid.encode(), hashlib.sha256
    ).hexdigest()
    return sid if hmac.compare_digest(expected, sig) else None


@lru_cache(maxsize=1)
def _clerk_issuer() -> str:
    pk = (
        os.environ.get("CLERK_PUBLISHABLE_KEY")
        or os.environ.get("VITE_CLERK_PUBLISHABLE_KEY")
        or ""
    )
    if not pk.startswith("pk_"):
        raise RuntimeError(
            "Clerk publishable key missing or malformed in backend env "
            "(checked CLERK_PUBLISHABLE_KEY and VITE_CLERK_PUBLISHABLE_KEY)"
        )
    encoded = pk.split("_", 2)[2].rstrip("$")
    padding = "=" * (-len(encoded) % 4)
    frontend_api = base64.urlsafe_b64decode(encoded + padding).decode("utf-8").rstrip("$")
    return f"https://{frontend_api}"


@lru_cache(maxsize=1)
def _jwks_client() -> PyJWKClient:
    return PyJWKClient(
        f"{_clerk_issuer()}/.well-known/jwks.json",
        cache_jwk_set=True,
        lifespan=3600,
    )


def require_clerk_user(creds: HTTPAuthorizationCredentials = Depends(_bearer)) -> dict:
    if creds is None or creds.scheme.lower() != "bearer":
        raise HTTPException(status_code=401, detail="Missing bearer token")
    try:
        signing_key = _jwks_client().get_signing_key_from_jwt(creds.credentials).key
        claims = jwt.decode(
            creds.credentials,
            signing_key,
            algorithms=["RS256"],
            issuer=_clerk_issuer(),
            options={"require": ["exp", "iat", "sub"]},
        )
    except jwt.PyJWTError as exc:
        raise HTTPException(status_code=401, detail=f"Invalid token: {exc}")
    return claims


def require_admin_user(claims: dict = Depends(require_clerk_user)) -> dict:
    """S0.5: a valid Clerk token is necessary but not sufficient — the user's
    id must be in ``config.ADMIN_USER_IDS``. Empty allow-list => nobody."""
    sub = claims.get("sub")
    if not sub or sub not in config.ADMIN_USER_IDS:
        raise HTTPException(status_code=403, detail="Admin privileges required.")
    return claims


def allow_clerk_or_guest(
    request: Request,
    response: Response,
    creds: HTTPAuthorizationCredentials = Depends(_bearer),
) -> dict:
    # Invite-only kill-switch: when guest mode is disabled, ignore the guest
    # header entirely so a valid Clerk token becomes mandatory below.
    guest_requested = (
        config.GUEST_MODE_ENABLED
        and request.headers.get(GUEST_HEADER, "").strip() == "1"
    )

    # A valid Clerk identity always wins over the guest header: a signed-in
    # user can never be silently demoted to a guest, and the guest header can
    # never be used to skip auth on an endpoint that has a real token present.
    if creds is not None:
        try:
            claims = require_clerk_user(creds)
            return {**claims, "session_key": f"user:{claims.get('sub', 'unknown')}"}
        except HTTPException:
            if not guest_requested:
                raise

    if guest_requested:
        sid = _verified_guest_sid(request.headers.get(GUEST_SID_HEADER))
        if sid is None:
            # No server-signed id presented — mint a fresh one. A forged or
            # foreign id can never collide with another guest's history; the
            # signed token is handed back for the client to store and resend.
            sid = secrets.token_urlsafe(18)
            response.headers[GUEST_SID_HEADER] = _sign_guest_sid(sid)
        return {"sub": "guest", "guest": True, "session_key": f"guest:{sid}"}

    raise HTTPException(status_code=401, detail="Authentication required")


def guest_sid_header(user: dict) -> dict:
    """Re-advertise the signed guest session id as a response header.

    ``allow_clerk_or_guest`` sets this header on the temporal ``Response``, but
    FastAPI drops dependency-set headers when an endpoint returns a ``Response``
    object (e.g. ``JSONResponse``) directly — so the guest never learns the id
    it must resend, and every request mints a fresh session (the "Job not
    found" failure on the async upload path). Endpoints that return a Response
    directly merge this dict into the response headers. No-op for Clerk users.
    """
    if not user.get("guest"):
        return {}
    session_key = user.get("session_key", "")
    raw = session_key.split(":", 1)[1] if ":" in session_key else session_key
    if not raw:
        return {}
    return {GUEST_SID_HEADER: _sign_guest_sid(raw)}


def owner_key(user: dict) -> str:
    """Phase 10 S10.4 — history ownership scope.

    Org-scoped when a Clerk org token is present (members of an org share
    history — multi-tenancy), else the Clerk user, else the guest session.
    """
    org = user.get("org_id") or user.get("orgId")
    if org:
        return f"org:{org}"
    if user.get("guest"):
        return user.get("session_key", "guest:anonymous")
    sub = user.get("sub", "unknown")
    return f"user:{sub}"
