"""Clerk JWT verification for FastAPI."""
import base64
import os
from functools import lru_cache

import jwt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jwt import PyJWKClient

_bearer = HTTPBearer(auto_error=False)


@lru_cache(maxsize=1)
def _clerk_issuer() -> str:
    pk = os.environ.get("CLERK_PUBLISHABLE_KEY", "")
    if not pk.startswith("pk_"):
        raise RuntimeError("CLERK_PUBLISHABLE_KEY missing or malformed in backend env")
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
