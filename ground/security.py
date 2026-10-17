"""AARIS ground security — JWT users, static edge-device tokens, RBAC.

Small by design. Everything the ground station needs to authenticate and
authorize requests, nothing it doesn't:

* Human users  -> short-lived JWT access tokens (HS256, default 60 min)
                  issued by POST /auth/login (OAuth2 password form).
* Edge devices -> static bearer tokens from environment variables
                  (EDGE_<AIRCRAFT>_TOKEN, e.g. EDGE_AC017_TOKEN -> AC-017),
                  each bound to exactly one aircraft_id.
* RBAC         -> deny-by-default role checks on FastAPI dependencies.
                  Roles: edge_device, maintenance_officer, commander, admin.
* Audit        -> a pluggable sink (the ground station registers a writer for
                  the audit_logs table). Tokens/passwords/payloads are NEVER
                  logged — only actor id, role, action, resource, outcome.

Secrets come from environment variables only:
  AEROGUARD_SECRET_KEY   JWT signing key (required in real use; an insecure
                         dev default is used with a loud warning otherwise)
  EDGE_AC017_TOKEN       static device token for AC-017 (dev default provided)
  AEROGUARD_JWT_TTL_MIN  access-token lifetime in minutes (default 60)

DEMO USERS (explicitly requested for the prototype; passwords are demo-grade
and stored only as SHA-256 digests of the literal demo values):
  maintainer / maintainer123  -> maintenance_officer
  commander  / commander123   -> commander
  admin      / admin123       -> admin
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional

import jwt  # PyJWT
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

# ---------------------------------------------------------------------------
# Configuration (environment only — no production secrets in code)
# ---------------------------------------------------------------------------
_ALG = "HS256"

_dev_secret = "aeroguard-dev-secret-DO-NOT-USE-IN-PRODUCTION"
SECRET_KEY = os.environ.get("AEROGUARD_SECRET_KEY") or _dev_secret
if SECRET_KEY == _dev_secret:
    print("[security] WARNING: AEROGUARD_SECRET_KEY not set — using an INSECURE "
          "dev default. Set the env var (see .env.example) before any real use.")

JWT_TTL_MINUTES = int(os.environ.get("AEROGUARD_JWT_TTL_MIN", "60"))

ROLES = ("edge_device", "maintenance_officer", "commander", "admin")


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


# Demo users. Digests are computed from the literals at import so the raw
# passwords never sit in the dict; these accounts are demo-grade by design.
DEMO_USERS: dict[str, dict] = {
    "maintainer": {"pwd_sha256": _sha("maintainer123"), "role": "maintenance_officer"},
    "commander": {"pwd_sha256": _sha("commander123"), "role": "commander"},
    "admin": {"pwd_sha256": _sha("admin123"), "role": "admin"},
}


def dev_token_for(aircraft_id: str) -> str:
    """Deterministic LOCAL-DEMO device token for one aircraft.

    AC-017 -> edge-ac017-dev-token (the value this project has always used).
    Every roster aircraft gets the same shape so a multi-aircraft live feed works
    with zero setup on local ports. These are guessable by design: set
    EDGE_<AIRCRAFT>_TOKEN (e.g. EDGE_AC017_TOKEN) for anything beyond a demo."""
    tag = aircraft_id.replace("-", "").replace(" ", "").lower()
    return f"edge-{tag}-dev-token"


def _roster_ids() -> list[str]:
    """Aircraft ids from fleet_config.json (empty-safe: the demo keeps working
    if the file is missing, by falling back to AC-017 alone)."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "fleet_config.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            fleet = json.load(f).get("fleet", [])
        ids = [a["aircraft_id"] for a in fleet if a.get("aircraft_id")]
        return ids or ["AC-017"]
    except (OSError, ValueError, KeyError):
        return ["AC-017"]


def _env_slot(aircraft_id: str) -> str:
    """Env var name that overrides the device token for this aircraft."""
    return f"EDGE_{aircraft_id.replace('-', '').upper()}_TOKEN"


def _device_token_registry() -> dict[str, str]:
    """token -> aircraft_id.

    Each roster aircraft gets one device token: `EDGE_<ID>_TOKEN` from the
    environment when set, otherwise the deterministic local-demo default from
    dev_token_for(). A token is bound to exactly one aircraft either way."""
    reg: dict[str, str] = {}
    for craft in _roster_ids():
        tok = os.environ.get(_env_slot(craft), dev_token_for(craft))
        if tok:
            reg[tok] = craft
    for name, value in os.environ.items():
        if name.startswith("EDGE_") and name.endswith("_TOKEN") and value:
            craft = name[len("EDGE_"):-len("_TOKEN")]
            reg[value] = f"{craft[:-3]}-{craft[-3:]}" if len(craft) > 3 and craft[-3:].isdigit() else craft
    return reg


DEVICE_TOKENS = _device_token_registry()

# True when no EDGE_*_TOKEN env var was provided, i.e. every device is using a
# guessable dev default. The ground station prints a loud one-line warning at
# startup in that case (tokens themselves are never logged).
USING_DEV_DEVICE_TOKENS = not any(
    name.startswith("EDGE_") and name.endswith("_TOKEN") and value
    for name, value in os.environ.items()
)

# ---------------------------------------------------------------------------
# Identity + audit sink
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Identity:
    kind: str               # "device" | "user"
    actor_id: str           # aircraft tag for devices, username for users
    role: str               # one of ROLES
    aircraft_id: Optional[str] = None   # set only for edge_device tokens


_audit_sink: Optional[Callable[..., None]] = None


def register_audit_sink(fn: Callable[..., None]) -> None:
    """Ground station wires this to its audit_logs writer at import time."""
    global _audit_sink
    _audit_sink = fn


def audit(action: str, outcome: str, *, actor_id: str = "anonymous",
          actor_role: str = "anonymous", resource: str = "",
          aircraft_id: Optional[str] = None, details: Optional[dict] = None) -> None:
    """Emit one audit event through the registered sink. Never pass tokens,
    passwords, or raw report payloads in `details`."""
    if _audit_sink is not None:
        _audit_sink(actor_id=actor_id, actor_role=actor_role, action=action,
                    resource=resource, aircraft_id=aircraft_id,
                    outcome=outcome,
                    details_json=json.dumps(details) if details else None)


# ---------------------------------------------------------------------------
# JWT helpers
# ---------------------------------------------------------------------------


def create_access_token(username: str, role: str, ttl_minutes: Optional[int] = None) -> str:
    now = datetime.now(timezone.utc)
    ttl = ttl_minutes if ttl_minutes is not None else JWT_TTL_MINUTES
    payload = {
        "sub": username,
        "role": role,
        "typ": "access",
        "iat": now,
        "exp": now + timedelta(minutes=ttl),
        "jti": secrets.token_hex(8),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=_ALG)


def decode_access_token(token: str) -> dict:
    payload = jwt.decode(token, SECRET_KEY, algorithms=[_ALG],
                         options={"require": ["exp", "sub"]})
    if payload.get("typ") != "access":
        raise jwt.InvalidTokenError("not an access token")
    if payload.get("role") not in ROLES:
        raise jwt.InvalidTokenError("unknown role")
    return payload


def verify_password(username: str, password: str) -> Optional[str]:
    """Return the user's role on success, None on failure."""
    user = DEMO_USERS.get(username or "")
    if user is None:
        return None
    ok = hmac.compare_digest(user["pwd_sha256"], _sha(password or ""))
    return user["role"] if ok else None


# ---------------------------------------------------------------------------
# FastAPI dependencies (deny-by-default)
# ---------------------------------------------------------------------------

_bearer = HTTPBearer(auto_error=False)


def get_identity(
    creds: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> Identity:
    """Validate the Authorization: Bearer header.

    Order: static device tokens first, then JWT. 401 on anything missing or
    invalid (audited as failed_auth without any token material)."""
    if creds is None or creds.scheme.lower() != "bearer" or not creds.credentials:
        audit("auth", "denied", resource="bearer_header")
        raise HTTPException(status_code=401, detail="Not authenticated",
                            headers={"WWW-Authenticate": "Bearer"})
    token = creds.credentials

    # 1) static edge-device token?
    for tok, aircraft_id in DEVICE_TOKENS.items():
        if hmac.compare_digest(tok, token):
            return Identity(kind="device", actor_id=f"edge-{aircraft_id}",
                            role="edge_device", aircraft_id=aircraft_id)

    # 2) human JWT?
    try:
        payload = decode_access_token(token)
    except jwt.PyJWTError as exc:
        audit("auth", "denied", resource="token",
              details={"reason": "invalid_or_expired_jwt"})
        raise HTTPException(status_code=401,
                            detail=f"Invalid or expired token: {type(exc).__name__}",
                            headers={"WWW-Authenticate": "Bearer"})
    return Identity(kind="user", actor_id=payload["sub"], role=payload["role"])


def require_roles(*allowed: str) -> Callable[..., Identity]:
    """Dependency factory: identity must carry one of `allowed` roles,
    otherwise 403 (audited as permission_denied)."""
    assert allowed, "require_roles needs at least one role (deny-by-default)"

    def _dep(identity: Identity = Depends(get_identity)) -> Identity:
        if identity.role not in allowed:
            audit("access", "denied", actor_id=identity.actor_id,
                  actor_role=identity.role,
                  resource=",".join(allowed),
                  aircraft_id=identity.aircraft_id,
                  details={"needed_role": list(allowed)})
            raise HTTPException(status_code=403,
                                detail=f"Role '{identity.role}' is not permitted; "
                                       f"requires one of {list(allowed)}")
        return identity

    return _dep
