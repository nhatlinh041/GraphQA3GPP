"""
Passwords and session tokens for the review tool. stdlib only.

Passwords are stored as salted PBKDF2-SHA256 ("pbkdf2$iters$salt$hash"). A
session token is base64(kind|subject|exp|sig) where sig is an HMAC keyed by
sha256(REVIEW_SECRET + per-account material): the stored password hash for a
rater, the configured password for an admin. So resetting a password, or
rotating REVIEW_SECRET, invalidates every outstanding token of that account
without any server-side session table.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import os
import secrets
import time

PBKDF2_ITERS = 200_000
RATER_TTL_S = 7 * 24 * 3600      # raters work across days; keep them signed in
ADMIN_TTL_S = 12 * 3600


def _secret() -> str:
    s = os.getenv("REVIEW_SECRET", "")
    if not s:
        raise RuntimeError("REVIEW_SECRET is not set")
    return s


def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), PBKDF2_ITERS).hex()
    return f"pbkdf2${PBKDF2_ITERS}${salt}${digest}"


def verify_password(password: str, stored: str | None) -> bool:
    if not stored or not password:
        return False
    try:
        _, iters, salt, digest = stored.split("$", 3)
        calc = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), int(iters)).hex()
    except Exception:  # noqa: BLE001 — malformed hash is simply "no match"
        return False
    return hmac.compare_digest(calc, digest)


def generate_password(n: int = 10) -> str:
    """Readable one-time password for the admin to hand over (no 0/O/1/l)."""
    alphabet = "abcdefghjkmnpqrstuvwxyzABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    return "".join(secrets.choice(alphabet) for _ in range(n))


def _sig(kind: str, subject: str, exp: int, material: str) -> str:
    key = hashlib.sha256((_secret() + "|" + material).encode()).digest()
    return hmac.new(key, f"{kind}|{subject}|{exp}".encode(), hashlib.sha256).hexdigest()


def make_token(kind: str, subject: str, material: str, ttl_s: int) -> tuple[str, int]:
    exp = int(time.time()) + ttl_s
    raw = f"{kind}|{subject}|{exp}|{_sig(kind, subject, exp, material)}"
    return base64.urlsafe_b64encode(raw.encode()).decode(), exp


def parse_token(token: str | None, kind: str) -> tuple[str, int, str] | None:
    """(subject, exp, sig) of a syntactically valid token of this kind, or None.
    Signature is NOT checked here — the caller looks up the account material."""
    if not token:
        return None
    try:
        k, subject, exp_s, sig = base64.urlsafe_b64decode(token.encode()).decode().split("|", 3)
        exp = int(exp_s)
    except Exception:  # noqa: BLE001
        return None
    if k != kind or exp < time.time():
        return None
    return subject, exp, sig


def check_sig(kind: str, subject: str, exp: int, sig: str, material: str) -> bool:
    return hmac.compare_digest(sig, _sig(kind, subject, exp, material))
