"""Password hashing (bcrypt) and JWT — FYP demo; set CARDIOSENSE_SECRET_KEY in production."""

import os
from datetime import datetime, timedelta, timezone

import bcrypt
from jose import JWTError, jwt

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_DAYS = 7

DEMO_EMAIL = "demo@example.com"
DEMO_PASSWORD = "demo123"


def secret_key() -> str:
    return os.environ.get("CARDIOSENSE_SECRET_KEY", "fyp-dev-secret-change-me")


def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))


def create_access_token(*, user_id: int, email: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(days=ACCESS_TOKEN_EXPIRE_DAYS)
    payload = {"sub": str(user_id), "email": email, "exp": expire}
    return jwt.encode(payload, secret_key(), algorithm=ALGORITHM)


def decode_token(token: str) -> dict:
    return jwt.decode(token, secret_key(), algorithms=[ALGORITHM])


def parse_user_id_from_token(token: str) -> int | None:
    try:
        data = decode_token(token)
        sub = data.get("sub")
        if sub is None:
            return None
        return int(sub)
    except (JWTError, ValueError, TypeError):
        return None
