"""Registration OTP constants and code generation."""

import secrets
from datetime import timedelta

OTP_EXPIRE = timedelta(minutes=15)
MAX_OTP_VERIFY_ATTEMPTS = 5
OTP_DIGITS = 6


def generate_otp_code() -> str:
    return f"{secrets.randbelow(900000) + 100000:0{OTP_DIGITS}d}"
