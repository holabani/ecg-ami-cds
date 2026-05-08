"""
Send registration OTP emails.

Environment (SMTP):
  CARDIOSENSE_EMAIL_MODE=smtp   (default smtp when SMTP_HOST set, else console)
  CARDIOSENSE_SMTP_HOST
  CARDIOSENSE_SMTP_PORT        default 587
  CARDIOSENSE_SMTP_USER
  CARDIOSENSE_SMTP_PASSWORD
  CARDIOSENSE_SMTP_FROM        From address

Or force console logs only:
  CARDIOSENSE_EMAIL_MODE=console

Testing (never enable in production):
  CARDIOSENSE_TEST_CAPTURE_OTP=1  — last OTP available via peek_last_otp_sent()
"""

from __future__ import annotations

import logging
import os
import ssl
import smtplib
from email.message import EmailMessage

logger = logging.getLogger(__name__)

_LAST_OTP_SENT: str | None = None


def peek_last_otp_sent() -> str | None:
    """Used only when CARDIOSENSE_TEST_CAPTURE_OTP=1 (pytest)."""
    return _LAST_OTP_SENT


def _smtp_configured() -> bool:
    return bool(os.environ.get("CARDIOSENSE_SMTP_HOST", "").strip())


def send_registration_otp(*, to_email: str, otp_plain: str) -> None:
    global _LAST_OTP_SENT
    subj = "CardioSense — your verification code"
    body = (
        f"Your CardioSense registration code is: {otp_plain}\n\n"
        "It expires in 15 minutes. If you did not request this, ignore this email."
    )

    if os.environ.get("CARDIOSENSE_TEST_CAPTURE_OTP", "").strip() == "1":
        _LAST_OTP_SENT = otp_plain
        logger.info("[test] OTP captured for %s", to_email)

    mode = os.environ.get("CARDIOSENSE_EMAIL_MODE", "").strip().lower()
    if mode == "console" or (not _smtp_configured()):
        logger.info("[email:console] to=%s subject=%s\n%s", to_email, subj, body)
        print(f"\n{'=' * 52}\n[CardioSense OTP] to={to_email}\ncode={otp_plain}\n{'=' * 52}\n")
        return

    host = os.environ["CARDIOSENSE_SMTP_HOST"].strip()
    port = int(os.environ.get("CARDIOSENSE_SMTP_PORT", "587"))
    user = os.environ["CARDIOSENSE_SMTP_USER"].strip()
    pw = os.environ["CARDIOSENSE_SMTP_PASSWORD"]
    sender = os.environ.get("CARDIOSENSE_SMTP_FROM", user).strip()

    msg = EmailMessage()
    msg["Subject"] = subj
    msg["From"] = sender
    msg["To"] = to_email
    msg.set_content(body)

    ctx = ssl.create_default_context()
    with smtplib.SMTP(host, port, timeout=30) as smtp:
        smtp.starttls(context=ctx)
        smtp.login(user, pw)
        smtp.send_message(msg)

    logger.info("Registration OTP emailed to %s", to_email)
