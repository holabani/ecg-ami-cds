"""
Registration OTP mail — console, SMTP (SES/Gmail/etc.), or Resend (HTTPS API).

Gmail SMTP (local dev typical):
  CARDIOSENSE_EMAIL_MODE=smtp
  CARDIOSENSE_SMTP_HOST=smtp.gmail.com
  CARDIOSENSE_SMTP_PORT=587
  CARDIOSENSE_SMTP_ENCRYPTION=starttls
  CARDIOSENSE_SMTP_USER=you@gmail.com
  CARDIOSENSE_SMTP_PASSWORD=<16-char App Password>
  CARDIOSENSE_SMTP_FROM=CardioSense <you@gmail.com>
  Match USER / FROM mailbox to reduce spam flags.

Resend:
  CARDIOSENSE_EMAIL_MODE=resend
  CARDIOSENSE_RESEND_API_KEY=re_xxxxx
  CARDIOSENSE_EMAIL_FROM=CardioSense <onboarding@resend.dev>

  If HTTPS fails with SSL verify errors (corporate MITM / strict Python 3.14+ OpenSSL):
  CARDIOSENSE_RESEND_VERIFY_SSL=0   (**local dev only** — disables TLS verification)

SMTP / console — see CARDIOSENSE_EMAIL_MODE and CARDIOSENSE_SMTP_* env vars below.

Testing (never in production): CARDIOSENSE_TEST_CAPTURE_OTP=1
"""

from __future__ import annotations

import html
import logging
import os
import ssl
import smtplib
from email.message import EmailMessage

import certifi
import httpx

logger = logging.getLogger(__name__)

_LAST_OTP_SENT: str | None = None


def peek_last_otp_sent() -> str | None:
    """Used only when CARDIOSENSE_TEST_CAPTURE_OTP=1 (pytest)."""
    return _LAST_OTP_SENT


def _smtp_configured() -> bool:
    return bool(os.environ.get("CARDIOSENSE_SMTP_HOST", "").strip())


def _resend_configured() -> bool:
    return bool(os.environ.get("CARDIOSENSE_RESEND_API_KEY", "").strip())


def effective_email_mode() -> str:
    """
    console | smtp | resend.
    Honors CARDIOSENSE_EMAIL_MODE if set to one of these; otherwise auto-selects.
    """
    raw = os.environ.get("CARDIOSENSE_EMAIL_MODE", "").strip().lower()
    if raw in ("console", "smtp", "resend"):
        return raw
    if _resend_configured():
        return "resend"
    if _smtp_configured():
        return "smtp"
    return "console"


def _smtp_login(smtp: smtplib.SMTP, user: str, password: str) -> None:
    if not user.strip():
        return
    smtp.login(user, password)


def _resend_tls_verify() -> bool | str:
    """
    Use certifi's bundle by default (more reliable than some macOS/Homebrew Python stores).
    CARDIOSENSE_RESEND_VERIFY_SSL=0|false disables verification (dev / broken proxy only).
    """
    off = os.environ.get("CARDIOSENSE_RESEND_VERIFY_SSL", "").strip().lower()
    if off in ("0", "false", "no"):
        logger.warning(
            "CARDIOSENSE_RESEND_VERIFY_SSL disables TLS verification for Resend — local dev only"
        )
        return False
    return certifi.where()


def email_delivery_user_hint(exc: BaseException) -> str:
    """
    Non-secret clue for operators (503 detail). Prefer server logs for full tracebacks.
    """
    if isinstance(exc, ValueError) and os.environ.get("CARDIOSENSE_TEST_CAPTURE_OTP", "").strip() != "1":
        return str(exc)

    if isinstance(exc, httpx.ConnectError):
        cs = str(exc)
        cu = cs.upper()
        if "CERTIFICATE" in cu or "SSL" in cu or "TLS" in cu:
            return (
                "HTTPS to Resend failed certificate checks — set CARDIOSENSE_RESEND_VERIFY_SSL=0 in "
                "backend/.env for local dev only, or trust your corporate root CA / try another network."
            )
        return "Could not reach Resend over HTTPS — check network and firewall."

    if isinstance(exc, ssl.SSLCertVerificationError):
        return (
            "TLS certificate verification failed calling Resend — common with corporate SSL inspection "
            "or strict OpenSSL on Python 3.14+. Try another network/VPN, install your org root CA, "
            "or for local dev only set CARDIOSENSE_RESEND_VERIFY_SSL=0 (never in production)."
        )

    s = str(exc)
    if "CERTIFICATE_VERIFY_FAILED" in s or "certificate verify failed" in s.lower():
        return (
            "HTTPS to Resend failed certificate checks — see backend logs. "
            "For local dev behind MITM proxies you may set CARDIOSENSE_RESEND_VERIFY_SSL=0 "
            "(insecure; never in production)."
        )

    if isinstance(exc, httpx.HTTPStatusError):
        r = exc.response
        code = r.status_code
        res_name = ""
        res_msg = ""
        try:
            payload = r.json()
            if isinstance(payload, dict):
                res_msg = str(payload.get("message", "") or "").strip()
                res_name = str(payload.get("name", "") or "").strip()
        except Exception:
            pass
        mlow = res_msg.lower()

        # Resend uses 403/422 with validation_error for invalid "from" domains (not the API key).
        if (
            res_name == "validation_error"
            or "domain" in mlow
            or "verified" in mlow and "domain" in mlow
        ):
            return (
                "Resend rejected the sender address (CARDIOSENSE_EMAIL_FROM) — personal Gmail/other "
                "addresses are not accepted as From. For local testing use "
                '`CardioSense <onboarding@resend.dev>`; for production add and verify a domain at '
                "https://resend.com/domains."
            )

        if code == 401 or (
            code == 403
            and res_msg
            and ("api key" in mlow or "unauthorized" in mlow or "invalid" in mlow and "key" in mlow)
        ):
            return (
                "Resend rejected the API key — regenerate CARDIOSENSE_RESEND_API_KEY at "
                "https://resend.com/api-keys."
            )

        if code == 422:
            return (
                "Resend did not accept the payload — verify CARDIOSENSE_EMAIL_FROM is "
                "a sender allowed in your Resend account."
            )
        return f"Resend HTTP error {code}; see backend logs."

    if isinstance(exc, smtplib.SMTPAuthenticationError):
        host = os.environ.get("CARDIOSENSE_SMTP_HOST", "").strip().lower()
        if "gmail.com" in host:
            return (
                "Gmail SMTP login failed — confirm 2FA is on, create a 16-character App Password "
                "(Google Account → Security → App passwords), "
                "and set CARDIOSENSE_SMTP_USER to the full Gmail address and CARDIOSENSE_SMTP_PASSWORD to that App Password "
                "(not your normal Gmail password)."
            )
        return (
            "SMTP login failed — verify CARDIOSENSE_SMTP_USER and CARDIOSENSE_SMTP_PASSWORD. "
            "For SES use SMTP IAM credentials + correct region; avoid the console/sign-in password."
        )
    if isinstance(exc, (smtplib.SMTPRecipientsRefused, smtplib.SMTPDataError)):
        return (
            "SMTP refused this recipient — check CARDIOSENSE_SMTP_FROM and provider policies "
            "(e.g. SES sandbox allow-list). Inspect backend logs for the SMTP diagnostic code."
        )
    if isinstance(exc, ssl.SSLError):
        return (
            "SMTP TLS failed — try port 587 with CARDIOSENSE_SMTP_ENCRYPTION=starttls "
            "(or port 465 with ssl)."
        )
    if isinstance(exc, (TimeoutError, ConnectionError, OSError)) and not isinstance(
        exc, ssl.SSLError
    ):
        return "Network error contacting the mail provider; check outbound connectivity."
    if isinstance(exc, smtplib.SMTPException):
        return (
            "SMTP rejected the message — verify CARDIOSENSE_SMTP_FROM matches the authenticated mailbox "
            "and see backend logs for the SMTP reply."
        )
    return f"{type(exc).__name__} — check backend traceback."


# Backwards-compat for older callers
def smtp_user_visible_hint(exc: BaseException) -> str:
    return email_delivery_user_hint(exc)


def _resend_sender() -> str:
    return (
        os.environ.get("CARDIOSENSE_EMAIL_FROM", "").strip()
        or os.environ.get("CARDIOSENSE_RESEND_FROM", "").strip()
        or os.environ.get("CARDIOSENSE_SMTP_FROM", "").strip()
    )


def _otp_subject_plain_and_html(otp_plain: str) -> tuple[str, str, str]:
    """Subject, text/plain body, HTML body for the registration OTP."""
    subject = "CardioSense — your verification code"
    safe = html.escape(str(otp_plain), quote=True)
    text = (
        f"Your CardioSense registration code is: {otp_plain}\n\n"
        "It expires in 15 minutes. If you did not request this, ignore this email."
    )
    html_body = (
        "<html><body><p>Your CardioSense registration code is: "
        f'<strong style="letter-spacing:0.15em">{safe}</strong></p>'
        "<p>It expires in 15 minutes. If you did not request this, ignore this email.</p>"
        "</body></html>"
    )
    return subject, text, html_body


def _build_otp_email_message(*, subject: str, text: str, html_body: str, sender: str, to_email: str) -> EmailMessage:
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = to_email
    msg.set_content(text)
    msg.add_alternative(html_body, subtype="html")
    return msg


def _send_via_smtp(
    *,
    host: str,
    port: int,
    enc: str,
    user: str,
    password: str,
    msg: EmailMessage,
    to_email: str,
) -> None:
    """
    Sends via SMTP with TLS appropriate for Gmail (STARTTLS + EHLO handshake).
    SSL context prefers certifi (helps some macOS/Python OpenSSL setups).
    """
    ctx = ssl.create_default_context(cafile=certifi.where())
    frm = msg.get("From", "")
    logger.info(
        "SMTP: OTP send host=%r port=%s encryption=%s to=%s from=%s login_user_present=%s",
        host,
        port,
        enc,
        to_email,
        frm,
        bool(user),
    )
    try:
        if enc == "ssl":
            with smtplib.SMTP_SSL(host, port, timeout=45, context=ctx) as smtp:
                _smtp_login(smtp, user, password)
                refuse = smtp.send_message(msg)
                if refuse:
                    logger.warning("SMTP: partial refusal %s", refuse)
                return

        with smtplib.SMTP(host, port, timeout=45) as smtp:
            smtp.ehlo()
            if enc == "starttls":
                smtp.starttls(context=ctx)
                smtp.ehlo()
            elif enc == "none":
                logger.warning("SMTP encryption=none — unsafe; use only for local lab tests.")
            _smtp_login(smtp, user, password)
            refuse = smtp.send_message(msg)
            if refuse:
                logger.warning("SMTP: partial refusal %s", refuse)

    except smtplib.SMTPAuthenticationError as e:
        smtp_detail = getattr(e, "smtp_code", None)
        err_part = getattr(e, "smtp_error", b"") or b""
        err_txt = (
            err_part.decode(errors="replace")[:240]
            if isinstance(err_part, (bytes, bytearray))
            else str(err_part)[:240]
        )
        logger.error(
            "SMTP AUTH failed host=%s port=%s encryption=%s smtp_reply=%s detail=%r "
            "(Gmail? CARDIOSENSE_SMTP_USER=full Gmail + App Password)",
            host,
            port,
            enc,
            smtp_detail,
            err_txt,
        )
        raise

    except smtplib.SMTPException as e:
        logger.error("SMTP protocol error host=%s port=%s: %s", host, port, e)
        raise

    except (TimeoutError, ConnectionError, OSError) as e:
        logger.error("SMTP connectivity error host=%s port=%s: %s", host, port, e)
        raise


def _send_via_resend(
    *,
    api_key: str,
    from_email: str,
    to_email: str,
    subject: str,
    text: str,
    html_body: str,
) -> None:
    verify = _resend_tls_verify()
    payload = {
        "from": from_email,
        "to": [to_email],
        "subject": subject,
        "text": text,
        "html": html_body,
    }
    with httpx.Client(timeout=30.0, verify=verify) as client:
        r = client.post(
            "https://api.resend.com/emails",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
    if r.status_code >= 400:
        logger.warning("Resend error body: %s", (r.text or "")[:600])
    r.raise_for_status()


def send_registration_otp(*, to_email: str, otp_plain: str) -> None:
    global _LAST_OTP_SENT
    subj, body_plain, body_html = _otp_subject_plain_and_html(otp_plain)

    if os.environ.get("CARDIOSENSE_TEST_CAPTURE_OTP", "").strip() == "1":
        _LAST_OTP_SENT = otp_plain
        logger.info("[test] OTP captured for %s", to_email)

    testing = os.environ.get("CARDIOSENSE_TEST_CAPTURE_OTP", "").strip() == "1"
    mode = effective_email_mode()

    if testing or mode == "console":
        logger.info("[email:console] to=%s subject=%s\n%s", to_email, subj, body_plain)
        print(f"\n{'=' * 52}\n[CardioSense OTP] to={to_email}\ncode={otp_plain}\n{'=' * 52}\n")
        return

    if mode == "resend":
        api_key = os.environ.get("CARDIOSENSE_RESEND_API_KEY", "").strip()
        if not api_key:
            raise ValueError("CARDIOSENSE_EMAIL_MODE=resend but CARDIOSENSE_RESEND_API_KEY is unset.")
        from_email = _resend_sender()
        if not from_email:
            raise ValueError(
                "Set CARDIOSENSE_EMAIL_FROM (e.g. 'CardioSense <onboarding@resend.dev>') "
                "to a verified Resend sender."
            )
        _send_via_resend(
            api_key=api_key,
            from_email=from_email,
            to_email=to_email,
            subject=subj,
            text=body_plain,
            html_body=body_html,
        )
        logger.info("Registration OTP sent via Resend to %s", to_email)
        return

    if mode == "smtp":
        if not _smtp_configured():
            raise ValueError("CARDIOSENSE_EMAIL_MODE=smtp but CARDIOSENSE_SMTP_HOST is not set.")

    host = os.environ["CARDIOSENSE_SMTP_HOST"].strip()
    port = int(os.environ.get("CARDIOSENSE_SMTP_PORT", "587"))
    user = os.environ.get("CARDIOSENSE_SMTP_USER", "").strip()
    pw = os.environ.get("CARDIOSENSE_SMTP_PASSWORD", "")
    sender = os.environ.get("CARDIOSENSE_SMTP_FROM", user or "noreply@localhost").strip()

    if mode == "smtp" and user and not pw.strip():
        logger.warning("SMTP: CARDIOSENSE_SMTP_PASSWORD is blank — LOGIN will usually fail.")

    enc = os.environ.get("CARDIOSENSE_SMTP_ENCRYPTION", "starttls").strip().lower()
    if enc not in ("starttls", "ssl", "none"):
        enc = "starttls"

    msg = _build_otp_email_message(
        subject=subj, text=body_plain, html_body=body_html, sender=sender, to_email=to_email
    )

    try:
        _send_via_smtp(
            host=host,
            port=port,
            enc=enc,
            user=user,
            password=pw,
            msg=msg,
            to_email=to_email,
        )
    except Exception:
        logger.exception(
            "SMTP OTP send failed (host=%r port=%s encryption=%s to=%s)",
            host,
            port,
            enc,
            to_email,
        )
        raise

    logger.info("Registration OTP emailed via SMTP to %s", to_email)
