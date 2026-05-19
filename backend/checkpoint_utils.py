"""
Locate pretrained CardioSense weights: bundled files under backend/checkpoints/
or a one-time download from CARDIOSENSE_CHECKPOINT_URL.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"
CHECKPOINT_CANDIDATES = ("best_model_fold1.pt", "best_model.pt")
ENV_CHECKPOINT_URL = "CARDIOSENSE_CHECKPOINT_URL"


def resolve_pretrained_checkpoint() -> Path | None:
    """
    Return path to a pretrained checkpoint file if one is available after:
    - using an existing bundled file, or
    - downloading from CARDIOSENSE_CHECKPOINT_URL (saved as best_model_fold1.pt).

    Returns None if no file exists and no URL is configured or download fails.
    """
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    for name in CHECKPOINT_CANDIDATES:
        p = CHECKPOINT_DIR / name
        if p.is_file() and p.stat().st_size > 0:
            return p

    url = os.environ.get(ENV_CHECKPOINT_URL, "").strip()
    if not url:
        return None

    dest = CHECKPOINT_DIR / CHECKPOINT_CANDIDATES[0]
    try:
        _download_url(url, dest)
    except (OSError, URLError, ValueError) as e:
        logger.warning("Could not download checkpoint from %s: %s", url, e)
        return None

    if dest.is_file() and dest.stat().st_size > 0:
        return dest
    return None


def iter_checkpoint_paths():
    """Yield existing checkpoint paths (resolve/download first, then legacy locations)."""
    seen: set[Path] = set()

    primary = resolve_pretrained_checkpoint()
    if primary is not None and primary.is_file():
        key = primary.resolve()
        seen.add(key)
        yield primary

    backend_root = Path(__file__).resolve().parent
    extras = [
        backend_root / "checkpoints" / "best_model_fold1.pt",
        backend_root / "checkpoints" / "best_model.pt",
        Path("checkpoints") / "best_model_fold1.pt",
        Path("checkpoints") / "best_model.pt",
    ]
    for p in extras:
        try:
            key = p.resolve()
        except OSError:
            continue
        if key in seen:
            continue
        if p.is_file() and p.stat().st_size > 0:
            seen.add(key)
            yield p


def _download_url(url: str, dest: Path) -> None:
    req = Request(url, headers={"User-Agent": "ecg-ami-cds-cardiosense/1.0"})
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with urlopen(req, timeout=120) as resp:
            data = resp.read()
        if not data:
            raise ValueError("empty response body")
        tmp.write_bytes(data)
        tmp.replace(dest)
    finally:
        if tmp.is_file():
            try:
                tmp.unlink()
            except OSError:
                pass
    logger.info(
        "Downloaded pretrained checkpoint to %s (%d bytes)",
        dest,
        dest.stat().st_size,
    )
