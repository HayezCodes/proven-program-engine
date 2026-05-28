"""
utils.py — Shared utilities for Proven Program Engine.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

from .safety import assert_safe_write

_LOG_DIR = Path(__file__).parent.parent / "logs"
_INITIALIZED: set[str] = set()


def get_logger(name: str) -> logging.Logger:
    """Return a named logger with console + daily file handlers (configured once)."""
    logger = logging.getLogger(name)
    if name in _INITIALIZED:
        return logger

    _INITIALIZED.add(name)
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)-35s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = _LOG_DIR / f"ppe_{datetime.now().strftime('%Y%m%d')}.log"
    assert_safe_write(log_file)
    try:
        fh = logging.FileHandler(log_file, encoding="utf-8")
    except OSError as exc:
        logger.warning(f"File logging disabled for {log_file}: {exc}")
    else:
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


def read_file_lines(path: Path) -> list[str] | None:
    """Read all lines from a file, trying UTF-8 then latin-1 encoding."""
    for encoding in ("utf-8", "latin-1"):
        try:
            return path.read_text(encoding=encoding).splitlines()
        except UnicodeDecodeError:
            continue
        except OSError as exc:
            get_logger(__name__).warning(f"Cannot read {path}: {exc}")
            return None
    get_logger(__name__).warning(f"Encoding failure — skipping: {path}")
    return None
