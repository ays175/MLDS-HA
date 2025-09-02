#!/usr/bin/env python3
"""
Text sanitization utilities for Customer Care data
- Unicode NFKC normalization
- Optional ftfy fix for mojibake/cp1252 smart quotes (if available)
- Strip control characters
- Collapse excessive whitespace
"""
from typing import List, Dict, Any, Tuple
import re

try:
    import unicodedata
except Exception:  # pragma: no cover
    unicodedata = None  # type: ignore

try:
    import ftfy  # type: ignore
except Exception:  # pragma: no cover
    ftfy = None  # type: ignore


_CONTROL_CHARS_RE = re.compile(r"[\u0000-\u001F\u007F]")
_WS_RE = re.compile(r"\s+")


def _sanitize_text(value: Any) -> str:
    try:
        s = "" if value is None else str(value)
        if ftfy:
            s = ftfy.fix_text(s)
        if unicodedata:
            s = unicodedata.normalize("NFKC", s)
        # Strip control chars
        s = _CONTROL_CHARS_RE.sub(" ", s)
        # Collapse whitespace
        s = _WS_RE.sub(" ", s).strip()
        return s
    except Exception:
        return ""


def sanitize_text_columns(df, columns: List[str]) -> Tuple[Any, Dict[str, Any]]:
    """Sanitize specified text columns in-place; returns (df, stats)."""
    stats: Dict[str, Any] = {"columns_processed": [], "cells_modified": 0}
    for col in columns:
        if col in df.columns:
            before = df[col].astype(str).fillna("")
            after = before.apply(_sanitize_text)
            mods = (before != after).sum()
            df[col] = after
            stats["columns_processed"].append(col)
            stats["cells_modified"] += int(mods)
    return df, stats
