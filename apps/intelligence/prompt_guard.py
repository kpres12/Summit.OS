"""
prompt_guard.py — Shared prompt injection defences for Heli.OS Intelligence.

Imported by both brain.py and context_builder.py to avoid duplication.

Two levels of protection are applied:
  1. _safe_str()       — strips control chars + injection patterns from short strings
                         (entity names, metadata values, alert severity labels)
  2. sanitize_text()   — full sanitization for multi-line user/external input
                         (mission objectives, alert descriptions, additional context)

Structural isolation (in brain.py) is the primary defence. Sanitization is the
secondary, belt-and-suspenders layer.
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger("heli.intelligence.prompt_guard")

# ---------------------------------------------------------------------------
# Control-character stripper (shared)
# ---------------------------------------------------------------------------

_CTRL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# ---------------------------------------------------------------------------
# Injection pattern denylist
# Patterns that attempt to override system instructions or switch LLM persona.
# Applied to ALL untrusted text before it enters any prompt section.
# ---------------------------------------------------------------------------

_INJECTION_PATTERNS: list[re.Pattern] = [
    # Classic "ignore previous instructions" variants
    re.compile(r"ignore\s+(all\s+)?(above|previous|prior|instructions)", re.I),
    re.compile(r"disregard\s+(your\s+)?(previous|prior|system|instructions)", re.I),
    re.compile(r"forget\s+(all|your|the\s+above|everything)", re.I),
    # New instruction injection
    re.compile(r"new\s+(system\s+)?instructions?\s*:", re.I),
    re.compile(r"updated?\s+instructions?\s*:", re.I),
    # Role/persona switching
    re.compile(r"act\s+as\s+(a\s+)?(different|new|evil|unrestricted|rogue)", re.I),
    re.compile(r"pretend\s+(you\s+are|to\s+be)", re.I),
    re.compile(
        r"you\s+are\s+now\s+(a\s+)?(?!summit)", re.I
    ),  # "you are now X" (not Heli)
    # Structural tag injection — attempt to escape section boundaries
    re.compile(r"</?\s*(system|user|assistant|instruction|context|mission)\s*>", re.I),
    re.compile(
        r"\[\s*(system|user|assistant|instruction|end.?context|end.?mission)\s*\]", re.I
    ),
    # Known jailbreak keywords
    re.compile(r"\bDAN\s+mode\b", re.I),
    re.compile(r"\bjailbreak\b", re.I),
    re.compile(r"developer\s+mode\s*(enabled|on|activated)", re.I),
    re.compile(r"unrestricted\s+mode", re.I),
    # Prompt exfiltration attempts
    re.compile(
        r"(print|repeat|output|reveal|show|display)\s+(your\s+)?(system\s+)?prompt",
        re.I,
    ),
    re.compile(
        r"what\s+(are|were)\s+your\s+(initial|original|system)\s+instructions", re.I
    ),
]


def _safe_str(value: Any, max_len: int = 200) -> str:
    """
    Sanitize a short string field (entity name, label, metadata value) for
    embedding in LLM context.

    - Casts to str and truncates to max_len
    - Strips C0 control characters
    - Redacts injection patterns
    """
    s = str(value)[:max_len]
    s = _CTRL_CHARS.sub("", s)
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(s):
            logger.warning(
                "Prompt injection pattern in entity field — redacting: %r", s[:80]
            )
            s = pattern.sub("[REDACTED]", s)
    return s


def sanitize_text(text: str, max_len: int = 1000, label: str = "input") -> str:
    """
    Full sanitization for multi-line text (alert descriptions, additional context,
    mission objectives, or any other external/user-supplied block).

    - Truncates to max_len
    - Strips C0 control characters (keeps \\n and \\t)
    - Redacts injection patterns with [REDACTED]
    - Logs a warning on each match so security teams can monitor
    """
    if not text:
        return text

    text = text[:max_len]
    text = _CTRL_CHARS.sub("", text)

    for pattern in _INJECTION_PATTERNS:
        if pattern.search(text):
            logger.warning(
                "Prompt injection pattern detected in %s — redacting. Preview: %r",
                label,
                text[:120],
            )
            text = pattern.sub("[REDACTED]", text)

    return text
