"""Loaded-terms ablation.

A single chokepoint that rewrites every LLM-bound text. Applied right before
`openai_client.chat.completions.create(messages=...)` in
`BasePlayer.get_completion` and `BasicLLMPlayer.get_basic_completion`.

Themes:

  - "original"  (default): no substitution. The simulator behaves exactly as
    before; the loaded terms ("Hitler", "Secret Hitler", "Fascist",
    "Liberal", "President", "Chancellor", ...) reach the LLM verbatim.

  - "neutral":  every loaded term is rewritten to a neutral counterpart so
    that the LLM never sees "Hitler" or "Secret Hitler". The rules of the
    game and all role/office labels are paraphrased to a neutral
    "Council / Reds vs Blues" framing. Game mechanics and vote tokens
    (Ja / Nein, JA / NEIN) are kept verbatim because the simulator's
    response parser depends on them.

Substitution strategy
---------------------

Word-boundary regex replacements applied in the listed order. Longer/more-
specific keys go first so that "Secret Hitler" is replaced before "Hitler",
"Fascists" before "Fascist", etc. Case is preserved for the four canonical
patterns (Title, lower, UPPER, Mixed) using the case-mapping helper below;
both lowercase and capitalised forms appear in the simulator prompts, so we
have to handle both.

After substitution the result is asserted to contain no forbidden tokens
("Hitler", "Secret Hitler", or their lowercase variants) when the theme is
"neutral" — assertion failures are logged loudly and (in strict mode) raise.
"""
from __future__ import annotations

import os
import re
from typing import Any

# ---------------------------------------------------------------------------
# Theme maps  (key -> replacement;  ORDER MATTERS — longest first)
# ---------------------------------------------------------------------------

# Each entry is (lower_key, lower_replacement). We expand to capitalised /
# uppercase / titlecased variants at build-time.
_NEUTRAL_BASE: list[tuple[str, str]] = [
    # multi-word / specific phrases first
    ("secret hitler", "the council"),
    # role names
    ("hitler", "saboteur"),
    ("fascists", "reds"),
    ("fascist", "red"),
    ("liberals", "blues"),
    ("liberal", "blue"),
    # office names
    ("ex-president", "ex-speaker"),
    ("ex president", "ex speaker"),
    ("president", "speaker"),
    ("chancellor", "deputy"),
]


def _build_patterns(base_map: list[tuple[str, str]]) -> list[tuple[re.Pattern, str]]:
    """Compile word-boundary regexes for every casing variant of each key.

    Order: original list order, but for each key we emit Title / lower /
    UPPER variants. We rely on Python regex \b for word boundaries so
    "republican" does not match "red"; "presidency" does not match
    "president" (the trailing \\b after "president" doesn't fire because
    "y" is a word char).
    """
    out: list[tuple[re.Pattern, str]] = []
    for lower_key, lower_repl in base_map:
        for fmt in ("title", "lower", "upper"):
            if fmt == "title":
                key = lower_key.title()
                repl = lower_repl.title()
            elif fmt == "lower":
                key = lower_key
                repl = lower_repl
            elif fmt == "upper":
                key = lower_key.upper()
                repl = lower_repl.upper()
            else:
                continue
            pattern = re.compile(r"\b" + re.escape(key) + r"\b")
            out.append((pattern, repl))
    return out


_NEUTRAL_PATTERNS = _build_patterns(_NEUTRAL_BASE)

# A few extra one-shot rewrites for phrasings the strict word-boundary regex
# does not catch (possessives, contractions). These are applied last.
_EXTRA_NEUTRAL = [
    (re.compile(r"\bHitler['’]s\b"), "Saboteur's"),
    (re.compile(r"\bhitler['’]s\b"), "saboteur's"),
    (re.compile(r"\bFascist['’]s\b"), "Red's"),
    (re.compile(r"\bfascist['’]s\b"), "red's"),
    (re.compile(r"\bLiberal['’]s\b"), "Blue's"),
    (re.compile(r"\bliberal['’]s\b"), "blue's"),
    (re.compile(r"\bPresident['’]s\b"), "Speaker's"),
    (re.compile(r"\bpresident['’]s\b"), "speaker's"),
    (re.compile(r"\bChancellor['’]s\b"), "Deputy's"),
    (re.compile(r"\bchancellor['’]s\b"), "deputy's"),
]


# ---------------------------------------------------------------------------
# Forbidden-token sanity check (applied AFTER substitution in "neutral" mode)
# ---------------------------------------------------------------------------

_FORBIDDEN_TOKENS = ("Hitler", "hitler", "HITLER",
                     "Secret Hitler", "secret hitler")

# strict-mode raises; soft mode (default) just logs the first 200 chars.
THEME_STRICT = os.environ.get("THEME_STRICT", "0") == "1"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

ACTIVE_THEME: str | None = None  # set by set_theme(...)


def set_theme(theme: str | None) -> None:
    """Switch the global active theme. Idempotent."""
    global ACTIVE_THEME
    if theme is None or theme.lower() in ("", "original", "default"):
        ACTIVE_THEME = None
    elif theme.lower() == "neutral":
        ACTIVE_THEME = "neutral"
    else:
        raise ValueError(f"Unknown theme: {theme!r} (expected 'neutral' or None)")


def apply(text: str) -> str:
    """Apply the active theme to *text*. Identity for the default theme."""
    if not text or ACTIVE_THEME is None:
        return text
    if ACTIVE_THEME == "neutral":
        for pat, repl in _NEUTRAL_PATTERNS:
            text = pat.sub(repl, text)
        for pat, repl in _EXTRA_NEUTRAL:
            text = pat.sub(repl, text)
    return text


def assert_clean(text: str, *, source: str = "") -> None:
    """When theme is 'neutral', complain loudly if any forbidden token leaks
    through. We do not block the API call by default — but in strict mode
    (THEME_STRICT=1) we raise. Either way, the leak is logged via the
    simulator logger so any contaminated prompt is visible during dev runs.
    """
    if ACTIVE_THEME != "neutral":
        return
    for tok in _FORBIDDEN_TOKENS:
        if tok in text:
            # Local import to avoid circular references at module load time.
            from .game_logging import logger
            preview = text[: max(0, text.find(tok) + 40)]
            msg = (f"theme.assert_clean: forbidden token {tok!r} leaked in "
                   f"{source or '<unspecified>'}; preview: …{preview!r}")
            if THEME_STRICT:
                raise RuntimeError(msg)
            logger.warning(msg)
            return


def apply_to_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Walk an OpenAI-style messages list and theme every text field.

    Supported shapes:
      [{"role": "system" | "user" | "assistant",
        "content": "..." | [{"type": "text", "text": "...", "cache_control": {...}}, ...]}]
    """
    if ACTIVE_THEME is None:
        return messages
    out: list[dict[str, Any]] = []
    for m in messages:
        new_m = dict(m)
        c = new_m.get("content")
        if isinstance(c, str):
            new_m["content"] = apply(c)
            assert_clean(new_m["content"], source=f"messages[{new_m.get('role')}]")
        elif isinstance(c, list):
            new_parts = []
            for part in c:
                if isinstance(part, dict) and "text" in part:
                    new_part = dict(part)
                    new_part["text"] = apply(part["text"])
                    assert_clean(new_part["text"],
                                 source=f"messages[{new_m.get('role')}].text")
                    new_parts.append(new_part)
                else:
                    new_parts.append(part)
            new_m["content"] = new_parts
        out.append(new_m)
    return out
