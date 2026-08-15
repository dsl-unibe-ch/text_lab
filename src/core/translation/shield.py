"""
Syntax / structure shielding for machine translation.

Neural MT models frequently mangle Markdown links, LaTeX equations, inline
code, HTML tags, URLs, and printf-style placeholders. This module replaces
those spans with short opaque sentinel tokens *before* translation and
restores them *after* translation, so the natural-language text is the only
thing actually sent to the model.

The sentinel format is ``[TL_<N>]``: pure ASCII, brackets are almost always
preserved atomically by SentencePiece / BPE tokenizers used by NLLB, M2M,
MarianMT and by chat LLMs.

Usage
-----

    from core.translation.shield import shield, unshield

    masked, table = shield(source_text)
    translated_masked = my_translate(masked)
    restored = unshield(translated_masked, table)

The module is deliberately dependency-free and Streamlit-free.
"""

from __future__ import annotations

import re
from typing import Dict, List, Mapping, Optional, Tuple

# Sentinel: [TL_0], [TL_1], ... — using control characters to avoid collision
_SENTINEL_FMT = "\x02TL_{i}\x03"
_SENTINEL_RE = re.compile(r"\x02TL_(\d+)\x03")

# Distinct sentinel for glossary/term-lock so it doesn't collide with the
# generic shield table. Applied *before* :func:`shield` and restored *after*
# :func:`unshield`.
_GLOSSARY_FMT = "\x02GL_{i}\x03"
_GLOSSARY_RE = re.compile(r"\x02GL_(\d+)\x03")

# Order matters. Higher-priority (longer / structural) patterns first.
# Each pattern is applied globally to the text before the next one runs.
_PATTERNS: List[Tuple[str, re.Pattern]] = [
    # Fenced code blocks (```lang ... ``` or ~~~ ... ~~~) — whole thing kept.
    ("fenced_code", re.compile(r"(?ms)^[ \t]*(?:```|~~~)[^\n]*\n.*?^[ \t]*(?:```|~~~)[ \t]*$")),
    # Display math $$...$$ and \[...\]
    ("math_display", re.compile(r"\$\$[\s\S]+?\$\$")),
    ("math_display_bracket", re.compile(r"\\\[[\s\S]+?\\\]")),
    # Inline math $...$ (single line, no blank $) and \(...\)
    ("math_inline", re.compile(r"(?<!\\)\$(?!\s)[^\$\n]+?(?<!\s)\$")),
    ("math_inline_paren", re.compile(r"\\\([\s\S]+?\\\)")),
    # Inline code `...`
    ("inline_code", re.compile(r"`[^`\n]+`")),
    # Markdown image ![alt](url)  — kept whole (alt text is usually alt for a
    # figure caption and the caption will still be translated as prose).
    ("md_image", re.compile(r"!\[[^\]]*\]\([^)]+\)")),
    # HTML/XML tag
    ("html_tag", re.compile(r"</?[A-Za-z][^>]*>")),
    # URLs and file paths
    ("url", re.compile(r"https?://\S+|ftp://\S+|www\.\S+")),
    ("path", re.compile(r"(?<![\w/])(?:/[A-Za-z0-9_.\-]+){2,}/?")),
    # Emails
    ("email", re.compile(r"[\w.+\-]+@[\w\-]+(?:\.[\w\-]+)+")),
    # Printf / f-string / format placeholders
    ("placeholder", re.compile(
        r"\{[A-Za-z_][A-Za-z0-9_.]*\}"          # {name}
        r"|%\([^)]+\)[sdif]"                    # %(name)s
        r"|%[0-9.\-+ #]*[sdifxXoeEgG]"          # %s, %5.2f, ...
    )),
]


def _shield_pattern(text: str, pattern: re.Pattern, placeholders: List[str]) -> str:
    def _sub(m: re.Match) -> str:
        placeholders.append(m.group(0))
        return _SENTINEL_FMT.format(i=len(placeholders) - 1)
    return pattern.sub(_sub, text)


def _shield_md_links(text: str, placeholders: List[str]) -> str:
    """
    Special-case Markdown links: ``[label](url)``.

    We keep the visible ``label`` translatable but shield the ``](url)`` part
    so the URL is preserved verbatim.
    """
    md_link_re = re.compile(r"\[([^\]\n]+)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")

    def _sub(m: re.Match) -> str:
        label, url = m.group(1), m.group(2)
        placeholders.append(f"]({url})")
        sentinel = _SENTINEL_FMT.format(i=len(placeholders) - 1)
        return f"[{label}{sentinel}"

    return md_link_re.sub(_sub, text)


def shield(text: str) -> Tuple[str, List[str]]:
    """
    Return (masked_text, placeholder_table).

    Feed ``masked_text`` to the MT model. Then call :func:`unshield` on the
    translated output with the same ``placeholder_table``.
    """
    placeholders: List[str] = []
    # Links first: their URL must survive even if the label is translated.
    text = _shield_md_links(text, placeholders)
    for _name, pat in _PATTERNS:
        text = _shield_pattern(text, pat, placeholders)
    return text, placeholders


def unshield(text: str, placeholders: List[str]) -> str:
    """Replace ``[TL_N]`` sentinels back with their originals."""
    def _sub(m: re.Match) -> str:
        idx = int(m.group(1))
        if 0 <= idx < len(placeholders):
            return placeholders[idx]
        return m.group(0)  # unknown sentinel — leave as-is
    return _SENTINEL_RE.sub(_sub, text)


def shielded_translate(
    text: str,
    translate_fn,
    glossary: Optional[Mapping[str, str]] = None,
    glossary_case_sensitive: bool = False,
) -> str:
    """
    Convenience wrapper: glossary-mask → shield → translate → unshield →
    glossary-restore.

    ``translate_fn`` is any callable ``str -> str`` (typically a partially
    applied :func:`core.translation.engine.translate`).

    ``glossary`` is an optional mapping from source-language term to the
    exact target-language term the user wants forced into the output. Each
    occurrence of a source term in ``text`` is replaced with an opaque
    ``[GL_N]`` sentinel before translation and swapped back to the target
    term after — a lightweight, decoder-agnostic implementation of the
    "term lock" feature offered by commercial MT services.

    ``glossary_case_sensitive`` controls whether source-term matching is
    case-sensitive. Default: case-insensitive.
    """
    if not text or not text.strip():
        return text

    glossary_placeholders: List[str] = []
    if glossary:
        text = _apply_glossary(
            text, glossary, glossary_placeholders,
            case_sensitive=glossary_case_sensitive,
        )

    masked, table = shield(text)
    translated = translate_fn(masked)
    restored = unshield(translated, table)

    if glossary_placeholders:
        restored = _restore_glossary(restored, glossary_placeholders)
    return restored


def shielded_translate_many(
    texts: List[str],
    translate_fn,
    glossary: Optional[Mapping[str, str]] = None,
    glossary_case_sensitive: bool = False,
) -> List[str]:
    """
    Batched counterpart of :func:`shielded_translate`.

    Shields every text (glossary-mask + structure-shield), translates them
    all in one batched call when ``translate_fn`` exposes a ``.many``
    (``List[str] -> List[str]``) attribute, then unshields each. Returns one
    output per input, in order. Empty / whitespace inputs pass through.

    This is the primitive the format-preserving pipelines use to avoid one
    model call per paragraph/line/cell.
    """
    n = len(texts)
    result: List[str] = list(texts)

    # Shield only the non-empty inputs; remember their positions.
    positions: List[int] = []
    masked_list: List[str] = []
    tables: List[List[str]] = []
    glossary_tables: List[List[str]] = []

    for i, text in enumerate(texts):
        if not text or not text.strip():
            continue
        gloss_ph: List[str] = []
        work = text
        if glossary:
            work = _apply_glossary(
                work, glossary, gloss_ph,
                case_sensitive=glossary_case_sensitive,
            )
        masked, table = shield(work)
        positions.append(i)
        masked_list.append(masked)
        tables.append(table)
        glossary_tables.append(gloss_ph)

    if not masked_list:
        return result

    many = getattr(translate_fn, "many", None)
    if callable(many):
        translated_list = many(masked_list)
    else:
        translated_list = [translate_fn(m) for m in masked_list]

    for pos, translated, table, gloss_ph in zip(
        positions, translated_list, tables, glossary_tables
    ):
        restored = unshield(translated, table)
        if gloss_ph:
            restored = _restore_glossary(restored, gloss_ph)
        result[pos] = restored

    return result


# ---------------------------------------------------------------------------
# Glossary / term-lock helpers
# ---------------------------------------------------------------------------


def _apply_glossary(
    text: str,
    glossary: Mapping[str, str],
    placeholders: List[str],
    case_sensitive: bool = False,
) -> str:
    """
    Replace every occurrence of each glossary source term in ``text`` with a
    ``[GL_N]`` sentinel and append the *target* term to ``placeholders``.

    * Latin/Cyrillic/Greek terms are matched with ``\\b`` word boundaries so
      short terms don't match inside larger words (e.g. ``"sun"`` won't hit
      ``"sunday"``).
    * For terms that start/end with a character where ``\\b`` never fires
      (CJK ideographs, Thai, most non-alphabetic scripts) we fall back to
      substring matching — pragmatic default.
    * Longer source terms are matched first, so a glossary containing both
      ``"University of Bern"`` and ``"Bern"`` always picks the longer one
      at any overlap.
    * Empty source keys are skipped.
    """
    if not glossary:
        return text

    # Sort by descending length so multi-word terms win over sub-terms.
    items = sorted(
        ((k, v) for k, v in glossary.items() if k and k.strip()),
        key=lambda kv: len(kv[0]),
        reverse=True,
    )

    flags = re.UNICODE | (0 if case_sensitive else re.IGNORECASE)

    for src, tgt in items:
        pattern = re.compile(_glossary_pattern(src), flags)

        def _sub(_m: re.Match, tgt_local: str = tgt) -> str:
            placeholders.append(tgt_local)
            return _GLOSSARY_FMT.format(i=len(placeholders) - 1)

        text = pattern.sub(_sub, text)
    return text


# Characters where Python's ``\b`` boundary reliably fires. If the source
# term begins/ends with one of these, we anchor with ``\b``; otherwise we
# fall back to plain substring matching (relevant for CJK, Thai, etc.).
_BOUNDARY_CHAR_RE = re.compile(r"[A-Za-z0-9_\u00C0-\u024F\u0400-\u052F\u0370-\u03FF]")


def _glossary_pattern(src: str) -> str:
    """Return the regex source string for a single glossary key."""
    esc = re.escape(src)
    starts_with_word = bool(_BOUNDARY_CHAR_RE.match(src[:1]))
    ends_with_word = bool(_BOUNDARY_CHAR_RE.match(src[-1:]))
    left = r"\b" if starts_with_word else ""
    right = r"\b" if ends_with_word else ""
    return f"{left}{esc}{right}"


def _restore_glossary(text: str, placeholders: List[str]) -> str:
    """Replace ``[GL_N]`` sentinels back with their glossary target terms."""
    def _sub(m: re.Match) -> str:
        idx = int(m.group(1))
        if 0 <= idx < len(placeholders):
            return placeholders[idx]
        return m.group(0)
    return _GLOSSARY_RE.sub(_sub, text)
