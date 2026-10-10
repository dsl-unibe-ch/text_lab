"""Markdown and plain-text translation that keeps the document structure.

Only the prose of each line is translated: heading markers, list bullets,
block quotes, code fences, math blocks and HTML comments are kept as they
are, and :mod:`textlab.features.translation.shield` protects links, inline
code and similar spans. Sentences a source hard-wrapped over several lines
are rejoined first (:func:`reflow_soft_wraps`), so the model sees them
whole.
"""

from __future__ import annotations

import re

from ..shield import shielded_translate_many
from .base import Glossary, ProgressCb, TranslateFn
from .base import report_progress as _report

# Structural line matchers. We only translate the *prose* portion of a line
# and re-attach the structural prefix afterwards.
_MD_HEADING_RE = re.compile(r"^(\s{0,3}#{1,6}\s+)(.*?)(\s+#+\s*)?$")
_MD_LIST_RE = re.compile(r"^(\s*(?:[-*+]|\d+[\.\)])\s+(?:\[[ xX]\]\s+)?)(.+)$")
_MD_BLOCKQUOTE_RE = re.compile(r"^(\s*>+\s*)(.*)$")
_MD_FENCE_RE = re.compile(r"^\s*(?:```|~~~)")
_MD_HR_RE = re.compile(r"^\s*(?:-{3,}|_{3,}|\*{3,})\s*$")
_LITERAL_BLOCKS = (("$$", "$$"), (r"\[", r"\]"), ("<!--", "-->"))


def _literal_block(line: str):
    for opening, closing in _LITERAL_BLOCKS:
        if line.lstrip().startswith(opening):
            tail = line.lstrip()[len(opening) :]
            return True, None if closing in tail else closing
    return False, None


def translate_markdown(
    md_text: str,
    translate_fn: TranslateFn,
    progress_cb: ProgressCb = None,
    glossary: Glossary = None,
    *,
    glossary_case_sensitive: bool = False,
) -> str:
    """Translate a Markdown document while preserving structure.

    A paragraph the source hard-wrapped over several lines is rejoined first
    (:func:`reflow_soft_wraps`), so the model sees whole sentences; markdown
    treats those breaks as cosmetic anyway. Structure -- headings, lists,
    tables, block quotes, fenced code -- keeps its own lines.

    Fenced code blocks are passed through untouched. Every other line is
    routed through :func:`shielded_translate` so links, math, inline code,
    HTML, and placeholders survive the round-trip. The optional
    ``glossary`` maps source-language terms to forced target-language
    replacements.

    All translatable line bodies are collected first and translated in a
    single batched call (:func:`shielded_translate_many`), which on GPU is
    dramatically faster than one model call per line.
    """
    # Rejoin sentences the source hard-wrapped across lines. Without this each
    # line goes to the model on its own, with no subject and no verb, and the
    # translation is as broken as the fragment it came from.
    lines = reflow_soft_wraps(md_text).splitlines(keepends=False)
    out: list[str | None] = []
    in_fence = False
    literal_end = None
    total = len(lines)

    # Gather translatable bodies; each slot records where/how to reinsert.
    bodies: list[str] = []
    slots: list[tuple[int, str, str]] = []  # (out_index, prefix, suffix)

    def _defer(prefix: str, body: str, suffix: str) -> None:
        out.append(None)
        slots.append((len(out) - 1, prefix, suffix))
        bodies.append(body)

    for i, line in enumerate(lines, start=1):
        _report(progress_cb, i, total, "parsing markdown")

        if literal_end:
            out.append(line)
            if literal_end in line:
                literal_end = None
            continue
        if not in_fence:
            protected, literal_end = _literal_block(line)
            if protected:
                out.append(line)
                continue

        # Code fences: toggle and passthrough (fence + contents).
        if _MD_FENCE_RE.match(line):
            in_fence = not in_fence
            out.append(line)
            continue
        if in_fence:
            out.append(line)
            continue

        # Blank line / HR / structural-only lines: passthrough.
        if not line.strip() or _MD_HR_RE.match(line):
            out.append(line)
            continue

        # Heading:  ## Title
        m = _MD_HEADING_RE.match(line)
        if m:
            prefix, body, suffix = m.group(1), m.group(2), (m.group(3) or "")
            _defer(prefix, body, suffix)
            continue

        # Blockquote:  > text
        m = _MD_BLOCKQUOTE_RE.match(line)
        if m:
            prefix, body = m.group(1), m.group(2)
            if body.strip():
                _defer(prefix, body, "")
            else:
                out.append(line)
            continue

        # List item:  - text  |  1. text  |  * [x] text
        m = _MD_LIST_RE.match(line)
        if m:
            prefix, body = m.group(1), m.group(2)
            _defer(prefix, body, "")
            continue

        # Regular paragraph line.
        _defer("", line, "")

    _report(progress_cb, total, total, "translating markdown")
    translated = shielded_translate_many(
        bodies,
        translate_fn,
        glossary=glossary,
        glossary_case_sensitive=glossary_case_sensitive,
    )
    for (idx, prefix, suffix), tr in zip(slots, translated, strict=False):
        out[idx] = f"{prefix}{tr}{suffix}"

    return "\n".join(s if s is not None else "" for s in out)


ENDS_SENTENCE_RE = re.compile(
    r"[\.\!\?\u3002\uFF01\uFF1F\u203C\u2049\uFF0E]\s*[\"'\)\]]?\s*$"
)


# Structure that markdown marks with a line, on top of the _MD_* patterns
# above: a table row, and the four-space indent of a code block.
_MD_TABLE_ROW_RE = re.compile(r"^\s*\|")
_INDENTED_CODE_RE = re.compile(r"^(?: {4,}|\t)")


def _is_soft_wrap(line: str, following: str) -> bool:
    """Did *line* stop mid-sentence, with *following* continuing it?

    A hard-wrapped paragraph is one sentence the writer's editor happened to
    break in the middle. Translating the halves separately gives the model no
    subject, no verb and no context, and the output shows it -- which is the
    whole reason this exists.

    Conservative on purpose: a line break that follows a completed sentence is
    left alone, so deliberately line-structured prose keeps its shape.
    """
    if not line.strip() or not following.strip():
        return False
    if ENDS_SENTENCE_RE.search(line.rstrip()):
        return False
    # An explicit markdown hard break ("two trailing spaces") is deliberate.
    # Checked before any rstrip, which would erase the evidence.
    if line.endswith("  "):
        return False
    # Structural lines are breaks in their own right, whatever they end with.
    # Probed with their indentation intact: that is what marks an indented
    # code block, and what tells a list continuation from a new paragraph.
    for probe in (line, following):
        if (
            _MD_HEADING_RE.match(probe)
            or _MD_LIST_RE.match(probe)
            or _MD_BLOCKQUOTE_RE.match(probe)
            or _MD_FENCE_RE.match(probe)
            or _MD_HR_RE.match(probe)
            or _MD_TABLE_ROW_RE.match(probe)
            or _INDENTED_CODE_RE.match(probe)
        ):
            return False
    return True


def reflow_soft_wraps(text: str) -> str:
    """Rejoin sentences that a hard-wrapped source split across lines.

    Blank lines, and any break that follows a finished sentence, are kept, so
    paragraph structure survives. A word hyphenated across the break is put
    back together.

    Fenced code blocks pass through untouched: the lines inside one are not
    prose, and nothing there ends in a full stop.

    Not safe for line-oriented formats -- subtitles, for one, where every line
    break carries meaning -- so callers opt in rather than get this for free.
    """
    if not text or "\n" not in text:
        return text

    lines = text.split("\n")
    out: list[str] = []
    in_fence = False
    literal_end = None
    previous_protected = False
    for i, line in enumerate(lines):
        was_protected = previous_protected
        previous_protected = False
        if literal_end:
            out.append(line)
            previous_protected = True
            if literal_end in line:
                literal_end = None
            continue
        if not in_fence:
            protected, literal_end = _literal_block(line)
            if protected:
                out.append(line)
                previous_protected = True
                continue
        if _MD_FENCE_RE.match(line):
            # The fence markers themselves are structural, so _is_soft_wrap
            # already refuses them; this is about the arbitrary code between.
            in_fence = not in_fence
            out.append(line)
            continue
        if in_fence:
            out.append(line)
            continue
        if out and not was_protected and _is_soft_wrap(lines[i - 1], line):
            previous = out.pop().rstrip()
            if previous.endswith("-") and line.lstrip()[:1].islower():
                out.append(previous[:-1] + line.strip())  # de-hyphenate
            else:
                out.append(f"{previous} {line.strip()}")
        else:
            out.append(line)
    return "\n".join(out)
