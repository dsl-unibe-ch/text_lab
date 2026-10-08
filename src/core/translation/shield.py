"""Lossless, fail-closed protection of translation syntax and glossary terms.

``shield`` returns masked text and a list-compatible restoration table. Keep
that table intact for ``unshield``: its per-input namespace prevents marker
swaps between translation units. Markers use uppercase, whitespace-free
ASCII, not control characters. They are not guaranteed tokenizer tokens;
models must reproduce every marker exactly once or the output is blocked.
Structural markers must retain source order; glossary markers may move.

Supported constructs include fenced/inline code, math, inline Markdown
links/images, HTML tags/comments, URLs, paths, emails and format placeholders.
This is not a full Markdown parser. Link labels remain translatable;
image alt text, destinations and titles remain verbatim. Glossaries only
match translatable spans, never protected structure or generated markers.
"""

from __future__ import annotations

import re
from bisect import bisect_left
from contextlib import contextmanager
from contextvars import ContextVar
from secrets import token_hex
from typing import Iterator, List, Mapping, Optional, Tuple

from .chunking import InputTooLongError

_MARKER_START = r"(?:\[|\x02)(?:TL|GL)_"
_MARKER_START_RE = re.compile(_MARKER_START, re.IGNORECASE)
_MARKER_RE = re.compile(r"\[(?:TL|GL)_[0-9A-F]{16}_(?:0|[1-9][0-9]*)\]")
_LEGACY_RE = re.compile(r"\x02TL_([0-9]+)\x03|\[TL_([0-9]+)\]")
_LITERAL_MARKER = (
    r"(?i:" + _MARKER_START + r")[^\]\x03\r\n]*(?:\]|\x03)?"
)
_INLINE_CODE = r"(?P<ticks>`+)(?!`)[\s\S]+?(?<!`)(?P=ticks)(?!`)"
_MATH = (
    r"\$\$[\s\S]+?\$\$|\\\[[\s\S]+?\\\]"
    r"|(?<!\\)\$(?!\s)[^\$\n]+?(?<!\s)\$|\\\([\s\S]+?\\\)"
)
_HTML_COMMENT = r"<!--[\s\S]*?(?:-->|\Z)"
_HTML_TAG = r"</?[A-Za-z](?:[^>\"']|\"[^\"]*\"|'[^']*')*>"
_OPAQUE_LABEL_RE = re.compile(
    "|".join((_INLINE_CODE, _MATH, _HTML_COMMENT, _HTML_TAG,
              _LITERAL_MARKER))
)
_TOKEN_RE = re.compile(
    r"^[ \t]*(?P<fence>`{3,})[^\r\n]*\r?\n[\s\S]*?"
    r"(?:^[ \t]*(?P=fence)`*[ \t]*(?=\r?$)|\Z)"
    r"|^[ \t]*(?P<tilde>~{3,})[^\r\n]*\r?\n[\s\S]*?"
    r"(?:^[ \t]*(?P=tilde)~*[ \t]*(?=\r?$)|\Z)"
    r"|" + _INLINE_CODE
    + r"|" + _MATH
    + r"|" + _HTML_COMMENT
    + r"|" + _HTML_TAG
    + r"|https?://\S+|ftp://\S+|www\.\S+"
    r"|(?<![\w/])(?:/[A-Za-z0-9_.\-]+){2,}/?"
    r"|[\w.+\-]+@[\w\-]+(?:\.[\w\-]+)+"
    r"|\{[A-Za-z_][A-Za-z0-9_.]*\}|%\([^)]+\)[sdif]"
    r"|%[0-9.\-+ #]*[sdifxXoeEgG]"
    r"|" + _LITERAL_MARKER
    + r"|(?P<link>!?\[)",
    re.MULTILINE,
)
_BOUNDARY_CHAR_RE = re.compile(
    r"[A-Za-z0-9_\u00C0-\u024F\u0400-\u052F\u0370-\u03FF]"
)


class ProtectedContentError(ValueError):
    """Protected output is unsafe; error messages never contain source text.

    Batch failures expose ``partial_results`` in input order, with ``None``
    for blocked units, and zero-based ``failed_indices``. A cardinality
    failure blocks every nonempty unit because alignment cannot be trusted.
    """

    def __init__(self, message: str, *, partial_results=None,
                 failed_indices=()):
        super().__init__(message)
        self.partial_results = partial_results
        self.failed_indices = tuple(failed_indices)


class _PlaceholderTable(list[str]):
    def __init__(self, text: str):
        super().__init__()
        self.namespace = token_hex(8).upper()
        while self.namespace in text.upper():
            self.namespace = token_hex(8).upper()
        self.markers: List[str] = []

    def protect(self, text: str, kind: str = "TL") -> str:
        marker = f"[{kind}_{self.namespace}_{len(self)}]"
        self.append(text)
        self.markers.append(marker)
        return marker


def _link_bounds(text: str, start: int) -> Optional[Tuple[int, int, int]]:
    """Find a balanced inline link, skipping opaque/escaped label content."""
    label = start + (2 if text.startswith("![", start) else 1)
    pos, depth = label, 1
    while pos < len(text):
        opaque = _OPAQUE_LABEL_RE.match(text, pos)
        if opaque:
            pos = opaque.end()
            continue
        char = text[pos]
        if char == "\\":
            pos += 2
            continue
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                break
        pos += 1
    if depth or text[pos:pos + 2] != "](":
        return None

    close = pos
    pos, depth = pos + 2, 1
    quote = ""
    angle = False
    while pos < len(text):
        char = text[pos]
        if char == "\\":
            pos += 2
            continue
        if quote:
            if char == quote:
                quote = ""
        elif angle:
            if char == ">":
                angle = False
        elif char == "<":
            angle = True
        elif char in "\"'" and text[pos - 1].isspace():
            quote = char
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return label, close, pos + 1
        pos += 1
    return None


def _glossary_pattern(src: str) -> str:
    left = r"\b" if _BOUNDARY_CHAR_RE.match(src[:1]) else ""
    right = r"\b" if _BOUNDARY_CHAR_RE.match(src[-1:]) else ""
    return f"{left}{re.escape(src)}{right}"


def _prepare(text: str, glossary=None, case_sensitive: bool = False):
    table = _PlaceholderTable(text)
    items = sorted(
        ((src, tgt) for src, tgt in (glossary or {}).items()
         if src and src.strip()),
        key=lambda item: len(item[0]), reverse=True,
    )
    patterns = [
        (re.compile(_glossary_pattern(src),
                    0 if case_sensitive else re.IGNORECASE), target)
        for src, target in items
    ]

    def plain(value: str) -> str:
        starts = []
        selected = []
        for pattern, target in patterns:
            for match in pattern.finditer(value):
                start, end = match.span()
                index = bisect_left(starts, start)
                if index and selected[index - 1][1] > start:
                    continue
                if index < len(starts) and starts[index] < end:
                    continue
                starts.insert(index, start)
                selected.insert(index, (start, end, target))
        parts = []
        pos = 0
        for start, end, target in selected:
            parts.extend((value[pos:start], table.protect(target, "GL")))
            pos = end
        parts.append(value[pos:])
        return "".join(parts)

    def scan(value: str) -> str:
        parts = []
        start = pos = 0
        while match := _TOKEN_RE.search(value, pos):
            bounds = None
            if match.lastgroup == "link":
                bounds = _link_bounds(value, match.start())
                if bounds is None:
                    pos = match.end()
                    continue
            parts.append(plain(value[start:match.start()]))
            if bounds is None:
                parts.append(table.protect(match.group()))
                pos = match.end()
            else:
                label, close, pos = bounds
                if value.startswith("![", match.start()):
                    parts.append(table.protect(value[match.start():pos]))
                else:
                    parts.append(table.protect(value[match.start():label]))
                    parts.append(scan(value[label:close]))
                    parts.append(table.protect(value[close:pos]))
            start = pos
        parts.append(plain(value[start:]))
        return "".join(parts)

    return scan(text), table


def shield(text: str) -> Tuple[str, List[str]]:
    """Return masked text and its list-compatible, per-input restoration table.

    Source text resembling markers is itself protected as literal content.
    No table entry contains generated markers: restoration is one pass.
    """
    return _prepare(text)


def unshield(text: str, placeholders: List[str]) -> str:
    """Validate marker counts/order, then restore without recursion.

    Structural TL markers must retain source order; GL markers may move for
    target grammar. Use the intact table returned by ``shield``. Plain
    positional lists support historical ``\\x02TL_N\\x03`` and ``[TL_N]``
    markers in table order, but cannot detect cross-unit swaps.
    """
    if not isinstance(text, str):
        raise ProtectedContentError("Translation output must be text.")
    if isinstance(placeholders, _PlaceholderTable):
        if len(placeholders.markers) != len(placeholders):
            raise ProtectedContentError("Protection table was modified.")
        expected = dict(zip(placeholders.markers, placeholders))
        structural_order = [marker for marker in placeholders.markers
                            if marker.startswith("[TL_")]
        marker_re = _MARKER_RE
    else:
        expected = dict(enumerate(placeholders))
        structural_order = list(expected)
        marker_re = _LEGACY_RE

    structural_keys = set(structural_order)
    observed_order = []
    counts = dict.fromkeys(expected, 0)
    replacements = []
    for start in _MARKER_START_RE.finditer(text):
        match = marker_re.match(text, start.start())
        if match is None:
            raise ProtectedContentError("Malformed protected marker.")
        try:
            key = match.group() if marker_re is _MARKER_RE else int(
                match.group(1) or match.group(2)
            )
        except ValueError:
            raise ProtectedContentError("Unknown protected marker.") from None
        if key not in expected:
            raise ProtectedContentError("Unknown protected marker.")
        counts[key] += 1
        if counts[key] > 1:
            raise ProtectedContentError("Duplicate protected marker.")
        if key in structural_keys:
            observed_order.append(key)
        replacements.append((match.start(), match.end(), expected[key]))
    if any(count != 1 for count in counts.values()):
        raise ProtectedContentError("Missing protected marker.")
    if observed_order != structural_order:
        raise ProtectedContentError(
            "Protected structural marker order changed."
        )

    parts = []
    pos = 0
    for start, end, original in replacements:
        parts.extend((text[pos:start], original))
        pos = end
    parts.append(text[pos:])
    return "".join(parts)


def _restore_nonempty(text: str, placeholders: List[str]) -> str:
    restored = unshield(text, placeholders)
    if not restored.strip():
        raise ProtectedContentError(
            "Empty translation output for nonempty input."
        )
    return restored


# Units with more markers than this, or with runs of touching markers
# (HTML tables from OCR, nested tags), skip the marker round-trip: seq2seq
# models drop or repeat such markers, and a whitespace-free run of markers
# is one indivisible "word" that can exceed the encoder's input budget.
MAX_INLINE_MARKERS = 4
_MARKER_RUN_RE = re.compile(f"(?:{_MARKER_RE.pattern}){{4}}")
_MARKER_SPLIT_RE = re.compile(f"({_MARKER_RE.pattern})")
_HAS_LETTER_RE = re.compile(r"[^\W\d_]")


def _needs_segments(masked: str, table: List[str]) -> bool:
    return (len(table) > MAX_INLINE_MARKERS
            or bool(_MARKER_RUN_RE.search(masked)))


def _segment_sources(masked: str) -> List[str]:
    """Translatable text between markers; protected spans never reach the model."""
    return [part.strip() for part in _MARKER_SPLIT_RE.split(masked)[::2]
            if _HAS_LETTER_RE.search(part)]


def _segment_restore(masked: str, table: List[str], outputs) -> str:
    """Reassemble a segment-mode unit: originals for markers, translations
    between them, with each segment's surrounding whitespace preserved."""
    expected = dict(zip(table.markers, table))
    outputs = iter(outputs)
    parts = []
    for index, part in enumerate(_MARKER_SPLIT_RE.split(masked)):
        if index % 2:
            parts.append(expected[part])
        elif _HAS_LETTER_RE.search(part):
            translated = next(outputs)
            if not isinstance(translated, str):
                raise ProtectedContentError("Translation output must be text.")
            if not translated.strip():
                raise ProtectedContentError(
                    "Empty translation output for nonempty input."
                )
            leading = part[:len(part) - len(part.lstrip())]
            trailing = part[len(part.rstrip()):]
            parts.append(leading + translated.strip() + trailing)
        else:
            parts.append(part)
    restored = "".join(parts)
    if not restored.strip():
        raise ProtectedContentError(
            "Empty translation output for nonempty input."
        )
    return restored


def _call_many(translate_fn, texts: List[str]):
    many = getattr(translate_fn, "many", None)
    return many(texts) if callable(many) else [
        translate_fn(text) for text in texts
    ]


def _translate_segments(units, translate_fn) -> List[str]:
    """Translate (masked, table) units piecewise in one batched call."""
    sources = [_segment_sources(masked) for masked, _ in units]
    flat = [source for unit in sources for source in unit]
    outputs = list(_call_many(translate_fn, flat)) if flat else []
    if len(outputs) != len(flat):
        raise ProtectedContentError(
            f"Translation batch output count mismatch: expected "
            f"{len(flat)}, received {len(outputs)}."
        )
    restored = []
    offset = 0
    for (masked, table), unit in zip(units, sources):
        restored.append(_segment_restore(
            masked, table, outputs[offset:offset + len(unit)],
        ))
        offset += len(unit)
    return restored


def shielded_translate(
    text: str,
    translate_fn,
    glossary: Optional[Mapping[str, str]] = None,
    glossary_case_sensitive: bool = False,
    *,
    fallback: bool = True,
) -> str:
    """Translate with structure and exact glossary terms protected together.

    Longest glossary terms win at overlaps. Matching defaults to
    case-insensitive and retains word boundaries for Latin, Cyrillic and
    Greek terms, with substring matching for other scripts. Empty keys are
    ignored. Invalid markers or blank output for nonempty source raise
    ``ProtectedContentError``; output is never repaired heuristically or
    replaced with untranslated source.

    With ``fallback`` (the default), marker-dense text, and text whose
    markers the model corrupted, is retranslated piecewise between the
    protected spans, so protected content never passes through the model.
    """
    return shielded_translate_many(
        [text], translate_fn, glossary, glossary_case_sensitive,
        fallback=fallback,
    )[0]


_RECORDER: ContextVar[Optional[List[Tuple[str, str]]]] = ContextVar(
    "translation_recorder", default=None,
)


@contextmanager
def record_translations() -> Iterator[List[Tuple[str, str]]]:
    """Collect ``(source, translation)`` pairs of every unit translated here.

    Used for side-by-side review files. Nested scopes record only into the
    innermost list.
    """
    pairs: List[Tuple[str, str]] = []
    token = _RECORDER.set(pairs)
    try:
        yield pairs
    finally:
        _RECORDER.reset(token)


def shielded_translate_many(
    texts: List[str],
    translate_fn,
    glossary: Optional[Mapping[str, str]] = None,
    glossary_case_sensitive: bool = False,
    *,
    fallback: bool = True,
) -> List[str]:
    """Translate units in order, using ``translate_fn.many`` when available.

    Empty/whitespace inputs pass through without model calls. Output count,
    marker namespaces/counts/order and nonempty restored units are checked.
    With ``fallback``, units that are marker-dense or fail marker checks are
    translated piecewise between protected spans (see ``shielded_translate``).
    On failure, ``ProtectedContentError.partial_results`` retains only
    independently verified units; ``None`` entries must not be published.
    """
    if fallback:
        result = _translate_with_fallback(
            texts, translate_fn, glossary, glossary_case_sensitive,
        )
    else:
        result = _translate_with_markers(
            texts, translate_fn, glossary, glossary_case_sensitive,
        )
    recorder = _RECORDER.get()
    if recorder is not None:
        recorder.extend(
            (text, output) for text, output in zip(texts, result)
            if text.strip()
        )
    return result


def _translate_with_fallback(
    texts: List[str],
    translate_fn,
    glossary: Optional[Mapping[str, str]],
    glossary_case_sensitive: bool,
) -> List[str]:
    result = [text if not text.strip() else None for text in texts]
    prepared = {
        i: _prepare(text, glossary, glossary_case_sensitive)
        for i, text in enumerate(texts) if text.strip()
    }
    segmented = {i for i, (masked, table) in prepared.items()
                 if _needs_segments(masked, table)}
    inline = [i for i in prepared if i not in segmented]
    failed: List[int] = []
    error = None
    if inline:
        try:
            outputs = _translate_with_markers(
                [texts[i] for i in inline], translate_fn, glossary,
                glossary_case_sensitive,
                prepared=[prepared[i] for i in inline],
            )
            for i, output in zip(inline, outputs):
                result[i] = output
        except ProtectedContentError as exc:
            error = exc
            partial = exc.partial_results or [None] * len(inline)
            for i, output in zip(inline, partial):
                result[i] = output
            failed = [inline[j] for j in exc.failed_indices]
        except InputTooLongError:
            # A marker run formed an indivisible span. Segments avoid it;
            # a genuinely oversized word still raises from the retry.
            segmented = set(prepared)

    # Text without markers gains nothing from a piecewise retry.
    unrecoverable = [i for i in failed if not prepared[i][1]]
    retry = sorted(segmented | (set(failed) - set(unrecoverable)))
    if retry:
        for i, output in zip(retry, _translate_segments(
                [prepared[i] for i in retry], translate_fn)):
            result[i] = output
    if unrecoverable:
        raise ProtectedContentError(
            str(error), partial_results=result,
            failed_indices=unrecoverable,
        )
    return result


def _translate_with_markers(
    texts: List[str],
    translate_fn,
    glossary: Optional[Mapping[str, str]] = None,
    glossary_case_sensitive: bool = False,
    *,
    prepared=None,
) -> List[str]:
    """Strict marker round-trip: any corrupted marker blocks its unit."""
    result = [text if not text.strip() else None for text in texts]
    positions: List[int] = []
    masked_list: List[str] = []
    tables: List[List[str]] = []
    for i, text in enumerate(texts):
        if not text.strip():
            continue
        masked, table = (prepared[len(positions)] if prepared is not None
                         else _prepare(text, glossary,
                                       glossary_case_sensitive))
        positions.append(i)
        masked_list.append(masked)
        tables.append(table)
    if not masked_list:
        return result

    outputs = _call_many(translate_fn, masked_list)
    if isinstance(outputs, (str, bytes)) or outputs is None:
        raise ProtectedContentError(
            "Translation batch must contain one output per input.",
            partial_results=result, failed_indices=positions,
        )
    try:
        outputs = list(outputs)
    except TypeError:
        raise ProtectedContentError(
            "Translation batch must contain one output per input.",
            partial_results=result, failed_indices=positions,
        ) from None
    if len(outputs) != len(positions):
        raise ProtectedContentError(
            f"Translation batch output count mismatch: expected "
            f"{len(positions)}, received {len(outputs)}.",
            partial_results=result, failed_indices=positions,
        )

    failures = []
    failed_indices = []
    for pos, output, table in zip(positions, outputs, tables):
        try:
            result[pos] = _restore_nonempty(output, table)
        except ProtectedContentError as exc:
            failures.append(f"Unit {pos + 1}: {exc}")
            failed_indices.append(pos)
    if failures:
        raise ProtectedContentError(
            " ".join(failures), partial_results=result,
            failed_indices=failed_indices,
        )
    return result
