"""Types and helpers shared by the document translators."""

from __future__ import annotations

from collections.abc import Callable, Mapping

#: Translates one text; carries a batched ``.many`` attribute when made by
#: :func:`textlab.features.translation.engine.make_translate_fn`.
TranslateFn = Callable[[str], str]
#: Progress of a document: ``(done, total, stage)``.
ProgressCb = Callable[[int, int, str], None] | None
#: Source terms mapped to the translations to force.
Glossary = Mapping[str, str] | None


def report_progress(cb: ProgressCb, done: int, total: int, stage: str) -> None:
    """Send a progress update, never letting the receiver fail the run."""
    if cb is not None:
        try:
            cb(done, total, stage)
        except Exception:
            pass
