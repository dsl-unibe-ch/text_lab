"""Results of the Knowledge Graph steps."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pandas as pd


@dataclasses.dataclass
class CorpusSummary:
    """The outcome of parsing a papers folder into a corpus.

    Attributes:
        processed: Papers parsed in this run.
        skipped: Papers already in the corpus.
        errors: Papers Grobid could not parse.
        table: The rebuilt corpus table, one row per parsed paper.
    """

    processed: int
    skipped: int
    errors: int
    table: pd.DataFrame


@dataclasses.dataclass
class TopicSummary:
    """The outcome of extracting topics for a corpus.

    Attributes:
        processed: Papers with an abstract, now with topics.
        skipped: Papers without an abstract (no topics).
        path: The table with topics that was written.
    """

    processed: int
    skipped: int
    path: Path
