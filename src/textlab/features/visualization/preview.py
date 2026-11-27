"""A quick look at an uploaded table before it is analyzed.

:func:`preview_table` reads the first rows of a table file as the agents
read it, and describes each column as the agents treat it (numbers, dates,
categories or free text); :func:`example_requests` suggests requests that
name the table's own columns. :func:`read_preview` and
:func:`column_profile` give a first look at an upload in memory, as data
frames.
"""

from __future__ import annotations

import dataclasses
import io
import math
import os
import re

import pandas as pd

from textlab.features.visualization.plot_data import (
    CATEGORICAL,
    DATETIME,
    NUMERIC,
    TEXT,
    column_kind,
)
from textlab.features.visualization.viz_utils import (
    drop_empty_unnamed_columns,
    get_fast_data_preview,
    shorten_text,
)

#: Rows read for the preview and the column profile.
PROFILE_ROWS = 2000

#: Rows of a table shown as its first rows.
FIRST_ROWS = 10

#: Characters of a cell shown among the first rows.
CELL_CHARS = 60

#: Characters of a category shown among a column's values.
VALUE_CHARS = 24

#: The most frequent categories shown among a column's values.
SHOWN_VALUES = 3

#: Most groups a column of categories may have to suggest comparing them.
GROUPS_AT_MOST = 12

#: The kinds of column a preview names, by the agents' kinds.
KIND_NAMES = {
    NUMERIC: "number",
    DATETIME: "date",
    CATEGORICAL: "category",
    TEXT: "text",
}

#: Column names of identifiers, which make no sense to plot or test.
_IDENTIFIER = re.compile(r"(?i)(id|index|key)|.+[_\s-]id|id[_\s-].+")


@dataclasses.dataclass(frozen=True)
class ColumnInfo:
    """One column of a table, described from its first rows.

    Attributes:
        name: The column's name.
        kind: How the agents treat it: ``"number"``, ``"date"``,
            ``"category"`` or ``"text"`` (free text, for word clouds but
            not for statistics); ``"nested"`` for lists or objects (JSON).
        filled: The share of rows with a value, from 0 to 1.
        distinct: How many different values it has.
        values: What its values are like: the range and mean of numbers,
            the first and last date, the most frequent categories, or the
            average length of texts; empty for a column without values.
    """

    name: str
    kind: str
    filled: float
    distinct: int
    values: str


@dataclasses.dataclass(frozen=True)
class TablePreview:
    """The first rows of a table, and its columns.

    Attributes:
        columns: The columns, in the table's order.
        rows: The first rows (at most :data:`FIRST_ROWS`), each cell as
            text.
        rows_read: How many rows the columns are described from.
        complete: Whether those are all of the table's rows.
    """

    columns: tuple[ColumnInfo, ...]
    rows: tuple[tuple[str, ...], ...]
    rows_read: int
    complete: bool

    @property
    def names(self) -> tuple[str, ...]:
        """The columns' names, in order."""
        return tuple(column.name for column in self.columns)


def preview_table(
    name: str, path: str | os.PathLike, nrows: int = PROFILE_ROWS
) -> TablePreview:
    """Read the first rows of a table file and describe its columns.

    The file is read as the agents read it, with the same readers and
    without empty header-less columns, so the preview shows the table the
    agents will analyze.

    Args:
        name: The file's name as uploaded; its extension picks the reader
            (CSV, TSV, Excel, JSON or JSON lines).
        path: The file.
        nrows: Rows to read.

    Returns:
        The preview.

    Raises:
        ValueError: If the file cannot be read as a table, or holds no
            columns or rows; the message is for users.
    """
    table = get_fast_data_preview(os.fspath(path), name, nrows=nrows)
    if table is None:
        raise ValueError(
            f"{name} could not be read as a table. Check that it is a CSV, "
            "TSV, Excel or JSON file with the column names in its first "
            "row."
        )
    table = drop_empty_unnamed_columns(table)
    if len(table.columns) == 0 or len(table) == 0:
        raise ValueError(
            f"{name} holds no data: it needs column names in its first row "
            "and at least one row of values."
        )
    columns = tuple(
        _describe(str(column), table.iloc[:, index])
        for index, column in enumerate(table.columns)
    )
    rows = tuple(
        tuple(_cell(value) for value in row)
        for row in table.head(FIRST_ROWS).itertuples(index=False)
    )
    return TablePreview(columns, rows, len(table), len(table) < nrows)


def example_requests(preview: TablePreview, limit: int = 4) -> tuple[str, ...]:
    """Suggest requests that name a table's own columns.

    Args:
        preview: The table's preview.
        limit: The most requests to suggest.

    Returns:
        Requests for what the table's columns allow: a distribution, a
        comparison of groups, the relation of two numbers, the association
        of two categories, a word cloud. None for a table without numbers,
        groups or texts.
    """
    numbers = [
        column.name
        for column in preview.columns
        if column.kind == "number"
        and column.distinct > 2
        and not _IDENTIFIER.fullmatch(column.name.strip())
    ]
    groups = [
        column.name
        for column in preview.columns
        if column.kind == "category" and 2 <= column.distinct <= GROUPS_AT_MOST
    ]
    texts = [
        column.name for column in preview.columns if column.kind == "text"
    ]
    found = []
    if numbers:
        found.append(f"Plot the distribution of {numbers[0]}.")
    if numbers and groups:
        found.append(
            f"Is {numbers[0]} different between the groups of {groups[0]}? "
            "Run a test and show box plots."
        )
    if len(numbers) > 1:
        found.append(
            f"How are {numbers[0]} and {numbers[1]} related? Show a scatter "
            "plot and test the correlation."
        )
    if len(groups) > 1:
        found.append(f"Is {groups[0]} associated with {groups[1]}?")
    if texts:
        found.append(f"Make a word cloud of {texts[0]}.")
    return tuple(found[:limit])


def _describe(name: str, series: pd.Series) -> ColumnInfo:
    """Describe one column of a table's first rows."""
    present = series.dropna()
    filled = len(present) / len(series) if len(series) else 0.0
    if not present.empty and isinstance(present.iloc[0], list | dict):
        distinct = int(present.astype(str).nunique())
        return ColumnInfo(name, "nested", filled, distinct, "lists or objects")
    kind = column_kind(series)
    try:
        distinct = int(present.nunique())
    except TypeError:
        distinct = int(present.astype(str).nunique())
    values = ""
    if present.empty:
        pass
    elif kind == NUMERIC:
        values = (
            f"{_number(present.min())} to {_number(present.max())}, "
            f"mean {_number(present.mean())}"
        )
    elif kind == DATETIME:
        values = f"{present.min():%Y-%m-%d} to {present.max():%Y-%m-%d}"
    elif kind == TEXT:
        length = present.astype(str).str.len().mean()
        values = f"{length:,.0f} characters on average"
    else:
        counts = present.astype(str).value_counts()
        shown = [
            shorten_text(value, VALUE_CHARS)
            for value in counts.index[:SHOWN_VALUES]
        ]
        values = ", ".join(shown) + (
            ", ..." if len(counts) > SHOWN_VALUES else ""
        )
    return ColumnInfo(name, KIND_NAMES[kind], filled, distinct, values)


def _number(value: object) -> str:
    """Return a number as a column's values show it."""
    number = float(value)
    if not math.isfinite(number):
        return str(number)
    if number.is_integer() or abs(number) >= 10_000:
        return f"{number:,.0f}"
    return f"{number:.4g}"


def _cell(value: object) -> str:
    """Return a cell of the first rows as text: one line, shortened.

    Whole numbers show without decimals, also in columns that pandas reads
    as decimals because some of their cells are empty.
    """
    if not isinstance(value, list | dict | tuple):
        try:
            if pd.isna(value):
                return ""
        except (TypeError, ValueError):
            pass
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return shorten_text(value, CELL_CHARS)


def read_preview(name: str, data: bytes, nrows: int = PROFILE_ROWS):
    """Read the first rows of an uploaded table.

    Args:
        name: The file name, which picks the parser (CSV, TSV, Excel,
            JSON or JSON lines).
        data: The file's content.
        nrows: Rows to read.

    Returns:
        A ``pandas.DataFrame`` without empty header-less columns (see
        :func:`.viz_utils.drop_empty_unnamed_columns`), or ``None`` if the
        file cannot be read.
    """
    df = _read_table(name, data, nrows)
    return drop_empty_unnamed_columns(df) if df is not None else None


def _read_table(name: str, data: bytes, nrows: int):
    """Parse the first ``nrows`` rows of a table, or return ``None``."""
    lower = name.lower()
    try:
        if lower.endswith(".csv"):
            return pd.read_csv(io.BytesIO(data), nrows=nrows)
        if lower.endswith(".tsv"):
            return pd.read_csv(io.BytesIO(data), sep="\t", nrows=nrows)
        if lower.endswith((".xls", ".xlsx")):
            return pd.read_excel(io.BytesIO(data), nrows=nrows)
        if lower.endswith(".json"):
            try:
                return pd.read_json(io.BytesIO(data), lines=True, nrows=nrows)
            except Exception:
                return pd.read_json(io.BytesIO(data)).head(nrows)
    except Exception:
        return None
    return None


def column_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize each column: type, completeness, distinct and typical values.

    Args:
        df: The table, or its first rows.

    Returns:
        One row per column with ``Column``, ``Type``, ``Non-Null %``,
        ``Unique``, ``Range (min / mean / max)`` (numbers) and ``Top
        Value`` (other columns).
    """
    total = len(df)
    rows = []
    for col in df.columns:
        series = df[col]
        non_null = int(series.notna().sum())
        null_pct = (total - non_null) / total * 100 if total > 0 else 0.0
        is_numeric = pd.api.types.is_numeric_dtype(series)

        # JSON columns can contain lists/dicts (unhashable). Coerce to str
        # for stats.
        first_val = series.dropna().iloc[0] if non_null > 0 else None
        is_nested = isinstance(first_val, list | dict)
        safe_series = (
            series.dropna().astype(str) if is_nested else series.dropna()
        )

        try:
            unique = int(safe_series.nunique())
        except TypeError:
            unique = -1

        if is_numeric and not is_nested:
            s = series.dropna()
            range_str = (
                f"{s.min():.4g} / {s.mean():.4g} / {s.max():.4g}"
                if len(s) > 0
                else "—"
            )
            top_str = ""
        else:
            try:
                top_vals = safe_series.value_counts()
                top_str = (
                    str(top_vals.index[0])[:50] if len(top_vals) > 0 else "—"
                )
            except TypeError:
                top_str = "nested"
            range_str = "nested" if is_nested else ""

        rows.append(
            {
                "Column": col,
                "Type": "nested (list/dict)"
                if is_nested
                else str(series.dtype),
                "Non-Null %": f"{100 - null_pct:.1f}%",
                "Unique": unique if unique >= 0 else "—",
                "Range (min / mean / max)": range_str,
                "Top Value": top_str,
            }
        )
    return pd.DataFrame(rows)
