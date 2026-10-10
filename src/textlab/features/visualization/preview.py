"""A quick look at an uploaded table before it is analyzed."""

from __future__ import annotations

import io

import pandas as pd

#: Rows read for the preview and the column profile.
PROFILE_ROWS = 2000


def read_preview(name: str, data: bytes, nrows: int = PROFILE_ROWS):
    """Read the first rows of an uploaded table.

    Args:
        name: The file name, which picks the parser (CSV, TSV, Excel,
            JSON or JSON lines).
        data: The file's content.
        nrows: Rows to read.

    Returns:
        A ``pandas.DataFrame``, or ``None`` if the file cannot be read.
    """
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
