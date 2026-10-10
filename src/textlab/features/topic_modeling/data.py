"""Reading uploaded collections: tables, ZIP archives and timestamps.

Used by the page, in the app process, and by the worker; it needs only
pandas.
"""

from __future__ import annotations

import csv
import io
import os
import zipfile
from typing import Any

import pandas as pd

from textlab.common import upload_safety

# Range of numeric values accepted as calendar years in a timestamp column.
_MIN_YEAR = 1000

_MAX_YEAR = 2999

# Encodings tried, in order, before falling back to Latin-1 (which accepts any
# byte sequence). Windows-1252 covers files saved by Excel on Windows.
_TEXT_ENCODINGS = ("utf-8-sig", "cp1252")

_CSV_DELIMITERS = ",;\t|"

_SNIFF_SAMPLE_CHARS = 64 * 1024


def decode_text_bytes(data: bytes) -> str:
    """Decode uploaded text without silently dropping characters.

    UTF-8 (with or without a byte-order mark) is tried first, then
    Windows-1252, and finally Latin-1, which accepts any byte sequence.

    Args:
        data: The raw file content.

    Returns:
        The decoded text.
    """
    for encoding in _TEXT_ENCODINGS:
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1")


def _sniff_delimiter(text: str) -> str:
    """Guess the delimiter of CSV text, defaulting to a comma.

    Args:
        text: The decoded CSV content.

    Returns:
        One of the supported delimiters (comma, semicolon, tab or pipe).
    """
    sample = text[:_SNIFF_SAMPLE_CHARS]
    if len(text) > _SNIFF_SAMPLE_CHARS and "\n" in sample:
        # Only sniff complete lines; a truncated last line skews the guess.
        sample = sample[: sample.rfind("\n")]
    try:
        return (
            csv.Sniffer().sniff(sample, delimiters=_CSV_DELIMITERS).delimiter
        )
    except csv.Error:
        return ","


def read_uploaded_table(filename: str, data: bytes) -> pd.DataFrame:
    """Read an uploaded CSV or Excel (.xlsx) file into a pandas DataFrame.

    CSV files may use a comma, semicolon, tab or pipe as delimiter and may be
    encoded in UTF-8, Windows-1252 or Latin-1.

    Args:
        filename: The uploaded file name.
        data: The raw file content.

    Returns:
        The loaded pandas DataFrame.

    Raises:
        ValueError: If the file format is not supported.
    """
    lower_name = filename.lower()
    if lower_name.endswith(".csv"):
        text = decode_text_bytes(data)
        return pd.read_csv(io.StringIO(text), sep=_sniff_delimiter(text))
    if lower_name.endswith(".xlsx"):
        return pd.read_excel(io.BytesIO(data))
    if lower_name.endswith(".xls"):
        raise ValueError(
            "Legacy Excel files (.xls) are not supported. Please save the "
            "file as .xlsx or .csv and upload it again."
        )
    raise ValueError("Unsupported tabular file format.")


def load_zip_texts(zip_bytes: bytes) -> pd.DataFrame:
    """Load non-empty text files from a ZIP archive into a DataFrame.

    Files whose basename starts with ``._`` are ignored. Each file is decoded
    with :func:`decode_text_bytes`.

    Args:
        zip_bytes: The ZIP archive content as bytes.

    Returns:
        A pandas DataFrame with columns ``Filename`` and ``Text``.
    """
    data = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as archive:
        for info in upload_safety.safe_zip_members(
            archive, allowed_extensions={".txt"}
        ):
            filename = info.filename
            if os.path.basename(filename).startswith("._"):
                continue
            content = decode_text_bytes(archive.read(info))
            if content.strip():
                data.append({"Filename": filename, "Text": content})
    return pd.DataFrame(data, columns=["Filename", "Text"])


def drop_empty_text_rows(df: pd.DataFrame, text_column: str) -> pd.DataFrame:
    """Remove rows with missing values in the selected text column.

    Args:
        df: The input DataFrame.
        text_column: The column containing the source text.

    Returns:
        A cleaned DataFrame with reset index.

    Raises:
        ValueError: If no valid rows remain after filtering.
    """
    cleaned_df = df.dropna(subset=[text_column]).reset_index(drop=True)

    if cleaned_df.empty:
        raise ValueError(
            "No valid documents remain after removing empty text rows."
        )

    return cleaned_df


def _parse_timestamp_column(values: pd.Series) -> pd.Series:
    """Parse a column of dates, timestamps or years into naive datetimes.

    Numeric columns are only accepted when they hold whole years (e.g. 2019),
    which are mapped to 1 January of that year. pandas would otherwise read
    numbers as nanoseconds since 1970, collapsing every row into the same
    instant without any error.

    Args:
        values: The raw timestamp column.

    Returns:
        A datetime Series in which unparseable values are ``NaT``.

    Raises:
        ValueError: If the column is numeric but does not contain whole years.
    """
    if pd.api.types.is_numeric_dtype(values):
        numbers = values.dropna()
        is_year = (numbers % 1 == 0) & numbers.between(_MIN_YEAR, _MAX_YEAR)
        if not is_year.all():
            raise ValueError(
                "The timestamp column contains numbers that are not years. "
                "Use a column with dates (e.g. 2019-05-31) or whole years "
                "(e.g. 2019)."
            )
        years = values.astype("Int64").astype("string")
        return pd.to_datetime(years, format="%Y", errors="coerce")

    return pd.to_datetime(values, errors="coerce", utc=True).dt.tz_localize(
        None
    )


def prepare_timestamps(
    df: pd.DataFrame,
    date_column: str,
) -> tuple[pd.DataFrame, list[pd.Timestamp], int]:
    """Parse and validate timestamps from a selected date column.

    Dates, date-times and whole years (numeric or text) are supported.

    Args:
        df: The input DataFrame.
        date_column: The column containing timestamps, dates or years.

    Returns:
        A tuple containing:
            - The filtered and sorted DataFrame
            - A list of parsed timestamps
            - The number of dropped rows

    Raises:
        ValueError: If the column is numeric but does not hold years, or if
            no valid timestamps remain after parsing.
    """
    parsed = _parse_timestamp_column(df[date_column])
    valid_mask = parsed.notna()
    dropped = int((~valid_mask).sum())

    filtered_df = df.loc[valid_mask].copy()
    filtered_df[date_column] = parsed.loc[valid_mask]
    filtered_df = filtered_df.sort_values(date_column).reset_index(drop=True)

    if filtered_df.empty:
        raise ValueError(
            "No valid timestamps remained after parsing the selected "
            "timestamp column."
        )

    return filtered_df, filtered_df[date_column].tolist(), dropped


def resolve_time_bins(
    timestamps: list[Any], requested_bins: int
) -> int | None:
    """Decide how many intervals the topics-over-time analysis should use.

    Args:
        timestamps: The parsed document timestamps.
        requested_bins: The number of intervals selected by the user.

    Returns:
        ``requested_bins`` if there are more distinct timestamps than that,
        otherwise ``None`` so that every distinct timestamp is kept as its own
        point instead of being spread over mostly empty intervals.
    """
    return requested_bins if len(set(timestamps)) > requested_bins else None


def validate_minimum_documents(
    texts: list[str], minimum_docs: int = 5
) -> None:
    """Validate that the dataset contains a minimum number of documents.

    Args:
        texts: The raw text documents.
        minimum_docs: The minimum number of required documents.

    Raises:
        ValueError: If too few documents are provided.
    """
    if len(texts) < minimum_docs:
        raise ValueError(
            f"A minimum of {minimum_docs} valid documents is required to "
            "perform topic modeling."
        )


def load_table(filename: str, data: bytes) -> pd.DataFrame:
    """Read an uploaded table, dropping rows that are entirely empty.

    Args:
        filename: The uploaded file name, which picks the parser.
        data: The file's content.

    Returns:
        The table.

    Raises:
        ValueError: If the format is not supported or the table is empty.
    """
    table = read_uploaded_table(filename, data).dropna(how="all")
    if table.empty or len(table.columns) == 0:
        raise ValueError(
            "The uploaded file does not contain any usable rows or columns."
        )
    return table


def load_archive(data: bytes) -> pd.DataFrame:
    """Read the text files of an uploaded ZIP archive.

    Args:
        data: The archive's content.

    Returns:
        A table with the columns ``Filename`` and ``Text``.

    Raises:
        ValueError: If the archive holds no non-empty ``.txt`` file.
    """
    table = load_zip_texts(data)
    if table.empty:
        raise ValueError(
            "The ZIP archive does not contain any non-empty .txt files."
        )
    return table
