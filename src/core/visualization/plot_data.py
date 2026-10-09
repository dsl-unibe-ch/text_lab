"""
Data Exploration Module for the AI Visualization Engine.
Provides tools for the LLM to inspect dataset columns before plotting.
"""

import pandas as pd

from core.visualization.viz_config import MAX_ROWS
from core.visualization.viz_utils import load_data_safely, shorten_text, was_last_load_truncated

# A non-numeric column whose values average more than this many characters is
# reported as free text (no sample values) rather than as categorical.
TEXT_COLUMN_MIN_AVG_CHARS: int = 50
# Maximum characters shown per sample value in schema summaries.
SAMPLE_VALUE_MAX_CHARS: int = 40
# Rows inspected when measuring average text length (keeps wide files fast).
TEXT_LENGTH_SAMPLE_ROWS: int = 1000


def _average_text_length(series: pd.Series) -> float:
    """Return the mean string length of the first non-null values of ``series``."""
    sample = series.dropna().head(TEXT_LENGTH_SAMPLE_ROWS)
    if sample.empty:
        return 0.0
    return float(sample.astype(str).str.len().mean())


def get_column_summary_impl(data_file_path: str, column: str) -> str:
    """
    Analyzes a specific column in the dataset and returns a statistical summary.
    
    For numeric columns, it calculates the minimum, maximum, mean, and null count.
    For categorical columns, it calculates the number of unique values, lists 
    the top 10 unique values, and the null count.

    Args:
        data_file_path: The absolute path to the data file.
        column: The name of the column to analyze.

    Returns:
        A formatted string containing the column statistics, or an error message
        if the column cannot be found or analyzed.
    """
    try:
        df = load_data_safely(data_file_path)

        truncation_note = ""
        if was_last_load_truncated(data_file_path):
            truncation_note = (
                f"\n\nNote: dataset was truncated to the first {MAX_ROWS:,} rows for "
                "memory safety. Mention this in your final summary."
            )

        if column not in df.columns:
            return f"Error: Column '{column}' not found in the dataset."
            
        col_data = df[column]
        null_count = col_data.isna().sum()
        
        if pd.api.types.is_numeric_dtype(col_data):
            min_val = col_data.min()
            max_val = col_data.max()
            mean_val = col_data.mean()
            
            return (
                f"Numeric Column '{column}': Min={min_val}, Max={max_val}, "
                f"Mean={mean_val:.2f}, Nulls={null_count}{truncation_note}"
            )
        else:
            unique_vals = col_data.unique()
            total_unique = len(unique_vals)
            
            shown = [shorten_text(v, SAMPLE_VALUE_MAX_CHARS) for v in unique_vals[:10]]
            val_str = ", ".join(shown)
            if total_unique > 10:
                val_str += f"... (+ {total_unique - 10} more)"
                
            return (
                f"Categorical Column '{column}': {total_unique} unique values. "
                f"Top values: {val_str}. Nulls={null_count}{truncation_note}"
            )

    except Exception as e:
        return f"Error analyzing column '{column}': {str(e)}"


def get_all_columns_summary_impl(data_file_path: str) -> str:
    """
    Returns a compact schema of every column grouped by type.

    Intentionally terse to minimise token load on the model. Categorical
    columns show a few shortened sample values; long free-text columns only
    show their average length, so no raw text reaches the model's prompt.
    Use get_column_summary for detailed stats on a specific column.
    """
    try:
        df = load_data_safely(data_file_path)

        truncation_note = ""
        if was_last_load_truncated(data_file_path):
            truncation_note = f" (truncated to {MAX_ROWS:,} rows)"

        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        datetime_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
        other_cols = [c for c in df.columns if c not in numeric_cols and c not in datetime_cols]
        text_lengths = {c: _average_text_length(df[c]) for c in other_cols}
        text_cols = [c for c in other_cols if text_lengths[c] > TEXT_COLUMN_MIN_AVG_CHARS]
        categorical_cols = [c for c in other_cols if c not in text_cols]

        lines = [
            f"Dataset: {len(df):,} rows × {len(df.columns)} columns{truncation_note}",
            f"Numeric columns ({len(numeric_cols)}): {', '.join(numeric_cols)}",
        ]
        if categorical_cols:
            cat_details = []
            for c in categorical_cols:
                unique_vals = df[c].dropna().unique()
                sample = ", ".join(
                    shorten_text(v, SAMPLE_VALUE_MAX_CHARS)
                    for v in sorted(unique_vals, key=str)[:5]
                )
                cat_details.append(f"{c} [{sample}]")
            lines.append(f"Categorical columns ({len(categorical_cols)}): {'; '.join(cat_details)}")
        if text_cols:
            text_details = "; ".join(f"{c} (avg {text_lengths[c]:.0f} chars)" for c in text_cols)
            lines.append(
                f"Text columns ({len(text_cols)}, free text: use for word clouds, "
                f"not for statistics): {text_details}"
            )
        if datetime_cols:
            lines.append(f"Datetime columns ({len(datetime_cols)}): {', '.join(datetime_cols)}")

        return "\n".join(lines)

    except Exception as e:
        return f"Error summarizing columns: {str(e)}"