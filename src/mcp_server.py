import os
import uuid
import shutil

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt

from mcp.server.fastmcp import FastMCP

# -----------------------------------------------------------------------------
# Server setup
# -----------------------------------------------------------------------------
mcp = FastMCP("Data Scientist MCP")

# In-memory dataset registry
_DATASETS: dict[str, pd.DataFrame] = {}

# Default artifacts directory for plots / exports
_ARTIFACTS_ROOT = os.environ.get("MCP_ARTIFACTS_DIR", "./mcp_artifacts")
os.makedirs(_ARTIFACTS_ROOT, exist_ok=True)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _new_dataset_id(prefix: str = "ds") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def _ensure_output_dir(dataset_id: str, sub: str | None = None) -> str:
    base = os.path.join(_ARTIFACTS_ROOT, dataset_id)
    if sub:
        base = os.path.join(base, sub)
    os.makedirs(base, exist_ok=True)
    return base


def _safe_get_df(dataset_id: str) -> pd.DataFrame:
    if dataset_id not in _DATASETS:
        raise ValueError(f"Unknown dataset_id: {dataset_id}. Load or register a dataset first.")
    return _DATASETS[dataset_id]


def _infer_column_roles(df: pd.DataFrame) -> dict:
    """Infer basic roles: numeric, categorical, datetime."""
    roles = {"numeric": [], "categorical": [], "datetime": [], "other": []}
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_datetime64_any_dtype(s):
            roles["datetime"].append(col)
        elif pd.api.types.is_numeric_dtype(s):
            roles["numeric"].append(col)
        elif pd.api.types.is_categorical_dtype(s) or s.dtype == "object":
            # treat low-cardinality object as categorical
            nunique = s.nunique(dropna=True)
            if nunique <= max(20, int(len(s) * 0.05)):
                roles["categorical"].append(col)
            else:
                # high-cardinality strings often act like "other" until proven useful (e.g., IDs)
                roles["other"].append(col)
        else:
            roles["other"].append(col)
    return roles


def _sniff_and_read(path: str, sample_rows: int = 5000) -> pd.DataFrame:
    """Read CSV/TSV/Parquet/JSON with a few conveniences."""
    p = path.lower()
    if p.endswith(".parquet"):
        return pd.read_parquet(path)
    if p.endswith(".json") or p.endswith(".jsonl"):
        orient = "records"
        lines = p.endswith(".jsonl")
        return pd.read_json(path, orient=orient, lines=lines)
    if p.endswith(".tsv"):
        return pd.read_csv(path, sep="\t", low_memory=False)
    if p.endswith(".csv") or any(p.endswith(ext) for ext in [".txt", ".data"]):
        # Try to sniff delimiter
        try:
            import csv
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                sample = "".join([next(f) for _ in range(20)])
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
            sep = dialect.delimiter
        except Exception:
            sep = ","
        return pd.read_csv(path, sep=sep, low_memory=False)
    # Fallback: attempt pandas auto
    return pd.read_table(path, low_memory=False)


def _mk_figure(figsize=(8, 5)):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    return fig, ax


def _save_fig(fig: plt.Figure, out_dir: str, slug: str) -> str:
    fname = f"{slug}.png"
    path = os.path.join(out_dir, fname)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _to_markdown_table(df: pd.DataFrame, max_rows: int = 10) -> str:
    return df.head(max_rows).to_markdown(index=False)


# -----------------------------------------------------------------------------
# Tools
# -----------------------------------------------------------------------------

@mcp.tool()
def register_dataset_from_file(file_path: str, dataset_id: str | None = None) -> dict:
    """Register a dataset from a local file.

    Reads CSV/TSV/Parquet/JSON into memory and assigns a `dataset_id`.

    Args:
        file_path: Path to a local tabular file (CSV/TSV/Parquet/JSON/JSONL/TXT).
        dataset_id: Optional custom ID. If not provided, one will be generated.

    Returns:
        dict: { "dataset_id": str, "shape": [rows, cols], "columns": [..], "preview_markdown": str }

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file cannot be parsed as a supported tabular format.

    Examples:
        >>> register_dataset_from_file("data/customers.csv")
        {'dataset_id': 'ds-1a2b3c4d', 'shape': [1000, 12], 'columns': [...], 'preview_markdown': '...'}
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    df = _sniff_and_read(file_path)

    # Try datetime inference for object columns
    for col in df.columns:
        s = df[col]
        if s.dtype == "object":
            try:
                parsed = pd.to_datetime(s, errors="raise", utc=False)
                # treat as datetime only if a majority parsed successfully
                valid_ratio = (~pd.isna(parsed)).mean()
                if valid_ratio > 0.9:
                    df[col] = parsed
            except Exception:
                pass

    dsid = dataset_id or _new_dataset_id()
    _DATASETS[dsid] = df

    return {
        "dataset_id": dsid,
        "shape": [int(df.shape[0]), int(df.shape[1])],
        "columns": df.columns.tolist(),
        "preview_markdown": _to_markdown_table(df, max_rows=10),
    }


@mcp.tool()
def list_datasets() -> list[dict]:
    """List registered datasets.

    Returns:
        list[dict]: Each item includes dataset_id, shape, and first few columns.

    Examples:
        >>> list_datasets()
        [{'dataset_id': 'ds-1a2b3c4d', 'shape': [1000, 12], 'columns_sample': ['id','age','income']}]
    """
    items: list[dict] = []
    for dsid, df in _DATASETS.items():
        items.append({
            "dataset_id": dsid,
            "shape": [int(df.shape[0]), int(df.shape[1])],
            "columns_sample": df.columns[:8].tolist(),
        })
    return items


@mcp.tool()
def drop_dataset(dataset_id: str) -> bool:
    """Remove a dataset from memory and its artifact folder.

    Args:
        dataset_id: The dataset ID to remove.

    Returns:
        bool: True if removed, False if ID not found.

    Examples:
        >>> drop_dataset("ds-1a2b3c4d")
        True
    """
    existed = dataset_id in _DATASETS
    _DATASETS.pop(dataset_id, None)
    art_dir = os.path.join(_ARTIFACTS_ROOT, dataset_id)
    if os.path.isdir(art_dir):
        shutil.rmtree(art_dir, ignore_errors=True)
    return existed


@mcp.tool()
def head(dataset_id: str, n: int = 10) -> dict:
    """Return a preview of the dataset.

    Args:
        dataset_id: Registered dataset ID.
        n: Number of rows to preview.

    Returns:
        dict: { "shape": [rows, cols], "markdown": str }

    Examples:
        >>> head("ds-1a2b3c4d", n=5)
        {'shape': [1000, 12], 'markdown': '| col1 | col2 | ...'}
    """
    df = _safe_get_df(dataset_id)
    return {"shape": [int(df.shape[0]), int(df.shape[1])], "markdown": _to_markdown_table(df, n)}


@mcp.tool()
def schema(dataset_id: str) -> dict:
    """Infer and summarize schema information.

    Computes basic types, null counts, distinct counts, and sample values.

    Args:
        dataset_id: Registered dataset ID.

    Returns:
        dict: {
          "columns": [
            {
              "name": str,
              "dtype": str,
              "role": "numeric|categorical|datetime|other",
              "nulls": int,
              "null_pct": float,
              "distinct": int,
              "sample_values": [.. up to 5 ..]
            }, ...
          ],
          "roles": {"numeric":[...], "categorical":[...], "datetime":[...], "other":[...]}
        }

    Examples:
        >>> schema("ds-1a2b3c4d")
        {'columns': [...], 'roles': {...}}
    """
    df = _safe_get_df(dataset_id)
    roles = _infer_column_roles(df)

    cols = []
    for c in df.columns:
        s = df[c]
        nulls = int(s.isna().sum())
        null_pct = float((nulls / len(s)) * 100) if len(s) else 0.0
        distinct = int(s.nunique(dropna=True))
        dtype = str(s.dtype)
        # role lookup
        if c in roles["numeric"]:
            role = "numeric"
        elif c in roles["categorical"]:
            role = "categorical"
        elif c in roles["datetime"]:
            role = "datetime"
        else:
            role = "other"
        sample = pd.Series(s.dropna().unique()[:5]).astype(str).tolist()
        cols.append({
            "name": c,
            "dtype": dtype,
            "role": role,
            "nulls": nulls,
            "null_pct": round(null_pct, 2),
            "distinct": distinct,
            "sample_values": sample,
        })

    return {"columns": cols, "roles": roles}


@mcp.tool()
def summary_stats(dataset_id: str, columns: list[str] | None = None) -> dict:
    """Compute summary statistics for numeric columns.

    Args:
        dataset_id: Registered dataset ID.
        columns: Optional list of numeric column names. If omitted, all numeric columns are used.

    Returns:
        dict: Keys are column names mapping to summary stats (count, mean, std, min, quartiles, max).

    Raises:
        ValueError: If no numeric columns are available.

    Examples:
        >>> summary_stats("ds-1a2b3c4d", columns=["age","income"])
        {'age': {'count': 1000, 'mean': 35.2, ...}, 'income': {...}}
    """
    df = _safe_get_df(dataset_id)
    if columns is None:
        columns = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    subset = df[columns]
    if subset.shape[1] == 0:
        raise ValueError("No numeric columns to summarize.")

    desc = subset.describe(include=[np.number]).to_dict()
    # pandas describe returns nested dict; restructure a bit
    out: dict[str, dict] = {}
    for stat, mapping in desc.items():
        for col, val in mapping.items():
            out.setdefault(col, {})
            # cast numpy types to Python scalars
            out[col][stat] = None if pd.isna(val) else float(val)
    return out


@mcp.tool()
def suggest_charts(dataset_id: str, max_suggestions: int = 10) -> list[dict]:
    """Suggest suitable charts based on inferred schema.

    Heuristics:
      - numeric: histogram, boxplot
      - datetime + numeric: line plot (agg mean)
      - categorical + numeric: bar plot (agg mean or count)
      - numeric x numeric: scatter
      - correlation heatmap if >= 3 numeric columns

    Args:
        dataset_id: Registered dataset ID.
        max_suggestions: Maximum number of suggestions to return.

    Returns:
        list[dict]: Each suggestion has:
            - chart: str  (e.g., 'histogram', 'box', 'line', 'bar', 'scatter', 'corr_heatmap')
            - columns: list[str]
            - reason: str

    Examples:
        >>> suggest_charts("ds-1a2b3c4d")
        [{'chart':'histogram','columns':['age'], 'reason':'numeric distribution'}, ...]
    """
    df = _safe_get_df(dataset_id)
    roles = _infer_column_roles(df)

    suggestions: list[dict] = []

    # Numeric-only suggestions
    for col in roles["numeric"]:
        suggestions.append({"chart": "histogram", "columns": [col], "reason": "Numeric distribution"})
        suggestions.append({"chart": "box", "columns": [col], "reason": "Numeric spread/outliers"})

    # Numeric vs numeric scatter (limit pair count)
    num_cols = roles["numeric"]
    for i in range(min(len(num_cols), 4)):
        for j in range(i + 1, min(len(num_cols), 4)):
            suggestions.append({
                "chart": "scatter",
                "columns": [num_cols[i], num_cols[j]],
                "reason": "Relationship between two numeric features",
            })

    # Datetime + numeric line
    if roles["datetime"] and roles["numeric"]:
        dt = roles["datetime"][0]
        for num in roles["numeric"][:4]:
            suggestions.append({
                "chart": "line",
                "columns": [dt, num],
                "reason": "Trend over time of numeric variable",
            })

    # Categorical + numeric bar
    if roles["categorical"] and roles["numeric"]:
        cat = roles["categorical"][0]
        for num in roles["numeric"][:4]:
            suggestions.append({
                "chart": "bar",
                "columns": [cat, num],
                "reason": "Mean/Count by category",
            })

    # Correlation heatmap
    if len(num_cols) >= 3:
        suggestions.append({
            "chart": "corr_heatmap",
            "columns": num_cols[:10],
            "reason": "Correlation among numeric features",
        })

    return suggestions[:max_suggestions]


# ----------------------------- Plotting tools --------------------------------

@mcp.tool()
def plot_histogram(dataset_id: str, column: str, bins: int = 30, title: str | None = None) -> dict:
    """Plot a histogram for a numeric column.

    Args:
        dataset_id: Registered dataset ID.
        column: Numeric column to plot.
        bins: Number of histogram bins.
        title: Optional title override.

    Returns:
        dict: { "path": str, "description": str }

    Raises:
        ValueError: If the column is not numeric or missing.

    Examples:
        >>> plot_histogram("ds-1a2b3c4d", "age")
        {'path': 'mcp_artifacts/ds-.../plots/hist_age.png', 'description': 'Histogram of age'}
    """
    df = _safe_get_df(dataset_id)
    if column not in df.columns or not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"Column '{column}' must exist and be numeric.")

    out_dir = _ensure_output_dir(dataset_id, "plots")
    fig, ax = _mk_figure()
    ax.hist(df[column].dropna().values, bins=bins)
    ax.set_xlabel(column)
    ax.set_ylabel("Count")
    ax.set_title(title or f"Histogram of {column}")
    path = _save_fig(fig, out_dir, f"hist_{column}")
    return {"path": path, "description": f"Histogram of {column}"}


@mcp.tool()
def plot_box(dataset_id: str, column: str, title: str | None = None) -> dict:
    """Plot a boxplot for a numeric column.

    Args:
        dataset_id: Registered dataset ID.
        column: Numeric column to plot.
        title: Optional title override.

    Returns:
        dict: { "path": str, "description": str }

    Raises:
        ValueError: If the column is not numeric or missing.

    Examples:
        >>> plot_box("ds", "income")
        {'path': '.../box_income.png', 'description': 'Boxplot of income'}
    """
    df = _safe_get_df(dataset_id)
    if column not in df.columns or not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"Column '{column}' must exist and be numeric.")

    out_dir = _ensure_output_dir(dataset_id, "plots")
    fig, ax = _mk_figure()
    ax.boxplot(df[column].dropna().values, vert=True, labels=[column])
    ax.set_title(title or f"Boxplot of {column}")
    path = _save_fig(fig, out_dir, f"box_{column}")
    return {"path": path, "description": f"Boxplot of {column}"}


@mcp.tool()
def plot_scatter(dataset_id: str, x: str, y: str, hue: str | None = None, title: str | None = None, sample: int | None = 5000) -> dict:
    """Plot a scatter plot of two numeric columns, optionally colored by a categorical column.

    Args:
        dataset_id: Registered dataset ID.
        x: Numeric column for x-axis.
        y: Numeric column for y-axis.
        hue: Optional categorical column for color grouping (up to 10 categories).
        title: Optional title override.
        sample: Optional row sampling cap for performance (None to disable).

    Returns:
        dict: { "path": str, "description": str }

    Raises:
        ValueError: If x/y are not numeric or missing; or hue has too many categories.

    Examples:
        >>> plot_scatter("ds", "age", "income", hue="segment")
        {'path':'.../scatter_age_income.png', 'description':'Scatter age vs income (hue=segment)'}
    """
    df = _safe_get_df(dataset_id)
    for col in [x, y]:
        if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"Column '{col}' must exist and be numeric.")

    if sample is not None and len(df) > sample:
        df = df.sample(sample, random_state=42)

    out_dir = _ensure_output_dir(dataset_id, "plots")
    fig, ax = _mk_figure(figsize=(7, 6))

    if hue and hue in df.columns:
        cats = df[hue].astype("category")
        cats = cats.cat.remove_unused_categories()
        if len(cats.cat.categories) > 10:
            raise ValueError(f"hue column '{hue}' has too many categories (>10)")
        for cat in cats.cat.categories:
            mask = cats == cat
            ax.scatter(df.loc[mask, x], df.loc[mask, y], label=str(cat), s=10, alpha=0.8)
        ax.legend(title=hue, fontsize=8)
        desc = f"Scatter {x} vs {y} (hue={hue})"
        slug = f"scatter_{x}_{y}_by_{hue}"
    else:
        ax.scatter(df[x], df[y], s=10, alpha=0.8)
        desc = f"Scatter {x} vs {y}"
        slug = f"scatter_{x}_{y}"

    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(title or desc)
    path = _save_fig(fig, out_dir, slug)
    return {"path": path, "description": desc}


@mcp.tool()
def plot_line(dataset_id: str, x: str, y: str, agg: str = "mean", title: str | None = None) -> dict:
    """Plot a line chart for a datetime x and numeric y (aggregated by x).

    Args:
        dataset_id: Registered dataset ID.
        x: Datetime column for x-axis.
        y: Numeric column for y-axis.
        agg: Aggregation ('mean','sum','median','min','max').
        title: Optional title override.

    Returns:
        dict: { "path": str, "description": str }

    Raises:
        ValueError: If x is not datetime or y is not numeric.

    Examples:
        >>> plot_line("ds","date","sales","sum")
        {'path':'.../line_date_sales.png', 'description':'Sum of sales over date'}
    """
    df = _safe_get_df(dataset_id)
    if x not in df.columns or not pd.api.types.is_datetime64_any_dtype(df[x]):
        raise ValueError(f"Column '{x}' must exist and be datetime-like.")
    if y not in df.columns or not pd.api.types.is_numeric_dtype(df[y]):
        raise ValueError(f"Column '{y}' must exist and be numeric.")

    group = df.dropna(subset=[x, y]).groupby(x)[y]
    if agg not in {"mean", "sum", "median", "min", "max"}:
        raise ValueError("agg must be one of 'mean','sum','median','min','max'")
    series = getattr(group, agg)().sort_index()

    out_dir = _ensure_output_dir(dataset_id, "plots")
    fig, ax = _mk_figure(figsize=(8, 4))
    ax.plot(series.index, series.values)
    ax.set_xlabel(x)
    ax.set_ylabel(f"{agg}({y})")
    desc = f"{agg.capitalize()} of {y} over {x}"
    ax.set_title(title or desc)
    path = _save_fig(fig, out_dir, f"line_{x}_{y}_{agg}")
    return {"path": path, "description": desc}


@mcp.tool()
def plot_bar(dataset_id: str, x: str, y: str | None = None, agg: str = "count", top_k: int = 20, title: str | None = None) -> dict:
    """Plot a bar chart for categorical vs numeric (aggregated) or pure counts of a categorical.

    Args:
        dataset_id: Registered dataset ID.
        x: Categorical column for x-axis.
        y: Optional numeric column to aggregate per category. If None, counts per category are plotted.
        agg: Aggregation if y is provided ('mean','sum','median','min','max','count').
        top_k: Limit categories to top K by count/aggregate.
        title: Optional title override.

    Returns:
        dict: { "path": str, "description": str }

    Raises:
        ValueError: If x is missing or y is not numeric when provided.

    Examples:
        >>> plot_bar("ds","segment","sales","mean", top_k=10)
        {'path':'.../bar_segment_sales_mean.png', 'description':'Mean sales by segment (top 10)'}
    """
    df = _safe_get_df(dataset_id)
    if x not in df.columns:
        raise ValueError(f"Column '{x}' must exist.")
    if y is not None and (y not in df.columns or not pd.api.types.is_numeric_dtype(df[y])):
        raise ValueError(f"Column '{y}' must be numeric when provided.")

    out_dir = _ensure_output_dir(dataset_id, "plots")
    fig, ax = _mk_figure(figsize=(9, 5))

    if y is None:
        counts = df[x].value_counts(dropna=False).head(top_k)
        ax.bar(counts.index.astype(str), counts.values)
        ax.set_ylabel("Count")
        desc = f"Count by {x} (top {top_k})"
        slug = f"bar_{x}_count"
    else:
        if agg not in {"mean", "sum", "median", "min", "max", "count"}:
            raise ValueError("agg must be one of 'mean','sum','median','min','max','count'")
        grouped = df.dropna(subset=[x, y]).groupby(x)[y]
        vals = getattr(grouped, agg)().sort_values(ascending=False).head(top_k)
        ax.bar(vals.index.astype(str), vals.values)
        ax.set_ylabel(f"{agg}({y})")
        desc = f"{agg.capitalize()} of {y} by {x} (top {top_k})"
        slug = f"bar_{x}_{y}_{agg}"

    ax.set_xlabel(x)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_title(title or desc)
    path = _save_fig(fig, out_dir, slug)
    return {"path": path, "description": desc}


@mcp.tool()
def plot_corr_heatmap(dataset_id: str, columns: list[str] | None = None, title: str | None = None) -> dict:
    """Plot a correlation heatmap for numeric columns.

    Args:
        dataset_id: Registered dataset ID.
        columns: Optional subset of numeric columns to include. If None, all numeric columns are used.
        title: Optional title override.

    Returns:
        dict: { "path": str, "description": str }

    Raises:
        ValueError: If fewer than 2 numeric columns are available.

    Examples:
        >>> plot_corr_heatmap("ds", columns=["age","income","score"])
        {'path':'.../corr_heatmap.png', 'description':'Correlation heatmap of 3 variables'}
    """
    df = _safe_get_df(dataset_id)
    if columns is None:
        columns = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    else:
        columns = [c for c in columns if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]

    if len(columns) < 2:
        raise ValueError("Need at least 2 numeric columns to plot a correlation heatmap.")

    corr = df[columns].corr(numeric_only=True)

    out_dir = _ensure_output_dir(dataset_id, "plots")
    fig, ax = _mk_figure(figsize=(0.7 * len(columns) + 3, 0.7 * len(columns) + 3))
    cax = ax.imshow(corr.values, interpolation="nearest")
    ax.set_xticks(range(len(columns)))
    ax.set_yticks(range(len(columns)))
    ax.set_xticklabels(columns, rotation=45, ha="right")
    ax.set_yticklabels(columns)
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title or "Correlation Heatmap")
    path = _save_fig(fig, out_dir, "corr_heatmap")
    return {"path": path, "description": f"Correlation heatmap of {len(columns)} variables"}


@mcp.tool()
def auto_explore(dataset_id: str, max_plots: int = 6) -> dict:
    """Automatically create a small set of exploratory plots.

    Uses `suggest_charts` and renders a curated set (up to `max_plots`).

    Args:
        dataset_id: Registered dataset ID.
        max_plots: Maximum number of plots to generate.

    Returns:
        dict: {
          "generated": [ { "chart": str, "columns": [...], "path": str, "description": str }, ... ],
          "skipped": [ { "chart": str, "columns": [...], "reason": str }, ... ]
        }

    Examples:
        >>> auto_explore("ds", max_plots=5)
        {'generated': [...], 'skipped': [...]}
    """
    sugg = suggest_charts(dataset_id, max_suggestions=20)
    generated: list[dict] = []
    skipped: list[dict] = []

    for s in sugg:
        if len(generated) >= max_plots:
            break
        chart = s["chart"]
        cols = s["columns"]
        try:
            if chart == "histogram":
                res = plot_histogram(dataset_id, column=cols[0])
            elif chart == "box":
                res = plot_box(dataset_id, column=cols[0])
            elif chart == "scatter":
                res = plot_scatter(dataset_id, x=cols[0], y=cols[1])
            elif chart == "line":
                res = plot_line(dataset_id, x=cols[0], y=cols[1], agg="mean")
            elif chart == "bar":
                res = plot_bar(dataset_id, x=cols[0], y=cols[1] if len(cols) > 1 else None)
            elif chart == "corr_heatmap":
                res = plot_corr_heatmap(dataset_id, columns=cols)
            else:
                skipped.append({"chart": chart, "columns": cols, "reason": "Unknown chart type"})
                continue

            generated.append({"chart": chart, "columns": cols, **res})
        except Exception as e:
            skipped.append({"chart": chart, "columns": cols, "reason": str(e)})

    return {"generated": generated, "skipped": skipped}


@mcp.tool()
def export_dataset(dataset_id: str, out_path: str) -> dict:
    """Export a registered dataset to disk.

    Format inferred from file extension: .csv, .parquet, .json.

    Args:
        dataset_id: Registered dataset ID.
        out_path: Destination path, e.g., 'mcp_artifacts/ds-x/export.csv'

    Returns:
        dict: { "path": str, "rows": int, "cols": int }

    Raises:
        ValueError: If extension is unsupported.

    Examples:
        >>> export_dataset("ds-1a2b3c4d", "mcp_artifacts/ds-1a2b3c4d/export.parquet")
        {'path': '...', 'rows': 1000, 'cols': 12}
    """
    df = _safe_get_df(dataset_id)
    out_dir = os.path.dirname(out_path) or "."
    os.makedirs(out_dir, exist_ok=True)

    p = out_path.lower()
    if p.endswith(".csv"):
        df.to_csv(out_path, index=False)
    elif p.endswith(".parquet"):
        df.to_parquet(out_path, index=False)
    elif p.endswith(".json"):
        df.to_json(out_path, orient="records")
    else:
        raise ValueError("Unsupported export format. Use .csv, .parquet, or .json")

    return {"path": out_path, "rows": int(df.shape[0]), "cols": int(df.shape[1])}


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("--- Data Scientist MCP server over stdio ---")
    print(f"Artifacts directory: {_ARTIFACTS_ROOT}")
    mcp.run(transport="stdio")
