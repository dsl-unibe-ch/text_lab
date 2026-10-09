"""Plot tool edge cases seen in agent runs, on a tiny in-memory dataset."""

import conftest_path  # noqa: F401

import base64
import os

import numpy as np
import pandas as pd
import plotly.io as pio
import pytest

from core.visualization import viz_utils
from core.visualization.plot_data import get_all_columns_summary_impl
from core.visualization.plot_interactive import generate_custom_plotly_impl, plot_barchart_impl


def _trace_values(values):
    """Return trace data as a list, decoding Plotly 6's typed-array encoding.

    Plotly >= 6 serialises numeric arrays as ``{"dtype": ..., "bdata": ...}``
    and ``pio.read_json`` returns that dict unchanged.
    """
    if hasattr(values, "keys") and "bdata" in values:
        raw = base64.b64decode(values["bdata"])
        return np.frombuffer(raw, dtype=values["dtype"]).tolist()
    return list(values)


@pytest.fixture
def data_file(tmp_path):
    path = tmp_path / "uploaded_data.csv"
    pd.DataFrame({
        "diagnosis": ["M", "B", "B", "M"],
        "radius_mean": [17.0, 12.0, 13.0, 20.0],
    }).to_csv(path, index=False)
    return str(path)


def test_long_plot_names_are_shortened_but_stay_distinct(data_file):
    columns = "_".join(f"feature_number_{i}_mean" for i in range(30))
    first = viz_utils.get_plot_path(data_file, f"corr_heatmap_{columns}_a", ext=".png")
    second = viz_utils.get_plot_path(data_file, f"corr_heatmap_{columns}_b", ext=".png")

    assert len(os.path.basename(first)) < 255
    assert first != second
    assert os.path.basename(first).startswith("corr_heatmap_feature_number_0")


def test_short_plot_names_are_unchanged(data_file):
    path = viz_utils.get_plot_path(data_file, "hist radius_mean", ext=".json")
    assert os.path.basename(path) == "hist_radius_mean.json"


def test_barchart_counts_rows_when_y_is_the_grouping_column(data_file):
    output = plot_barchart_impl(data_file, "diagnosis", "diagnosis", "Count", aggregation="count")

    assert not output.startswith("Error"), output
    path, code = output.split("|||", 1)
    fig = pio.read_json(path)
    assert sorted(_trace_values(fig.data[0].y)) == [2, 2]
    assert ".size().reset_index(name='count')" in code


def test_barchart_drops_colour_equal_to_the_x_column(data_file):
    output = plot_barchart_impl(
        data_file, "diagnosis", "radius_mean", "Radius", color_column="diagnosis"
    )

    assert not output.startswith("Error"), output
    assert "color=" not in output.split("|||", 1)[1]


def test_barchart_regular_aggregation_still_works(data_file):
    output = plot_barchart_impl(data_file, "diagnosis", "radius_mean", "Radius")

    assert not output.startswith("Error"), output
    assert "['radius_mean'].mean()" in output.split("|||", 1)[1]


@pytest.fixture
def text_data_file(tmp_path):
    path = tmp_path / "uploaded_data.csv"
    long_text = "This review talks about the product <b>at length</b> & more. " * 20
    pd.DataFrame({
        "rating": [1, 5, 4],
        "sentiment": ["neg", "pos", "pos"],
        "review": [long_text, long_text + " again", "Short\nmulti-line " + long_text],
    }).to_csv(path, index=False)
    return str(path)


def test_schema_reports_long_text_columns_without_raw_text(text_data_file):
    schema = get_all_columns_summary_impl(text_data_file)

    assert "Text columns (1, free text" in schema
    assert "review (avg" in schema
    assert "Categorical columns (1): sentiment [neg, pos]" in schema
    assert "at length" not in schema


def test_data_preview_shortens_text_cells_to_one_line(text_data_file):
    df = pd.read_csv(text_data_file)
    preview = viz_utils.format_data_preview(df, max_cell_chars=30)

    assert "at length" not in preview
    assert len(preview.splitlines()) == len(df) + 1  # header + one line per row
    assert "neg" in preview and "5" in preview


def test_custom_code_cannot_modify_the_cached_dataset(data_file):
    code = (
        "df.drop(columns=['radius_mean'], inplace=True)\n"
        "fig = px.histogram(df, x='diagnosis')"
    )
    output = generate_custom_plotly_impl(data_file, code, "mutating")

    assert not output.startswith("Error"), output
    assert "radius_mean" in viz_utils.load_data_safely(data_file).columns
