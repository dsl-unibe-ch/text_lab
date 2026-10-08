"""Plot tool edge cases seen in agent runs, on a tiny in-memory dataset."""

import conftest_path  # noqa: F401

import os

import pandas as pd
import plotly.io as pio
import pytest

from core.visualization import viz_utils
from core.visualization.plot_interactive import plot_barchart_impl


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
    assert sorted(fig.data[0].y) == [2, 2]
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
