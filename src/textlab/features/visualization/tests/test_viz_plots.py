"""Plot and stats tool behaviour (incl. optional R code), on tiny datasets."""

import base64
import os

import numpy as np
import pandas as pd
import plotly.io as pio
import pytest
from scipy import stats as stats_module

from textlab.features.visualization import (
    plot_interactive,
    plot_static,
    r_code,
    stats_analysis,
    viz_utils,
)
from textlab.features.visualization.plot_data import (
    get_all_columns_summary_impl,
)
from textlab.features.visualization.plot_interactive import (
    generate_custom_plotly_impl,
    plot_barchart_impl,
)


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
    pd.DataFrame(
        {
            "diagnosis": ["M", "B", "B", "M"],
            "radius_mean": [17.0, 12.0, 13.0, 20.0],
        }
    ).to_csv(path, index=False)
    return str(path)


def test_long_plot_names_are_shortened_but_stay_distinct(data_file):
    columns = "_".join(f"feature_number_{i}_mean" for i in range(30))
    first = viz_utils.get_plot_path(
        data_file, f"corr_heatmap_{columns}_a", ext=".png"
    )
    second = viz_utils.get_plot_path(
        data_file, f"corr_heatmap_{columns}_b", ext=".png"
    )

    assert len(os.path.basename(first)) < 255
    assert first != second
    assert os.path.basename(first).startswith("corr_heatmap_feature_number_0")


def test_short_plot_names_are_unchanged(data_file):
    path = viz_utils.get_plot_path(data_file, "hist radius_mean", ext=".json")
    assert os.path.basename(path) == "hist_radius_mean.json"


def test_barchart_counts_rows_when_y_is_the_grouping_column(data_file):
    output = plot_barchart_impl(
        data_file, "diagnosis", "diagnosis", "Count", aggregation="count"
    )

    assert not output.startswith("Error"), output
    path, code = output.split("|||", 1)
    fig = pio.read_json(path)
    assert sorted(_trace_values(fig.data[0].y)) == [2, 2]
    assert ".size().reset_index(name='count')" in code


def test_barchart_drops_colour_equal_to_the_x_column(data_file):
    output = plot_barchart_impl(
        data_file,
        "diagnosis",
        "radius_mean",
        "Radius",
        color_column="diagnosis",
    )

    assert not output.startswith("Error"), output
    assert "color=" not in output.split("|||", 1)[1]


def test_barchart_regular_aggregation_still_works(data_file):
    output = plot_barchart_impl(
        data_file, "diagnosis", "radius_mean", "Radius"
    )

    assert not output.startswith("Error"), output
    assert "['radius_mean'].mean()" in output.split("|||", 1)[1]


@pytest.fixture
def text_data_file(tmp_path):
    path = tmp_path / "uploaded_data.csv"
    long_text = (
        "This review talks about the product <b>at length</b> & more. " * 20
    )
    pd.DataFrame(
        {
            "rating": [1, 5, 4],
            "sentiment": ["neg", "pos", "pos"],
            "review": [
                long_text,
                long_text + " again",
                "Short\nmulti-line " + long_text,
            ],
        }
    ).to_csv(path, index=False)
    return str(path)


def test_schema_reports_long_text_columns_without_raw_text(text_data_file):
    schema = get_all_columns_summary_impl(text_data_file)

    assert "Text columns (1, free text" in schema
    assert "review (avg" in schema
    assert "Categorical columns (1): sentiment [neg, pos]" in schema
    assert "at length" not in schema


def test_custom_code_cannot_modify_the_cached_dataset(data_file):
    code = (
        "df.drop(columns=['radius_mean'], inplace=True)\n"
        "fig = px.histogram(df, x='diagnosis')"
    )
    output = generate_custom_plotly_impl(data_file, code, "mutating")

    assert not output.startswith("Error"), output
    assert "radius_mean" in viz_utils.load_data_safely(data_file).columns


def test_existing_plot_files_are_never_overwritten(data_file):
    first = viz_utils.get_plot_path(data_file, "hist_radius_mean", ext=".json")
    open(first, "w").close()
    second = viz_utils.get_plot_path(
        data_file, "hist_radius_mean", ext=".json"
    )
    open(second, "w").close()
    third = viz_utils.get_plot_path(data_file, "hist_radius_mean", ext=".json")

    assert os.path.basename(second) == "hist_radius_mean_2.json"
    assert os.path.basename(third) == "hist_radius_mean_3.json"


def test_time_limit_interrupts_a_runaway_loop():
    with pytest.raises(TimeoutError):
        with viz_utils.time_limit(0.2):
            while True:
                pass


def test_endless_custom_plotly_code_is_stopped(data_file, monkeypatch):
    monkeypatch.setattr(plot_interactive, "CUSTOM_CODE_TIMEOUT", 0.3)
    output = generate_custom_plotly_impl(
        data_file, "while True:\n    pass", "loop"
    )

    assert output.startswith("Error") and "did not finish" in output


def test_endless_custom_static_code_is_stopped(data_file, monkeypatch):
    monkeypatch.setattr(plot_static, "CUSTOM_CODE_TIMEOUT", 0.3)
    output = plot_static.generate_custom_static_plot_impl(
        data_file, "while True:\n    pass", "loop"
    )

    assert output.startswith("Error") and "did not finish" in output


@pytest.mark.parametrize("code", ["import sys\nsys.exit(1)", "exit()"])
def test_exit_in_custom_code_returns_an_error_instead_of_stopping(
    data_file, code
):
    output = generate_custom_plotly_impl(data_file, code, "exit")
    static_output = plot_static.generate_custom_static_plot_impl(
        data_file, code, "exit"
    )

    for result in (output, static_output):
        assert result.startswith("Error") and "exit()" in result


def test_custom_code_errors_name_the_line_and_the_columns(data_file):
    code = "x = 1\nfig = px.histogram(df, x=df['Radius_mean'])"
    output = generate_custom_plotly_impl(data_file, code, "typo")

    assert "KeyError: 'Radius_mean'" in output
    assert "At line 2: fig = px.histogram(df, x=df['Radius_mean'])" in output
    assert "Available columns: diagnosis, radius_mean" in output


def test_custom_code_syntax_errors_name_the_line(data_file):
    output = plot_static.generate_custom_static_plot_impl(
        data_file, "x = 1\nif x\n    y = 2", "syntax"
    )

    assert "SyntaxError" in output and "At line 2: if x" in output


# =========================================================================
# R CODE
# =========================================================================


@pytest.fixture
def r_data_file(tmp_path):
    path = tmp_path / "uploaded_data.csv"
    pd.DataFrame(
        {
            "diagnosis": ["M", "B", "B", "M", "B", "M"],
            "radius_mean": [17.0, 12.0, 13.0, 20.0, 11.5, 18.2],
            "concave points_mean": [0.14, 0.05, 0.04, 0.18, 0.03, 0.12],
            "notes": [
                "large irregular mass",
                "small smooth",
                "small round",
                "large irregular",
                "small smooth round",
                "irregular mass",
            ],
        }
    ).to_csv(path, index=False)
    return str(path)


@pytest.fixture
def r_enabled(monkeypatch):
    monkeypatch.setenv(r_code.R_CODE_ENV, "1")


def _plot_calls(path):
    """Every standard plot tool with valid arguments for ``r_data_file``."""
    return {
        "interactive histogram": lambda: plot_interactive.plot_histogram_impl(
            path, "radius_mean", "Radius", "diagnosis"
        ),
        "interactive scatter": lambda: plot_interactive.plot_scatterplot_impl(
            path, "radius_mean", "concave points_mean", "Scatter", "diagnosis"
        ),
        "interactive box": lambda: plot_interactive.plot_boxplot_impl(
            path, "diagnosis", "radius_mean", "Box"
        ),
        "interactive line": lambda: plot_interactive.plot_lineplot_impl(
            path, "radius_mean", "concave points_mean", "Line"
        ),
        "interactive bar": lambda: plot_interactive.plot_barchart_impl(
            path, "diagnosis", "radius_mean", "Bar"
        ),
        "interactive matrix": lambda: (
            plot_interactive.plot_scatter_matrix_impl(
                path, "radius_mean,concave points_mean", "Matrix", "diagnosis"
            )
        ),
        "interactive heatmap": lambda: (
            plot_interactive.plot_correlation_heatmap_impl(
                path, "Heatmap", "pearson", "_mean"
            )
        ),
        "static histogram": lambda: plot_static.plot_static_histogram_impl(
            path, "radius_mean", "Radius", "Radius"
        ),
        "static scatter": lambda: plot_static.plot_static_scatterplot_impl(
            path,
            "radius_mean",
            "concave points_mean",
            "Scatter",
            "R",
            "C",
            "diagnosis",
        ),
        "static box": lambda: plot_static.plot_static_boxplot_impl(
            path, "diagnosis", "radius_mean", "Box", "D", "R"
        ),
        "static line": lambda: plot_static.plot_static_lineplot_impl(
            path, "radius_mean", "concave points_mean", "Line", "R", "C"
        ),
        "static bar": lambda: plot_static.plot_static_barchart_impl(
            path, "diagnosis", "radius_mean", "Bar", "D", "R"
        ),
        "static wordcloud": lambda: plot_static.plot_static_wordcloud_impl(
            path, "notes", "Words"
        ),
        "static pairplot": lambda: plot_static.plot_static_pairplot_impl(
            path, "radius_mean,concave points_mean", "Pairs", "diagnosis"
        ),
        "static heatmap": lambda: (
            plot_static.plot_static_correlation_heatmap_impl(
                path, "Heatmap", "spearman"
            )
        ),
    }


def test_every_standard_plot_tool_returns_r_code_when_enabled(
    r_data_file, r_enabled
):
    for name, call in _plot_calls(r_data_file).items():
        output = call()
        assert not output.startswith("Error"), (name, output)
        parts = output.split("|||")
        assert len(parts) == 3, name
        r_snippet = parts[2]
        assert "check.names = FALSE" in r_snippet, name
        assert (
            "library(" in r_snippet
            and "# The R code could not" not in r_snippet
        ), name


def test_plot_tools_return_no_r_code_by_default(r_data_file):
    output = plot_interactive.plot_histogram_impl(
        r_data_file, "radius_mean", "Radius"
    )
    assert len(output.split("|||")) == 2


def test_custom_code_plots_have_no_r_code(r_data_file, r_enabled):
    output = generate_custom_plotly_impl(
        r_data_file, "fig = px.histogram(df, x='radius_mean')", "custom"
    )
    assert len(output.split("|||")) == 2


def test_t_test_r_code_matches_pingouins_choice_of_test(
    r_data_file, r_enabled
):
    equal = stats_analysis.run_group_comparison_impl(
        r_data_file, "radius_mean", "diagnosis"
    )
    assert "```r\n" in equal
    assert (
        "var.equal = TRUE" in equal
    )  # 3 vs 3 rows: pingouin uses Student's t-test
    assert 'groups <- c("B", "M")' in equal  # same group order as Python


def test_unequal_groups_use_welch_in_r(tmp_path, r_enabled):
    path = tmp_path / "uploaded_data.csv"
    pd.DataFrame(
        {"g": ["a", "a", "a", "b", "b"], "y": [1.0, 2.0, 3.0, 4.0, 6.0]}
    ).to_csv(path, index=False)
    output = stats_analysis.run_group_comparison_impl(str(path), "y", "g")
    assert "var.equal = FALSE" in output


def test_rank_correlations_r_code_repeats_the_binary_encoding(
    r_data_file, r_enabled
):
    output = stats_analysis.rank_target_correlations_impl(
        r_data_file, "diagnosis"
    )
    assert '== "M", 1' in output and '== "B", 0' in output


def test_stats_have_no_r_block_by_default(r_data_file):
    output = stats_analysis.run_correlation_impl(
        r_data_file, "radius_mean", "concave points_mean"
    )
    assert "```r" not in output and "```python" in output


def test_r_literals_quote_names_and_values_safely():
    assert r_code.r_col("concave points_mean") == "`concave points_mean`"
    assert r_code.r_col("a`b") == "`a\\`b`"
    assert r_code.r_str('say "hi"\n') == '"say \\"hi\\"\\n"'
    assert r_code.r_value(np.bool_(True)) == "TRUE"
    assert r_code.r_value(np.int64(3)) == "3"
    assert r_code.r_value("M") == '"M"'


def test_r_generation_errors_never_break_a_tool(r_enabled):
    def broken(*args):
        raise ValueError("boom")

    assert r_code.build(broken) == "# The R code could not be generated: boom"


# =========================================================================
# EXTENDED STATISTICS
# =========================================================================


@pytest.fixture
def survey_file(tmp_path):
    """60 deterministic survey-like rows: numeric, group and binary columns."""
    rng = np.random.RandomState(0)
    faculty = np.repeat(["Arts", "Law", "Science"], 20)
    score = np.concatenate([rng.normal(m, 1.0, 20) for m in (3.0, 3.5, 4.5)])
    sex = np.tile(["F", "M"], 30)
    smoker = np.where(rng.rand(60) < 0.3, "yes", "no")
    age = rng.randint(18, 40, 60)
    dropout = np.where(score + rng.normal(0, 1, 60) < 3.5, "yes", "no")
    path = tmp_path / "survey.csv"
    pd.DataFrame(
        {
            "faculty": faculty,
            "score": score,
            "sex": sex,
            "smoker": smoker,
            "age": age,
            "dropout": dropout,
        }
    ).to_csv(path, index=False)
    return str(path)


def test_parametric_two_groups_reports_assumptions_and_effect_size(
    survey_file,
):
    output = stats_analysis.run_group_comparison_impl(
        survey_file, "score", "sex"
    )

    assert "Independent t-test (Student's)" in output  # 30 vs 30 rows
    # The effect-size column of pingouin's t-test table is "cohen-d" in some
    # releases and "cohen_d" in others; the image does not pin pingouin.
    assert "cohen-d" in output or "cohen_d" in output
    assert "Assumption checks" in output and "Levene's test" in output


def test_nonparametric_two_groups_matches_scipy(survey_file, r_enabled):
    output = stats_analysis.run_group_comparison_impl(
        survey_file, "score", "sex", method="nonparametric"
    )
    df = pd.read_csv(survey_file)
    u, p = stats_module.mannwhitneyu(
        df.loc[df.sex == "F", "score"],
        df.loc[df.sex == "M", "score"],
        use_continuity=True,
        alternative="two-sided",
        method="asymptotic",
    )

    assert "Mann-Whitney U test" in output
    assert f"{u:.1f}" in output or f"{u:g}" in output
    assert (
        "wilcox.test(samples[[1]], samples[[2]], exact = FALSE, correct = "
        "TRUE)" in output
    )


def test_three_groups_get_post_hoc_tests(survey_file):
    anova = stats_analysis.run_group_comparison_impl(
        survey_file, "score", "faculty"
    )
    kruskal = stats_analysis.run_group_comparison_impl(
        survey_file, "score", "faculty", method="nonparametric"
    )

    assert "One-way ANOVA" in anova and "Tukey HSD post-hoc" in anova
    assert "Kruskal-Wallis test" in kruskal and "Holm-corrected" in kruskal
    assert "p-holm" in kruskal


def test_group_comparison_rejects_bad_input(survey_file):
    assert "method must be" in stats_analysis.run_group_comparison_impl(
        survey_file, "score", "sex", method="bayesian"
    )
    assert "run_association_test" in stats_analysis.run_group_comparison_impl(
        survey_file, "smoker", "sex"
    )


def test_association_test_uses_chi_square_and_reports_cramers_v(
    survey_file, r_enabled
):
    output = stats_analysis.run_association_test_impl(
        survey_file, "faculty", "smoker"
    )
    df = pd.read_csv(survey_file)
    table = pd.crosstab(df.faculty, df.smoker)
    chi2_raw = stats_module.chi2_contingency(table, correction=False)[0]
    cramers_v = (
        chi2_raw / (table.values.sum() * (min(table.shape) - 1))
    ) ** 0.5

    assert "Chi-square test" in output and "Row percentages" in output
    assert f"{cramers_v:.4f}"[:5] in output
    assert "result <- chisq.test(tab)" in output


def test_small_two_by_two_table_uses_fisher(tmp_path, r_enabled):
    path = tmp_path / "small.csv"
    pd.DataFrame(
        {
            "treated": ["yes"] * 6 + ["no"] * 6,
            "cured": ["yes"] * 5 + ["no"] + ["yes"] + ["no"] * 5,
        }
    ).to_csv(path, index=False)
    output = stats_analysis.run_association_test_impl(
        str(path), "treated", "cured"
    )

    assert "Fisher's exact test" in output
    assert "fisher.test(tab)" in output


def test_association_test_rejects_numeric_columns(survey_file):
    output = stats_analysis.run_association_test_impl(
        survey_file, "score", "age"
    )
    assert output.startswith("Error") and "run_correlation" in output


def test_logistic_regression_reports_odds_ratios(survey_file, r_enabled):
    output = stats_analysis.run_logistic_regression_impl(
        survey_file, "dropout", ["score", "faculty"]
    )

    assert not output.startswith("Error"), output
    assert "odds ratio" in output and "McFadden" in output
    assert "'dropout' coded yes = 1, no = 0" in output
    assert "reference" in output  # faculty is categorical
    assert "family = binomial" in output and "confint.default(model)" in output


def test_logistic_regression_handles_perfect_separation(tmp_path):
    path = tmp_path / "separated.csv"
    pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            "y": [0, 0, 0, 0, 1, 1, 1, 1],
        }
    ).to_csv(path, index=False)
    output = stats_analysis.run_logistic_regression_impl(str(path), "y", ["x"])

    assert "separation" in output.lower()


def test_logistic_regression_needs_a_binary_outcome(survey_file):
    output = stats_analysis.run_logistic_regression_impl(
        survey_file, "faculty", ["score"]
    )
    assert output.startswith("Error") and "binary outcome" in output


@pytest.mark.parametrize(
    "values, expected",
    [
        (["no", "yes", "no"], ("yes", "no")),
        (["B", "M"], ("M", "B")),
        ([1, 2, 2], (2, 1)),
        (["apple", "Banana"], ("Banana", "apple")),
        (["a", "b", "c"], None),
    ],
)
def test_binary_encoding_is_deterministic(values, expected):
    assert stats_analysis._binary_encoding(values) == expected
