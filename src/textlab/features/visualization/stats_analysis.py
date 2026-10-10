"""
Statistical Analysis Module for the AI Visualization Engine.
Provides tools for the LLM to run statistical tests using Pingouin, SciPy and Statsmodels.

Every test returns a markdown result, a reproducible Python snippet and, when
requested, an R snippet (see r_code.py). Where Python and R defaults differ,
the Python call is configured so both give the same numbers (noted inline).
"""

import math
import warnings

import numpy as np
import pandas as pd
import pingouin as pg
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.tools.sm_exceptions import (
    ConvergenceWarning,
    PerfectSeparationError,
    PerfectSeparationWarning,
)

from textlab.features.visualization import r_code
from textlab.features.visualization.viz_utils import load_data_safely

# Values that represent the "positive" class in binary categorical columns.
# Used for deterministic 1/0 encoding of binary targets.
_KNOWN_POSITIVE_VALUES = {
    "m", "malignant", "yes", "true", "1", "positive", "pos",
    "disease", "infected", "bad", "high", "abnormal"
}

_GROUP_METHODS = ("parametric", "nonparametric")
# Post-hoc tables grow quadratically with the number of groups.
_POSTHOC_MAX_GROUPS = 10
# Shapiro-Wilk is only reliable (and only allowed in R) for 3 to 5000 values.
_SHAPIRO_MIN_N, _SHAPIRO_MAX_N = 3, 5000
# Crosstabs larger than this usually mean a numeric column was passed.
_MAX_CROSSTAB_CELLS = 400


def _py_literal(value) -> str:
    """Return a Python literal for a data value (NumPy scalars as plain Python)."""
    if hasattr(value, "item"):
        value = value.item()
    return repr(value)


def _binary_encoding(values) -> tuple | None:
    """Return ``(positive, negative)`` for a column with exactly two distinct values.

    For numbers, the larger value is positive (coded 1). For text, a known
    positive label such as "yes" or "M" is positive if present, otherwise the
    alphabetically later value. Returns None if there are not exactly two values.
    """
    unique = list(pd.unique(pd.Series(values).dropna()))
    if len(unique) != 2:
        return None
    if all(pd.api.types.is_number(v) for v in unique):
        low, high = sorted(unique)
        return high, low
    ordered = sorted(unique, key=lambda v: str(v).lower())
    pos_idx = next(
        (i for i, v in enumerate(ordered) if str(v).lower() in _KNOWN_POSITIVE_VALUES),
        1,
    )
    return ordered[pos_idx], ordered[1 - pos_idx]


def _stats_code_header(data_file_path: str) -> str:
    """Returns the import/load preamble for a reproducible stats snippet."""
    ext = data_file_path.rsplit(".", 1)[-1].lower()
    if ext == "tsv":
        loader = "df = pd.read_csv('your_data.tsv', sep='\\t')"
    elif ext in ("xls", "xlsx"):
        loader = f"df = pd.read_excel('your_data.{ext}')"
    elif ext == "json":
        loader = "df = pd.read_json('your_data.json')"
    else:
        loader = "df = pd.read_csv('your_data.csv')"
    return f"import pandas as pd\n\n# Load Data\n{loader}\n\n"


def _with_r_block(output: str, r_snippet: str) -> str:
    """Append the R equivalent as a fenced ```r block when one was generated."""
    return f"{output}\n\n```r\n{r_snippet}\n```" if r_snippet else output


def run_correlation_impl(
    data_file_path: str, x_column: str, y_column: str, method: str = "pearson"
) -> str:
    """
    Computes the correlation between two numeric columns.

    Args:
        data_file_path: The absolute path to the data file.
        x_column: The name of the first numeric column.
        y_column: The name of the second numeric column.
        method: The correlation method ('pearson', 'spearman', or 'kendall').

    Returns:
        A markdown-formatted string of the correlation results + reproducible code,
        or an error message.
    """
    try:
        df = load_data_safely(data_file_path)
        if x_column not in df.columns or y_column not in df.columns:
            return f"Error: Columns '{x_column}' and/or '{y_column}' not found."

        clean_df = df[[x_column, y_column]].dropna()
        if len(clean_df) < 3:
            return "Error: Not enough valid data points to calculate correlation."

        res = pg.corr(clean_df[x_column], clean_df[y_column], method=method)

        code = (
            _stats_code_header(data_file_path)
            + "import pingouin as pg\n\n"
            + f"clean_df = df[['{x_column}', '{y_column}']].dropna()\n"
            + f"result = pg.corr(clean_df['{x_column}'], clean_df['{y_column}'], method='{method}')\n"
            + "print(result)"
        )
        r_snippet = r_code.build(
            r_code.correlation_test, data_file_path, x_column, y_column, method
        )
        return _with_r_block(
            f"Correlation Analysis ({method}) between '{x_column}' and '{y_column}':\n\n"
            f"{res.to_markdown()}\n\n"
            f"```python\n{code}\n```",
            r_snippet,
        )
    except Exception as e:
        return f"Error computing correlation: {str(e)}"


def _assumption_checks(samples: list[pd.Series], groups: list) -> tuple[str, list[str], str]:
    """Shapiro-Wilk per group and Levene's test, as markdown plus plain notes.

    Returns:
        ``(markdown, notes, python_code)`` where ``notes`` flags violated
        assumptions in words the model and the user can act on.
    """
    rows = []
    for group, values in zip(groups, samples):
        row = {"Group": group, "n": len(values), "Shapiro-Wilk W": None, "p-val": None}
        if _SHAPIRO_MIN_N <= len(values) <= _SHAPIRO_MAX_N:
            w, p = stats.shapiro(values)
            row.update({"Shapiro-Wilk W": w, "p-val": p})
        rows.append(row)
    normality = pd.DataFrame(rows)
    # center="median" is the Brown-Forsythe variant, SciPy's default.
    lev_w, lev_p = stats.levene(*samples, center="median")

    notes = []
    if (normality["p-val"].dropna() < 0.05).any():
        notes.append(
            "Normality is doubtful in at least one group (Shapiro-Wilk p < 0.05); "
            "with small groups, consider method='nonparametric'."
        )
    if lev_p < 0.05:
        notes.append("Group variances differ (Levene p < 0.05).")
    untested = [str(g) for g, v in zip(groups, samples) if not
                _SHAPIRO_MIN_N <= len(v) <= _SHAPIRO_MAX_N]
    if untested:
        notes.append(
            f"Normality not tested for group(s) {', '.join(untested)} "
            f"(needs {_SHAPIRO_MIN_N} to {_SHAPIRO_MAX_N:,} values)."
        )

    markdown = (
        "Assumption checks:\n\n"
        f"{normality.to_markdown(index=False)}\n\n"
        f"Levene's test for equal variances: W = {lev_w:.4f}, p = {lev_p:.4g}"
    )
    code = (
        "# Assumption checks\n"
        "for group, values in zip(groups, samples):\n"
        f"    if {_SHAPIRO_MIN_N} <= len(values) <= {_SHAPIRO_MAX_N}:\n"
        "        print(group, stats.shapiro(values))\n"
        "print(stats.levene(*samples, center='median'))\n"
    )
    return markdown, notes, code


def run_group_comparison_impl(
    data_file_path: str, target_col: str, group_col: str, method: str = "parametric"
) -> str:
    """
    Compares a numeric variable between the groups of a categorical variable.

    * ``method="parametric"``: independent t-test (2 groups; pingouin picks
      Student's t-test for equal group sizes, otherwise Welch's) or one-way
      ANOVA (3+ groups) followed by Tukey HSD post-hoc tests.
    * ``method="nonparametric"``: Mann-Whitney U test (2 groups) or
      Kruskal-Wallis test (3+ groups) followed by pairwise Mann-Whitney tests
      with Holm correction. Suited to ordinal data (e.g. Likert scales), small
      samples and clearly non-normal data.

    Both report assumption checks (Shapiro-Wilk per group, Levene) and an
    effect size.

    Args:
        data_file_path: The absolute path to the data file.
        target_col: The numeric column to compare.
        group_col: The categorical column defining the groups.
        method: 'parametric' (default) or 'nonparametric'.

    Returns:
        A markdown-formatted string of the results + reproducible code,
        or an error message.
    """
    try:
        df = load_data_safely(data_file_path)
        if target_col not in df.columns or group_col not in df.columns:
            return f"Error: Columns '{target_col}' or '{group_col}' not found."

        method = (method or "parametric").strip().lower()
        if method not in _GROUP_METHODS:
            return "Error: method must be 'parametric' or 'nonparametric'."

        clean_df = df[[target_col, group_col]].dropna()
        if not pd.api.types.is_numeric_dtype(clean_df[target_col]):
            return (
                f"Error: '{target_col}' is not numeric. To test whether two categorical "
                "columns are related, use run_association_test."
            )
        groups = sorted(clean_df[group_col].unique(), key=str)
        if len(groups) < 2:
            return "Error: The grouping column must have at least 2 unique values."

        samples = [clean_df.loc[clean_df[group_col] == g, target_col] for g in groups]
        too_small = [str(g) for g, v in zip(groups, samples) if len(v) < 2]
        if too_small:
            return (
                f"Error: Group(s) {', '.join(too_small)} have fewer than 2 values; "
                "remove them or merge rare groups."
            )

        checks_md, notes, checks_code = _assumption_checks(samples, groups)
        sections: list[str] = []
        equal_sizes = len(samples[0]) == len(samples[1]) if len(groups) == 2 else None
        posthoc = 3 <= len(groups) <= _POSTHOC_MAX_GROUPS
        k, n = len(groups), len(clean_df)

        group_literals = ", ".join(_py_literal(g) for g in groups)
        code_logic = (
            f"clean_df = df[[{target_col!r}, {group_col!r}]].dropna()\n"
            f"groups = [{group_literals}]\n"
            f"samples = [clean_df.loc[clean_df[{group_col!r}] == g, {target_col!r}] "
            "for g in groups]\n\n"
        )

        if method == "parametric" and k == 2:
            res = pg.ttest(samples[0], samples[1])
            test_name = "Welch's" if not equal_sizes else "Student's"
            sections.append(
                f"Independent t-test ({test_name}) for '{target_col}' grouped by "
                f"'{group_col}' (Groups: {groups[0]} vs {groups[1]}):\n\n{res.to_markdown()}"
            )
            code_logic += "print(pg.ttest(samples[0], samples[1]))\n"
        elif method == "parametric":
            res = pg.anova(dv=target_col, between=group_col, data=clean_df, detailed=True)
            sections.append(
                f"One-way ANOVA for '{target_col}' grouped by '{group_col}' "
                f"({k} groups; np2 = partial eta squared):\n\n{res.to_markdown()}"
            )
            code_logic += (
                f"print(pg.anova(dv={target_col!r}, between={group_col!r}, "
                "data=clean_df, detailed=True))\n"
            )
            if posthoc:
                tukey = pg.pairwise_tukey(data=clean_df, dv=target_col, between=group_col)
                sections.append(
                    f"Tukey HSD post-hoc comparisons:\n\n{tukey.to_markdown(index=False)}"
                )
                code_logic += (
                    f"print(pg.pairwise_tukey(data=clean_df, dv={target_col!r}, "
                    f"between={group_col!r}))\n"
                )
        elif k == 2:
            n1, n2 = len(samples[0]), len(samples[1])
            # method="asymptotic" matches R's wilcox.test(exact = FALSE); SciPy's
            # default switches to an exact test at a different sample size.
            u, p = stats.mannwhitneyu(
                samples[0], samples[1], use_continuity=True,
                alternative="two-sided", method="asymptotic",
            )
            rbc = 2 * u / (n1 * n2) - 1
            table = pd.DataFrame([{
                "U": u, "p-val": p, "rank-biserial r": rbc, "n1": n1, "n2": n2,
            }])
            sections.append(
                f"Mann-Whitney U test for '{target_col}' grouped by '{group_col}' "
                f"(Groups: {groups[0]} vs {groups[1]}; rank-biserial r > 0 means "
                f"'{groups[0]}' tends to have larger values):\n\n{table.to_markdown(index=False)}"
            )
            code_logic += (
                "u, p = stats.mannwhitneyu(samples[0], samples[1], use_continuity=True,\n"
                "                          alternative='two-sided', method='asymptotic')\n"
                "rbc = 2 * u / (len(samples[0]) * len(samples[1])) - 1\n"
                "print(u, p, rbc)\n"
            )
        else:
            h, p = stats.kruskal(*samples)
            eta2 = (h - k + 1) / (n - k)
            table = pd.DataFrame([{"H": h, "dof": k - 1, "p-val": p, "eta2[H]": eta2, "n": n}])
            sections.append(
                f"Kruskal-Wallis test for '{target_col}' grouped by '{group_col}' "
                f"({k} groups):\n\n{table.to_markdown(index=False)}"
            )
            code_logic += (
                "h, p = stats.kruskal(*samples)\n"
                f"print(h, p, (h - {k} + 1) / ({n} - {k}))\n"
            )
            if posthoc:
                pairs, pvals = [], []
                for i in range(k):
                    for j in range(i + 1, k):
                        u_ij, p_ij = stats.mannwhitneyu(
                            samples[i], samples[j], use_continuity=True,
                            alternative="two-sided", method="asymptotic",
                        )
                        pairs.append({"A": groups[i], "B": groups[j], "U": u_ij, "p-unc": p_ij})
                        pvals.append(p_ij)
                pairwise = pd.DataFrame(pairs)
                pairwise["p-holm"] = multipletests(pvals, method="holm")[1]
                sections.append(
                    "Pairwise Mann-Whitney U tests (Holm-corrected):\n\n"
                    f"{pairwise.to_markdown(index=False)}"
                )
                code_logic += (
                    "k = len(groups)\n"
                    "pairs = [(i, j) for i in range(k) for j in range(i + 1, k)]\n"
                    "pvals = [stats.mannwhitneyu(samples[i], samples[j], use_continuity=True,\n"
                    "                            alternative='two-sided', method='asymptotic')[1]\n"
                    "         for i, j in pairs]\n"
                    "print(multipletests(pvals, method='holm')[1])\n"
                )
        if k >= 3 and not posthoc:
            notes.append(
                f"Post-hoc comparisons are skipped for more than {_POSTHOC_MAX_GROUPS} groups."
            )

        sections.append(checks_md)
        if notes:
            sections.append("Notes:\n" + "\n".join(f"- {note}" for note in notes))
        code_logic += "\n" + checks_code

        code = (
            _stats_code_header(data_file_path)
            + "import pingouin as pg\n"
            + "from scipy import stats\n"
            + "from statsmodels.stats.multitest import multipletests\n\n"
            + code_logic.rstrip()
        )
        r_snippet = r_code.build(
            r_code.group_comparison, data_file_path, target_col, group_col,
            list(groups), equal_sizes, method, posthoc,
        )
        return _with_r_block(
            "\n\n".join(sections) + f"\n\n```python\n{code}\n```", r_snippet
        )

    except Exception as e:
        return f"Error computing group comparison: {str(e)}"


def run_association_test_impl(data_file_path: str, x_column: str, y_column: str) -> str:
    """
    Tests whether two categorical columns are associated.

    Builds the crosstab, then runs a chi-square test of independence (with
    Yates' continuity correction for 2x2 tables, as R's ``chisq.test``). For a
    2x2 table with an expected count below 5, Fisher's exact test is used
    instead. Cramer's V (from the uncorrected chi-square) measures the strength.

    Args:
        data_file_path: The absolute path to the data file.
        x_column: The first categorical column (crosstab rows).
        y_column: The second categorical column (crosstab columns).

    Returns:
        A markdown-formatted string of the results + reproducible code,
        or an error message.
    """
    try:
        df = load_data_safely(data_file_path)
        if x_column not in df.columns or y_column not in df.columns:
            return f"Error: Columns '{x_column}' and/or '{y_column}' not found."
        if x_column == y_column:
            return "Error: Choose two different columns."

        clean_df = df[[x_column, y_column]].dropna()
        table = pd.crosstab(clean_df[x_column], clean_df[y_column])
        if table.shape[0] < 2 or table.shape[1] < 2:
            return "Error: Both columns need at least 2 different values."
        if table.size > _MAX_CROSSTAB_CELLS:
            return (
                f"Error: The crosstab has {table.size} cells. Both columns must be "
                "categorical with few categories; for numeric columns use "
                "run_correlation or run_group_comparison."
            )

        chi2, p, dof, expected = stats.chi2_contingency(table)
        chi2_raw = stats.chi2_contingency(table, correction=False)[0]
        n = int(table.values.sum())
        cramers_v = math.sqrt(chi2_raw / (n * (min(table.shape) - 1)))
        small_expected = bool((expected < 5).any())
        use_fisher = table.shape == (2, 2) and small_expected

        row_pct = pd.crosstab(
            clean_df[x_column], clean_df[y_column], normalize="index"
        ).mul(100).round(1)
        sections = [
            f"Crosstab of '{x_column}' (rows) by '{y_column}' (columns), counts:\n\n"
            f"{table.to_markdown()}",
            f"Row percentages:\n\n{row_pct.to_markdown()}",
        ]
        if use_fisher:
            odds_ratio, p_fisher = stats.fisher_exact(table)
            result = pd.DataFrame([{
                "Test": "Fisher's exact test", "odds ratio": odds_ratio,
                "p-val": p_fisher, "Cramer's V": cramers_v, "n": n,
            }])
            note = "Fisher's exact test was used because an expected count is below 5."
        else:
            result = pd.DataFrame([{
                "Test": "Chi-square test", "chi2": chi2, "dof": dof,
                "p-val": p, "Cramer's V": cramers_v, "n": n,
            }])
            note = (
                "Some expected counts are below 5, so the chi-square p-value may be "
                "unreliable; consider merging rare categories."
                if small_expected else ""
            )
        sections.append(f"Test of association:\n\n{result.to_markdown(index=False)}")
        if note:
            sections.append(f"Note: {note}")

        test_code = (
            "odds_ratio, p = stats.fisher_exact(table)\nprint(odds_ratio, p)\n"
            if use_fisher else
            "chi2, p, dof, expected = stats.chi2_contingency(table)\nprint(chi2, dof, p)\n"
        )
        code = (
            _stats_code_header(data_file_path)
            + "import math\nfrom scipy import stats\n\n"
            + f"clean_df = df[[{x_column!r}, {y_column!r}]].dropna()\n"
            + f"table = pd.crosstab(clean_df[{x_column!r}], clean_df[{y_column!r}])\n"
            + "print(table)\n"
            + test_code
            + "chi2_raw = stats.chi2_contingency(table, correction=False)[0]\n"
            + "n = table.values.sum()\n"
            + "print('Cramer V:', math.sqrt(chi2_raw / (n * (min(table.shape) - 1))))"
        )
        r_snippet = r_code.build(
            r_code.association_test, data_file_path, x_column, y_column, use_fisher
        )
        return _with_r_block(
            "\n\n".join(sections) + f"\n\n```python\n{code}\n```", r_snippet
        )
    except Exception as e:
        return f"Error computing association test: {str(e)}"


def run_linear_regression_impl(
    data_file_path: str, target_col: str, predictor_cols: list[str]
) -> str:
    """
    Performs an Ordinary Least Squares (OLS) linear regression.

    Args:
        data_file_path: The absolute path to the data file.
        target_col: The dependent variable (Y).
        predictor_cols: A list of independent variables (X).

    Returns:
        A formatted string of the statsmodels regression summary + reproducible code,
        or an error message.
    """
    try:
        df = load_data_safely(data_file_path)

        # Coerce predictor_cols to a list in case the LLM passed a comma-separated string.
        if isinstance(predictor_cols, str):
            predictor_cols = [c.strip() for c in predictor_cols.split(",") if c.strip()]

        missing_cols = [col for col in [target_col] + predictor_cols if col not in df.columns]
        if missing_cols:
            return f"Error: Missing columns in data: {', '.join(missing_cols)}"

        cols = [target_col] + predictor_cols
        clean_df = df[cols].dropna()
        
        if len(clean_df) <= len(predictor_cols):
            return "Error: Not enough data points to run regression after dropping NaNs."

        X = sm.add_constant(clean_df[predictor_cols])
        y = clean_df[target_col]
        model = sm.OLS(y, X).fit()

        pred_list = str(predictor_cols)
        code = (
            _stats_code_header(data_file_path)
            + "import statsmodels.api as sm\n\n"
            + f"cols = ['{target_col}'] + {pred_list}\n"
            + "clean_df = df[cols].dropna()\n"
            + f"X = sm.add_constant(clean_df[{pred_list}])\n"
            + f"y = clean_df['{target_col}']\n"
            + "model = sm.OLS(y, X).fit()\n"
            + "print(model.summary())"
        )
        r_snippet = r_code.build(
            r_code.regression, data_file_path, target_col, predictor_cols
        )
        return _with_r_block(
            f"OLS Linear Regression Results (Target: {target_col}):\n\n"
            f"{model.summary().as_text()}\n\n"
            f"```python\n{code}\n```",
            r_snippet,
        )
    except Exception as e:
        return f"Error computing linear regression: {str(e)}"


def _formula_term(name: str, categorical: bool) -> str:
    """Return a patsy formula term for a column name (quoted; C() if categorical)."""
    term = f"Q({name!r})"
    return f"C({term})" if categorical else term


def run_logistic_regression_impl(
    data_file_path: str, target_col: str, predictor_cols: list[str]
) -> str:
    """
    Fits a logistic regression for a binary outcome.

    The outcome may be 0/1 or any column with exactly two values (e.g. yes/no),
    which is coded 1/0 like in rank_target_correlations. Text predictors are
    treated as categorical, with the alphabetically first category as the
    reference (as in R). Reports coefficients, odds ratios with Wald 95%
    confidence intervals, and McFadden's pseudo R-squared.

    Args:
        data_file_path: The absolute path to the data file.
        target_col: The binary outcome column.
        predictor_cols: The predictor columns.

    Returns:
        A markdown-formatted string of the results + reproducible code,
        or an error message.
    """
    try:
        df = load_data_safely(data_file_path)
        if isinstance(predictor_cols, str):
            predictor_cols = [c.strip() for c in predictor_cols.split(",") if c.strip()]
        if not predictor_cols:
            return "Error: Provide at least one predictor column."
        missing_cols = [c for c in [target_col] + predictor_cols if c not in df.columns]
        if missing_cols:
            return f"Error: Missing columns in data: {', '.join(missing_cols)}"
        if target_col in predictor_cols:
            return "Error: The outcome column cannot also be a predictor."

        clean_df = df[[target_col] + predictor_cols].dropna().copy()
        values = set(pd.unique(clean_df[target_col]))
        encoding = None
        if values <= {0, 1} and pd.api.types.is_numeric_dtype(clean_df[target_col]):
            if len(values) < 2:
                return f"Error: '{target_col}' has only one value after removing missing data."
        else:
            encoding = _binary_encoding(clean_df[target_col])
            if encoding is None:
                return (
                    f"Error: '{target_col}' has {len(values)} different values. Logistic "
                    "regression needs a binary outcome (two values, e.g. yes/no); for a "
                    "numeric outcome use run_linear_regression."
                )
            positive, negative = encoding
            clean_df[target_col] = clean_df[target_col].map({positive: 1, negative: 0})

        if len(clean_df) <= len(predictor_cols) + 1:
            return "Error: Not enough data points to fit the model after dropping NaNs."

        categorical = {
            c: (not pd.api.types.is_numeric_dtype(clean_df[c])
                or pd.api.types.is_bool_dtype(clean_df[c]))
            for c in predictor_cols
        }
        formula = f"{_formula_term(target_col, False)} ~ " + " + ".join(
            _formula_term(c, categorical[c]) for c in predictor_cols
        )

        notes: list[str] = []
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                model = smf.logit(formula, data=clean_df).fit(disp=0)
            except (PerfectSeparationError, np.linalg.LinAlgError):
                return (
                    "Error: The model cannot be estimated, usually because of perfect "
                    "separation (a predictor predicts the outcome perfectly) or because "
                    "predictors duplicate each other. Remove the predictor concerned."
                )
        if any(issubclass(w.category, PerfectSeparationWarning) for w in caught):
            notes.append(
                "Perfect or quasi-perfect separation detected: some estimates and odds "
                "ratios are unreliable (extremely large)."
            )
        if any(issubclass(w.category, ConvergenceWarning) for w in caught):
            notes.append("The model did not converge; interpret the estimates with caution.")

        conf = model.conf_int()
        table = pd.DataFrame({
            "coef": model.params,
            "std err": model.bse,
            "z": model.tvalues,
            "p-val": model.pvalues,
            # np.exp gives inf (not OverflowError) for the huge estimates that
            # perfect separation produces, so the warning below still shows.
            "odds ratio": np.exp(model.params),
            "OR 2.5%": np.exp(conf[0]),
            "OR 97.5%": np.exp(conf[1]),
        })
        coding = (
            f" ('{target_col}' coded {encoding[0]!s} = 1, {encoding[1]!s} = 0)" if encoding else ""
        )
        sections = [
            f"Logistic Regression Results (Outcome: {target_col}{coding}):\n\n"
            f"{table.to_markdown()}",
            f"n = {int(model.nobs)}, McFadden's pseudo R-squared = {model.prsquared:.4f}, "
            f"likelihood-ratio test p = {model.llr_pvalue:.4g}",
        ]
        if any(categorical.values()):
            sections.append(
                "Note: categorical predictors are compared with their alphabetically "
                "first category (the reference)."
            )
        if notes:
            sections.append("Notes:\n" + "\n".join(f"- {note}" for note in notes))

        encode_code = ""
        if encoding:
            encode_code = (
                f"clean_df[{target_col!r}] = clean_df[{target_col!r}].map("
                f"{{{_py_literal(encoding[0])}: 1, {_py_literal(encoding[1])}: 0}})\n"
            )
        code = (
            _stats_code_header(data_file_path)
            + "import numpy as np\nimport statsmodels.formula.api as smf\n\n"
            + f"cols = [{target_col!r}] + {predictor_cols!r}\n"
            + "clean_df = df[cols].dropna().copy()\n"
            + encode_code
            + f"model = smf.logit({formula!r}, data=clean_df).fit()\n"
            + "print(model.summary())\n"
            + "print(np.exp(model.params))      # odds ratios\n"
            + "print(np.exp(model.conf_int()))  # 95% CI of the odds ratios"
        )
        r_snippet = r_code.build(
            r_code.logistic_regression, data_file_path, target_col, predictor_cols, encoding
        )
        return _with_r_block(
            "\n\n".join(sections) + f"\n\n```python\n{code}\n```", r_snippet
        )
    except Exception as e:
        return f"Error computing logistic regression: {str(e)}"


def rank_target_correlations_impl(
    data_file_path: str, target_col: str, method: str = "pearson"
) -> str:
    """
    Calculates the correlation between a target column and all other numeric features,
    sorting them by absolute correlation strength. Automatically encodes binary text
    deterministically (known positive class → 1, otherwise alphabetically lower → 0).

    Args:
        data_file_path: The absolute path to the data file.
        target_col: The column to correlate all other features against.
        method: The correlation method ('pearson' or 'spearman').

    Returns:
        A markdown table ranking the features + reproducible code, or an error message.
    """
    try:
        df = load_data_safely(data_file_path)
        if target_col not in df.columns:
            return f"Error: Target column '{target_col}' not found in the dataset."

        working_df = df.copy()
        binary_encode_snippet = ""
        encoding = None

        if not pd.api.types.is_numeric_dtype(working_df[target_col]):
            encoding = _binary_encoding(working_df[target_col])
            if encoding is None:
                n_unique = working_df[target_col].nunique()
                return (
                    f"Error: Target column '{target_col}' is non-numeric and contains "
                    f"{n_unique} unique values. It must be strictly binary to auto-encode."
                )
            positive, negative = encoding
            working_df[target_col] = working_df[target_col].map({positive: 1, negative: 0})
            # Build the encoding step to include in the reproducible snippet.
            binary_encode_snippet = (
                f"# Binary encode target column\n"
                f"val_map = {{{_py_literal(positive)}: 1, {_py_literal(negative)}: 0}}\n"
                f"df['{target_col}'] = df['{target_col}'].map(val_map)\n\n"
            )

        numeric_df = working_df.select_dtypes(include=["number"])
        if target_col not in numeric_df.columns:
            return f"Error: Target column '{target_col}' could not be evaluated numerically."

        correlations = numeric_df.corr(method=method)[target_col].drop(target_col)

        if correlations.empty:
            return "Error: No other numeric columns found to correlate against."

        ranking_df = pd.DataFrame({
            "Feature": correlations.index,
            "Correlation Coefficient": correlations.values,
            "Absolute Strength": correlations.abs().values
        }).sort_values(by="Absolute Strength", ascending=False).drop(columns=["Absolute Strength"])

        code = (
            _stats_code_header(data_file_path)
            + binary_encode_snippet
            + f"corr = df.select_dtypes(include='number').corr(method='{method}')['{target_col}']\n"
            + f"corr = corr.drop('{target_col}').abs().sort_values(ascending=False)\n"
            + "print(corr)"
        )
        r_snippet = r_code.build(
            r_code.rank_correlations, data_file_path, target_col, method, encoding
        )
        return _with_r_block(
            f"Correlation Ranking with respect to target column '{target_col}' ({method}):\n\n"
            f"{ranking_df.to_markdown(index=False)}\n\n"
            f"```python\n{code}\n```",
            r_snippet,
        )
    except Exception as e:
        return f"Error ranking correlations: {str(e)}"