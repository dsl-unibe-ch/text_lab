"""
R equivalents of the visualisation and statistics tools.

When a Visualize Data run asks for R code, its MCP server is started with
``TEXTLAB_R_CODE=1`` and every standard tool adds an R snippet next to its
Python snippet: ggplot2 for plots, base R ``stats`` for tests.

The snippets are built from the same resolved values the Python tools use
(group order and sizes, binary encodings, sampling, the row cap), and they
mirror the Python defaults that change numbers, so statistical results can be
cross-checked exactly. Plots show the same data, but default styling such as
bin counts and colours differs between the libraries.

Custom-code tools run arbitrary model-written Python and have no R snippet.
"""

import numbers
import os
from typing import Any, Callable

from core.visualization.viz_config import MAX_ROWS
from core.visualization.viz_utils import was_last_load_truncated

# Environment variable that switches R snippet generation on in the MCP server.
R_CODE_ENV = "TEXTLAB_R_CODE"

# Sampling used by the Python pair plot; R draws its own random sample.
_PAIRPLOT_SEED = 42


def is_enabled() -> bool:
    """Return True when this process should generate R snippets."""
    return os.environ.get(R_CODE_ENV) == "1"


def build(generator: Callable[..., str], *args: Any, **kwargs: Any) -> str:
    """Return ``generator``'s R snippet if R code is enabled, otherwise ``""``.

    Never raises: a failure here must not break the plot or test it belongs
    to, so it is reported inside the snippet as an R comment instead.
    """
    if not is_enabled():
        return ""
    try:
        return generator(*args, **kwargs)
    except Exception as exc:
        return f"# The R code could not be generated: {exc}"


# =========================================================================
# R LITERALS
# =========================================================================

def r_str(value: Any) -> str:
    """Return ``value`` as a double-quoted R string literal."""
    text = str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    return f'"{text}"'


def r_col(name: Any) -> str:
    """Return a column name as a backtick-quoted R name (safe for spaces)."""
    return "`" + str(name).replace("\\", "\\\\").replace("`", "\\`") + "`"


def r_value(value: Any) -> str:
    """Return a data value (number, boolean or string) as an R literal."""
    is_numpy_bool = getattr(getattr(value, "dtype", None), "kind", "") == "b"
    if isinstance(value, bool) or is_numpy_bool:
        return "TRUE" if value else "FALSE"
    if isinstance(value, numbers.Number):
        return repr(value.item() if hasattr(value, "item") else value)
    return r_str(value)


def r_vector(values: list[Any]) -> str:
    """Return a character vector literal, e.g. ``c("a", "b")``."""
    return "c(" + ", ".join(r_str(v) for v in values) + ")"


# =========================================================================
# SHARED PIECES
# =========================================================================

def _header(data_file_path: str, packages: list[str]) -> str:
    """Return the install hint, ``library()`` calls and data-loading lines."""
    ext = os.path.splitext(data_file_path)[1].lower()
    needed = list(packages)
    if ext == ".tsv":
        loader = ['df <- read.delim("your_data.tsv", check.names = FALSE)']
    elif ext in (".xls", ".xlsx"):
        needed.append("readxl")
        loader = [f'df <- as.data.frame(readxl::read_excel("your_data{ext}"))']
    elif ext == ".json":
        needed.append("jsonlite")
        loader = [
            "# JSON Lines file; for a single JSON array use:",
            '# df <- jsonlite::fromJSON("your_data.json")',
            'df <- jsonlite::stream_in(file("your_data.json"), verbose = FALSE)',
        ]
    else:
        loader = ['df <- read.csv("your_data.csv", check.names = FALSE)']

    if was_last_load_truncated(data_file_path):
        loader.append(f"# The analysis used only the first {MAX_ROWS:,} rows:")
        loader.append(f"df <- head(df, {MAX_ROWS})")

    lines: list[str] = []
    if needed:
        lines.append("# Required packages (install once):")
        lines.append(f"# install.packages({r_vector(needed)})")
    lines += [f"library({pkg})" for pkg in packages]
    if lines:
        lines.append("")
    lines.append("# Load data (check.names = FALSE keeps the original column names)")
    lines += loader
    return "\n".join(lines) + "\n\n"


def _finish_ggplot(interactive: bool, width: int = 10, height: int = 6) -> str:
    """Return the lines that display (and, for static plots, save) plot ``p``."""
    if interactive:
        return (
            "print(p)\n"
            "# Optional interactive version (needs the plotly package):\n"
            "# plotly::ggplotly(p)"
        )
    return (
        "print(p)\n"
        f'ggsave("output.png", p, width = {width}, height = {height}, dpi = 300)'
    )


def _theme(interactive: bool) -> str:
    """Return the ggplot2 theme closest to the Python version's style."""
    return "theme_minimal()" if interactive else "theme_bw()"


def _labs(title: str, x_label: str | None = None, y_label: str | None = None,
          **extra: str | None) -> str:
    """Return a ``labs(...)`` call; ``None`` values keep ggplot2's default label."""
    parts = [f"title = {r_str(title)}"]
    if x_label:
        parts.append(f"x = {r_str(x_label)}")
    if y_label:
        parts.append(f"y = {r_str(y_label)}")
    parts += [f"{key} = {r_str(value)}" for key, value in extra.items() if value]
    return f"labs({', '.join(parts)})"


def _rotate_x() -> str:
    """Return the theme tweak that rotates x-axis labels by 45 degrees."""
    return "theme(axis.text.x = element_text(angle = 45, hjust = 1))"


def _select_numeric(filters: list[str] | None) -> str:
    """Return R lines building ``num``: numeric columns, optionally filtered.

    Mirrors the Python ``column_filter``: exact names or name suffixes.
    """
    lines = "num <- df[, sapply(df, is.numeric), drop = FALSE]\n"
    if filters:
        lines += (
            f"filters <- {r_vector(filters)}\n"
            "keep <- sapply(names(num), function(n) n %in% filters || any(endsWith(n, filters)))\n"
            "num <- num[, keep, drop = FALSE]\n"
        )
    return lines


# =========================================================================
# PLOTS
# =========================================================================

def histogram(data_file_path: str, column: str, title: str, color_column: str | None,
              numeric: bool, interactive: bool, x_label: str | None = None) -> str:
    """Histogram (interactive: stacked by colour; static: with a density curve)."""
    if interactive:
        fill = f", fill = factor({r_col(color_column)})" if color_column else ""
        geom = "geom_histogram(bins = 30)" if numeric else "geom_bar()"
        body = (
            f"p <- ggplot(df, aes(x = {r_col(column)}{fill})) +\n"
            f"  {geom} +\n"
            f"  {_labs(title, fill=color_column)} +\n"
            f"  {_theme(True)}\n"
        )
    elif numeric:
        body = (
            "# Bin width for 30 bins; the density curve is scaled to counts,\n"
            "# like seaborn's histplot(kde=True).\n"
            f"binwidth <- diff(range(df[[{r_str(column)}]], na.rm = TRUE)) / 30\n"
            f"p <- ggplot(df, aes(x = {r_col(column)})) +\n"
            '  geom_histogram(binwidth = binwidth, fill = "steelblue", colour = "white") +\n'
            "  geom_density(aes(y = after_stat(count) * binwidth)) +\n"
            f'  {_labs(title, x_label, "Frequency")} +\n'
            f"  {_theme(False)}\n"
        )
    else:
        body = (
            f"p <- ggplot(df, aes(x = {r_col(column)})) +\n"
            '  geom_bar(fill = "steelblue") +\n'
            f'  {_labs(title, x_label, "Frequency")} +\n'
            f"  {_theme(False)}\n"
        )
    return _header(data_file_path, ["ggplot2"]) + body + _finish_ggplot(interactive)


def scatter(data_file_path: str, x_column: str, y_column: str, title: str,
            color_column: str | None, color_numeric: bool, interactive: bool,
            x_label: str | None = None, y_label: str | None = None) -> str:
    """Scatter plot; a numeric colour column is shown as a continuous scale."""
    colour = ""
    if color_column:
        mapped = r_col(color_column) if color_numeric else f"factor({r_col(color_column)})"
        colour = f", colour = {mapped}"
    body = (
        f"p <- ggplot(df, aes(x = {r_col(x_column)}, y = {r_col(y_column)}{colour})) +\n"
        "  geom_point() +\n"
        f"  {_labs(title, x_label, y_label, colour=color_column)} +\n"
        f"  {_theme(interactive)}\n"
    )
    return _header(data_file_path, ["ggplot2"]) + body + _finish_ggplot(interactive)


def boxplot(data_file_path: str, x_column: str, y_column: str, title: str,
            color_column: str | None, interactive: bool,
            x_label: str | None = None, y_label: str | None = None) -> str:
    """Box plot of ``y_column`` per category of ``x_column``."""
    fill = f", fill = factor({r_col(color_column)})" if color_column else ""
    body = (
        f"p <- ggplot(df, aes(x = factor({r_col(x_column)}), y = {r_col(y_column)}{fill})) +\n"
        "  geom_boxplot() +\n"
        f"  {_labs(title, x_label or x_column, y_label, fill=color_column)} +\n"
        f"  {_theme(interactive)}"
    )
    if not interactive:
        body += f" +\n  {_rotate_x()}"
    return _header(data_file_path, ["ggplot2"]) + body + "\n" + _finish_ggplot(interactive)


def lineplot(data_file_path: str, x_column: str, y_column: str, title: str,
             color_column: str | None, interactive: bool,
             x_label: str | None = None, y_label: str | None = None) -> str:
    """Line plot (interactive: raw rows in data order; static: mean with 95% CI)."""
    group = f"factor({r_col(color_column)})" if color_column else None
    if interactive:
        colour = f", colour = {group}, group = {group}" if group else ""
        body = (
            "# Lines connect the rows in data order, as in the Python version.\n"
            f"p <- ggplot(df, aes(x = {r_col(x_column)}, y = {r_col(y_column)}{colour})) +\n"
            "  geom_path() +\n"
            f"  {_labs(title, colour=color_column)} +\n"
            f"  {_theme(True)}\n"
        )
        return _header(data_file_path, ["ggplot2"]) + body + _finish_ggplot(True)

    colour = f", colour = {group}, fill = {group}" if group else ""
    body = (
        "# Mean of y per x value with a 95% bootstrap confidence band, like\n"
        "# seaborn's lineplot (the band is random, so it differs slightly).\n"
        f"p <- ggplot(df, aes(x = {r_col(x_column)}, y = {r_col(y_column)}{colour})) +\n"
        '  stat_summary(fun.data = mean_cl_boot, geom = "ribbon", alpha = 0.2, colour = NA) +\n'
        '  stat_summary(fun = mean, geom = "line") +\n'
        f"  {_labs(title, x_label, y_label, colour=color_column, fill=color_column)} +\n"
        f"  {_theme(False)} +\n"
        f"  {_rotate_x()}\n"
    )
    return _header(data_file_path, ["ggplot2", "Hmisc"]) + body + _finish_ggplot(False)


# R expressions for the aggregations the bar chart tools support; ``{v}`` is
# the values. "count" counts non-missing values, like pandas.
_R_AGGREGATES = {
    "mean": "mean({v}, na.rm = TRUE)",
    "sum": "sum({v}, na.rm = TRUE)",
    "median": "median({v}, na.rm = TRUE)",
    "count": "sum(!is.na({v}))",
}


def barchart_interactive(data_file_path: str, x_column: str, y_column: str, title: str,
                         color_column: str | None, aggregation: str,
                         count_rows: bool) -> str:
    """Bar chart of an aggregated table, mirroring the pandas groupby in Python.

    Args:
        count_rows: True when y is also a grouping column, in which case the
            Python tool counts rows per group instead of aggregating y.
    """
    groups = [x_column] + ([color_column] if color_column else [])
    group_args = ", ".join(r_col(g) for g in groups)
    # pandas groupby drops rows whose group key is missing.
    not_missing = " & ".join(f"!is.na({r_col(g)})" for g in groups)
    if count_rows:
        agg = (
            f"agg_df <- df |>\n"
            f"  dplyr::filter({not_missing}) |>\n"
            f'  dplyr::count({group_args}, name = "count")\n'
        )
        y = "count"
    else:
        expr = _R_AGGREGATES[aggregation].format(v=r_col(y_column))
        agg = (
            f"agg_df <- df |>\n"
            f"  dplyr::filter({not_missing}) |>\n"
            f"  dplyr::group_by({group_args}) |>\n"
            f'  dplyr::summarise({r_col(y_column)} = {expr}, .groups = "drop")\n'
        )
        y = y_column
    fill = f", fill = factor({r_col(color_column)})" if color_column else ""
    position = '"dodge"' if color_column else '"stack"'
    body = (
        agg
        + f"p <- ggplot(agg_df, aes(x = {r_col(x_column)}, y = {r_col(y)}{fill})) +\n"
        f"  geom_col(position = {position}) +\n"
        f"  {_labs(title, fill=color_column)} +\n"
        f"  {_theme(True)}\n"
    )
    return _header(data_file_path, ["ggplot2", "dplyr"]) + body + _finish_ggplot(True)


def barchart_static(data_file_path: str, x_column: str, y_column: str, title: str,
                    x_label: str, y_label: str, hue_column: str | None,
                    aggregation: str) -> str:
    """Bar chart like seaborn's barplot: estimator per group, +/- 1 SD error bars."""
    fill = f", fill = factor({r_col(hue_column)})" if hue_column else ""
    dodge = "position_dodge(width = 0.9)"
    with_sd = aggregation in ("mean", "median")

    body = f"agg <- function(v) {_R_AGGREGATES[aggregation].format(v='v')}\n"
    if with_sd:
        body += (
            "# Error bars: estimate +/- one standard deviation (seaborn errorbar='sd').\n"
            "agg_sd <- function(v) {\n"
            "  m <- agg(v)\n"
            "  s <- sd(v, na.rm = TRUE)\n"
            "  data.frame(y = m, ymin = m - s, ymax = m + s)\n"
            "}\n"
        )
    body += (
        f"p <- ggplot(df, aes(x = factor({r_col(x_column)}), y = {r_col(y_column)}{fill})) +\n"
        f'  stat_summary(fun = agg, geom = "col", position = {dodge}) +\n'
    )
    if with_sd:
        body += (
            '  stat_summary(fun.data = agg_sd, geom = "errorbar", width = 0.2,\n'
            f"               position = {dodge}) +\n"
        )
    body += (
        f"  {_labs(title, x_label, y_label, fill=hue_column)} +\n"
        f"  {_theme(False)} +\n"
        f"  {_rotate_x()}\n"
    )
    return _header(data_file_path, ["ggplot2"]) + body + _finish_ggplot(False, width=12)


def pair_plot(data_file_path: str, columns: list[str], title: str,
              color_column: str | None, interactive: bool,
              sample_rows: int | None = None) -> str:
    """Scatter matrix (interactive: lower triangle only) or pair plot with densities."""
    mapping = f", mapping = aes(colour = factor({r_col(color_column)}))" if color_column else ""
    cols = columns + ([color_column] if color_column else [])
    body = f"plot_df <- na.omit(df[, {r_vector(cols)}])\n"
    if sample_rows:
        body += (
            f"# The Python plot used a random sample of {sample_rows:,} rows; R draws\n"
            "# its own sample, so individual points differ.\n"
            f"set.seed({_PAIRPLOT_SEED})\n"
            f"if (nrow(plot_df) > {sample_rows}) "
            f"plot_df <- plot_df[sample(nrow(plot_df), {sample_rows}), ]\n"
        )
    if interactive:
        panels = (
            '  upper = list(continuous = "blank"),\n'
            '  diag = list(continuous = "blankDiag"),\n'
        )
    else:
        panels = (
            '  upper = list(continuous = wrap("points", alpha = 0.6)),\n'
            '  lower = list(continuous = wrap("points", alpha = 0.6)),\n'
            '  diag = list(continuous = "densityDiag"),\n'
        )
    body += (
        f"p <- ggpairs(plot_df, columns = {r_vector(columns)}{mapping},\n"
        f"{panels}"
        f"  title = {r_str(title)})\n"
    )
    size = max(8, 2 * len(columns))
    return (
        _header(data_file_path, ["ggplot2", "GGally"])
        + body
        + _finish_ggplot(interactive, width=size, height=size)
    )


def correlation_heatmap(data_file_path: str, title: str, method: str,
                        filters: list[str] | None, interactive: bool) -> str:
    """Correlation matrix heatmap with the coefficients printed in each cell."""
    body = (
        _select_numeric(filters)
        + f'corr <- cor(num, use = "pairwise.complete.obs", method = {r_str(method)})\n'
        "corr_long <- as.data.frame(as.table(corr))\n"
        'names(corr_long) <- c("Var1", "Var2", "value")\n'
        "p <- ggplot(corr_long, aes(x = Var1, y = Var2, fill = value)) +\n"
        '  geom_tile(colour = "white") +\n'
        '  geom_text(aes(label = sprintf("%.2f", value)), size = 3) +\n'
        '  scale_fill_gradient2(low = "#3b4cc0", mid = "white", high = "#b40426",\n'
        "                       midpoint = 0, limits = c(-1, 1)) +\n"
        "  coord_fixed() +\n"
        f'  labs(title = {r_str(title)}, x = NULL, y = NULL, fill = "r") +\n'
        f"  {_theme(interactive)} +\n"
        f"  {_rotate_x()}\n"
    )
    return (
        _header(data_file_path, ["ggplot2"])
        + body
        + _finish_ggplot(interactive, width=10, height=10)
    )


def wordcloud(data_file_path: str, text_column: str, title: str,
              extra_stopwords: list[str] | None) -> str:
    """Word cloud of the most frequent words in a text column."""
    stop = 'tm::stopwords("en")'
    if extra_stopwords:
        stop = f"c({stop}, {r_vector(extra_stopwords)})"
    body = (
        "# Word frequencies are counted slightly differently than the Python\n"
        "# wordcloud package (which also merges common two-word phrases).\n"
        f"text <- tolower(paste(na.omit(df[[{r_str(text_column)}]]), collapse = \" \"))\n"
        "words <- unlist(strsplit(text, \"[^[:alnum:]']+\"))\n"
        f"words <- words[nchar(words) > 1 & !(words %in% {stop})]\n"
        "freq <- sort(table(words), decreasing = TRUE)\n"
        "\n"
        '# To save instead of display: png("output.png", width = 1200, height = 600)\n'
        "set.seed(42)\n"
        "wordcloud(names(freq), as.numeric(freq), max.words = 200,\n"
        "          random.order = FALSE, colors = viridisLite::viridis(8))\n"
        f"title(main = {r_str(title)})\n"
        "# dev.off()  # when saving with png()"
    )
    return _header(data_file_path, ["wordcloud", "tm", "viridisLite"]) + body


# =========================================================================
# STATISTICS
# =========================================================================

def _cor_exact(method: str) -> str:
    """Return ``, exact = FALSE`` for rank methods, matching scipy's p-values."""
    return ", exact = FALSE" if method in ("spearman", "kendall") else ""


def correlation_test(data_file_path: str, x_column: str, y_column: str,
                     method: str) -> str:
    """Correlation coefficient with p-value and confidence interval."""
    note = ""
    if _cor_exact(method):
        note = (
            "# exact = FALSE uses the same large-sample p-value as the Python\n"
            "# version (scipy), instead of R's exact test for small samples.\n"
        )
    body = (
        f"clean_df <- na.omit(df[, {r_vector([x_column, y_column])}])\n"
        + note
        + f"result <- cor.test(clean_df[[{r_str(x_column)}]], clean_df[[{r_str(y_column)}]],\n"
        f"                   method = {r_str(method)}{_cor_exact(method)})\n"
        "print(result)"
    )
    return _header(data_file_path, []) + body


def group_comparison(data_file_path: str, target_col: str, group_col: str,
                     groups: list[Any], equal_sizes: bool | None,
                     method: str = "parametric", posthoc: bool = False) -> str:
    """Group comparison mirroring ``run_group_comparison``.

    Args:
        groups: The group values in the order the Python tool compared them.
        equal_sizes: For two groups, whether both have the same size. pingouin
            then uses Student's t-test, otherwise Welch's; R's ``t.test``
            defaults to Welch, so ``var.equal`` is set to match.
        method: 'parametric' (t-test / ANOVA + Tukey) or 'nonparametric'
            (Mann-Whitney / Kruskal-Wallis + pairwise Mann-Whitney, Holm).
        posthoc: Whether the Python tool ran post-hoc comparisons.
    """
    target, group = r_str(target_col), r_str(group_col)
    formula = f"{r_col(target_col)} ~ factor({r_col(group_col)})"
    body = (
        f"clean_df <- na.omit(df[, {r_vector([target_col, group_col])}])\n"
        f"groups <- c({', '.join(r_value(g) for g in groups)})\n"
        f"samples <- lapply(groups, function(g) clean_df[[{target}]][clean_df[[{group}]] == g])\n\n"
    )
    if method == "parametric" and len(groups) == 2:
        body += (
            "# The Python version (pingouin) uses Student's t-test when both groups\n"
            "# have the same size and Welch's t-test otherwise.\n"
            f"result <- t.test(samples[[1]], samples[[2]], var.equal = "
            f"{'TRUE' if equal_sizes else 'FALSE'})\n"
            "print(result)\n"
        )
    elif method == "parametric":
        body += (
            f"model <- aov({formula}, data = clean_df)\n"
            "print(summary(model))\n"
            'ss <- summary(model)[[1]][["Sum Sq"]]\n'
            'cat("Partial eta squared:", ss[1] / sum(ss), "\\n")\n'
        )
        if posthoc:
            body += "print(TukeyHSD(model))\n"
    elif len(groups) == 2:
        body += (
            "# exact = FALSE gives the same large-sample p-value (with continuity\n"
            "# correction) as the Python version.\n"
            "result <- wilcox.test(samples[[1]], samples[[2]], exact = FALSE, correct = TRUE)\n"
            "print(result)\n"
            "n1 <- length(samples[[1]])\n"
            "n2 <- length(samples[[2]])\n"
            'cat("Rank-biserial r:", 2 * unname(result$statistic) / (n1 * n2) - 1, "\\n")\n'
        )
    else:
        body += (
            f"result <- kruskal.test({formula}, data = clean_df)\n"
            "print(result)\n"
            "k <- length(groups)\n"
            'cat("eta2[H]:", (unname(result$statistic) - k + 1) / (nrow(clean_df) - k), "\\n")\n'
        )
        if posthoc:
            body += (
                f"print(pairwise.wilcox.test(clean_df[[{target}]], clean_df[[{group}]],\n"
                '                           p.adjust.method = "holm", exact = FALSE))\n'
            )
    body += (
        "\n# Assumption checks: Shapiro-Wilk per group (3 to 5000 values) and\n"
        "# Levene's test on deviations from the group medians, as in Python.\n"
        "for (i in seq_along(samples)) {\n"
        "  if (length(samples[[i]]) >= 3 && length(samples[[i]]) <= 5000) {\n"
        '    cat("Group", format(groups[i]), "\\n")\n'
        "    print(shapiro.test(samples[[i]]))\n"
        "  }\n"
        "}\n"
        f"deviation <- abs(clean_df[[{target}]] - ave(clean_df[[{target}]], "
        f"clean_df[[{group}]], FUN = median))\n"
        f"print(summary(aov(deviation ~ factor(clean_df[[{group}]]))))"
    )
    return _header(data_file_path, []) + body


def association_test(data_file_path: str, x_column: str, y_column: str,
                     use_fisher: bool) -> str:
    """Crosstab with chi-square (or Fisher's exact) test and Cramer's V."""
    if use_fisher:
        test = (
            "# R reports the conditional maximum-likelihood odds ratio, which differs\n"
            "# slightly from the sample odds ratio in Python; the p-value is the same.\n"
            "result <- fisher.test(tab)\n"
        )
    else:
        test = (
            "# For 2x2 tables both R and Python apply Yates' continuity correction.\n"
            "result <- chisq.test(tab)\n"
        )
    body = (
        f"clean_df <- na.omit(df[, {r_vector([x_column, y_column])}])\n"
        f"tab <- table(clean_df[[{r_str(x_column)}]], clean_df[[{r_str(y_column)}]])\n"
        "print(tab)\n"
        "print(round(100 * prop.table(tab, 1), 1))  # row percentages\n"
        + test
        + "print(result)\n"
        "chi2_raw <- unname(chisq.test(tab, correct = FALSE)$statistic)\n"
        'cat("Cramer\'s V:", sqrt(chi2_raw / (sum(tab) * (min(dim(tab)) - 1))), "\\n")'
    )
    return _header(data_file_path, []) + body


def regression(data_file_path: str, target_col: str, predictor_cols: list[str]) -> str:
    """Ordinary least squares regression with an intercept."""
    formula = f"{r_col(target_col)} ~ " + " + ".join(r_col(c) for c in predictor_cols)
    body = (
        f"clean_df <- na.omit(df[, {r_vector([target_col] + predictor_cols)}])\n"
        f"model <- lm({formula}, data = clean_df)\n"
        "print(summary(model))"
    )
    return _header(data_file_path, []) + body


def logistic_regression(data_file_path: str, target_col: str, predictor_cols: list[str],
                        encoding: tuple[Any, Any] | None) -> str:
    """Logistic regression with odds ratios, mirroring ``run_logistic_regression``.

    Args:
        encoding: ``(positive, negative)`` values when the Python tool coded a
            two-value outcome as 1/0, otherwise None (outcome already 0/1).
    """
    target = r_str(target_col)
    formula = f"{r_col(target_col)} ~ " + " + ".join(r_col(c) for c in predictor_cols)
    body = f"clean_df <- na.omit(df[, {r_vector([target_col] + predictor_cols)}])\n"
    if encoding:
        positive, _ = encoding
        body += (
            "# Code the outcome as 1/0, as the Python version did\n"
            f"clean_df[[{target}]] <- ifelse(clean_df[[{target}]] == {r_value(positive)}, 1, 0)\n"
        )
    body += (
        "# Text predictors become factors; the alphabetically first category is\n"
        "# the reference, as in the Python version.\n"
        f"model <- glm({formula}, family = binomial, data = clean_df)\n"
        "print(summary(model))\n"
        "# Odds ratios with Wald 95% confidence intervals (confint.default), which\n"
        "# is what the Python version reports; confint() would differ slightly.\n"
        "print(exp(cbind(OR = coef(model), confint.default(model))))\n"
        'cat("McFadden pseudo R-squared:", 1 - model$deviance / model$null.deviance, "\\n")'
    )
    return _header(data_file_path, []) + body


def rank_correlations(data_file_path: str, target_col: str, method: str,
                      encoding: tuple[Any, Any] | None) -> str:
    """Correlation of every numeric column with the target, strongest first.

    Args:
        encoding: ``(positive, negative)`` values when the Python tool encoded
            a binary text target as 1/0, otherwise None.
    """
    target = r_str(target_col)
    body = ""
    if encoding:
        positive, negative = encoding
        body += (
            "# Encode the binary target column as 1/0, as the Python version did\n"
            f"df[[{target}]] <- ifelse(df[[{target}]] == {r_value(positive)}, 1,\n"
            f"                  ifelse(df[[{target}]] == {r_value(negative)}, 0, NA))\n"
        )
    body += (
        "num <- df[, sapply(df, is.numeric), drop = FALSE]\n"
        f'corr <- cor(num, use = "pairwise.complete.obs", method = {r_str(method)})[, {target}]\n'
        f"corr <- corr[names(corr) != {target}]\n"
        "ranking <- corr[order(abs(corr), decreasing = TRUE)]\n"
        "print(data.frame(Feature = names(ranking), Correlation = unname(ranking)))"
    )
    return _header(data_file_path, []) + body
