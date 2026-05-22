"""
Statistical Analysis Module for the AI Visualization Engine.
Provides tools for the LLM to run advanced statistical tests using Pingouin and Statsmodels.
"""

import pandas as pd
import pingouin as pg
import statsmodels.api as sm

from core.visualization.viz_utils import load_data_safely


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
        A markdown-formatted string of the correlation results, or an error message.
    """
    try:
        df = load_data_safely(data_file_path)
        if x_column not in df.columns or y_column not in df.columns:
            return f"Error: Columns '{x_column}' and/or '{y_column}' not found."

        # Drop NaNs to prevent calculation errors
        clean_df = df[[x_column, y_column]].dropna()
        if len(clean_df) < 3:
            return "Error: Not enough valid data points to calculate correlation."

        res = pg.corr(clean_df[x_column], clean_df[y_column], method=method)
        return (
            f"Correlation Analysis ({method}) between '{x_column}' and '{y_column}':\n\n"
            f"{res.to_markdown()}"
        )
    except Exception as e:
        return f"Error computing correlation: {str(e)}"


def run_group_comparison_impl(
    data_file_path: str, target_col: str, group_col: str
) -> str:
    """
    Automatically performs a T-test (for 2 groups) or ANOVA (for >2 groups) 
    on a target numeric variable grouped by a categorical variable.

    Args:
        data_file_path: The absolute path to the data file.
        target_col: The numeric column to test.
        group_col: The categorical column defining the groups.

    Returns:
        A markdown-formatted string of the test results, or an error message.
    """
    try:
        df = load_data_safely(data_file_path)
        if target_col not in df.columns or group_col not in df.columns:
            return f"Error: Columns '{target_col}' or '{group_col}' not found."

        clean_df = df[[target_col, group_col]].dropna()
        groups = clean_df[group_col].unique()

        if len(groups) == 2:
            group1 = clean_df[clean_df[group_col] == groups[0]][target_col]
            group2 = clean_df[clean_df[group_col] == groups[1]][target_col]
            res = pg.ttest(group1, group2)
            return (
                f"Independent T-test for '{target_col}' grouped by '{group_col}' "
                f"(Groups: {groups[0]} vs {groups[1]}):\n\n{res.to_markdown()}"
            )
        elif len(groups) > 2:
            res = pg.anova(dv=target_col, between=group_col, data=clean_df, detailed=True)
            return (
                f"One-way ANOVA for '{target_col}' grouped by '{group_col}' "
                f"({len(groups)} groups):\n\n{res.to_markdown()}"
            )
        else:
            return "Error: The grouping column must have at least 2 unique values."
    except Exception as e:
        return f"Error computing group comparison: {str(e)}"


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
        A formatted string of the statsmodels regression summary, or an error message.
    """
    try:
        df = load_data_safely(data_file_path)
        
        # Verify all columns exist
        missing_cols = [col for col in [target_col] + predictor_cols if col not in df.columns]
        if missing_cols:
            return f"Error: Missing columns in data: {', '.join(missing_cols)}"

        # Drop rows with NaNs in the required columns
        cols = [target_col] + predictor_cols
        clean_df = df[cols].dropna()
        
        if len(clean_df) <= len(predictor_cols):
            return "Error: Not enough data points to run regression after dropping NaNs."

        X = clean_df[predictor_cols]
        y = clean_df[target_col]

        # Add a constant (intercept) to the model
        X = sm.add_constant(X)
        model = sm.OLS(y, X).fit()
        
        return (
            f"OLS Linear Regression Results (Target: {target_col}):\n\n"
            f"{model.summary().as_text()}"
        )
    except Exception as e:
        return f"Error computing linear regression: {str(e)}"