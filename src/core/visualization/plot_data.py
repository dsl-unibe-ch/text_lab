"""
Data Exploration Module for the AI Visualization Engine.
Provides tools for the LLM to inspect dataset columns before plotting.
"""

import pandas as pd

from core.visualization.viz_utils import load_data_safely


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
                f"Mean={mean_val:.2f}, Nulls={null_count}"
            )
        else:
            unique_vals = col_data.unique()
            total_unique = len(unique_vals)
            
            if total_unique > 10:
                top_vals = ", ".join(map(str, unique_vals[:10]))
                val_str = f"{top_vals}... (+ {total_unique - 10} more)"
            else:
                val_str = ", ".join(map(str, unique_vals))
                
            return (
                f"Categorical Column '{column}': {total_unique} unique values. "
                f"Top values: {val_str}. Nulls={null_count}"
            )
            
    except Exception as e:
        return f"Error analyzing column '{column}': {str(e)}"