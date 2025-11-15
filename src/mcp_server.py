import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mcp.server.fastmcp import FastMCP
import logging

# Set logging level for MCP
logging.basicConfig(level=logging.ERROR)
logging.getLogger("mcp").setLevel(logging.ERROR)

# 1. Create an MCP server instance
mcp = FastMCP("Data Visualisation MCP Server")

# --- Helper Function to Load Data ---
def _load_data(file_path: str) -> pd.DataFrame:
    """Internal helper to load data from various file formats."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found at: {file_path}")
        
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith('.tsv'):
        return pd.read_csv(file_path, sep='\t')
    elif file_path.endswith(('.xls', '.xlsx')):
        return pd.read_excel(file_path)
    elif file_path.endswith('.json'):
        return pd.read_json(file_path)
    else:
        raise ValueError(f"Unsupported file type: {os.path.basename(file_path)}")

def _get_plot_path(data_file_path: str, plot_name: str) -> str:
    """Internal helper to create a unique plot path."""
    # Assumes data_file_path is like ".../ds-xxxxxxxx/uploaded_data.csv"
    run_dir = os.path.dirname(data_file_path)
    plot_dir = os.path.join(run_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    
    # Clean the plot_name to be a valid filename
    safe_plot_name = "".join(c for c in plot_name if c.isalnum() or c in ('_', '-')).rstrip()
    if not safe_plot_name:
        safe_plot_name = "plot"
        
    plot_path = os.path.join(plot_dir, f"{safe_plot_name}.png")
    return plot_path

# --- 2. Define Plotting Tools ---

@mcp.tool()
def plot_histogram(data_file_path: str, column: str, title: str, x_label: str) -> str:
    """
    Generates and saves a histogram for a single numerical column.
    Use this to show the distribution of a variable.
    
    Parameters
    ----------
    data_file_path : str
        The path to the data file (this is injected by the client).
    column : str
        The name of the numerical column to plot.
    title : str
        The title for the plot (e.g., "Distribution of Age").
    x_label : str
        The label for the x-axis (e.g., "Age").
    
    Returns
    -------
    str
        The file path to the saved plot image.
    """
    try:
        df = _load_data(data_file_path)
        if column not in df.columns:
            return f"Error: Column '{column}' not found in data."
            
        plt.figure(figsize=(10, 6))
        sns.histplot(df[column], kde=True)
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel("Frequency")
        
        plot_path = _get_plot_path(data_file_path, f"hist_{column}")
        plt.savefig(plot_path)
        plt.close()
        return plot_path
    except Exception as e:
        return f"Error plotting histogram: {str(e)}"

@mcp.tool()
def plot_scatterplot(data_file_path: str, x_column: str, y_column: str, title: str, x_label: str, y_label: str, hue_column: str = None) -> str:
    """
    Generates and saves a scatter plot for two numerical columns.
    Use this to show the relationship between two variables.
    
    Parameters
    ----------
    data_file_path : str
        The path to the data file (this is injected by the client).
    x_column : str
        The name of the column for the x-axis.
    y_column : str
        The name of the column for the y-axis.
    title : str
        The title for the plot (e.g., "Salary vs. Experience").
    x_label : str
        The label for the x-axis (e.g., "Years of Experience").
    y_label : str
        The label for the y-axis (e.g., "Salary").
    hue_column : str, optional
        A categorical column to use for coloring the points.
    
    Returns
    -------
    str
        The file path to the saved plot image.
    """
    try:
        df = _load_data(data_file_path)
        if x_column not in df.columns or y_column not in df.columns:
            return f"Error: Columns '{x_column}' or '{y_column}' not found."
        
        if hue_column and hue_column not in df.columns:
            hue_column = None # Ignore if bad column name
            
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x=x_column, y=y_column, hue=hue_column)
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        
        plot_path = _get_plot_path(data_file_path, f"scatter_{x_column}_vs_{y_column}")
        plt.savefig(plot_path)
        plt.close()
        return plot_path
    except Exception as e:
        return f"Error plotting scatterplot: {str(e)}"

@mcp.tool()
def plot_boxplot(data_file_path: str, x_column: str, y_column: str, title: str, x_label: str, y_label: str) -> str:
    """
    Generates and saves a box plot.
    Use this to show the distribution of a numerical column (y_column) across different
    categories (x_column).
    
    Parameters
    ----------
    data_file_path : str
        The path to the data file (this is injected by the client).
    x_column : str
        The categorical column for the x-axis (e.g., "Department").
    y_column : str
        The numerical column for the y-axis (e.g., "Salary").
    title : str
        The title for the plot (e.g., "Salary Distribution by Department").
    x_label : str
        The label for the x-axis.
    y_label : str
        The label for the y-axis.
    
    Returns
    -------
    str
        The file path to the saved plot image.
    """
    try:
        df = _load_data(data_file_path)
        if x_column not in df.columns or y_column not in df.columns:
            return f"Error: Columns '{x_column}' or '{y_column}' not found."

        plt.figure(figsize=(12, 7))
        sns.boxplot(data=df, x=x_column, y=y_column)
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.xticks(rotation=45)
        
        plot_path = _get_plot_path(data_file_path, f"boxplot_{y_column}_by_{x_column}")
        plt.savefig(plot_path)
        plt.close()
        return plot_path
    except Exception as e:
        return f"Error plotting boxplot: {str(e)}"
    

@mcp.tool()
def plot_countplot(data_file_path: str, x_column: str, title: str, x_label: str) -> str:
    """
    Generates and saves a count plot (bar chart) for a single categorical column.
    Use this to show the frequency (count) of each category.
    
    Parameters
    ----------
    data_file_path : str
        The path to the data file (this is injected by the client).
    x_column : str
        The name of the categorical column to plot.
    title : str
        The title for the plot (e.g., "Count of Employees by Department").
    x_label : str
        The label for the x-axis (e.g., "Department").
    
    Returns
    -------
    str
        The file path to the saved plot image.
    """
    try:
        df = _load_data(data_file_path)
        if x_column not in df.columns:
            return f"Error: Column '{x_column}' not found in data."
            
        plt.figure(figsize=(12, 7))
        sns.countplot(data=df, x=x_column)
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        
        plot_path = _get_plot_path(data_file_path, f"count_{x_column}")
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        return plot_path
    except Exception as e:
        return f"Error plotting countplot: {str(e)}"

@mcp.tool()
def plot_lineplot(data_file_path: str, x_column: str, y_column: str, title: str, x_label: str, y_label: str, hue_column: str = None) -> str:
    """
    Generates and saves a line plot.
    Use this to show the trend of a numerical variable (y_column) over a
    continuous or ordered variable (x_column, like time or sequence).
    
    Parameters
    ----------
    data_file_path : str
        The path to the data file (this is injected by the client).
    x_column : str
        The column for the x-axis (e.g., "Date", "Year", "Timestamp").
    y_column : str
        The numerical column for the y-axis (e.g., "Stock Price", "Temperature").
    title : str
        The title for the plot (e.g., "Stock Price Over Time").
    x_label : str
        The label for the x-axis.
    y_label : str
        The label for the y-axis.
    hue_column : str, optional
        A categorical column to use for coloring, creating multiple lines (e.g., "Stock_Symbol").
    
    Returns
    -------
    str
        The file path to the saved plot image.
    """
    try:
        df = _load_data(data_file_path)
        if x_column not in df.columns or y_column not in df.columns:
            return f"Error: Columns '{x_column}' or '{y_column}' not found."
        
        if hue_column and hue_column not in df.columns:
            hue_column = None # Ignore if bad column name
            
        plt.figure(figsize=(12, 7))
        sns.lineplot(data=df, x=x_column, y=y_column, hue=hue_column)
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.xticks(rotation=45)
        
        plot_path = _get_plot_path(data_file_path, f"lineplot_{y_column}_over_{x_column}")
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        return plot_path
    except Exception as e:
        return f"Error plotting lineplot: {str(e)}"

@mcp.tool()
def plot_correlation_heatmap(data_file_path: str, title: str = "Correlation Heatmap") -> str:
    """
    Generates and saves a heatmap of the correlation matrix for all numerical columns.
    Use this to get a quick overview of the linear relationships between all numerical variables.
    
    Parameters
    ----------
    data_file_path : str
        The path to the data file (this is injected by the client).
    title : str, optional
        The title for the plot.
    
    Returns
    -------
    str
        The file path to the saved plot image.
    """
    try:
        df = _load_data(data_file_path)
        
        # Select only numerical columns for correlation
        numeric_df = df.select_dtypes(include='number')
        if numeric_df.shape[1] < 2:
            return "Error: Need at least two numerical columns to plot a correlation heatmap."
            
        corr_matrix = numeric_df.corr()
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
        plt.title(title)
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        plot_path = _get_plot_path(data_file_path, "correlation_heatmap")
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        return plot_path
    except Exception as e:
        return f"Error plotting correlation heatmap: {str(e)}"

@mcp.tool()
def plot_violinplot(data_file_path: str, x_column: str, y_column: str, title: str, x_label: str, y_label: str) -> str:
    """
    Generates and saves a violin plot.
    Use this as an alternative to a boxplot. It combines a boxplot with a
    kernel density plot (histogram) to show the distribution shape of the data.
    
    Parameters
    ----------
    data_file_path : str
        The path to the data file (this is injected by the client).
    x_column : str
        The categorical column for the x-axis (e.g., "Department").
    y_column : str
        The numerical column for the y-axis (e.g., "Salary").
    title : str
        The title for the plot (e.g., "Salary Distribution by Department").
    x_label : str
        The label for the x-axis.
    y_label : str
        The label for the y-axis.
    
    Returns
    -------
    str
        The file path to the saved plot image.
    """
    try:
        df = _load_data(data_file_path)
        if x_column not in df.columns or y_column not in df.columns:
            return f"Error: Columns '{x_column}' or '{y_column}' not found."

        plt.figure(figsize=(12, 7))
        sns.violinplot(data=df, x=x_column, y=y_column)
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.xticks(rotation=45)
        
        plot_path = _get_plot_path(data_file_path, f"violinplot_{y_column}_by_{x_column}")
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        return plot_path
    except Exception as e:
        return f"Error plotting violinplot: {str(e)}"

@mcp.tool()
def plot_pairplot(data_file_path: str, columns: list[str] = None, hue_column: str = None, title: str = "Pairwise Relationships") -> str:
    """
    Generates and saves a grid of scatterplots for pairs of numerical columns.
    The diagonal shows histograms or KDEs for each variable.
    Use this for a comprehensive initial exploration of numerical variables.
    
    Parameters
    ----------
    data_file_path : str
        The path to the data file (this is injected by the client).
    columns : list[str], optional
        A list of numerical column names to include. If None, all numerical
        columns will be used.
    hue_column : str, optional
        A categorical column to use for coloring the points.
    title : str, optional
        The main title for the entire plot grid.
    
    Returns
    -------
    str
        The file path to the saved plot image.
    """
    try:
        df = _load_data(data_file_path)
        
        plot_vars = columns
        if plot_vars:
            missing = [col for col in plot_vars if col not in df.columns]
            if missing:
                return f"Error: Columns {missing} not found."
        else:
            plot_vars = df.select_dtypes(include='number').columns.tolist()
            if not plot_vars:
                return "Error: No numerical columns found to plot."
        
        if hue_column and hue_column not in df.columns:
            return f"Error: Hue column '{hue_column}' not found."
        
        # Pairplot creates its own Figure/Grid, so we don't use plt.figure()
        g = sns.pairplot(data=df, vars=plot_vars, hue=hue_column)
        g.fig.suptitle(title, y=1.02) # Adjust title position
        
        plot_path = _get_plot_path(data_file_path, f"pairplot")
        g.savefig(plot_path)
        plt.close(g.fig)
        return plot_path
    except Exception as e:
        return f"Error plotting pairplot: {str(e)}"

# 3. Main entry point to run the server
if __name__ == "__main__":
    # This runs the server over standard input/output
    mcp.run(transport="stdio")