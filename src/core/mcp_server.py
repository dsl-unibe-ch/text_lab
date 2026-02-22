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

MAX_ROWS = 300000

# --- Helper Function to Load Data ---
def _load_data(file_path: str) -> pd.DataFrame:
    """
    Internal helper to load data. 
    Includes safety sampling for huge files to prevent OOM crashes.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found at: {file_path}")
        
    # Helper to check file size (optional logging)
    file_size = os.path.getsize(file_path) / (1024 * 1024) # in MB
    
    try:
        if file_path.endswith('.csv'):
            # Read only first N rows to ensure memory safety
            df = pd.read_csv(file_path, nrows=MAX_ROWS, encoding="latin1")
        elif file_path.endswith('.tsv'):
            df = pd.read_csv(file_path, sep='\t', nrows=MAX_ROWS, encoding="latin1")
        elif file_path.endswith(('.xls', '.xlsx')):
            # Excel doesn't support 'nrows' efficiently like CSV, but we can try
            df = pd.read_excel(file_path, nrows=MAX_ROWS)
        elif file_path.endswith('.json'):
            # JSON is harder to partial read, load normally (risky for huge JSON)
            df = pd.read_json(file_path)
        else:
            raise ValueError(f"Unsupported file type: {os.path.basename(file_path)}")
            
        return df
    except Exception as e:
        # Fallback: if loading fails, try standard load (or reraise)
        raise RuntimeError(f"Failed to load data (size {file_size:.2f}MB): {str(e)}")

def _get_plot_path(data_file_path: str, plot_name: str) -> str:
    """Internal helper to create a unique plot path."""
    run_dir = os.path.dirname(data_file_path)
    plot_dir = os.path.join(run_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    
    safe_plot_name = "".join(c for c in plot_name if c.isalnum() or c in ('_', '-')).rstrip()
    if not safe_plot_name:
        safe_plot_name = "plot"
        
    plot_path = os.path.join(plot_dir, f"{safe_plot_name}.png")
    return plot_path

# --- Helper to format code for the user ---
def _generate_code_snippet(import_lines, load_line, plot_lines):
    return f"""{import_lines}

# Load Data
{load_line}

# Generate Plot
{plot_lines}
plt.show()"""

# --- 2. Define Standard Plotting Tools ---

@mcp.tool()
def plot_histogram(data_file_path: str, column: str, title: str, x_label: str) -> str:
    """Generates and saves a histogram."""
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

        # Generate Code Snippet
        code = _generate_code_snippet(
            "import pandas as pd\nimport seaborn as sns\nimport matplotlib.pyplot as plt",
            f"df = pd.read_csv('your_data.csv') # Replaced temp path",
            f"plt.figure(figsize=(10, 6))\nsns.histplot(df['{column}'], kde=True)\nplt.title('{title}')\nplt.xlabel('{x_label}')\nplt.ylabel('Frequency')"
        )
        return f"{plot_path}|||{code}"
    except Exception as e:
        return f"Error plotting histogram: {str(e)}"

@mcp.tool()
def plot_scatterplot(data_file_path: str, x_column: str, y_column: str, title: str, x_label: str, y_label: str, hue_column: str = None) -> str:
    """Generates and saves a scatter plot."""
    try:
        df = _load_data(data_file_path)
        if x_column not in df.columns or y_column not in df.columns:
            return f"Error: Columns '{x_column}' or '{y_column}' not found."
        
        if hue_column and hue_column not in df.columns:
            hue_column = None 
            
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x=x_column, y=y_column, hue=hue_column)
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        
        plot_path = _get_plot_path(data_file_path, f"scatter_{x_column}_vs_{y_column}")
        plt.savefig(plot_path)
        plt.close()

        # Code
        hue_arg = f", hue='{hue_column}'" if hue_column else ""
        code = _generate_code_snippet(
            "import pandas as pd\nimport seaborn as sns\nimport matplotlib.pyplot as plt",
            f"df = pd.read_csv('your_data.csv')",
            f"plt.figure(figsize=(10, 6))\nsns.scatterplot(data=df, x='{x_column}', y='{y_column}'{hue_arg})\nplt.title('{title}')\nplt.xlabel('{x_label}')\nplt.ylabel('{y_label}')"
        )
        return f"{plot_path}|||{code}"
    except Exception as e:
        return f"Error plotting scatterplot: {str(e)}"

@mcp.tool()
def plot_boxplot(data_file_path: str, x_column: str, y_column: str, title: str, x_label: str, y_label: str) -> str:
    """Generates and saves a box plot."""
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

        code = _generate_code_snippet(
            "import pandas as pd\nimport seaborn as sns\nimport matplotlib.pyplot as plt",
            f"df = pd.read_csv('your_data.csv')",
            f"plt.figure(figsize=(12, 7))\nsns.boxplot(data=df, x='{x_column}', y='{y_column}')\nplt.title('{title}')\nplt.xlabel('{x_label}')\nplt.ylabel('{y_label}')\nplt.xticks(rotation=45)"
        )
        return f"{plot_path}|||{code}"
    except Exception as e:
        return f"Error plotting boxplot: {str(e)}"
    

@mcp.tool()
def plot_countplot(data_file_path: str, x_column: str, title: str, x_label: str) -> str:
    """Generates and saves a count plot."""
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

        code = _generate_code_snippet(
            "import pandas as pd\nimport seaborn as sns\nimport matplotlib.pyplot as plt",
            f"df = pd.read_csv('your_data.csv')",
            f"plt.figure(figsize=(12, 7))\nsns.countplot(data=df, x='{x_column}')\nplt.title('{title}')\nplt.xlabel('{x_label}')\nplt.ylabel('Count')\nplt.xticks(rotation=45)"
        )
        return f"{plot_path}|||{code}"
    except Exception as e:
        return f"Error plotting countplot: {str(e)}"

@mcp.tool()
def plot_lineplot(data_file_path: str, x_column: str, y_column: str, title: str, x_label: str, y_label: str, hue_column: str = None) -> str:
    """Generates and saves a line plot."""
    try:
        df = _load_data(data_file_path)
        if x_column not in df.columns or y_column not in df.columns:
            return f"Error: Columns '{x_column}' or '{y_column}' not found."
        
        if hue_column and hue_column not in df.columns:
            hue_column = None
            
        plt.figure(figsize=(12, 7))
        sns.lineplot(data=df, x=x_column, y=y_column, hue=hue_column)
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.xticks(rotation=45)
        
        plot_path = _get_plot_path(data_file_path, f"lineplot_{y_column}_over_{x_column}")
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()

        hue_arg = f", hue='{hue_column}'" if hue_column else ""
        code = _generate_code_snippet(
            "import pandas as pd\nimport seaborn as sns\nimport matplotlib.pyplot as plt",
            f"df = pd.read_csv('your_data.csv')",
            f"plt.figure(figsize=(12, 7))\nsns.lineplot(data=df, x='{x_column}', y='{y_column}'{hue_arg})\nplt.title('{title}')\nplt.xlabel('{x_label}')\nplt.ylabel('{y_label}')\nplt.xticks(rotation=45)"
        )
        return f"{plot_path}|||{code}"
    except Exception as e:
        return f"Error plotting lineplot: {str(e)}"

@mcp.tool()
def plot_correlation_heatmap(data_file_path: str, title: str = "Correlation Heatmap") -> str:
    """Generates and saves a correlation heatmap."""
    try:
        df = _load_data(data_file_path)
        numeric_df = df.select_dtypes(include='number')
        if numeric_df.shape[1] < 2:
            return "Error: Need at least two numerical columns."
            
        corr_matrix = numeric_df.corr()
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
        plt.title(title)
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        plot_path = _get_plot_path(data_file_path, "correlation_heatmap")
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()

        code = _generate_code_snippet(
            "import pandas as pd\nimport seaborn as sns\nimport matplotlib.pyplot as plt",
            f"df = pd.read_csv('your_data.csv')\nnumeric_df = df.select_dtypes(include='number')\ncorr_matrix = numeric_df.corr()",
            f"plt.figure(figsize=(12, 10))\nsns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)\nplt.title('{title}')\nplt.xticks(rotation=45)\nplt.yticks(rotation=0)"
        )
        return f"{plot_path}|||{code}"
    except Exception as e:
        return f"Error plotting correlation heatmap: {str(e)}"

@mcp.tool()
def plot_violinplot(data_file_path: str, x_column: str, y_column: str, title: str, x_label: str, y_label: str) -> str:
    """Generates and saves a violin plot."""
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

        code = _generate_code_snippet(
            "import pandas as pd\nimport seaborn as sns\nimport matplotlib.pyplot as plt",
            f"df = pd.read_csv('your_data.csv')",
            f"plt.figure(figsize=(12, 7))\nsns.violinplot(data=df, x='{x_column}', y='{y_column}')\nplt.title('{title}')\nplt.xlabel('{x_label}')\nplt.ylabel('{y_label}')\nplt.xticks(rotation=45)"
        )
        return f"{plot_path}|||{code}"
    except Exception as e:
        return f"Error plotting violinplot: {str(e)}"

@mcp.tool()
def plot_pairplot(data_file_path: str, columns: list[str] = None, hue_column: str = None, title: str = "Pairwise Relationships") -> str:
    """Generates and saves a pairplot."""
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
        
        g = sns.pairplot(data=df, vars=plot_vars, hue=hue_column)
        g.fig.suptitle(title, y=1.02)
        
        plot_path = _get_plot_path(data_file_path, f"pairplot")
        g.savefig(plot_path)
        plt.close(g.fig)

        hue_arg = f", hue='{hue_column}'" if hue_column else ""
        vars_arg = f", vars={plot_vars}" if columns else ""
        code = _generate_code_snippet(
            "import pandas as pd\nimport seaborn as sns\nimport matplotlib.pyplot as plt",
            f"df = pd.read_csv('your_data.csv')",
            f"g = sns.pairplot(data=df{vars_arg}{hue_arg})\ng.fig.suptitle('{title}', y=1.02)"
        )
        return f"{plot_path}|||{code}"
    except Exception as e:
        return f"Error plotting pairplot: {str(e)}"

# --- 3. Advanced Custom Plot Tool ---

@mcp.tool()
def generate_custom_plot(data_file_path: str, python_code: str, plot_filename_keyword: str) -> str:
    """
    Executes custom Python plotting code using matplotlib/seaborn.
    Use this when the user requests a specific visualization that is not covered
    by the standard tools (e.g., specific colors, complex subplots, regression lines).

    Parameters
    ----------
    data_file_path : str
        The path to the data file (injected).
    python_code : str
        The valid python code to execute.
        - You have access to pandas as `pd`, seaborn as `sns`, and matplotlib.pyplot as `plt`.
        - The data is ALREADY loaded into a variable named `df`. DO NOT reload it.
        - DO NOT call `plt.show()`. Just create the plot.
        - The code will be executed in a restricted scope.
    plot_filename_keyword : str
        A short string to use in the filename (e.g., "age_regression").

    Returns
    -------
    str
        The file path ||| the code used.
    """
    try:
        # 1. Load Data
        df = _load_data(data_file_path)

        # 2. Prepare Sandbox Scope
        local_scope = {
            "pd": pd,
            "sns": sns,
            "plt": plt,
            "df": df
        }

        # 3. Clean up the code (remove markdown backticks if LLM provides them)
        clean_code = python_code.replace("```python", "").replace("```", "").strip()

        # 4. Reset Figure (Important to not draw over previous plots)
        plt.clf()
        plt.figure(figsize=(10, 6)) # Default size, user code can override this

        # 5. Execute Code
        exec(clean_code, {}, local_scope)

        # 6. Save Result
        plot_path = _get_plot_path(data_file_path, f"custom_{plot_filename_keyword}")
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()

        # 7. Return Code Snippet (including the implicit imports/loading for the user)
        full_user_code = _generate_code_snippet(
            "import pandas as pd\nimport seaborn as sns\nimport matplotlib.pyplot as plt",
            f"df = pd.read_csv('your_data.csv')",
            clean_code
        )

        return f"{plot_path}|||{full_user_code}"

    except Exception as e:
        return f"Error executing custom plot code: {str(e)}"

if __name__ == "__main__":
    mcp.run(transport="stdio")