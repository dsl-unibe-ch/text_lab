import streamlit as st
import pandas as pd
import os
import uuid
import asyncio
import pathlib
import ollama
import sys
import tempfile
import zipfile
from io import BytesIO

from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
from utils import ensure_ollama_server

# Make sure we can import auth from parent dir
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from auth import check_token

# --- Auth & Ollama ---
check_token()
ensure_ollama_server()

# --- Configuration ---
MODEL = "qwen3:32b"

# Dynamic Path Configuration
_CURRENT_SCRIPT_DIR = pathlib.Path(__file__).parent       # src/pages/
_SRC_DIR = _CURRENT_SCRIPT_DIR.parent                     # src/
MCP_SERVER_SCRIPT = str(_SRC_DIR / "mcp_server.py")
ARTIFACTS_DIR = str(_SRC_DIR / "mcp_artifacts")

# Ensure artifacts base dir exists (required for TemporaryDirectory(dir=...))
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# --- System & Default Prompts ---
SYSTEM_PROMPT = """
You are an expert data analyst. Your task is to generate visualisations based on a user's request and the first 5 rows of their dataset.

1.  You have access to a suite of plotting tools:
    * `plot_histogram` (for numerical distributions)
    * `plot_countplot` (for categorical distributions/counts)
    * `plot_scatterplot` (for relationships between two numerical variables)
    * `plot_boxplot` (for numerical-by-categorical distributions)
    * `plot_violinplot` (an alternative to boxplot, showing distribution shape)
    * `plot_lineplot` (for trends, often over time or a sequence)
    * `plot_correlation_heatmap` (for a single overview of all numerical relationships)
    * `plot_pairplot` (for a detailed grid of all pairwise numerical relationships)

2.  The user will provide a prompt and the `head()` of their data.

3.  Based on the prompt and the data columns (names and types), you must decide which plotting tools to call. Choose the most appropriate tools for the user's request and the data provided.

4.  **CRITICAL:** Your tools require a `data_file_path` argument. You DO NOT need to provide this. It will be injected for you. You only need to provide the *other* arguments (like `column`, `x_column`, `y_column`, `title`, etc.) based on the data head.

5.  Call multiple tools if it makes sense. For example, if the user asks for a general analysis, you could call `plot_correlation_heatmap` for a numerical overview, `plot_countplot` for key categorical columns, and `plot_histogram` for key numerical columns.

6.  After you call the tools, you will receive their output (which are file paths).

7.  You must then provide a final, single response to the user. This response should be a brief, non-technical summary in Markdown, describing what you did (e.g., "I generated a histogram for the 'Age' column, a count plot for 'Department', and a scatter plot to explore 'Age' vs. 'Salary'.").

8.  Do not just list the tool names. Provide a helpful, narrative summary. Do not mention file paths or errors.
"""

DEFAULT_PROMPT = "Please perform a basic exploratory data analysis. Generate a few useful plots to understand the data's distribution and relationships."


# --- Helper Functions ---

@st.cache_data
def load_dataframe(uploaded_file):
    """Loads an uploaded file into a pandas DataFrame."""
    try:
        if uploaded_file.name.endswith(".csv"):
            return pd.read_csv(uploaded_file, encoding="latin1")
        elif uploaded_file.name.endswith(".tsv"):
            return pd.read_csv(uploaded_file, sep="\t", encoding="latin1")
        elif uploaded_file.name.endswith((".xls", ".xlsx")):
            uploaded_file.seek(0)
            return pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith(".json"):
            return pd.read_json(uploaded_file)
        else:
            st.error(f"Unsupported file type: {uploaded_file.name}")
            return None
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None


def save_uploaded_file(uploaded_file, run_dir):
    """Saves the uploaded file to the specific run directory."""
    file_extension = os.path.splitext(uploaded_file.name)[1]
    data_file_path = os.path.join(run_dir, f"uploaded_data{file_extension}")

    with open(data_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return data_file_path


async def get_mcp_tools(session: ClientSession) -> list:
    """Fetches tools from the MCP server and formats them for Ollama."""
    tool_list_response = await session.list_tools()
    ollama_tools = []
    for tool in tool_list_response.tools:
        ollama_tools.append(
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema,
                },
            }
        )
    return ollama_tools


async def run_analysis(messages, data_file_path):
    """
    Main async logic to connect to MCP, call Ollama, and execute tools.
    """
    server_params = StdioServerParameters(
        command="python3",
        args=[MCP_SERVER_SCRIPT],
    )

    plot_paths = []
    summary = ""

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await get_mcp_tools(session)

            # 1. First call to Ollama to get tool calls
            try:
                response = ollama.chat(
                    model=MODEL,
                    messages=messages,
                    tools=tools,
                )
                messages.append(response["message"])
            except Exception as e:
                st.error(f"Error calling Ollama: {e}")
                return "Failed to get analysis from LLM.", []

            # 2. Check for and execute tool calls
            if response["message"].get("tool_calls"):
                tool_calls = response["message"]["tool_calls"]

                tool_results = []
                for tool_call in tool_calls:
                    tool_name = tool_call["function"]["name"]
                    tool_args = tool_call["function"]["arguments"]

                    # Inject the data_file_path
                    tool_args["data_file_path"] = data_file_path

                    try:
                        # 3. Execute the tool by calling the MCP server
                        result = await session.call_tool(
                            tool_name, arguments=tool_args
                        )

                        tool_output = ""
                        if result.content and isinstance(
                            result.content[0], types.TextContent
                        ):
                            tool_output = result.content[0].text

                        # Check if tool returned an error
                        if "Error:" in tool_output:
                            st.warning(
                                f"Tool '{tool_name}' failed: {tool_output}"
                            )
                            tool_results.append(
                                {
                                    "role": "tool",
                                    "content": f"Tool Error: {tool_output}",
                                }
                            )
                        else:
                            # Success, save the plot path
                            plot_paths.append(tool_output)
                            tool_results.append(
                                {
                                    "role": "tool",
                                    "content": f"Successfully generated plot at {tool_output}",
                                }
                            )

                    except Exception as e:
                        st.error(f"Failed to execute tool '{tool_name}': {e}")
                        tool_results.append(
                            {
                                "role": "tool",
                                "content": f"Failed to execute tool: {str(e)}",
                            }
                        )

                # 4. Send tool results back to Ollama for final summary
                messages.extend(tool_results)
                try:
                    final_response = ollama.chat(model=MODEL, messages=messages)
                    summary = final_response["message"]["content"]
                except Exception as e:
                    st.error(f"Error getting final summary from Ollama: {e}")
                    summary = "Failed to generate summary, but plots were created."

            else:
                # No tool was called, just use the response content
                summary = response["message"]["content"]

    return summary, plot_paths


# --- Streamlit Page UI ---

st.set_page_config(layout="wide")
st.title("🤖 AI Data Visualiser")

st.info(f"Using Model: **{MODEL}**")

uploaded_file = st.file_uploader(
    "Upload your data file (CSV, TSV, Excel, JSON)",
    type=["csv", "tsv", "xls", "xlsx", "json"],
)

user_prompt = st.text_area(
    "Describe what you want to do (optional)",
    placeholder=DEFAULT_PROMPT,
)

if st.button("Generate Visualisations", type="primary", disabled=(not uploaded_file)):
    with st.spinner("AI is analyzing your data and generating plots..."):
        run_id = f"ds-{uuid.uuid4().hex[:8]}"

        try:
            # 1. Create a truly temporary directory (auto-deleted)
            with tempfile.TemporaryDirectory(dir=ARTIFACTS_DIR) as run_dir:
                # Optional: create a 'plots' subdir
                plot_dir = os.path.join(run_dir, "plots")
                os.makedirs(plot_dir, exist_ok=True)

                # Save uploaded file ONLY inside this temp dir
                data_file_path = save_uploaded_file(uploaded_file, run_dir)

                # 2. Load data head for the prompt (in memory)
                df = load_dataframe(uploaded_file)
                if df is None:
                    st.stop()  # error already shown

                data_head = df.head().to_string()

                # 3. Format messages for Ollama
                final_user_prompt = (
                    user_prompt if user_prompt.strip() else DEFAULT_PROMPT
                )

                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"User Request: {final_user_prompt}\n\n"
                            f"Data Head:\n{data_head}"
                        ),
                    },
                ]

                # 4. Run the full async analysis
                summary, plot_paths = asyncio.run(
                    run_analysis(messages, data_file_path)
                )

                # 5. Load all plots into memory BEFORE the temp dir is deleted
                images = []  # list of (filename, bytes)
                for plot_path in plot_paths:
                    if os.path.exists(plot_path):
                        filename = os.path.basename(plot_path)
                        with open(plot_path, "rb") as f:
                            img_bytes = f.read()
                        images.append((filename, img_bytes))
                    else:
                        st.error(f"Could not find plot at: {plot_path}")

                # When we exit the 'with TemporaryDirectory' block,
                # run_dir, the uploaded data file, and all plot files
                # are deleted from disk automatically.

        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            # Stop execution cleanly in Streamlit
            st.stop()

        # ========== After this line, NOTHING is left on disk ==========
        # Only 'images' (bytes in RAM) and 'summary' remain.

        st.success("Analysis Complete!")

        st.subheader("Analysis Summary")
        st.markdown(summary)

        st.subheader("Generated Visualisations")
        if not images:
            st.warning("No plots were generated. Try refining your prompt.")
        else:
            # Display images from RAM
            for filename, img_bytes in images:
                st.image(img_bytes, caption=filename)

            # Build ZIP in-memory
            zip_buffer = BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                for filename, img_bytes in images:
                    zf.writestr(filename, img_bytes)

            zip_buffer.seek(0)

            st.download_button(
                label="Download All Plots (.zip)",
                data=zip_buffer,
                file_name=f"{run_id}_plots.zip",
                mime="application/zip",
            )
