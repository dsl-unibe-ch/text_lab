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
import subprocess 
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

# --- Dynamic Path Configuration ---
_CURRENT_SCRIPT_DIR = pathlib.Path(__file__).parent       # src/pages/
_SRC_DIR = _CURRENT_SCRIPT_DIR.parent                     # src/
MCP_SERVER_SCRIPT = str(_SRC_DIR / "mcp_server.py")
ARTIFACTS_DIR = str(_SRC_DIR / "mcp_artifacts")

# Ensure artifacts base dir exists
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# --- System & Default Prompts ---
SYSTEM_PROMPT = """
You are an expert data analyst. Your task is to generate visualisations based on a user's request and the first 5 rows of their dataset.

1.  You have access to a suite of plotting tools:
    * `plot_histogram` (for numerical distributions)
    * `plot_countplot` (for categorical distributions/counts)
    * `plot_scatterplot` (for relationships between two numerical variables)
    * `plot_boxplot` (for numerical-by-categorical distributions)
    * `plot_violinplot` (for distribution shape)
    * `plot_lineplot` (for trends)
    * `plot_correlation_heatmap` (for numerical overview)
    * `plot_pairplot` (for pairwise relationships)

2.  The user will provide a prompt and the `head()` of their data.

3.  Based on the prompt and the data columns (names and types), choose the most appropriate plotting tools.

4.  **CRITICAL:** Your tools require a `data_file_path` argument. You DO NOT need to provide this. It will be injected for you. You only need to provide the *other* arguments.

5.  Call multiple tools if it makes sense.

6.  After you call the tools, you will receive their output. Provide a final, single response summarizing what you did in Markdown.
"""

DEFAULT_PROMPT = "Please perform a basic exploratory data analysis. Generate a few useful plots to understand the data's distribution and relationships."


# --- Helper Functions ---

def get_gpu_name():
    try:
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], 
            encoding="utf-8"
        )
        return result.strip()
    except Exception:
        return "Unknown/CPU"

@st.cache_data
def load_dataframe(uploaded_file):
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
    file_extension = os.path.splitext(uploaded_file.name)[1]
    data_file_path = os.path.join(run_dir, f"uploaded_data{file_extension}")
    with open(data_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return data_file_path


async def get_mcp_tools(session: ClientSession) -> list:
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


async def run_analysis(messages, data_file_path, model_name):
    server_params = StdioServerParameters(
        command="python3",
        args=[MCP_SERVER_SCRIPT],
    )

    # plot_results will store dicts: {'path': str, 'code': str}
    plot_results = []
    summary = ""

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await get_mcp_tools(session)

            # 1. First call to Ollama
            try:
                response = ollama.chat(
                    model=model_name,
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
                mcp_tool_results = []

                for tool_call in tool_calls:
                    tool_name = tool_call["function"]["name"]
                    tool_args = tool_call["function"]["arguments"]
                    tool_args["data_file_path"] = data_file_path

                    try:
                        # 3. Execute tool
                        result = await session.call_tool(tool_name, arguments=tool_args)

                        tool_output_raw = ""
                        if result.content and isinstance(result.content[0], types.TextContent):
                            tool_output_raw = result.content[0].text

                        if "Error:" in tool_output_raw:
                            st.warning(f"Tool '{tool_name}' failed: {tool_output_raw}")
                            mcp_tool_results.append({
                                "role": "tool",
                                "content": f"Tool Error: {tool_output_raw}",
                            })
                        else:
                            # -----------------------------------------------------
                            # PARSE RESULT: Check for "|||" delimiter (Path ||| Code)
                            # -----------------------------------------------------
                            if "|||" in tool_output_raw:
                                path_part, code_part = tool_output_raw.split("|||", 1)
                                plot_results.append({
                                    "path": path_part.strip(),
                                    "code": code_part.strip(),
                                    "name": tool_name
                                })
                                # Return only the path to the LLM so it doesn't get confused
                                mcp_tool_results.append({
                                    "role": "tool",
                                    "content": f"Successfully generated plot at {path_part.strip()}",
                                })
                            else:
                                # Fallback for old style
                                plot_results.append({
                                    "path": tool_output_raw.strip(),
                                    "code": "# Code transparency not available for this plot.",
                                    "name": tool_name
                                })
                                mcp_tool_results.append({
                                    "role": "tool",
                                    "content": f"Successfully generated plot at {tool_output_raw}",
                                })

                    except Exception as e:
                        st.error(f"Failed to execute tool '{tool_name}': {e}")
                        mcp_tool_results.append({
                            "role": "tool",
                            "content": f"Failed to execute tool: {str(e)}",
                        })

                # 4. Send results back to Ollama
                messages.extend(mcp_tool_results)
                try:
                    final_response = ollama.chat(model=model_name, messages=messages)
                    summary = final_response["message"]["content"]
                except Exception as e:
                    st.error(f"Error getting final summary: {e}")
                    summary = "Failed to generate summary, but plots were created."

            else:
                summary = response["message"]["content"]

    return summary, plot_results


# --- Streamlit Page UI ---

st.set_page_config(layout="wide")

st.sidebar.title("Model Selection")
current_gpu = get_gpu_name()
is_high_memory_gpu = any(x in current_gpu for x in ["A100", "H100", "H200"])

small_models = ["ministral-3:14b"]
large_models = ["qwen3-next:80b", "qwen3-coder-next:latest"]

if is_high_memory_gpu:
    available_models = small_models + large_models
    gpu_badge = f"🚀 **High-Performance Mode** ({current_gpu})"
else:
    available_models = small_models
    gpu_badge = f"⚠️ **Standard Mode** ({current_gpu})"

st.sidebar.markdown(gpu_badge)

selected_model = st.sidebar.selectbox(
    "Select Analysis Model:",
    options=available_models,
    index=0
)

# Auto-pull logic
try:
    current_models = [m.get('model') for m in ollama.list().get('models', [])]
except:
    pass

st.title("🤖 AI Data Visualiser")
st.info(f"Using Model: **{selected_model}**")

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
            ollama.pull(selected_model)
        except Exception as e:
            st.error(f"Failed to pull model {selected_model}: {e}")
            st.stop()

        try:
            with tempfile.TemporaryDirectory(dir=ARTIFACTS_DIR) as run_dir:
                plot_dir = os.path.join(run_dir, "plots")
                os.makedirs(plot_dir, exist_ok=True)

                data_file_path = save_uploaded_file(uploaded_file, run_dir)
                df = load_dataframe(uploaded_file)
                if df is None:
                    st.stop()

                data_head = df.head().to_string()

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

                # Run Analysis
                summary, plot_results = asyncio.run(
                    run_analysis(messages, data_file_path, selected_model)
                )

                # Load results into memory
                final_artifacts = [] # list of dicts: {filename, bytes, code}
                for item in plot_results:
                    path = item['path']
                    code = item['code']
                    if os.path.exists(path):
                        filename = os.path.basename(path)
                        with open(path, "rb") as f:
                            img_bytes = f.read()
                        final_artifacts.append({
                            "filename": filename,
                            "bytes": img_bytes,
                            "code": code
                        })
                    else:
                        st.error(f"Could not find plot at: {path}")

        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            st.stop()

        # Display Results
        st.success("Analysis Complete!")
        st.subheader("Analysis Summary")
        st.markdown(summary)

        st.subheader("Generated Visualisations")
        if not final_artifacts:
            st.warning("No plots were generated. Try refining your prompt.")
        else:
            # Display Loop
            for artifact in final_artifacts:
                # Use a container for better grouping
                with st.container():
                    st.image(artifact['bytes'], caption=artifact['filename'])
                    # --- CODE TRANSPARENCY BLOCK ---
                    with st.expander(f"🐍 View Source Code: {artifact['filename']}"):
                        st.code(artifact['code'], language="python")
                    st.divider()

            # Zip Download
            zip_buffer = BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                for artifact in final_artifacts:
                    zf.writestr(artifact['filename'], artifact['bytes'])
                    # Also save the code as a separate .py file in the zip
                    code_filename = artifact['filename'].replace('.png', '.py')
                    zf.writestr(code_filename, artifact['code'])

            zip_buffer.seek(0)

            st.download_button(
                label="Download Plots & Code (.zip)",
                data=zip_buffer,
                file_name=f"{run_id}_analysis.zip",
                mime="application/zip",
            )