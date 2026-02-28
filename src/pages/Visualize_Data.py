import streamlit as st
import os
import uuid
import asyncio
import pathlib
import ollama
import sys
import tempfile
import zipfile
from io import BytesIO
from PIL import Image


current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
favicon_path = os.path.join(src_dir, "assets", "text_lab_logo.png")

favicon = Image.open(favicon_path)

st.set_page_config(page_title="Visualise Data",page_icon=favicon, layout="wide")

# Make sure we can import auth from parent dir
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from auth import check_token

# Import core engine logic (No duplicated backend logic here!)
from core.chat_engine import check_ollama_server, get_gpu_name
from core.viz_engine import (
    SYSTEM_PROMPT, 
    DEFAULT_PROMPT, 
    parse_dataframe, 
    save_data_file, 
    run_analysis
)

# --- Auth & Ollama ---
check_token()
if not check_ollama_server():
    st.error("Could not connect to Ollama server.")
    st.info("Please check the log file: text_lab/ollama_server.log")
    st.stop()

# --- Dynamic Path Configuration ---
_CURRENT_SCRIPT_DIR = pathlib.Path(__file__).parent       # src/pages/
_SRC_DIR = _CURRENT_SCRIPT_DIR.parent                     # src/
MCP_SERVER_SCRIPT = str(_SRC_DIR / "core" / "mcp_server.py")  # <-- Points to new location
ARTIFACTS_DIR = str(_SRC_DIR / "mcp_artifacts")

# Ensure artifacts base dir exists and is restricted to the owner for privacy
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
try:
    os.chmod(ARTIFACTS_DIR, 0o700) 
except Exception:
    pass

# --- Streamlit Caching ---
@st.cache_data(show_spinner=False)
def load_dataframe_cached(file_name, file_bytes):
    try:
        return parse_dataframe(file_name, file_bytes)
    except Exception as e:
        st.error(str(e))
        return None

# --- Streamlit Page UI ---

st.sidebar.title("Model Selection")
current_gpu = get_gpu_name()
is_high_memory_gpu = any(x in current_gpu for x in ["A100", "H100", "H200"])

small_models = ["ministral-3:14b"]
large_models = ["qwen3-coder-next:latest"]

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

                # Use Core Engine to process files
                file_bytes = uploaded_file.getvalue()
                data_file_path = save_data_file(file_bytes, uploaded_file.name, run_dir)
                
                df = load_dataframe_cached(uploaded_file.name, file_bytes)
                if df is None:
                    st.stop()

                data_head = df.head().to_string()
                final_user_prompt = user_prompt if user_prompt.strip() else DEFAULT_PROMPT

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

                # Run Analysis via Core Engine
                summary, plot_results, logs = asyncio.run(
                    run_analysis(messages, data_file_path, selected_model, MCP_SERVER_SCRIPT)
                )

                # Output any engine warnings/errors to UI
                for log_type, msg in logs:
                    if log_type == "error":
                        st.error(msg)
                    else:
                        st.warning(msg)

                # Load results into memory
                final_artifacts = []
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
                with st.container():
                    st.image(artifact['bytes'], caption=artifact['filename'])
                    with st.expander(f"🐍 View Source Code: {artifact['filename']}"):
                        st.code(artifact['code'], language="python")
                    st.divider()

            # Zip Download
            zip_buffer = BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                for artifact in final_artifacts:
                    zf.writestr(artifact['filename'], artifact['bytes'])
                    code_filename = artifact['filename'].replace('.png', '.py')
                    zf.writestr(code_filename, artifact['code'])

            zip_buffer.seek(0)

            st.download_button(
                label="Download Plots & Code (.zip)",
                data=zip_buffer,
                file_name=f"{run_id}_analysis.zip",
                mime="application/zip",
            )
