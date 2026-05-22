import asyncio
import os
import pathlib
import sys
import tempfile
import uuid
import zipfile
from io import BytesIO

import ollama
import pandas as pd
import plotly.io as pio
import streamlit as st
from PIL import Image

# Ensure absolute imports resolve correctly
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.append(src_dir)

from auth import check_token
from core.chat_engine import check_ollama_server, get_gpu_name
from core.visualization.viz_agent import run_analysis
from core.visualization.viz_config import DEFAULT_PROMPT
from core.visualization.viz_utils import get_fast_data_preview, save_data_file

# --- Page Configuration ---
favicon_path = os.path.join(src_dir, "assets", "text_lab_logo.png")
try:
    favicon = Image.open(favicon_path)
    st.set_page_config(page_title="Visualise Data", page_icon=favicon, layout="wide")
except FileNotFoundError:
    st.set_page_config(page_title="Visualise Data", layout="wide")

# --- Dynamic Path Configuration ---
_CURRENT_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
_SRC_DIR = _CURRENT_SCRIPT_DIR.parent
MCP_SERVER_SCRIPT = str(_SRC_DIR / "core" / "visualization" / "mcp_server.py")
ARTIFACTS_DIR = str(_SRC_DIR / "mcp_artifacts")

os.makedirs(ARTIFACTS_DIR, exist_ok=True)
try:
    os.chmod(ARTIFACTS_DIR, 0o700)
except Exception:
    pass


def render_sidebar() -> str:
    """
    Renders the sidebar for model selection based on hardware capabilities.

    Returns:
        The name of the selected Ollama model.
    """
    st.sidebar.title("Model Selection")
    
    current_gpu = get_gpu_name()
    is_high_memory_gpu = any(x in current_gpu for x in ["A100", "H100", "H200"])

    small_models = ["gemma4:26b", "ministral-3:14b"]
    large_models = ["qwen3-coder-next:latest", "gemma4:31b", "deepseek-r1:70b", "llama3.3", "firefunction-v2", "qwen3.6:35b", "qwen3.5:122b", "mistral-medium-3.5:128b"]

    if is_high_memory_gpu:
        available_models = small_models + large_models
        gpu_badge = f"High-Performance Mode ({current_gpu})"
    else:
        available_models = small_models
        gpu_badge = f"Standard Mode ({current_gpu})"

    st.sidebar.markdown(f"**{gpu_badge}**")

    selected_model = st.sidebar.selectbox(
        "Select Analysis Model:",
        options=available_models,
        index=0
    )
    
    return str(selected_model)


def render_results(
    summary: str, 
    final_artifacts: list[dict[str, str | bytes]], 
    run_id: str
) -> None:
    """
    Renders the final LLM summary, generated plots, and the download button.

    Args:
        summary: The markdown summary returned by the LLM.
        final_artifacts: A list of dictionaries containing the plot metadata and bytes.
        run_id: The unique identifier for the current analysis run.
    """
    st.success("Analysis Complete.")
    st.subheader("Analysis Summary")
    st.markdown(summary)

    st.subheader("Generated Visualisations")
    
    if not final_artifacts:
        st.warning("No plots were generated. Try refining your prompt.")
        return

    # Render visualizations to the UI
    for idx, artifact in enumerate(final_artifacts):
        filename = artifact["filename"]
        file_bytes = artifact["bytes"]
        code = artifact["code"]
        
        with st.container():
            if filename.endswith(".json"):
                # Render interactive Plotly chart
                json_str = file_bytes.decode("utf-8")
                fig = pio.from_json(json_str)
                # ADDED UNIQUE KEY HERE using the run_id and loop index
                st.plotly_chart(fig, use_container_width=True, key=f"plotly_{run_id}_{idx}")
            else:
                # Render static Matplotlib/Seaborn image
                st.image(file_bytes, caption=filename)
                
            with st.expander(f"View Source Code: {filename}"):
                st.code(code, language="python")
            
            st.divider()

    # Build the ZIP archive for downloading
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for artifact in final_artifacts:
            filename = artifact["filename"]
            file_bytes = artifact["bytes"]
            code = artifact["code"]
            
            if filename.endswith(".json"):
                # Convert Plotly JSON back to an interactive HTML file for offline viewing
                fig = pio.from_json(file_bytes.decode("utf-8"))
                html_filename = filename.replace(".json", ".html")
                zf.writestr(html_filename, fig.to_html(include_plotlyjs="cdn"))
            else:
                # Save standard static images
                zf.writestr(filename, file_bytes)
                
            # Save the associated Python script
            code_filename = filename.replace(".json", ".py").replace(".png", ".py")
            zf.writestr(code_filename, code)

    zip_buffer.seek(0)
    st.download_button(
        label="Download Dashboards & Code (.zip)",
        data=zip_buffer,
        file_name=f"{run_id}_analysis.zip",
        mime="application/zip",
    )


def main() -> None:
    """Main execution function for the Data Visualization page."""
    check_token()
    
    if not check_ollama_server():
        st.error("Could not connect to Ollama server.")
        st.info("Please check the log file: text_lab/ollama_server.log")
        st.stop()

    selected_model = render_sidebar()

    st.title("AI Data Visualiser")
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
        with st.spinner("Multi-Agent Team is analyzing your data... this may take a moment as agents coordinate."):
            run_id = f"ds-{uuid.uuid4().hex[:8]}"
            
            try:
                ollama.pull(selected_model)
            except Exception as e:
                st.error(f"Failed to pull model {selected_model}: {e}")
                st.stop()

            try:
                with tempfile.TemporaryDirectory(dir=ARTIFACTS_DIR) as run_dir:
                    file_bytes = uploaded_file.getvalue()
                    data_file_path = save_data_file(file_bytes, uploaded_file.name, run_dir)
                    
                    df = get_fast_data_preview(data_file_path, uploaded_file.name)
                    if df is None:
                        st.error("Failed to generate a data preview.")
                        st.stop()

                    data_head = df.to_string()
                    final_user_prompt = user_prompt if user_prompt.strip() else DEFAULT_PROMPT

                    # We removed SYSTEM_PROMPT here because the Supervisor prompt 
                    # is now injected directly inside run_analysis() in viz_agent.py
                    messages = [
                        {
                            "role": "user",
                            "content": (
                                f"User Request: {final_user_prompt}\n\n"
                                f"Data Head:\n{data_head}"
                            ),
                        },
                    ]

                    # Execute the autonomous MAS loop
                    analysis_result = asyncio.run(
                        run_analysis(messages, data_file_path, selected_model, MCP_SERVER_SCRIPT)
                    )

                    # Unpack the typed result
                    summary = analysis_result["summary"]
                    plot_results = analysis_result["plots"]
                    logs = analysis_result["logs"]

                    # --- NEW: Display Agent Execution Logs ---
                    with st.expander("🤖 View Multi-Agent Execution Logs", expanded=False):
                        for log_type, msg in logs:
                            if log_type == "error":
                                st.error(msg)
                            elif log_type == "warning":
                                st.warning(msg)
                            else:
                                st.info(msg)

                    # Read generated artifacts into memory
                    final_artifacts = []
                    for item in plot_results:
                        path = item["path"]
                        code = item["code"]
                        
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

            # Display final results
            render_results(summary, final_artifacts, run_id)


if __name__ == "__main__":
    main()