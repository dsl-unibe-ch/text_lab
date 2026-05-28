import asyncio
import io
import os
import pathlib
import shutil
import sys
import tempfile
import time
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
from core.visualization.viz_config import DEFAULT_PROMPT, MAX_ROWS, get_tool_label
from core.visualization.viz_utils import get_fast_data_preview, save_data_file
from core.model_config import get_available_models, is_high_memory_gpu

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

# Max seconds before the analysis is cancelled and an error is shown.
ANALYSIS_TIMEOUT_SECONDS = 600

os.makedirs(ARTIFACTS_DIR, exist_ok=True)
try:
    os.chmod(ARTIFACTS_DIR, 0o700)
except Exception:
    pass


def _cleanup_orphaned_artifacts(max_age_hours: int = 12) -> None:
    """
    Remove any mcp_artifacts/tmp* directories that are older than max_age_hours.
    These can accumulate when Streamlit crashes mid-analysis before the
    TemporaryDirectory context manager can run its cleanup.
    Called once per browser session via session_state guard.
    """
    cutoff = time.time() - max_age_hours * 3600
    artifacts = pathlib.Path(ARTIFACTS_DIR)
    for entry in artifacts.iterdir():
        if entry.is_dir() and entry.name.startswith("tmp"):
            try:
                if entry.stat().st_mtime < cutoff:
                    shutil.rmtree(entry, ignore_errors=True)
            except Exception:
                pass


def render_sidebar() -> str:
    st.sidebar.title("Model Selection")
    
    current_gpu = get_gpu_name()
    available_models = get_available_models(current_gpu)

    if is_high_memory_gpu(current_gpu):
        gpu_badge = f"High-Performance Mode ({current_gpu})"
    else:
        gpu_badge = f"Standard Mode ({current_gpu})"

    if not available_models:
        st.sidebar.error("No models are configured. Please check src/config/models.json.")
        st.stop()

    st.sidebar.markdown(f"**{gpu_badge}**")

    selected_model = st.sidebar.selectbox(
        "Select Analysis Model:",
        options=available_models,
        index=0
    )
    
    return str(selected_model)


def render_results(
    summary: str, 
    final_artifacts: list[dict], 
    run_id: str
) -> None:
    st.success("Analysis Complete.")
    st.subheader("Analysis Summary")
    st.markdown(summary)

    st.subheader("Generated Visualisations")
    
    if not final_artifacts:
        st.warning("No plots were generated. Try refining your prompt.")
        return

    for idx, artifact in enumerate(final_artifacts):
        filename = artifact["filename"]
        file_bytes = artifact["bytes"]
        code = artifact["code"]
        fig = artifact.get("fig")
        tool_label = get_tool_label(artifact.get("tool_name", ""))
        
        with st.container():
            if tool_label:
                st.markdown(f"**{tool_label}**")
            if filename.endswith(".json") and fig is not None:
                st.plotly_chart(fig, use_container_width=True, key=f"plotly_{run_id}_{idx}")
            else:
                st.image(file_bytes, caption=filename)
                
            with st.expander(f"View Source Code: {tool_label or filename}"):
                st.code(code, language="python")
            
            st.divider()

    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for artifact in final_artifacts:
            filename = artifact["filename"]
            file_bytes = artifact["bytes"]
            code = artifact["code"]
            fig = artifact.get("fig")
            
            if filename.endswith(".json") and fig is not None:
                html_filename = filename.replace(".json", ".html")
                # include_plotlyjs=True embeds the JS so the HTML works offline.
                zf.writestr(html_filename, fig.to_html(include_plotlyjs=True))
            else:
                zf.writestr(filename, file_bytes)
                
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
    check_token()

    # Orphaned artifact cleanup — once per browser session.
    if "artifacts_cleaned" not in st.session_state:
        _cleanup_orphaned_artifacts()
        st.session_state["artifacts_cleaned"] = True

    if not check_ollama_server():
        st.error("Could not connect to Ollama server.")
        st.info("Please check the log file: text_lab/ollama_server.log")
        st.stop()

    selected_model = render_sidebar()

    st.title("AI Data Visualiser")
    st.info(f"Using Model: **{selected_model}**")

    # --- CAPABILITIES EXPANDER ---
    with st.expander("View Available AI Capabilities"):
        st.markdown("""
        This tool uses a **Multi-Agent System** to analyze your data. A Supervisor AI reads your prompt and delegates tasks to three specialist agents:

        * **Interactive Agent (Default):** Generates web-ready, interactive Plotly charts (Scatter, Bar, Line, Box, Scatter Matrix, Correlation Heatmap, etc.). Best for exploring data on this page.
        * **Static Agent:** Generates publication-ready Matplotlib/Seaborn charts, Pair Plots, and Word Clouds. Triggered when you explicitly ask for "static", "publication figures", "pair plot", or "word cloud".
        * **Statistical Agent:** Runs rigorous mathematical tests including Correlations, T-tests, ANOVA, and OLS Linear Regression. Each result includes reproducible Python code.

        **Prompting Tip:** Be specific about what you want!
        *(e.g., "Run a t-test on column X grouped by Y, then plot an interactive bar chart of the means.")*
        """)

    uploaded_file = st.file_uploader(
        "Upload your data file (CSV, TSV, Excel, JSON)",
        type=["csv", "tsv", "xls", "xlsx", "json"],
    )

    if uploaded_file:
        file_size_mb = uploaded_file.size / (1024 * 1024)
        # Parse column count from raw bytes in memory — never write to /tmp.
        try:
            file_bytes_for_preview = uploaded_file.getvalue()
            name_lower = uploaded_file.name.lower()
            if name_lower.endswith(".csv"):
                _preview_df = pd.read_csv(io.BytesIO(file_bytes_for_preview), nrows=1)
            elif name_lower.endswith(".tsv"):
                _preview_df = pd.read_csv(io.BytesIO(file_bytes_for_preview), sep="\t", nrows=1)
            elif name_lower.endswith((".xls", ".xlsx")):
                _preview_df = pd.read_excel(io.BytesIO(file_bytes_for_preview), nrows=1)
            elif name_lower.endswith(".json"):
                try:
                    _preview_df = pd.read_json(io.BytesIO(file_bytes_for_preview), lines=True, nrows=1)
                except Exception:
                    _preview_df = pd.read_json(io.BytesIO(file_bytes_for_preview)).head(1)
            else:
                _preview_df = None
            n_cols = len(_preview_df.columns) if _preview_df is not None else "?"
        except Exception:
            n_cols = "?"

        st.caption(
            f"📄 **{uploaded_file.name}** · {file_size_mb:.1f} MB · {n_cols} columns"
        )
        if file_size_mb > 100:
            st.warning(
                f"Large file detected ({file_size_mb:.0f} MB). "
                f"Data will be capped at {MAX_ROWS:,} rows for memory safety."
            )

    user_prompt = st.text_area(
        "Describe what you want to do (optional)",
        placeholder=DEFAULT_PROMPT,
    )

    if st.button("Generate Visualisations", type="primary", disabled=(not uploaded_file)):
        summary = ""
        final_artifacts: list[dict] = []

        with st.status("Orchestrating Multi-Agent Team...", expanded=True) as status_box:

            def live_log_callback(log_type: str, msg: str):
                if log_type == "error":
                    status_box.error(msg)
                elif log_type == "warning":
                    status_box.warning(msg)
                else:
                    status_box.write(msg)

            run_id = f"ds-{uuid.uuid4().hex[:8]}"

            # Only pull if the model isn't already present locally.
            try:
                local_models = {m["model"] for m in ollama.list().get("models", [])}
                if selected_model not in local_models:
                    status_box.write(f"Pulling model '{selected_model}'...")
                    ollama.pull(selected_model)
            except Exception as e:
                status_box.error(f"Failed to pull model {selected_model}: {e}")
                st.stop()

            try:
                with tempfile.TemporaryDirectory(dir=ARTIFACTS_DIR) as run_dir:
                    file_bytes = uploaded_file.getvalue()
                    data_file_path = save_data_file(file_bytes, uploaded_file.name, run_dir)

                    df = get_fast_data_preview(data_file_path, uploaded_file.name)
                    if df is None:
                        status_box.error("Failed to generate a data preview.")
                        st.stop()

                    data_head = df.to_string()
                    final_user_prompt = user_prompt if user_prompt.strip() else DEFAULT_PROMPT

                    messages = [
                        {
                            "role": "user",
                            "content": (
                                f"User Request: {final_user_prompt}\n\n"
                                f"Data Head:\n{data_head}\n\n"
                                f"Note: datasets larger than {MAX_ROWS:,} rows will be truncated."
                            ),
                        },
                    ]

                    status_box.write("Starting Supervisor Agent...")

                    try:
                        analysis_result = asyncio.run(
                            asyncio.wait_for(
                                run_analysis(
                                    messages,
                                    data_file_path,
                                    selected_model,
                                    MCP_SERVER_SCRIPT,
                                    log_callback=live_log_callback,
                                ),
                                timeout=ANALYSIS_TIMEOUT_SECONDS,
                            )
                        )
                    except asyncio.TimeoutError:
                        status_box.update(label="Analysis Timed Out", state="error", expanded=True)
                        status_box.error(
                            f"Analysis exceeded the {ANALYSIS_TIMEOUT_SECONDS // 60}-minute limit. "
                            "Try a simpler prompt or a smaller dataset."
                        )
                        st.stop()

                    status_box.update(label="Analysis Complete", state="complete", expanded=True)

                    summary = analysis_result["summary"]
                    plot_results = analysis_result["plots"]

                    for item in plot_results:
                        path = item["path"]
                        code = item["code"]

                        if os.path.exists(path):
                            filename = os.path.basename(path)
                            with open(path, "rb") as f:
                                img_bytes = f.read()

                            artifact: dict = {
                                "filename": filename,
                                "bytes": img_bytes,
                                "code": code,
                                "fig": None,
                                "tool_name": item.get("tool_name", ""),
                            }
                            # Parse Plotly JSON once and reuse for both render + export.
                            if filename.endswith(".json"):
                                try:
                                    artifact["fig"] = pio.from_json(img_bytes.decode("utf-8"))
                                except Exception as e:
                                    status_box.warning(f"Failed to parse plot {filename}: {e}")
                                    continue
                            final_artifacts.append(artifact)
                        else:
                            st.error(f"Could not find plot at: {path}")

            except Exception as e:
                status_box.update(label="Analysis Failed", state="error", expanded=True)
                status_box.error(f"An unexpected error occurred: {e}")
                st.stop()

        render_results(summary, final_artifacts, run_id)


if __name__ == "__main__":
    main()