import asyncio
import io
import os
import pathlib
import shutil
import sys
import tempfile
import threading
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
    stats_results: list[dict],
    run_id: str
) -> None:
    st.success("Analysis Complete.")
    st.subheader("Analysis Summary")
    st.markdown(summary)

    if stats_results:
        st.subheader("Statistical Analysis Results")
        for item in stats_results:
            with st.expander(item["title"], expanded=True):
                st.markdown(item["result"])
                if item["code"]:
                    st.code(item["code"], language="python")
        st.divider()

    if final_artifacts:
        st.subheader("Generated Visualisations")

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


def _start_analysis_thread(
    file_bytes: bytes,
    file_name: str,
    file_id: tuple[str, int],
    user_prompt: str,
    selected_model: str,
    selected_columns: list[str],
) -> None:
    """
    Capture all inputs, initialise shared session_state structures, then start the
    analysis in a daemon thread so the Streamlit UI remains responsive.
    """
    cancel_event: threading.Event = threading.Event()
    live_logs: list[tuple[str, str]] = []
    thread_result: dict = {
        "status": "running",
        "result": None,
        "final_artifacts": [],
        "error": None,
    }
    run_id = f"ds-{uuid.uuid4().hex[:8]}"

    st.session_state["viz_submission"] = {
        "file_name": file_name,
        "selected_columns": selected_columns,
        "user_prompt": user_prompt.strip(),
        "model": selected_model,
    }

    def _worker() -> None:
        try:
            local_models = {m["model"] for m in ollama.list().get("models", [])}
            if selected_model not in local_models:
                live_logs.append(("info", f"Pulling model '{selected_model}'..."))
                ollama.pull(selected_model)
        except Exception as e:
            thread_result["error"] = f"Failed to pull model '{selected_model}': {e}"
            thread_result["status"] = "error"
            return

        try:
            with tempfile.TemporaryDirectory(dir=ARTIFACTS_DIR) as run_dir:
                data_file_path = save_data_file(file_bytes, file_name, run_dir)

                df = get_fast_data_preview(data_file_path, file_name)
                if df is None:
                    thread_result["error"] = "Failed to generate a data preview."
                    thread_result["status"] = "error"
                    return

                final_user_prompt = user_prompt.strip() or DEFAULT_PROMPT

                valid_selected = [c for c in selected_columns if c in df.columns]
                if valid_selected:
                    head_df = df[valid_selected]
                    column_instruction = (
                        f"Column Selection: Focus ONLY on these columns chosen by the user: "
                        f"{', '.join(valid_selected)}\n\n"
                    )
                else:
                    head_df = df
                    column_instruction = ""

                messages = [
                    {
                        "role": "user",
                        "content": (
                            f"User Request: {final_user_prompt}\n\n"
                            f"{column_instruction}"
                            f"Data Head:\n{head_df.to_string()}\n\n"
                            f"Note: datasets larger than {MAX_ROWS:,} rows will be truncated."
                        ),
                    }
                ]

                live_logs.append(("info", "Starting Supervisor Agent..."))

                def _log_cb(log_type: str, msg: str) -> None:
                    live_logs.append((log_type, msg))

                try:
                    analysis_result = asyncio.run(
                        asyncio.wait_for(
                            run_analysis(
                                messages,
                                data_file_path,
                                selected_model,
                                MCP_SERVER_SCRIPT,
                                log_callback=_log_cb,
                                cancel_event=cancel_event,
                            ),
                            timeout=ANALYSIS_TIMEOUT_SECONDS,
                        )
                    )
                except asyncio.TimeoutError:
                    thread_result["status"] = "timeout"
                    return

                final_artifacts: list[dict] = []
                for item in analysis_result["plots"]:
                    path = item["path"]
                    if os.path.exists(path):
                        filename = os.path.basename(path)
                        with open(path, "rb") as f:
                            img_bytes = f.read()
                        artifact: dict = {
                            "filename": filename,
                            "bytes": img_bytes,
                            "code": item["code"],
                            "fig": None,
                            "tool_name": item.get("tool_name", ""),
                        }
                        if filename.endswith(".json"):
                            try:
                                artifact["fig"] = pio.from_json(img_bytes.decode("utf-8"))
                            except Exception:
                                live_logs.append(("warning", f"Failed to parse plot {filename}"))
                                continue
                        final_artifacts.append(artifact)
                    else:
                        live_logs.append(("warning", f"Could not find plot at: {path}"))

                thread_result["result"] = analysis_result
                thread_result["final_artifacts"] = final_artifacts
                thread_result["status"] = "cancelled" if cancel_event.is_set() else "done"

        except Exception as e:
            thread_result["error"] = str(e)
            thread_result["status"] = "error"

    thread = threading.Thread(target=_worker, daemon=True)

    st.session_state["viz_cancel_event"] = cancel_event
    st.session_state["viz_live_logs"] = live_logs
    st.session_state["viz_thread_result"] = thread_result
    st.session_state["viz_run_id"] = run_id
    st.session_state["viz_file_id"] = file_id
    st.session_state["viz_run_state"] = "running"

    thread.start()


def _render_log_box(live_logs: list, is_complete: bool = False) -> None:
    """
    Renders the log section. Kept expanded during analysis, but collapses
    automatically when complete so it doesn't push results off-screen.
    """
    log_state = "complete" if is_complete else "running"
    
    with st.status("Agent Activity Log", expanded=(not is_complete), state=log_state):
        if live_logs:
            for log_type, msg in live_logs:
                if log_type == "error":
                    st.error(msg)
                elif log_type == "warning":
                    st.warning(msg)
                else:
                    st.write(msg)
        else:
            st.caption("Starting agents...")


def _render_log_section() -> None:
    """
    Render the live log, cancel button, and polling logic.
    Called from main() BELOW the form area while analysis is running.
    Handles all terminal-state transitions when the thread finishes.
    """
    thread_result: dict = st.session_state.get("viz_thread_result", {})
    live_logs: list = st.session_state.get("viz_live_logs", [])
    run_state: str = st.session_state.get("viz_run_state", "running")
    status: str = thread_result.get("status", "running")

    st.divider()

    cancel_col, _ = st.columns([1, 5])
    with cancel_col:
        if run_state == "cancelling":
            st.warning("Cancelling...")
        else:
            if st.button("Cancel Analysis", type="secondary", use_container_width=True):
                st.session_state["viz_cancel_event"].set()
                st.session_state["viz_run_state"] = "cancelling"
                st.rerun()

    current_logs = list(live_logs)
    _render_log_box(current_logs, is_complete=False)

    if status == "running":
        time.sleep(1)
        st.rerun()
        return

    if status == "done":
        analysis_result = thread_result["result"]
        run_id = st.session_state.get("viz_run_id", "ds-unknown")
        st.session_state["viz_results"] = {
            "summary": analysis_result.get("summary", ""),
            "final_artifacts": thread_result.get("final_artifacts", []),
            "stats_results": analysis_result.get("stats", []),
            "run_id": run_id,
            "file_id": st.session_state.get("viz_file_id"),
        }
    elif status == "timeout":
        st.error(
            f"Analysis exceeded the {ANALYSIS_TIMEOUT_SECONDS // 60}-minute limit. "
            "Try a simpler prompt or a smaller dataset."
        )
    elif status == "cancelled":
        st.warning("Analysis was cancelled.")
    elif status == "error":
        st.error(f"An unexpected error occurred: {thread_result.get('error', '')}")

    st.session_state["viz_run_state"] = "idle"
    st.rerun()


def main() -> None:
    check_token()

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

    with st.expander("View Available AI Capabilities"):
        st.markdown("""
        This tool uses a **Multi-Agent System** to analyze your data. A Supervisor AI reads your prompt and delegates tasks to three specialist agents:

        * **Interactive Agent (Default):** Generates web-ready, interactive Plotly charts (Scatter, Bar, Line, Box, Scatter Matrix, Correlation Heatmap, etc.). Best for exploring data on this page.
        * **Static Agent:** Generates publication-ready Matplotlib/Seaborn charts, Pair Plots, and Word Clouds. Triggered when you explicitly ask for "static", "publication figures", "pair plot", or "word cloud".
        * **Statistical Agent:** Runs rigorous mathematical tests including Correlations, T-tests, ANOVA, and OLS Linear Regression. Each result includes reproducible Python code.

        **Prompting Tip:** Be specific about what you want!
        *(e.g., "Run a t-test on column X grouped by Y, then plot an interactive bar chart of the means.")*
        """)

    run_state = st.session_state.get("viz_run_state", "idle")
    is_running = run_state in ("running", "cancelling")

    # =========================================================
    # INPUT FORM — remains visible but disabled during execution
    # =========================================================
    uploaded_file = None
    _preview_df = None
    file_id: tuple[str, int] | None = None

    uploaded_file = st.file_uploader(
        "Upload your data file (CSV, TSV, Excel, JSON)",
        type=["csv", "tsv", "xls", "xlsx", "json"],
        disabled=is_running,
    )

    if uploaded_file:
        file_size_mb = uploaded_file.size / (1024 * 1024)
        file_id = (uploaded_file.name, uploaded_file.size)

        try:
            file_bytes_for_preview = uploaded_file.getvalue()
            name_lower = uploaded_file.name.lower()
            if name_lower.endswith(".csv"):
                _preview_df = pd.read_csv(io.BytesIO(file_bytes_for_preview), nrows=10)
            elif name_lower.endswith(".tsv"):
                _preview_df = pd.read_csv(io.BytesIO(file_bytes_for_preview), sep="\t", nrows=10)
            elif name_lower.endswith((".xls", ".xlsx")):
                _preview_df = pd.read_excel(io.BytesIO(file_bytes_for_preview), nrows=10)
            elif name_lower.endswith(".json"):
                try:
                    _preview_df = pd.read_json(io.BytesIO(file_bytes_for_preview), lines=True, nrows=10)
                except Exception:
                    _preview_df = pd.read_json(io.BytesIO(file_bytes_for_preview)).head(10)
            else:
                _preview_df = None
            n_cols = len(_preview_df.columns) if _preview_df is not None else "?"
        except Exception:
            n_cols = "?"

        st.caption(f"**{uploaded_file.name}** | {file_size_mb:.1f} MB | {n_cols} columns")
        if file_size_mb > 100:
            st.warning(
                f"Large file detected ({file_size_mb:.0f} MB). "
                f"Data will be capped at {MAX_ROWS:,} rows for memory safety."
            )

        if "viz_results" in st.session_state:
            stored_id = st.session_state["viz_results"].get("file_id")
            if stored_id != file_id:
                del st.session_state["viz_results"]
                if "viz_live_logs" in st.session_state:
                    del st.session_state["viz_live_logs"]

        if _preview_df is not None:
            with st.expander("Preview Data", expanded=False):
                st.dataframe(_preview_df, use_container_width=True)

    selected_columns: list[str] = []
    if _preview_df is not None:
        all_columns = list(_preview_df.columns)

        if st.session_state.get("viz_columns_file_id") != file_id:
            st.session_state["viz_col_multiselect"] = all_columns
            st.session_state["viz_columns_file_id"] = file_id

        st.markdown("**Select columns to include in the analysis:**")
        btn_col1, btn_col2, _ = st.columns([1, 1, 8])
        with btn_col1:
            if st.button("Select All", key="viz_sel_all_btn", disabled=is_running):
                st.session_state["viz_col_multiselect"] = all_columns
                st.rerun()
        with btn_col2:
            if st.button("Clear All", key="viz_sel_none_btn", disabled=is_running):
                st.session_state["viz_col_multiselect"] = []
                st.rerun()

        selected_columns = st.multiselect(
            "Columns",
            options=all_columns,
            key="viz_col_multiselect",
            label_visibility="collapsed",
            disabled=is_running,
        )
        if not selected_columns:
            st.caption("No columns selected -- all columns will be used.")

    user_prompt = st.text_area(
        "Describe what you want to do (optional)",
        placeholder=DEFAULT_PROMPT,
        key="viz_prompt",
        disabled=is_running,
    )

    if is_running:
        st.button("Generating...", type="primary", disabled=True)
    else:
        if st.button("Generate Visualisations", type="primary", disabled=(not uploaded_file)):
            _start_analysis_thread(
                file_bytes=uploaded_file.getvalue(),
                file_name=uploaded_file.name,
                file_id=(uploaded_file.name, uploaded_file.size),
                user_prompt=user_prompt,
                selected_model=selected_model,
                selected_columns=selected_columns,
            )
            st.rerun()

    # =========================================================
    # RUNNING STATE — log and cancel controls placed under the form
    # =========================================================
    if is_running:
        _render_log_section()
        return  # End execution here so results aren't rendered until done

    # =========================================================
    # RESULTS — persisted results shown below the form
    # =========================================================
    if "viz_results" in st.session_state:
        # Render the completed log, cleanly collapsed by default
        if "viz_live_logs" in st.session_state:
            _render_log_box(st.session_state["viz_live_logs"], is_complete=True)

        r = st.session_state["viz_results"]
        render_results(r["summary"], r["final_artifacts"], r["stats_results"], r["run_id"])


if __name__ == "__main__":
    main()