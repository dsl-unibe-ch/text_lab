"""Visualize Data page: charts and statistics of a table, made by LLM agents.

The page collects the table, the request and the model; the agents run in
the background through :mod:`textlab.features.visualization.service`, and
the page polls the run and shows its activity and results.
"""

import os
import time

import plotly.io as pio
import streamlit as st
from PIL import Image

from textlab.common import gpu_manager
from textlab.common.gpu_manager import get_gpu_name
from textlab.common.model_config import (
    get_available_models,
    is_high_memory_gpu,
)
from textlab.common.ollama import check_ollama_server
from textlab.features.visualization import service
from textlab.ui.streamlit.auth import check_token
from textlab.ui.streamlit.components.gpu import free_gpu_for

current_dir = os.path.dirname(os.path.abspath(__file__))
app_dir = os.path.dirname(current_dir)
favicon_path = os.path.join(app_dir, "assets", "text_lab_logo.png")
try:
    favicon = Image.open(favicon_path)
    st.set_page_config(
        page_title="Visualise Data", page_icon=favicon, layout="wide"
    )
except FileNotFoundError:
    st.set_page_config(page_title="Visualise Data", layout="wide")

#: Rows shown in the raw-data preview.
PREVIEW_ROWS = 10

CAPABILITIES = """
This tool uses a **Multi-Agent System** to analyze your data. A Supervisor AI \
reads your prompt and delegates tasks to three specialist agents:

* **Interactive Agent (Default):** Generates web-ready, interactive Plotly \
charts (Scatter, Bar, Line, Box, Scatter Matrix, Correlation Heatmap, etc.). \
Best for exploring data on this page.
* **Static Agent:** Generates publication-ready Matplotlib/Seaborn charts, \
Pair Plots, and Word Clouds. Triggered when you explicitly ask for "static", \
"publication figures", "pair plot", or "word cloud".
* **Statistical Agent:** Runs statistical tests: correlations, group \
comparisons (t-test, ANOVA, Mann-Whitney, Kruskal-Wallis), associations \
between categorical columns (chi-square, Fisher), and linear and logistic \
regression. Each result includes reproducible Python code.
* **R Code (optional):** Tick "Also generate equivalent R code" to get R \
code (ggplot2 plots, base R statistics) next to the Python code of every \
result. Plots made from custom Python code have no R version.

**Prompting Tip:** Be specific about what you want!
*(e.g., "Run a t-test on column X grouped by Y, then plot an interactive bar \
chart of the means.")*
"""


def _render_code(python_code: str, r_snippet: str, show_r: bool) -> None:
    """Show the reproducible code: Python only, or Python and R tabs.

    Args:
        python_code: The Python snippet.
        r_snippet: The R equivalent (empty for custom-code plots).
        show_r: Whether the user asked for R code for this run.
    """
    if not show_r:
        st.code(python_code, language="python")
        return
    tab_python, tab_r = st.tabs(["Python", "R"])
    with tab_python:
        st.code(python_code, language="python")
    with tab_r:
        if r_snippet:
            st.code(r_snippet, language="r")
        else:
            st.caption(service.R_NOT_AVAILABLE)


def render_sidebar() -> str:
    """Draw the model choice; return the selected model."""
    st.sidebar.title("Model Selection")

    current_gpu = get_gpu_name()
    available_models = get_available_models(current_gpu)

    if is_high_memory_gpu(current_gpu):
        gpu_badge = f"High-Performance Mode ({current_gpu})"
    else:
        gpu_badge = f"Standard Mode ({current_gpu})"

    if not available_models:
        st.sidebar.error(
            "No models are configured. Please check src/config/models.json."
        )
        st.stop()

    st.sidebar.markdown(f"**{gpu_badge}**")

    selected_model = st.sidebar.selectbox(
        "Select Analysis Model:", options=available_models, index=0
    )
    return str(selected_model)


def render_results(
    result: service.AnalysisResult, request: service.AnalysisRequest
) -> None:
    """Show the summary, statistics, charts and the download of a run."""
    show_r = request.include_r_code
    st.success("Analysis Complete.")
    st.subheader("Analysis Summary")
    st.markdown(result.summary)

    if result.stats:
        st.subheader("Statistical Analysis Results")
        for item in result.stats:
            with st.expander(item["title"], expanded=True):
                st.markdown(item["result"])
                if item["code"]:
                    _render_code(item["code"], item.get("r_code", ""), show_r)
        st.divider()

    if result.artifacts:
        st.subheader("Generated Visualisations")

        for idx, artifact in enumerate(result.artifacts):
            label = service.tool_label(artifact.tool_name)
            with st.container():
                if label:
                    st.markdown(f"**{label}**")
                if artifact.figure_json is not None:
                    st.plotly_chart(
                        pio.from_json(artifact.figure_json),
                        use_container_width=True,
                        key=f"plotly_{result.run_id}_{idx}",
                    )
                else:
                    st.image(artifact.data, caption=artifact.filename)

                with st.expander(
                    f"View Source Code: {label or artifact.filename}"
                ):
                    _render_code(artifact.code, artifact.r_code, show_r)

                st.divider()

    if result.artifacts or result.stats:
        if result.artifacts and result.stats:
            zip_label = "Download Report, Dashboards & Code (.zip)"
        elif result.artifacts:
            zip_label = "Download Dashboards & Code (.zip)"
        else:
            zip_label = "Download Report (.zip)"

        st.download_button(
            label=zip_label,
            data=service.results_zip(result, request),
            file_name=f"{result.run_id}_analysis.zip",
            mime="application/zip",
        )


def _render_log_box(logs: list, is_complete: bool = False) -> None:
    """Show the agents' activity log.

    Kept expanded during the analysis, and collapsed when it is complete so
    it does not push the results off-screen.
    """
    log_state = "complete" if is_complete else "running"
    with st.status(
        "Agent Activity Log", expanded=not is_complete, state=log_state
    ):
        if logs:
            for log_type, msg in logs:
                if log_type == "error":
                    st.error(msg)
                elif log_type == "warning":
                    st.warning(msg)
                else:
                    st.write(msg)
        else:
            st.caption("Starting agents...")


def _render_log_section() -> None:
    """Show the live log and the cancel button, and poll the run.

    When the run finishes, its result or error is kept in session state and
    the page is redrawn.
    """
    run: service.AnalysisRun = st.session_state["viz_run"]

    st.divider()

    cancel_col, _ = st.columns([1, 5])
    with cancel_col:
        if st.session_state.get("viz_cancelling"):
            st.warning("Cancelling...")
        elif st.button(
            "Cancel Analysis", type="secondary", use_container_width=True
        ):
            run.cancel()
            st.session_state["viz_cancelling"] = True
            st.rerun()

    _render_log_box(list(run.logs), is_complete=False)

    if run.running:
        time.sleep(1)
        st.rerun()
        return

    if run.status == service.DONE:
        st.session_state["viz_results"] = {
            "result": run.result,
            "file_id": st.session_state.get("viz_file_id"),
        }
    elif run.status == service.TIMEOUT:
        st.error(
            f"Analysis exceeded the {service.ANALYSIS_TIMEOUT_SECONDS // 60}"
            "-minute limit. Try a simpler prompt or a smaller dataset."
        )
    elif run.status == service.CANCELLED:
        st.warning("Analysis was cancelled.")
    elif run.status == service.ERROR:
        st.error(f"An unexpected error occurred: {run.error or ''}")

    st.session_state["viz_live_logs"] = list(run.logs)
    del st.session_state["viz_run"]
    st.session_state.pop("viz_cancelling", None)
    st.rerun()


def _render_preview(uploaded_file):
    """Show the file's size, a preview and the column profile.

    Returns:
        The first rows of the table, or ``None`` if it cannot be read.
    """
    file_size_mb = uploaded_file.size / (1024 * 1024)
    profile_df = service.read_preview(
        uploaded_file.name, uploaded_file.getvalue()
    )
    preview_df = (
        profile_df.head(PREVIEW_ROWS) if profile_df is not None else None
    )
    n_cols = len(preview_df.columns) if preview_df is not None else "?"

    st.caption(
        f"**{uploaded_file.name}** | {file_size_mb:.1f} MB | {n_cols} columns"
    )
    if file_size_mb > 100:
        st.warning(
            f"Large file detected ({file_size_mb:.0f} MB). "
            f"Data will be capped at {service.MAX_ROWS:,} rows for memory "
            "safety."
        )

    if preview_df is not None:
        with st.expander("Preview Data", expanded=False):
            tab_raw, tab_profile = st.tabs(
                ["Raw Data (first 10 rows)", "Column Profile"]
            )
            with tab_raw:
                st.dataframe(preview_df, use_container_width=True)
            with tab_profile:
                st.caption(
                    f"Summary based on first {len(profile_df):,} rows. "
                    "Numeric columns show min / mean / max; text columns "
                    "show the most frequent value."
                )
                st.dataframe(
                    service.column_profile(profile_df),
                    use_container_width=True,
                    hide_index=True,
                )
    return preview_df


def _render_column_selection(preview_df, file_id, is_running):
    """Let the user pick the columns to analyze; return them."""
    all_columns = list(preview_df.columns)

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
    return selected_columns


def main() -> None:
    """Draw the page."""
    check_token()

    if not check_ollama_server():
        st.error("Could not connect to Ollama server.")
        st.info("Please check the log file: text_lab/ollama_server.log")
        st.stop()

    selected_model = render_sidebar()

    st.title("AI Data Visualiser")
    st.info(f"Using Model: **{selected_model}**")

    with st.expander("View Available AI Capabilities"):
        st.markdown(CAPABILITIES)

    is_running = "viz_run" in st.session_state

    # Input form: stays visible, but disabled while an analysis runs.
    uploaded_file = st.file_uploader(
        "Upload your data file (CSV, TSV, Excel, JSON)",
        type=list(service.INPUT_TYPES),
        disabled=is_running,
    )

    preview_df = None
    file_id = None
    if uploaded_file:
        file_id = (uploaded_file.name, uploaded_file.size)
        stored = st.session_state.get("viz_results")
        if stored is not None and stored.get("file_id") != file_id:
            del st.session_state["viz_results"]
            st.session_state.pop("viz_live_logs", None)
        preview_df = _render_preview(uploaded_file)

    selected_columns: list[str] = []
    if preview_df is not None:
        selected_columns = _render_column_selection(
            preview_df, file_id, is_running
        )

    user_prompt = st.text_area(
        "Describe what you want to do (optional)",
        placeholder=service.DEFAULT_PROMPT,
        key="viz_prompt",
        disabled=is_running,
    )

    include_r_code = st.checkbox(
        "Also generate equivalent R code",
        key="viz_include_r",
        disabled=is_running,
        help=(
            "Adds R code (ggplot2 plots, base R statistics) next to the "
            "Python code of each result, so the analysis can be reproduced "
            "and checked in R. Plots made from custom Python code have no R "
            "version."
        ),
    )

    if is_running:
        st.button("Generating...", type="primary", disabled=True)
    elif st.button(
        "Generate Visualisations", type="primary", disabled=not uploaded_file
    ):
        free_gpu_for(gpu_manager.LLM, ollama_model=selected_model)
        request = service.AnalysisRequest(
            model=selected_model,
            prompt=user_prompt.strip(),
            columns=tuple(selected_columns),
            include_r_code=include_r_code,
            file_name=uploaded_file.name,
        )
        st.session_state["viz_request"] = request
        st.session_state["viz_file_id"] = file_id
        st.session_state.pop("viz_live_logs", None)
        st.session_state["viz_run"] = service.start_analysis(
            request, data=uploaded_file.getvalue()
        )
        st.rerun()

    if is_running:
        # The log and cancel button sit under the form; results follow
        # when the run is done.
        _render_log_section()
        return

    if "viz_results" in st.session_state:
        if "viz_live_logs" in st.session_state:
            _render_log_box(
                st.session_state["viz_live_logs"], is_complete=True
            )
        render_results(
            st.session_state["viz_results"]["result"],
            st.session_state["viz_request"],
        )


if __name__ == "__main__":
    main()
