import asyncio
import base64
import html as html_lib
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
from datetime import datetime
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


def _build_column_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Build a per-column summary DataFrame for the data profile tab."""
    total = len(df)
    rows = []
    for col in df.columns:
        series = df[col]
        non_null = int(series.notna().sum())
        null_pct = (total - non_null) / total * 100 if total > 0 else 0.0
        is_numeric = pd.api.types.is_numeric_dtype(series)

        # JSON columns can contain lists/dicts (unhashable). Coerce to str for stats.
        first_val = series.dropna().iloc[0] if non_null > 0 else None
        is_nested = isinstance(first_val, (list, dict))
        safe_series = series.dropna().astype(str) if is_nested else series.dropna()

        try:
            unique = int(safe_series.nunique())
        except TypeError:
            unique = -1

        if is_numeric and not is_nested:
            s = series.dropna()
            range_str = f"{s.min():.4g} / {s.mean():.4g} / {s.max():.4g}" if len(s) > 0 else "—"
            top_str = ""
        else:
            try:
                top_vals = safe_series.value_counts()
                top_str = str(top_vals.index[0])[:50] if len(top_vals) > 0 else "—"
            except TypeError:
                top_str = "nested"
            range_str = "nested" if is_nested else ""

        rows.append({
            "Column": col,
            "Type": "nested (list/dict)" if is_nested else str(series.dtype),
            "Non-Null %": f"{100 - null_pct:.1f}%",
            "Unique": unique if unique >= 0 else "—",
            "Range (min / mean / max)": range_str,
            "Top Value": top_str,
        })
    return pd.DataFrame(rows)


def _build_html_report(
    summary: str,
    final_artifacts: list[dict],
    stats_results: list[dict],
    submission_info: dict,
    run_id: str,
) -> str:
    """
    Build a fully self-contained HTML analysis report.
    Plotly charts are embedded as interactive divs (CDN JS, no inline bundle).
    Static images are embedded as base64 data URIs.
    Stats tables and code blocks are rendered as HTML.
    """
    try:
        import markdown as _md_lib
        def _md(text: str) -> str:
            return _md_lib.markdown(text, extensions=["tables", "fenced_code"])
    except ImportError:
        def _md(text: str) -> str:
            return f"<pre style='white-space:pre-wrap'>{html_lib.escape(text)}</pre>"

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    file_name = submission_info.get("file_name", "—")
    model = submission_info.get("model", "—")
    prompt = submission_info.get("user_prompt") or "(default exploratory analysis)"
    columns = submission_info.get("selected_columns") or []
    columns_str = ", ".join(columns) if columns else "All columns"

    has_plotly = any(a.get("fig") is not None for a in final_artifacts)
    plotly_cdn = (
        '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>'
        if has_plotly else ""
    )

    css = """
    body{font-family:system-ui,-apple-system,sans-serif;max-width:1200px;margin:40px auto;padding:0 24px;color:#1a1a1a;line-height:1.6}
    h1{color:#0f2346;border-bottom:3px solid #0f2346;padding-bottom:12px}
    h2{color:#1a3a6b;margin-top:40px}
    h3{color:#2a4a7f;margin-top:0}
    .meta{display:grid;grid-template-columns:max-content 1fr;gap:6px 20px;background:#f5f8ff;padding:16px 20px;border-radius:8px;border-left:4px solid #4a7fd4;margin:20px 0;font-size:.95em}
    .mk{font-weight:600;color:#4a6fa5}
    .mv{word-break:break-word}
    .summary-box{background:#fafafa;border:1px solid #e0e0e0;border-radius:8px;padding:20px 24px;margin:16px 0}
    .stat-card{border:1px solid #dde4f0;border-radius:8px;padding:16px 20px;margin:16px 0;background:#fff}
    pre,code{background:#f4f4f4;border-radius:4px;font-family:'SFMono-Regular',Consolas,monospace;font-size:.85em}
    pre{padding:12px 16px;overflow-x:auto;border:1px solid #e0e0e0}
    table{border-collapse:collapse;width:100%;margin:12px 0;font-size:.9em}
    th{background:#e8eef8;color:#1a3a6b;font-weight:600;text-align:left;padding:8px 12px;border:1px solid #c8d4e8}
    td{padding:7px 12px;border:1px solid #dde4f0}
    tr:nth-child(even) td{background:#f8faff}
    .chart-card{margin:24px 0;border:1px solid #e0e8f0;border-radius:8px;overflow:hidden}
    .chart-title{background:#e8eef8;padding:10px 16px;font-weight:600;color:#1a3a6b}
    .chart-body{padding:16px}
    img{max-width:100%;height:auto;display:block;margin:0 auto}
    details summary{cursor:pointer;font-weight:600;color:#4a6fa5;padding:6px 0;user-select:none}
    .footer{margin-top:48px;padding-top:16px;border-top:1px solid #e0e0e0;font-size:.8em;color:#888;text-align:center}
    """

    meta_html = f"""
    <div class="meta">
      <span class="mk">File</span><span class="mv">{html_lib.escape(file_name)}</span>
      <span class="mk">Model</span><span class="mv">{html_lib.escape(model)}</span>
      <span class="mk">Prompt</span><span class="mv">{html_lib.escape(prompt)}</span>
      <span class="mk">Columns analysed</span><span class="mv">{html_lib.escape(columns_str)}</span>
      <span class="mk">Generated</span><span class="mv">{timestamp}</span>
      <span class="mk">Run ID</span><span class="mv">{html_lib.escape(run_id)}</span>
    </div>"""

    summary_html = f'<div class="summary-box">{_md(summary)}</div>' if summary else ""

    stats_parts = []
    for item in stats_results:
        title = html_lib.escape(item.get("title", ""))
        code = item.get("code", "")
        code_block = (
            f"<details><summary>View reproducible code</summary>"
            f"<pre><code>{html_lib.escape(code)}</code></pre></details>"
            if code else ""
        )
        stats_parts.append(
            f'<div class="stat-card"><h3>{title}</h3>'
            f'{_md(item.get("result", ""))}{code_block}</div>'
        )
    stats_section = (
        f'<h2>Statistical Analysis</h2>{"".join(stats_parts)}' if stats_parts else ""
    )

    chart_parts = []
    for artifact in final_artifacts:
        filename = artifact["filename"]
        fig = artifact.get("fig")
        code = artifact.get("code", "")
        tool_label = get_tool_label(artifact.get("tool_name", "")) or filename
        code_block = (
            f"<details><summary>View source code</summary>"
            f"<pre><code>{html_lib.escape(code)}</code></pre></details>"
            if code else ""
        )
        if filename.endswith(".json") and fig is not None:
            chart_div = fig.to_html(full_html=False, include_plotlyjs=False)
        else:
            img_b64 = base64.b64encode(artifact["bytes"]).decode()
            ext = filename.rsplit(".", 1)[-1].lower()
            mime = f"image/{ext}" if ext != "jpg" else "image/jpeg"
            chart_div = f'<img src="data:{mime};base64,{img_b64}" alt="{html_lib.escape(filename)}">'
        chart_parts.append(
            f'<div class="chart-card">'
            f'<div class="chart-title">{html_lib.escape(tool_label)}</div>'
            f'<div class="chart-body">{chart_div}{code_block}</div>'
            f'</div>'
        )
    charts_section = (
        f'<h2>Visualisations</h2>{"".join(chart_parts)}' if chart_parts else ""
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Analysis Report — {html_lib.escape(run_id)}</title>
  {plotly_cdn}
  <style>{css}</style>
</head>
<body>
  <h1>Analysis Report</h1>
  {meta_html}
  <h2>Summary</h2>
  {summary_html}
  {stats_section}
  {charts_section}
  <div class="footer">
    Generated by Text Lab AI Visualiser &nbsp;·&nbsp;
    Run {html_lib.escape(run_id)} &nbsp;·&nbsp; {timestamp}
  </div>
</body>
</html>"""


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
    run_id: str,
    submission_info: dict,
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

    if final_artifacts or stats_results:
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

            report_html = _build_html_report(
                summary, final_artifacts, stats_results, submission_info, run_id
            )
            zf.writestr("report.html", report_html)

        zip_buffer.seek(0)
        has_plots = bool(final_artifacts)
        has_stats = bool(stats_results)
        if has_plots and has_stats:
            zip_label = "Download Report, Dashboards & Code (.zip)"
        elif has_plots:
            zip_label = "Download Dashboards & Code (.zip)"
        else:
            zip_label = "Download Report (.zip)"

        st.download_button(
            label=zip_label,
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
    _profile_df = None
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
            _PROFILE_NROWS = 2000
            if name_lower.endswith(".csv"):
                _profile_df = pd.read_csv(io.BytesIO(file_bytes_for_preview), nrows=_PROFILE_NROWS)
            elif name_lower.endswith(".tsv"):
                _profile_df = pd.read_csv(io.BytesIO(file_bytes_for_preview), sep="\t", nrows=_PROFILE_NROWS)
            elif name_lower.endswith((".xls", ".xlsx")):
                _profile_df = pd.read_excel(io.BytesIO(file_bytes_for_preview), nrows=_PROFILE_NROWS)
            elif name_lower.endswith(".json"):
                try:
                    _profile_df = pd.read_json(io.BytesIO(file_bytes_for_preview), lines=True, nrows=_PROFILE_NROWS)
                except Exception:
                    _profile_df = pd.read_json(io.BytesIO(file_bytes_for_preview)).head(_PROFILE_NROWS)
            else:
                _profile_df = None
            _preview_df = _profile_df.head(10) if _profile_df is not None else None
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
                tab_raw, tab_profile = st.tabs(["Raw Data (first 10 rows)", "Column Profile"])
                with tab_raw:
                    st.dataframe(_preview_df, use_container_width=True)
                with tab_profile:
                    profile_source = _profile_df if _profile_df is not None else _preview_df
                    st.caption(
                        f"Summary based on first {len(profile_source):,} rows. "
                        "Numeric columns show min / mean / max; text columns show the most frequent value."
                    )
                    st.dataframe(
                        _build_column_profile(profile_source),
                        use_container_width=True,
                        hide_index=True,
                    )

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
        render_results(r["summary"], r["final_artifacts"], r["stats_results"], r["run_id"],
                       st.session_state.get("viz_submission", {}))


if __name__ == "__main__":
    main()