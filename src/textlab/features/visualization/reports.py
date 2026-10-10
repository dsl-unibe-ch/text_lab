"""The downloads of an analysis: an HTML report and a ZIP of everything."""

from __future__ import annotations

import base64
import html as html_lib
import io
import zipfile
from datetime import datetime
from typing import Any

from textlab.features.visualization.models import (
    AnalysisRequest,
    AnalysisResult,
    Artifact,
)
from textlab.features.visualization.viz_config import get_tool_label

R_NOT_AVAILABLE = (
    "R code is not available for plots made from custom Python code."
)


def figure(artifact: Artifact) -> Any:
    """Return an interactive chart's Plotly figure, or ``None``."""
    if artifact.figure_json is None:
        return None
    import plotly.io as pio

    return pio.from_json(artifact.figure_json)


def build_html_report(result: AnalysisResult, request: AnalysisRequest) -> str:
    """Build a self-contained HTML report of an analysis.

    Plotly charts are embedded as interactive charts, with the Plotly
    library inlined once, so the report needs no internet connection.
    Static images are embedded as base64 data URIs; statistics tables and
    code blocks are rendered as HTML.

    Args:
        result: The finished analysis.
        request: What was asked, shown at the top of the report.

    Returns:
        The HTML document.
    """
    summary = result.summary
    final_artifacts = result.artifacts
    stats_results = result.stats
    run_id = result.run_id
    try:
        import markdown as _md_lib

        def _md(text: str) -> str:
            return _md_lib.markdown(text, extensions=["tables", "fenced_code"])
    except ImportError:

        def _md(text: str) -> str:
            return (
                "<pre "
                f"style='white-space:pre-wrap'>{html_lib.escape(text)}</pre>"
            )

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    file_name = request.file_name or "—"
    model = request.model or "—"
    prompt = request.prompt.strip() or "(default exploratory analysis)"
    columns = list(request.columns)
    columns_str = ", ".join(columns) if columns else "All columns"

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
    """  # noqa: E501

    meta_html = f"""
    <div class="meta">
      <span class="mk">File</span><span class="mv">{html_lib.escape(file_name)}</span>
      <span class="mk">Model</span><span class="mv">{html_lib.escape(model)}</span>
      <span class="mk">Prompt</span><span class="mv">{html_lib.escape(prompt)}</span>
      <span class="mk">Columns analysed</span><span class="mv">{html_lib.escape(columns_str)}</span>
      <span class="mk">Generated</span><span class="mv">{timestamp}</span>
      <span class="mk">Run ID</span><span class="mv">{html_lib.escape(run_id)}</span>
    </div>"""  # noqa: E501

    summary_html = (
        f'<div class="summary-box">{_md(summary)}</div>' if summary else ""
    )

    show_r = request.include_r_code

    def _r_block(r_snippet: str) -> str:
        if not show_r:
            return ""
        if not r_snippet:
            return f"<p><em>{html_lib.escape(R_NOT_AVAILABLE)}</em></p>"
        return (
            f"<details><summary>View R code</summary>"
            f"<pre><code>{html_lib.escape(r_snippet)}</code></pre></details>"
        )

    stats_parts = []
    for item in stats_results:
        title = html_lib.escape(item.get("title", ""))
        code = item.get("code", "")
        code_block = (
            f"<details><summary>View reproducible code</summary>"
            f"<pre><code>{html_lib.escape(code)}</code></pre></details>"
            if code
            else ""
        ) + _r_block(item.get("r_code", ""))
        stats_parts.append(
            f'<div class="stat-card"><h3>{title}</h3>'
            f"{_md(item.get('result', ''))}{code_block}</div>"
        )
    stats_section = (
        f"<h2>Statistical Analysis</h2>{''.join(stats_parts)}"
        if stats_parts
        else ""
    )

    chart_parts = []
    plotly_js_embedded = False
    for artifact in final_artifacts:
        filename = artifact.filename
        fig = figure(artifact)
        code = artifact.code
        tool_label = get_tool_label(artifact.tool_name) or filename
        code_block = (
            f"<details><summary>View source code</summary>"
            f"<pre><code>{html_lib.escape(code)}</code></pre></details>"
            if code
            else ""
        ) + _r_block(artifact.r_code)
        if filename.endswith(".json") and fig is not None:
            # Embed the full Plotly JS bundle with the first chart so the
            # report is completely self-contained and never requests external
            # resources.
            include_js = not plotly_js_embedded
            chart_div = fig.to_html(
                full_html=False, include_plotlyjs=include_js
            )
            plotly_js_embedded = True
        else:
            img_b64 = base64.b64encode(artifact.data).decode()
            ext = filename.rsplit(".", 1)[-1].lower()
            mime = f"image/{ext}" if ext != "jpg" else "image/jpeg"
            chart_div = (
                f'<img src="data:{mime};base64,{img_b64}" '
                f'alt="{html_lib.escape(filename)}">'
            )
        chart_parts.append(
            f'<div class="chart-card">'
            f'<div class="chart-title">{html_lib.escape(tool_label)}</div>'
            f'<div class="chart-body">{chart_div}{code_block}</div>'
            f"</div>"
        )
    charts_section = (
        f"<h2>Visualisations</h2>{''.join(chart_parts)}" if chart_parts else ""
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Analysis Report — {html_lib.escape(run_id)}</title>
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


def results_zip(result: AnalysisResult, request: AnalysisRequest) -> bytes:
    """Return a ZIP of an analysis: charts, their code and the report.

    Interactive charts become standalone HTML files; each chart's Python
    code (and R code, if requested) is a file next to it.

    Args:
        result: The finished analysis.
        request: What was asked.

    Returns:
        The ZIP archive.
    """
    show_r = request.include_r_code
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for artifact in result.artifacts:
            filename = artifact.filename
            fig = figure(artifact)
            if filename.endswith(".json") and fig is not None:
                html_filename = filename.replace(".json", ".html")
                zf.writestr(html_filename, fig.to_html(include_plotlyjs=True))
            else:
                zf.writestr(filename, artifact.data)

            code_filename = filename.replace(".json", ".py").replace(
                ".png", ".py"
            )
            zf.writestr(code_filename, artifact.code)
            if show_r and artifact.r_code:
                zf.writestr(code_filename[:-3] + ".R", artifact.r_code)

        zf.writestr("report.html", build_html_report(result, request))
    return zip_buffer.getvalue()
