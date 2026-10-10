"""Exporting a conversation as Markdown or HTML, with its charts."""

from __future__ import annotations

from typing import Any


def _artifact_to_markdown(artifact: dict[str, Any], index: int) -> str:
    """Render a single plot artifact as embeddable Markdown.

    Static images (PNG) are embedded inline as base64 data URIs. Interactive
    Plotly charts (stored as JSON) are converted to a static PNG via kaleido
    when available; otherwise a note and the reproducible source code are
    included instead.
    """
    import base64

    tool_label = artifact.get("tool_name", "") or f"Plot {index + 1}"
    md = f"**{tool_label.replace('_', ' ').title()}**\n\n"
    fig_json = artifact.get("fig_json")
    img_bytes = artifact.get("bytes")
    png_b64 = None

    if fig_json:
        try:
            import plotly.io as pio

            fig = pio.from_json(fig_json)
            png_bytes = fig.to_image(format="png")  # requires kaleido
            png_b64 = base64.b64encode(png_bytes).decode("ascii")
        except Exception:
            png_b64 = None
    elif img_bytes and artifact.get("filename", "").lower().endswith(
        (".png", ".jpg", ".jpeg")
    ):
        png_b64 = base64.b64encode(img_bytes).decode("ascii")

    if png_b64:
        md += f"![{tool_label}](data:image/png;base64,{png_b64})\n\n"
    elif fig_json:
        md += (
            "_Interactive chart — not embeddable as a static image in this "
            "export. "
            "Reproduce it with the code below._\n\n"
        )

    code = artifact.get("code")
    if code:
        md += f"```python\n{code}\n```\n\n"
    return md


def has_analysis_plots(messages: list[dict[str, Any]]) -> bool:
    """Return True if any assistant turn produced plot artifacts."""
    for msg in messages:
        analysis = msg.get("analysis") if isinstance(msg, dict) else None
        if analysis and analysis.get("artifacts"):
            return True
    return False


def format_chat_history_html(messages: list[dict[str, Any]]) -> str:
    """Convert the conversation into a self-contained HTML document.

    Interactive Plotly charts are rendered using Plotly's own ``fig.to_html``
    (which generates correct embedding code). Plotly.js is bundled inline on
    the first interactive chart so the file works without an internet
    connection. Static images are embedded as base64 data URIs.

    Args:
        messages (list): The conversation history.

    Returns:
        str: A complete HTML document string.
    """
    import base64
    import html as html_lib

    parts: list[str] = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        "<title>Text Lab Chat Export</title>",
        "<style>",
        "body{font-family:system-ui,Arial,sans-serif;max-width:900px;"
        "margin:2rem auto;"
        "padding:0 1rem;line-height:1.5;color:#1e1e1e;}",
        ".turn{border:1px solid #ddd;border-radius:8px;padding:1rem "
        "1.25rem;margin:1rem 0;}",
        ".user{background:#f5f5f5;}.assistant{background:#fff;}",
        ".role{font-weight:600;margin-bottom:.5rem;color:#555;}",
        "img{max-width:100%;height:auto;}",
        "pre{background:#f0f0f0;padding:.75rem;border-radius:6px;overflow-x:auto;}",
        "h4{margin-top:1.25rem;}",
        "</style>",
        "</head><body>",
        "<h1>Text Lab Chat Export</h1>",
    ]

    # Track whether Plotly.js has been bundled yet. First interactive chart
    # includes it inline; subsequent charts omit it to avoid duplication. This
    # keeps the file self-contained and offline-capable.
    plotlyjs_included = False

    for msg in messages:
        role = msg.get("role")
        if role not in ("user", "assistant"):
            continue
        content_html = html_lib.escape(msg.get("content", "")).replace(
            "\n", "<br>"
        )
        parts.append(f"<div class='turn {role}'>")
        parts.append(f"<div class='role'>{role.title()}</div>")
        parts.append(f"<div>{content_html}</div>")

        analysis = msg.get("analysis") if role == "assistant" else None
        if analysis:
            artifacts = analysis.get("artifacts", [])
            if artifacts:
                parts.append("<h4>Generated Visualisations</h4>")
            for artifact in artifacts:
                label = html_lib.escape(
                    (artifact.get("tool_name", "") or "Plot")
                    .replace("_", " ")
                    .title()
                )
                parts.append(f"<p><strong>{label}</strong></p>")
                fig_json = artifact.get("fig_json")
                if fig_json:
                    try:
                        import plotly.io as pio

                        fig = pio.from_json(fig_json)
                        # Bundle plotlyjs inline on the first chart so the
                        # exported file is self-contained. Subsequent charts
                        # skip it.
                        include_js = True if not plotlyjs_included else False
                        chart_html = fig.to_html(
                            full_html=False,
                            include_plotlyjs=include_js,
                        )
                        plotlyjs_included = True
                        parts.append(chart_html)
                    except Exception:
                        parts.append(
                            "<p><em>Could not render interactive "
                            "chart.</em></p>"
                        )
                else:
                    img_bytes = artifact.get("bytes")
                    if img_bytes:
                        b64 = base64.b64encode(img_bytes).decode("ascii")
                        parts.append(
                            f"<img src='data:image/png;base64,{b64}' "
                            f"alt='{label}'>"
                        )
                code = artifact.get("code")
                if code:
                    parts.append(
                        f"<details><summary>View source code</summary>"
                        f"<pre>{html_lib.escape(code)}</pre></details>"
                    )

            stats_results = analysis.get("stats", [])
            if stats_results:
                parts.append("<h4>Statistical Analysis Results</h4>")
                for item in stats_results:
                    parts.append(
                        "<p><strong>"
                        f"{html_lib.escape(item.get('title', 'Result'))}"
                        "</strong></p>"
                    )
                    result_html = html_lib.escape(
                        item.get("result", "")
                    ).replace("\n", "<br>")
                    parts.append(f"<div>{result_html}</div>")
                    if item.get("code"):
                        parts.append(
                            f"<details><summary>View source code</summary>"
                            f"<pre>{html_lib.escape(item['code'])}</pre>"
                            "</details>"
                        )

        parts.append("</div>")

    parts.append("</body></html>")
    return "".join(parts)


def format_chat_history(messages: list[dict[str, str]]) -> str:
    """Convert the conversation into a readable Markdown document.

    Assistant turns that produced data analysis embed their plots (as base64
    images) and statistical results so the exported document is self-contained.

    Args:
        messages (list): The conversation history.

    Returns:
        str: The formatted Markdown document.
    """
    formatted_text = "# Text Lab Chat Export\n\n"
    for msg in messages:
        if msg["role"] == "user":
            formatted_text += f"### User\n{msg['content']}\n\n---\n\n"
        elif msg["role"] == "assistant":
            formatted_text += f"### Assistant\n{msg['content']}\n\n"

            analysis = msg.get("analysis")
            if analysis:
                artifacts = analysis.get("artifacts", [])
                if artifacts:
                    formatted_text += "#### Generated Visualisations\n\n"
                    for idx, artifact in enumerate(artifacts):
                        formatted_text += _artifact_to_markdown(artifact, idx)

                stats_results = analysis.get("stats", [])
                if stats_results:
                    formatted_text += "#### Statistical Analysis Results\n\n"
                    for item in stats_results:
                        formatted_text += (
                            f"**{item.get('title', 'Result')}**\n\n"
                        )
                        formatted_text += f"{item.get('result', '')}\n\n"
                        if item.get("code"):
                            formatted_text += (
                                f"```python\n{item['code']}\n```\n\n"
                            )

            formatted_text += "---\n\n"
    return formatted_text
