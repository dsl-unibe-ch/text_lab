"""The result of one recognized document: notices, downloads and tabs."""

from __future__ import annotations

import base64

import streamlit as st

from textlab.common import html_safety
from textlab.features.ocr import doc_ir, service
from textlab.ui.streamlit.ocr.review import render_markup_tab
from textlab.ui.streamlit.ocr.state import SURVEY_EXTRACTION_UI_ENABLED


def _b64_bytes(b64):
    if not b64:
        return None
    try:
        return base64.b64decode(b64)
    except Exception:
        return None


def render_document_tab(document):
    """Show the document as it reads: text, tables, figures, checkboxes."""
    for page in document.pages:
        if len(document.pages) > 1:
            st.caption(f"— Page {page.page_number} · {page.source} —")
        for region in page.ordered_regions():
            rtype = region.type
            if rtype == doc_ir.TITLE:
                text = region.text.strip()
                if text:
                    st.markdown(f"### {text}")
            elif rtype == doc_ir.TABLE:
                html = html_safety.sanitize_table_html(
                    region.content.get("html", "").strip()
                )
                if html:
                    st.markdown(html, unsafe_allow_html=True)
            elif rtype == doc_ir.FORMULA:
                latex = region.content.get("latex", "").strip()
                if latex:
                    try:
                        st.latex(latex)
                    except Exception:
                        st.code(latex)
            elif rtype in (doc_ir.FIGURE, doc_ir.SEAL):
                data = _b64_bytes((region.asset or {}).get("b64"))
                if data:
                    st.image(data, caption=(region.text.strip() or rtype))
                if region.visual_description:
                    st.caption(
                        f"AI description: "
                        f"{region.visual_description.description}"
                    )
            elif rtype == doc_ir.CHECKBOX:
                state = (region.markup or {}).get("state", "uncertain")
                icon = {"checked": "☑", "unchecked": "☐"}.get(state, "?")
                st.markdown(f"{icon} {region.text.strip()}".rstrip())
            else:
                text = region.text.strip()
                if text:
                    st.markdown(text)
        if len(document.pages) > 1:
            st.divider()


def render_tables_tab(document):
    """Show each table with a CSV download."""
    tables = doc_ir.tables_to_dataframes(document)
    if not tables:
        st.info("No tables were detected in this document.")
        return
    for entry in tables:
        st.markdown(
            f"**Table — page {entry['page']}** (`{entry['region_id']}`)"
        )
        st.dataframe(entry["dataframe"], use_container_width=True)
        st.download_button(
            "Download this table (CSV)",
            entry["dataframe"].to_csv(index=False).encode("utf-8"),
            file_name=f"table_{entry['region_id']}.csv",
            mime="text/csv",
            key=f"tbl_csv_{entry['region_id']}",
        )
        st.divider()


def render_figures_tab(document):
    """Show the figures with their captions and generated descriptions."""
    figures = [
        (page, region)
        for page, region in document.all_regions()
        if region.type in (doc_ir.FIGURE, doc_ir.SEAL) and region.asset
    ]
    if not figures:
        st.info("No figures, charts, or seals were detected.")
        return
    cols = st.columns(2)
    for i, (page, region) in enumerate(figures):
        with cols[i % 2]:
            data = _b64_bytes(region.asset.get("b64"))
            if data:
                st.image(
                    data,
                    use_container_width=True,
                    caption=f"Page {page.page_number} · {region.type}",
                )
            caption = region.text.strip()
            if caption:
                st.caption(f"Printed caption/text: {caption[:300]}")
            generated = region.visual_description
            if generated:
                st.markdown(generated.description)
                if generated.visible_text:
                    st.caption(f"Visible text: {generated.visible_text[:500]}")
                st.caption(
                    f"AI description · {generated.source} · {generated.model}"
                )


def render_layout_tab(document):
    """Show a page with its regions outlined by type, with a legend."""
    if not document.pages:
        st.info("Nothing to preview.")
        return

    page_options = list(range(len(document.pages)))
    idx = st.selectbox(
        "Page",
        page_options,
        format_func=lambda i: f"Page {document.pages[i].page_number}",
        key="auto_layout_page_select",
    )
    page = document.pages[idx]
    img_bytes = _b64_bytes(page.image_b64)
    if not img_bytes:
        st.info(
            "No page image is available for this page (native text-only page)."
        )
        return

    regions = [
        {"bbox": r.bbox, "type": r.type} for r in page.regions if r.bbox
    ]
    preview = service.render_layout_preview(img_bytes, regions)
    if preview:
        st.image(preview, use_container_width=True)
    else:
        st.image(img_bytes, use_container_width=True)

    present_types = sorted({r.type for r in page.regions})
    if present_types:

        def swatch(region_type):
            blue, green, red = service.LAYOUT_TYPE_COLORS.get(
                region_type, (90, 90, 90)
            )
            return (
                f"<span style='color:rgb({red},{green},{blue})'>&#9632; "
                f"{region_type}</span>"
            )

        legend = "  ".join(swatch(t) for t in present_types)
        st.markdown(f"**Legend:** {legend}", unsafe_allow_html=True)


def render_document_result():
    """Show the document in session state: notices, downloads and tabs."""
    document = st.session_state.get("auto_document")
    if document is None:
        return
    summary = st.session_state.get("auto_summary", {})
    downloads = st.session_state.get("auto_downloads")

    n_pages = summary.get("n_pages", 0)
    counts = summary.get("region_counts", {})
    chips = " · ".join(f"{k}: {v}" for k, v in counts.items()) or "no regions"
    st.success(
        f"Parsed {n_pages} page(s) · "
        f"{', '.join(summary.get('routes', [])) or 'no route'} · {chips}"
    )

    # One collapsed block, so notices cannot push the downloads off-screen.
    notices = []
    if summary.get("n_form_groups"):
        notices.append(
            f"Extracted {summary['n_form_groups']} survey/form response "
            "group(s) — "
            "see the **Responses** tab."
        )
    if summary.get("n_markup_disagreements"):
        notices.append(
            f"{summary['n_markup_disagreements']} OCR/geometric mark "
            "disagreement(s) "
            "were left unchanged and flagged for review."
        )
    if summary.get("n_uncertain_marks"):
        notices.append(
            f"{summary['n_uncertain_marks']} checkbox/mark(s) flagged "
            "uncertain — "
            "see the **Responses** tab."
        )
    if notices:
        with st.expander(f"{len(notices)} thing(s) to check", expanded=False):
            for notice in notices:
                st.markdown(f"- {notice}")

    # --- Downloads ---
    stem = downloads.stem
    # Grouped by purpose; every slot always renders, so a missing output greys
    # out in place instead of reflowing the grid.
    searchable = downloads.searchable_pdf
    text_bytes = downloads.text
    docx_bytes = downloads.docx
    md_zip = downloads.markdown_zip
    tables_zip = downloads.tables_zip

    st.caption("**Read and edit**")
    d1, d2, d3, d4 = st.columns(4)
    with d1:
        st.download_button(
            "Plain text",
            text_bytes or b"",
            file_name=f"{stem}.txt",
            mime="text/plain",
            disabled=not text_bytes,
            use_container_width=True,
        )
    with d2:
        st.download_button(
            "Word",
            docx_bytes or b"",
            file_name=f"{stem}.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            disabled=not docx_bytes,
            use_container_width=True,
            help=None
            if docx_bytes
            else "Unavailable: python-docx is not installed.",
        )
    with d3:
        st.download_button(
            "Markdown",
            md_zip or b"",
            file_name=f"{stem}_markdown.zip",
            mime="application/zip",
            disabled=not md_zip,
            use_container_width=True,
            help="Markdown plus an `assets/` folder of cropped figures.",
        )
    with d4:
        st.download_button(
            "Searchable PDF",
            searchable or b"",
            file_name=f"{stem}_searchable.pdf",
            mime="application/pdf",
            disabled=not searchable,
            use_container_width=True,
            help=(
                "The original pages with an invisible, selectable text layer."
                if searchable
                else "Tick **Searchable PDF** before parsing to produce this."
            ),
        )

    st.caption("**Analyse**")
    d5, d6, d7 = st.columns(3)
    with d5:
        st.download_button(
            "JSON",
            downloads.json or b"{}",
            file_name=f"{stem}.json",
            mime="application/json",
            use_container_width=True,
            help="Regions, bounding boxes, confidence and markup states.",
        )
    with d6:
        st.download_button(
            "Tables (CSV)",
            tables_zip or b"",
            file_name=f"{stem}_tables.zip",
            mime="application/zip",
            disabled=not tables_zip,
            use_container_width=True,
            help=None
            if tables_zip
            else "No tables were detected in this document.",
        )
    with d7:
        st.download_button(
            "Everything",
            downloads.full or b"",
            file_name=f"{stem}_bundle.zip",
            mime="application/zip",
            disabled=not downloads.full,
            use_container_width=True,
            type="primary",
            help="Every format above in one ZIP.",
        )

    responses_csv = downloads.responses_csv
    if responses_csv:
        st.download_button(
            "Form responses (CSV)",
            responses_csv,
            file_name=f"{stem}_form_responses.csv",
            mime="text/csv",
        )

    # Read off the document, so the citation cannot drift from what ran.
    provenance = doc_ir.model_provenance(document)
    if provenance:
        with st.expander("Models used (for citation)", expanded=False):
            labels = {
                "text_recognition": "**Text recognition**",
                "figure_descriptions": "**Figure descriptions**",
                "text_layer": "**Searchable-PDF word geometry**",
            }
            for key, value in provenance.items():
                joined = (
                    ", ".join(value) if isinstance(value, list) else str(value)
                )
                st.markdown(f"- {labels.get(key, key)}: {joined}")
            st.caption(
                "Printed text is transcribed by the recognition model; figure "
                "descriptions are *generated* by a vision-language model and "
                "are "
                "not part of the document. The same summary ships as "
                "`models_used.txt` in the bundle and under `models` in the "
                "JSON."
            )

    # Without survey extraction the Responses tab would always be empty.
    labels = ["Document", "Tables", "Figures"]
    if SURVEY_EXTRACTION_UI_ENABLED:
        labels.append("Responses")
    labels.append("Layout preview")

    tabs = dict(zip(labels, st.tabs(labels), strict=False))
    with tabs["Document"]:
        render_document_tab(document)
    with tabs["Tables"]:
        render_tables_tab(document)
    with tabs["Figures"]:
        render_figures_tab(document)
    if SURVEY_EXTRACTION_UI_ENABLED:
        with tabs["Responses"]:
            render_markup_tab(document)
    with tabs["Layout preview"]:
        render_layout_tab(document)
