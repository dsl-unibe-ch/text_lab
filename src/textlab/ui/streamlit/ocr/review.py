"""Reviewing survey responses: the Responses tab and questionnaire batches.

Survey extraction is experimental and hidden in the app
(:data:`.state.SURVEY_EXTRACTION_UI_ENABLED`); the questionnaire batch
review (:func:`render_survey_review`) is shown after a batch run in survey
batch mode.
"""

from __future__ import annotations

import base64

import pandas as pd
import streamlit as st

from textlab.features.ocr import doc_ir, service


def _b64_bytes(b64):
    """Decode base64 image data, or return ``None``."""
    if not b64:
        return None
    try:
        return base64.b64decode(b64)
    except Exception:
        return None


def render_survey_review():
    """Let the user check the detected form and drop anything spurious."""
    survey = st.session_state.get("survey_batch")
    if survey is None or not survey.readings:
        return

    st.markdown("### The questionnaire TextLab detected")
    st.caption(
        f"{survey.control_count} response controls in {survey.answer_count} "
        f"answers, learned from the batch itself. Check the outlines below: "
        "printed text can occasionally be mistaken for an empty checkbox."
    )
    overlays = survey.overlays()
    if overlays:
        tabs = st.tabs([f"Page {i + 1}" for i in range(len(overlays))])
        for tab, (name, data) in zip(
            tabs, sorted(overlays.items()), strict=False
        ):
            with tab:
                st.image(data, caption=name, use_container_width=True)

    overview = survey.overview()
    dead = overview[overview["never_marked"]]
    if len(dead):
        st.warning(
            f"{len(dead)} answer(s) nobody in this batch marked (shown as "
            "`never_marked` below). That is either a question they all "
            "skipped, "
            "or printed text mistaken for a control — the image above settles "
            "which."
        )
    st.dataframe(
        overview.drop(columns=["control_ids"]),
        use_container_width=True,
        hide_index=True,
    )

    labels = {
        row["answer"]: row["control_ids"] for _, row in overview.iterrows()
    }
    chosen = st.multiselect(
        "Remove answers that are not really on the form",
        options=list(labels),
        # Deliberately not pre-selected: an answer nobody marked is just as
        # likely a question the whole batch skipped as a false positive, and
        # only the image above settles it.
        key="survey_drop",
        help=(
            "Removes them from every file that carries answers — the batch "
            "tables, each questionnaire's own answers and its document.md — "
            "and renumbers the rest. The questionnaires are not parsed again."
        ),
    )
    if chosen and st.button("Rebuild the exports", key="survey_rebuild"):
        ids = [i for name in chosen for i in labels[name].split(",") if i]
        zip_bytes, removed, summary = survey.drop_controls(
            ids, st.session_state.batch_auto_zip
        )
        st.session_state.batch_auto_zip = zip_bytes
        st.success(
            f"Removed {removed} control(s); {summary['controls']} remain in "
            f"{summary['answer_groups']} answers. Every export was rewritten "
            "— "
            "**download the ZIP again** to get them."
        )
        st.rerun()


def _group_is_multiselect(group):
    """Return True when a question's rows allow several answers."""
    return (
        group.question_type == "multiple"
        or group.selection_rule == "zero_or_more"
    )


REVIEW_REASON_LABELS = {
    "answer_geometry_disagreement": (
        "answer conflicts with geometric ink evidence"
    ),
    "ambiguous_mark": "ambiguous response mark",
    "extraction_failed": "response extraction failed",
    "missing_answer": "expected answer was not found",
    "selection_rule_violation": "answer violates the selection rule",
    "source_disagreement": "answer evidence disagrees",
    "structural_issue": "question structure needs checking",
    "unbenchmarked_model": "model has not passed the release benchmark",
    "unmapped_mark": "visible ink was not mapped to an answer",
    "validation_warning": "validation warning",
}


def _option_labels(row):
    """Unambiguous labels suitable for both widgets and correction mapping."""
    labels = [
        (option.label.strip() or f"option {index + 1}")
        for index, option in enumerate(row.options)
    ]
    duplicates = {label for label in labels if labels.count(label) > 1}
    return [
        f"{label} [{index + 1}]" if label in duplicates else label
        for index, label in enumerate(labels)
    ]


def _selected_answer(row):
    labels = _option_labels(row)
    selected = [
        labels[index]
        for index, option in enumerate(row.options)
        if option.state == "selected"
    ]
    return " | ".join(selected) if selected else "— no answer —"


def _group_answer_summary(group):
    answers = []
    for row in group.rows:
        answer = _selected_answer(row)
        answers.append(f"{row.label}: {answer}" if row.label else answer)
    return "; ".join(answers) if answers else "— extraction failed —"


def _matrix_uses_shared_single_choice_options(group):
    """Whether a matrix can be edited safely as one compact answer column."""
    if (
        group.question_type != "matrix"
        or not group.rows
        or _group_is_multiselect(group)
    ):
        return False
    first = _option_labels(group.rows[0])
    return bool(first) and all(
        _option_labels(row) == first
        and sum(option.state == "selected" for option in row.options) <= 1
        for row in group.rows
    )


def _render_matrix_editor(group):
    """Render a matrix as one table; return each row's chosen position."""
    labels = _option_labels(group.rows[0])
    choices = ["— none —", *labels]
    records = []
    for row in group.rows:
        selected_position = next(
            (
                index + 1
                for index, option in enumerate(row.options)
                if option.state == "selected"
            ),
            0,
        )
        reasons = [
            REVIEW_REASON_LABELS.get(reason, reason.replace("_", " "))
            for reason in row.review_reasons
        ]
        records.append(
            {
                "Row": row.label or row.id,
                "Answer": choices[selected_position],
                "Review": " · ".join(reasons),
            }
        )
    edited = st.data_editor(
        pd.DataFrame(records),
        key=f"rev_matrix_{group.id}",
        hide_index=True,
        use_container_width=True,
        disabled=["Row", "Review"],
        column_config={
            "Row": st.column_config.TextColumn(width="large"),
            "Answer": st.column_config.SelectboxColumn(
                options=choices,
                required=True,
                width="medium",
            ),
            "Review": st.column_config.TextColumn(width="large"),
        },
    )
    return {
        row.id: choices.index(answer) if answer in choices else 0
        for row, answer in zip(
            group.rows, edited["Answer"].tolist(), strict=False
        )
    }


def _render_form_review(document):
    """Summary-first response review with compact question-level editors."""
    form_groups = [
        (page, g) for page in document.pages for g in page.form_groups
    ]
    if not form_groups:
        return
    n_review = sum(g.status == "needs_review" for _, g in form_groups)
    if n_review:
        st.warning(
            f"{n_review} of {len(form_groups)} question(s) were flagged for "
            "review. "
            "Flagged questions are open below; accepted questions stay "
            "collapsed. "
            "Saving updates the downloads and retains the original model "
            "answer in JSON."
        )
    else:
        st.success(
            f"Extracted {len(form_groups)} question(s). Expand any row below "
            "to make a correction."
        )

    summary_records = [
        {
            "Question": group.question_text or group.id,
            "Answer": _group_answer_summary(group),
            "Status": "Needs review"
            if group.status == "needs_review"
            else "Accepted",
            "Page": page.page_number,
        }
        for page, group in form_groups
    ]
    st.dataframe(
        pd.DataFrame(summary_records),
        hide_index=True,
        use_container_width=True,
        column_config={
            "Question": st.column_config.TextColumn(width="large"),
            "Answer": st.column_config.TextColumn(width="large"),
            "Status": st.column_config.TextColumn(width="small"),
            "Page": st.column_config.NumberColumn(width="small"),
        },
    )

    matrix_edits = {}
    with st.form("survey_review_form"):
        for page, group in form_groups:
            multi = _group_is_multiselect(group)
            flag = "Needs review: " if group.status == "needs_review" else ""
            question = group.question_text or group.id
            answer = _group_answer_summary(group)
            expander_label = f"{flag}{question} — {answer}"
            if len(expander_label) > 180:
                expander_label = f"{expander_label[:177]}…"
            with st.expander(
                expander_label,
                expanded=(group.status == "needs_review"),
            ):
                needs_review = group.status == "needs_review"
                layout = (
                    st.columns([3, 2]) if needs_review else [st.container()]
                )
                with layout[0]:
                    st.caption(
                        f"`{group.question_type}` · `{group.selection_rule}` "
                        "· "
                        f"page {page.page_number}"
                    )
                    if group.condition_text:
                        st.caption(f"↳ conditional: {group.condition_text}")
                    if group.review_reasons:
                        reasons = [
                            REVIEW_REASON_LABELS.get(
                                reason, reason.replace("_", " ")
                            )
                            for reason in group.review_reasons
                        ]
                        st.caption(f"Review because: {'; '.join(reasons)}")
                    for warning in group.warnings:
                        st.caption(f"Warning: {warning}")
                    if not group.rows or not any(
                        row.options for row in group.rows
                    ):
                        st.info(
                            "Options for this question could not be "
                            "extracted automatically. "
                            "This item remains flagged after saving because "
                            "there is no safe "
                            "correction control."
                        )
                    elif _matrix_uses_shared_single_choice_options(group):
                        matrix_edits[group.id] = _render_matrix_editor(group)
                    else:
                        for row in group.rows:
                            key_base = f"rev_{group.id}_{row.id}"
                            labels = _option_labels(row)
                            if not row.options:
                                st.caption(
                                    f"_{row.label or 'row'}: no options "
                                    "detected_"
                                )
                                continue
                            if row.review_reasons:
                                reasons = [
                                    REVIEW_REASON_LABELS.get(
                                        reason, reason.replace("_", " ")
                                    )
                                    for reason in row.review_reasons
                                ]
                                label = row.label or "Answer"
                                st.caption(f"{label}: {'; '.join(reasons)}")
                            if multi:
                                if row.label:
                                    st.markdown(f"*{row.label}*")
                                n_cols = min(len(row.options), 4)
                                cols = st.columns(n_cols)
                                for index, option in enumerate(row.options):
                                    cols[index % n_cols].checkbox(
                                        labels[index],
                                        value=(option.state == "selected"),
                                        key=f"{key_base}_{option.id}",
                                    )
                            else:
                                selected_index = next(
                                    (
                                        index + 1
                                        for index, option in enumerate(
                                            row.options
                                        )
                                        if option.state == "selected"
                                    ),
                                    0,
                                )
                                st.radio(
                                    row.label or "Answer",
                                    options=list(range(len(row.options) + 1)),
                                    index=selected_index,
                                    format_func=lambda value, _labels=labels: (
                                        "— none —"
                                        if value == 0
                                        else _labels[value - 1]
                                    ),
                                    key=key_base,
                                    horizontal=(len(row.options) <= 6),
                                )
                            for index, option in enumerate(row.options):
                                if option.associated_text:
                                    st.text_input(
                                        f"Handwriting near “{labels[index]}”",
                                        value=option.associated_text,
                                        key=f"{key_base}_{option.id}_txt",
                                    )
                if needs_review:
                    with layout[1]:
                        st.caption("Source section")
                        crop = _b64_bytes(group.source_crop_b64)
                        if crop:
                            st.image(
                                crop,
                                caption=f"Page {page.page_number}",
                                use_container_width=True,
                            )
                        else:
                            st.warning(
                                "The source crop is unavailable; this item "
                                "cannot be "
                                "verified visually."
                            )
        submitted = st.form_submit_button("Save corrections", type="primary")

    if submitted:
        _apply_form_corrections(document, matrix_edits)
        st.session_state.auto_downloads.refresh_responses(document)
        st.session_state.auto_summary = service.document_summary(document)
        st.success(
            "Corrections saved. The downloads above now reflect your edits."
        )


def _apply_form_corrections(document, matrix_edits=None):
    """Write the reviewer's choices into the document, keeping the original."""
    matrix_edits = matrix_edits or {}
    for page in document.pages:
        for group in page.form_groups:
            multi = _group_is_multiselect(group)
            if "model_answer" not in group.provenance:
                group.provenance["model_answer"] = [
                    {
                        "row": row.label or row.id,
                        "selected": [
                            o.label
                            for o in row.options
                            if o.state == "selected"
                        ],
                        "states": {o.id: o.state for o in row.options},
                    }
                    for row in group.rows
                ]
                group.provenance["pre_review_reasons"] = {
                    "group": list(group.review_reasons),
                    "rows": {
                        row.id: list(row.review_reasons) for row in group.rows
                    },
                }
            reviewed_rows = 0
            unresolved_rows = 0
            for row in group.rows:
                key_base = f"rev_{group.id}_{row.id}"
                if not row.options:
                    unresolved_rows += 1
                    continue
                previous_states = {
                    option.id: option.state for option in row.options
                }
                if group.id in matrix_edits:
                    selected_position = matrix_edits[group.id].get(row.id, 0)
                    for index, option in enumerate(row.options):
                        option.state = (
                            "selected"
                            if selected_position == index + 1
                            else "unselected"
                        )
                elif multi:
                    for o in row.options:
                        chosen = st.session_state.get(
                            f"{key_base}_{o.id}", o.state == "selected"
                        )
                        o.state = "selected" if chosen else "unselected"
                else:
                    sel = st.session_state.get(key_base)
                    for i, o in enumerate(row.options):
                        o.state = "selected" if sel == i + 1 else "unselected"
                for o in row.options:
                    tkey = f"{key_base}_{o.id}_txt"
                    if tkey in st.session_state:
                        o.associated_text = st.session_state[tkey]
                    o.observations.append(
                        doc_ir.Observation(
                            source="human-review",
                            value=o.state,
                            method="responses-tab",
                            raw={"previous_state": previous_states[o.id]},
                        )
                    )
                row.status = "accepted"
                row.review_reasons.clear()
                reviewed_rows += 1
            if reviewed_rows and not unresolved_rows:
                group.status = "accepted"
                group.review_reasons.clear()
                group.provenance["human_reviewed"] = True
            elif unresolved_rows or not group.rows:
                group.status = "needs_review"


def _render_legacy_mark_summary(glyph_regions, checkbox_marks):
    """Compact, image-free note for the older geometric mark detector."""
    n_total = len(checkbox_marks) + sum(
        len((r.markup or {}).get("items", [])) for _, r in glyph_regions
    )
    n_uncertain = sum(
        1
        for _, r in checkbox_marks
        if (r.markup or {}).get("state") == "uncertain"
    ) + sum((r.markup or {}).get("n_uncertain", 0) for _, r in glyph_regions)
    with st.expander(
        "Geometric mark detector (legacy) — details in the JSON / Layout "
        "preview"
    ):
        st.caption(
            f"{n_total} geometric mark(s) across "
            f"{len(glyph_regions) + len(checkbox_marks)} region(s)"
            + (f"; {n_uncertain} uncertain" if n_uncertain else "")
            + "."
        )


def render_markup_tab(document):
    """Show the Responses tab: the form review and the mark summary."""
    form_groups = [
        (page, group) for page in document.pages for group in page.form_groups
    ]
    checkbox_marks = [
        (page, region)
        for page, region in document.all_regions()
        if region.type == doc_ir.CHECKBOX and region.markup
    ]
    glyph_regions = [
        (page, region)
        for page, region in document.all_regions()
        if region.type != doc_ir.CHECKBOX
        and (region.markup or {}).get("kind") == "glyph-marks"
    ]
    if not form_groups and not checkbox_marks and not glyph_regions:
        st.info(
            "No survey responses were extracted. Enable **Extract survey/form "
            "responses** before parsing to run enhanced response analysis."
        )
        return

    if form_groups:
        _render_form_review(document)

    if glyph_regions or checkbox_marks:
        _render_legacy_mark_summary(glyph_regions, checkbox_marks)
    return
