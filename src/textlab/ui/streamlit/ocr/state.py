"""Session state shared by the OCR page's workflows."""

from __future__ import annotations

import streamlit as st

#: Hides the survey/form controls and the Responses tab while the extractor
#: is validated; the backend stays reachable with ``extract_survey=True``.
SURVEY_EXTRACTION_UI_ENABLED = False


def clear_results(reset_running=False):
    """Forget the results of the previous run.

    Called when a new file is chosen or the workflow changes, so a new
    document never shows the previous one's results or review selections.

    Args:
        reset_running: Also clear the "a job is running" flag.
    """
    keys_to_clear = [
        # manual engines
        "manual_run",
        "manual_batch_zip",
        "manual_error",
        "manual_error_details",
        "manual_preview_page",
        # automatic pipeline
        "auto_complete",
        "auto_document",
        "auto_summary",
        "auto_downloads",
        "auto_error",
        "batch_auto_complete",
        "batch_auto_zip",
        "batch_auto_elapsed",
        # questionnaire batch, for the review of the detected form
        "survey_batch",
    ]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
    # Interactive survey-review widgets are keyed "rev_<group>_<row>..."; drop
    # them so a new document does not inherit the previous document's
    # selections.
    for key in [
        k
        for k in st.session_state
        if isinstance(k, str) and k.startswith("rev_")
    ]:
        del st.session_state[key]
    if reset_running:
        st.session_state.ocr_running = False


def init_state() -> None:
    """Set the session-state defaults the page relies on."""
    if "ocr_running" not in st.session_state:
        st.session_state.ocr_running = False
