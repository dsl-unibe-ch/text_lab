"""The disclaimer about AI output, shown in the sidebar of every page."""

import streamlit as st

DISCLAIMER = (
    "**Disclaimer**  \n"
    "AI models may produce inaccurate or misleading output. Please review "
    "all results before using them in your research."
)


def render_disclaimer() -> None:
    """Show the disclaimer in the sidebar.

    Pages call it right after ``check_token()``, so that it appears even on
    pages that stop early, e.g. when the model server is not available.
    """
    st.sidebar.markdown(DISCLAIMER)
