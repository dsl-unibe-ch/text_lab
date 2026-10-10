"""User interfaces for Text Lab.

Each interface lives in its own subpackage and calls the backend in
``textlab.features`` and ``textlab.common``; the backend never calls back into
this package. Today there is one interface, the Streamlit app in
``textlab.ui.streamlit``.
"""
