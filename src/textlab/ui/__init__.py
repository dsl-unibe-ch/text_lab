"""User interfaces for Text Lab.

Each interface lives in its own subpackage and calls the backend in
``textlab.features`` and ``textlab.common``; the backend never calls back
into this package. The Streamlit app moves to ``textlab.ui.streamlit``
during the refactor. Until then it lives in ``src/Home.py`` and
``src/pages/``.
"""
