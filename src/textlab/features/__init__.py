"""Backend packages, one per Text Lab feature.

Each feature package holds the processing logic for one tool and exposes it
through plain functions and data classes, so the Streamlit app, a future web
frontend and command-line tools can all call the same code. Nothing in this
package may import ``streamlit`` or ``textlab.ui``; the import-linter
contract in ``pyproject.toml`` enforces that.
"""
