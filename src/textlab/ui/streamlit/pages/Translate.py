"""Translate page for Text Lab.

Two clearly separated workflows:

* **Text** — DeepL-style split-screen editor with source-language
  auto-detection, character/word counters, glossary/term-lock,
  language-swap, and formality control (LLM backend).
* **Document** — drag-and-drop upload for .md / .txt / .srt / .vtt /
  .pdf / .docx / .xlsx / .pptx files (or ZIP of any). The tool parses,
  translates, and reconstructs the file in its original format with
  structural markup preserved. Supports multi-file batch upload.

Translation goes through :mod:`textlab.features.translation.service`, which
shields Markdown links, LaTeX equations, inline code, HTML tags, URLs and
placeholders so they survive intact, and enforces glossary terms the same
way.

State-management note
---------------------
This page follows the "session_state is the widget state" pattern:
every widget uses a single ``key=`` and its value is read/written via
``st.session_state[key]``. Swap operations use ``on_click`` callbacks so
mutations happen *before* the widgets re-render on the next frame — the
pattern that fixes the classic "value= is ignored after rerun" trap.
"""

from __future__ import annotations

import html
import os
import time

import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

current_dir = os.path.dirname(os.path.abspath(__file__))
app_dir = os.path.dirname(current_dir)
favicon_path = os.path.join(app_dir, "assets", "text_lab_logo.png")
favicon = Image.open(favicon_path)

st.set_page_config(page_title="Translate", page_icon=favicon, layout="wide")

from textlab.common import gpu_manager  # noqa: E402
from textlab.common.language_mappings import (  # noqa: E402
    TRANSLATE_LANGUAGE_MAPPING,
)
from textlab.features.translation import service  # noqa: E402
from textlab.ui.streamlit.auth import check_token  # noqa: E402
from textlab.ui.streamlit.components.gpu import free_gpu_for  # noqa: E402

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
TEXT_SOFT_CAP = 5_000  # characters; UI warns above this
GLOSSARY_MAX_ROWS = 50


check_token()
st.title("Translate")
st.caption(
    "Neural machine translation — all inference runs locally in your session."
)


# ---------------------------------------------------------------------------
# Session state — single source of truth per widget
# ---------------------------------------------------------------------------
_STATE_DEFAULTS = {
    # Language selectboxes (also the actual widget state — same key).
    "src_lang": "German",
    "tgt_lang": "English",
    # Text areas (also the actual widget state for the source).
    "source_text": "",
    "target_text": "",
    # Backend & options.
    "backend_label": next(iter(service.TRANSLATION_BACKENDS.values())),
    "formality": "default",
    "ollama_model": None,
    # Glossary.
    "glossary_rows": [{"source": "", "target": ""}],
    "glossary_case_sensitive": False,
    # Detection (transient).
    "detection": None,
    # Persistent translation error (survives the st.rerun after Translate).
    "translate_error": None,
    "translate_traceback": None,
    "translation_notices": [],
    # Explicit "load model" state for the Text tab. Compared against the
    # current (backend, langs, ollama_model) signature to gate translation.
    "loaded_signature": None,
    "load_error": None,
    "load_traceback": None,
}
for _k, _v in _STATE_DEFAULTS.items():
    st.session_state.setdefault(_k, _v)


def _swap_languages() -> None:
    """Callback for the ⇄ button.

    Runs BEFORE the language selectboxes / text_areas re-render on the
    next frame, so the widgets pick up the swapped values naturally.
    """
    st.session_state["src_lang"], st.session_state["tgt_lang"] = (
        st.session_state["tgt_lang"],
        st.session_state["src_lang"],
    )
    st.session_state["source_text"], st.session_state["target_text"] = (
        st.session_state["target_text"],
        st.session_state["source_text"],
    )
    # Stale detection of the previous direction no longer applies.
    st.session_state["detection"] = None


def _use_detected_source() -> None:
    det = st.session_state.get("detection")
    if det is not None and det.display_name:
        st.session_state["src_lang"] = det.display_name


# ---------------------------------------------------------------------------
# Shared controls (backend + languages)
# ---------------------------------------------------------------------------
lang_names = list(TRANSLATE_LANGUAGE_MAPPING.keys())

col_backend, col_src, col_swap, col_tgt = st.columns([2, 1, 0.4, 1])

with col_backend:
    st.selectbox(
        "Translation backend",
        list(service.TRANSLATION_BACKENDS.values()),
        key="backend_label",
        help=(
            "NLLB-200 covers 200 languages. MADLAD-400 is strong on "
            "low-resource languages. OPUS-MT is small/fast per pair. "
            "The Ollama LLM option is best for dialects (e.g. Swiss German)."
        ),
    )
backend_label = st.session_state["backend_label"]
backend_key = next(
    k for k, v in service.TRANSLATION_BACKENDS.items() if v == backend_label
)

with col_src:
    st.selectbox("Source language", lang_names, key="src_lang")

with col_swap:
    st.write("")
    st.write("")
    st.button(
        "⇄",
        help="Swap source and target languages (also swaps the text panels).",
        on_click=_swap_languages,
    )

with col_tgt:
    st.selectbox("Target language", lang_names, key="tgt_lang")

src_name = st.session_state["src_lang"]
tgt_name = st.session_state["tgt_lang"]
src_code = TRANSLATE_LANGUAGE_MAPPING[src_name]
tgt_code = TRANSLATE_LANGUAGE_MAPPING[tgt_name]

# Backend-specific options.
ollama_model = None
if backend_key == "ollama":
    try:
        from textlab.common.gpu_manager import get_gpu_name
        from textlab.common.model_config import get_available_models
        from textlab.common.ollama import check_ollama_server

        if check_ollama_server():
            models = get_available_models(get_gpu_name())
            if models:
                ollama_model = st.selectbox(
                    "LLM model (Ollama)", models, index=0
                )
            else:
                st.warning("No Ollama models are available on this GPU.")
        else:
            st.error("Ollama server is not reachable.")
    except Exception as exc:
        st.error(f"Could not query Ollama: {exc}")

# Formality control — only shown for backends that actually honour it.
formality = "default"
if backend_key in service.FORMALITY_CAPABLE_BACKENDS:
    formality = st.radio(
        "Formality",
        service.FORMALITY_CHOICES,
        key="formality",
        horizontal=True,
        help=(
            "Steer the LLM's register. 'Default' lets the model decide "
            "based on the source text."
        ),
    )

opus_mt_supported = service.supports_pair(backend_key, src_code, tgt_code)

# ---------------------------------------------------------------------------
# Glossary editor (shared between Text and Document tabs)
# ---------------------------------------------------------------------------


def _current_glossary() -> dict[str, str]:
    """Return the non-empty glossary rows as an ordered dict."""
    out: dict[str, str] = {}
    for row in st.session_state["glossary_rows"]:
        src = (row.get("source") or "").strip()
        tgt = (row.get("target") or "").strip()
        if src and tgt:
            out[src] = tgt
    return out


with st.expander(
    "Glossary / term lock  "
    f"({len(_current_glossary())} active term"
    f"{'s' if len(_current_glossary()) != 1 else ''})",
    expanded=False,
):
    st.markdown(
        "Force specific translations for domain-specific terms, proper "
        "nouns, or product names. Longer terms are matched first so "
        "`University of Bern` wins over `Bern`."
    )
    edited = st.data_editor(
        st.session_state["glossary_rows"],
        num_rows="dynamic",
        column_config={
            "source": st.column_config.TextColumn(
                f"Source ({src_name})",
                help="Word or phrase in the source text.",
            ),
            "target": st.column_config.TextColumn(
                f"Target ({tgt_name})",
                help="Exact translation to force into the output.",
            ),
        },
        use_container_width=True,
        key="glossary_editor",
    )
    st.session_state["glossary_rows"] = list(edited)[:GLOSSARY_MAX_ROWS]

    st.checkbox(
        "Case-sensitive matching",
        key="glossary_case_sensitive",
        help=(
            "When off (default), 'Bern' also matches 'BERN' or 'bern'. "
            "When on, only exact-case matches are locked."
        ),
    )

options = service.TranslationOptions(
    backend=backend_key,
    source_code=src_code,
    source_name=src_name,
    target_code=tgt_code,
    target_name=tgt_name,
    ollama_model=ollama_model,
    formality=formality,
    glossary=_current_glossary(),
    glossary_case_sensitive=st.session_state["glossary_case_sensitive"],
)

# What needs to be resident in VRAM for the Text tab, given the current
# controls above. Compared against ``st.session_state["loaded_signature"]``
# to know whether the user still needs to hit "Load model".
current_load_sig = service.load_signature(options)
model_is_loaded = st.session_state[
    "loaded_signature"
] == current_load_sig and service.backend_ready(options)

st.divider()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _count_words(text: str) -> int:
    return len(text.split()) if text and text.strip() else 0


# ---------------------------------------------------------------------------
# Workflow tabs
# ---------------------------------------------------------------------------
def _render_doc_results(results: service.DocumentResults) -> None:
    file_outputs = results.file_outputs
    errors = results.errors
    pdf_reports = results.pdf_reports
    outputs = results.outputs

    elapsed = int(results.elapsed)
    details = [f"Finished in {elapsed // 60}:{elapsed % 60:02d}"]
    details += [
        f"{os.path.basename(name)}: {language}"
        for name, language in results.languages.items()
    ]
    st.caption(" · ".join(details))

    blocked_count = sum(len(result.blocked) for _, result in pdf_reports)
    if pdf_reports:
        with st.expander(
            "PDF validation reports", expanded=bool(blocked_count)
        ):
            for name, result in pdf_reports:
                st.write(name)
                for issue in result.blocked:
                    st.warning(
                        f"{issue['output']} could not be created. "
                        + issue.get("message", issue["reason"])
                    )
                for warning in result.warnings:
                    st.caption(warning)

    # A fully blocked PDF is already explained in its validation report.
    reported = {name for name, _ in pdf_reports}
    shown_errors = [item for item in errors if item[0] not in reported]
    if shown_errors:
        with st.expander(
            f"{len(shown_errors)} file(s) failed",
            expanded=True,
        ):
            for name, msg in shown_errors:
                st.error(f"**{name}** — {msg}")

    if not outputs:
        st.error("No files were translated successfully.")
        return
    if len(outputs) == 1:
        tname, tbytes = outputs[0]
        st.success(f"Translated → {tname}")
        st.download_button(
            f"Download {tname}",
            data=tbytes,
            file_name=tname,
            mime=service.mime_type(tname),
            on_click="ignore",
            key="doc_download_single",
        )
        return
    summary = (
        f"{len(file_outputs)}/{results.total} files produced "
        "downloadable outputs."
    )
    if blocked_count:
        summary += f" {blocked_count} PDF output(s) could not be created."
    if errors or blocked_count:
        st.warning(summary)
    else:
        st.success(summary)
    st.download_button(
        "Download translated ZIP",
        data=service.outputs_zip(file_outputs, errors),
        file_name=f"translated_{results.target_code}.zip",
        mime="application/zip",
        on_click="ignore",
        key="doc_download_zip",
    )


text_tab, doc_tab = st.tabs(["Text", "Document"])


# ===========================================================================
# TEXT WORKFLOW
# ===========================================================================
with text_tab:
    st.markdown(
        "Paste text on the left and press **Translate** "
        "to see the result on the right. "
        "Markdown links, inline code, LaTeX, HTML tags, URLs, "
        "and placeholders are "
        "automatically shielded so the model can't corrupt them."
    )

    left, right = st.columns(2, gap="medium")

    with left:
        st.markdown(f"**Source — {src_name}**")
        # `key='source_text'` makes st.session_state['source_text'] the
        # single source of truth. The swap callback writes to that key and
        # the widget re-reads it on the next frame.
        st.text_area(
            "source",
            key="source_text",
            height=380,
            label_visibility="collapsed",
            placeholder=f"Enter {src_name} text here…",
        )

        source_value = st.session_state["source_text"]
        char_count = len(source_value)
        word_count = _count_words(source_value)
        over_cap = char_count > TEXT_SOFT_CAP
        cap_style = (
            "color:#c0392b;font-weight:600;" if over_cap else "color:#666;"
        )
        st.markdown(
            f"<div style='{cap_style}font-size:13px;'>"
            f"{char_count:,} / {TEXT_SOFT_CAP:,} characters &nbsp;·&nbsp; "
            f"{word_count:,} words"
            f"{' &nbsp;·&nbsp; exceeds soft limit' if over_cap else ''}"
            "</div>",
            unsafe_allow_html=True,
        )

        # Detect controls.
        det_col1, det_col2 = st.columns([1, 3])
        with det_col1:
            do_detect = st.button(
                "Detect",
                help="Auto-detect the source language of the pasted text.",
                disabled=not source_value.strip(),
                key="detect_btn",
            )
        with det_col2:
            det = st.session_state["detection"]
            if det is not None:
                if det.display_name and det.flores_code:
                    pct = int(round(det.confidence * 100))
                    badge_color = (
                        "#2e7d32"
                        if det.confidence >= 0.85
                        else "#f9a825"
                        if det.confidence >= 0.60
                        else "#c62828"
                    )
                    st.markdown(
                        f"<div style='display:inline-block;padding:4px 10px;"
                        f"border-radius:12px;background:{badge_color};"
                        f"color:white;font-size:13px;'>"
                        f"Detected: {det.display_name} · {pct}%"
                        "</div>",
                        unsafe_allow_html=True,
                    )
                    if det.display_name != src_name:
                        st.button(
                            f"Use {det.display_name} as source",
                            key="use_detected_lang",
                            on_click=_use_detected_source,
                        )
                else:
                    st.markdown(
                        f"<div style='color:#666;font-size:13px;'>"
                        f"Detected: <code>{det.iso639_1}</code> "
                        "(outside NLLB dropdown — pick manually)"
                        "</div>",
                        unsafe_allow_html=True,
                    )

        if do_detect:
            with st.spinner("Detecting language…"):
                st.session_state["detection"] = service.detect_language(
                    source_value
                )
            st.rerun()

    with right:
        st.markdown(f"**Target — {tgt_name}**")
        # Read-only, selectable/copyable target pane. We use a styled div
        # instead of `st.text_area(disabled=True)` because HTML `disabled`
        # blocks text selection in all browsers — and CSS `user-select`
        # cannot override that. The div mimics text_area styling closely.
        target_text_html = html.escape(st.session_state["target_text"])
        if st.session_state["target_text"]:
            st.markdown(
                f"""
                <div style="
                    border: 1px solid rgba(49, 51, 63, 0.2);
                    border-radius: 0.5rem;
                    padding: 12px 16px;
                    height: 380px;
                    overflow-y: auto;
                    background-color: rgba(240, 242, 246, 0.5);
                    white-space: pre-wrap;
                    word-wrap: break-word;
                    font-family: 'Source Sans Pro', system-ui, sans-serif;
                    color: rgba(49, 51, 63, 1);
                    font-size: 14px;
                    line-height: 1.5;
                    user-select: text;
                    -webkit-user-select: text;
                    -moz-user-select: text;
                ">{target_text_html}</div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div style="
                    border: 1px solid rgba(49, 51, 63, 0.2);
                    border-radius: 0.5rem;
                    padding: 12px 16px;
                    height: 380px;
                    background-color: rgba(240, 242, 246, 0.5);
                    color: rgba(49, 51, 63, 0.4);
                    font-size: 14px;
                    font-style: italic;
                ">Translation will appear here.</div>
                """,
                unsafe_allow_html=True,
            )

        tgt_char_count = len(st.session_state["target_text"])
        tgt_word_count = _count_words(st.session_state["target_text"])
        st.markdown(
            f"<div style='color:#666;font-size:13px;'>"
            f"{tgt_char_count:,} characters &nbsp;·&nbsp; "
            f"{tgt_word_count:,} words &nbsp;·&nbsp; "
            f"Translated by <b>{backend_label}</b>"
            "</div>",
            unsafe_allow_html=True,
        )

        if st.session_state["target_text"]:
            escaped = (
                st.session_state["target_text"]
                .replace("\\", "\\\\")
                .replace("`", "\\`")
                .replace("$", "\\$")
            )
            components.html(
                f"""
                <button id="tl-copy-btn"
                    style="padding:6px 14px;border-radius:6px;
                           border:1px solid #888;background:#f6f6f6;
                           cursor:pointer;font-size:14px;">
                    Copy translation
                </button>
                <span id="tl-copy-msg"
                    style="margin-left:10px;color:#0a0;font-size:13px;">
                </span>
                <script>
                    const btn = document.getElementById('tl-copy-btn');
                    const msg = document.getElementById('tl-copy-msg');
                    btn.addEventListener('click', async () => {{
                        try {{
                            await navigator.clipboard.writeText(`{escaped}`);
                            msg.textContent = 'Copied!';
                            setTimeout(() => msg.textContent = '', 1500);
                        }} catch (e) {{
                            msg.style.color = '#c00';
                            msg.textContent = 'Copy failed';
                        }}
                    }});
                </script>
                """,
                height=48,
            )

    # -------------------------------------------------------------------
    # Load-model gate (Text tab only)
    #
    # The first inference for any given (backend, model) pair pays the
    # cost of a HF download (if not cached) plus moving weights to VRAM —
    # anywhere from a few seconds to 1-2 min on a fresh Slurm allocation.
    # We surface that cost explicitly so users know exactly when they're
    # paying it. The Translate button stays disabled until the currently-
    # selected backend/model matches what's already loaded.
    # -------------------------------------------------------------------
    if src_code == tgt_code:
        st.warning("Source and target languages are the same.")
    elif not opus_mt_supported:
        st.error(
            "OPUS-MT does not support direct translation between "
            f"{src_name} and {tgt_name}."
        )
    elif model_is_loaded:
        st.success(
            f"**{backend_label}** is ready on the current compute device."
        )
    else:
        load_col, msg_col = st.columns([1, 3])
        with load_col:
            do_load = st.button(
                "Load model",
                type="primary",
                disabled=(not opus_mt_supported or src_code == tgt_code),
                key="load_model_btn",
            )
        with msg_col:
            if st.session_state["loaded_signature"] is None:
                st.info(
                    "Text translation runs on this session's compute "
                    "device. Click **Load model** to warm up "
                    f"**{backend_label}** before translating. First-time "
                    "downloads can take 1-2 minutes; subsequent loads hit "
                    "the shared cache and are seconds fast."
                )
            else:
                st.info(
                    "You changed the backend or the language pair. Click "
                    f"**Load model** to warm up **{backend_label}** before "
                    "translating again."
                )

        if do_load:
            free_gpu_for(
                gpu_manager.TRANSLATION,
                ollama_model=ollama_model,
            )
            with st.spinner(
                f"Loading {backend_label} onto the compute device… "
                "(first time can take 1-2 min)"
            ):
                try:
                    service.load_backend(options)
                    st.session_state["loaded_signature"] = current_load_sig
                    st.session_state["load_error"] = None
                    st.session_state["load_traceback"] = None
                except Exception as exc:
                    import traceback as _tb

                    st.session_state["load_error"] = (
                        f"Model load failed: {exc}"
                    )
                    st.session_state["load_traceback"] = _tb.format_exc()
            st.rerun()

    # Persisted load error, if any.
    if st.session_state["load_error"]:
        st.error(st.session_state["load_error"])
        if st.session_state["load_traceback"]:
            with st.expander("Show load traceback"):
                st.code(st.session_state["load_traceback"])

    do_translate = st.button(
        "Translate →",
        type="primary",
        disabled=(
            not st.session_state["source_text"].strip()
            or not model_is_loaded
            or not opus_mt_supported
            or src_code == tgt_code
        ),
        key="translate_btn",
        help=(
            "Source and target languages are the same."
            if src_code == tgt_code
            else "OPUS-MT pair unsupported."
            if not opus_mt_supported
            else None
            if model_is_loaded
            else "Load the model first (button above)."
        ),
    )

    # Show a persisted error from the previous translation attempt — must
    # be rendered from session_state because st.rerun() below discards any
    # inline st.error() emitted during the run that raised.
    if st.session_state["translate_error"]:
        st.error(st.session_state["translate_error"])
        if st.session_state["translate_traceback"]:
            with st.expander("Show traceback"):
                st.code(st.session_state["translate_traceback"])

    for notice in st.session_state["translation_notices"]:
        st.info(notice)

    if do_translate:
        free_gpu_for(
            gpu_manager.TRANSLATION,
            ollama_model=ollama_model,
        )
        st.session_state["translation_notices"] = []
        progress = st.progress(0.0, text="Translating…")
        retry_status = st.empty()

        def _text_retry_notice(message: str) -> None:
            notices = st.session_state["translation_notices"]
            if message not in notices:
                notices.append(message)
            retry_status.info(message)

        try:
            translated = service.translate_text(
                st.session_state["source_text"],
                options,
                on_progress=lambda update: progress.progress(
                    update.fraction,
                    text=update.message,
                ),
                on_notice=_text_retry_notice,
            )
            st.session_state["target_text"] = translated
            # Clear any previous error on a successful run.
            st.session_state["translate_error"] = None
            st.session_state["translate_traceback"] = None
        except (
            service.TranslationLimitError,
            service.ProtectedContentError,
        ) as exc:
            st.session_state["target_text"] = ""
            st.session_state["translate_error"] = service.describe_error(
                exc,
                backend_key,
            )
            st.session_state["translate_traceback"] = None
        except Exception as exc:
            import traceback as _tb

            st.session_state["target_text"] = ""
            st.session_state["translate_error"] = f"Translation failed: {exc}"
            st.session_state["translate_traceback"] = _tb.format_exc()
        finally:
            progress.empty()
        st.rerun()


# ===========================================================================
# DOCUMENT WORKFLOW
# ===========================================================================
with doc_tab:
    st.markdown(
        "Drop one or more documents below. "
        "PDF pages are checked individually. "
        "Markdown and reconstructed PDF are validated separately: an unsafe "
        "or overflowing output is blocked without discarding a valid sibling. "
        "PDF downloads include a validation report; reconstruction is "
        "best-effort, not a guarantee of identical formatting."
    )
    st.caption(
        "Supported: .md · .txt · .srt · .vtt · .pdf · .docx · .xlsx · .pptx "
        "— or a .zip of any of the above.  "
        "Upload multiple files to translate them as a batch."
    )

    docs = st.file_uploader(
        "Drop files here (or click to browse)",
        type=["md", "txt", "srt", "vtt", "pdf", "docx", "xlsx", "pptx", "zip"],
        accept_multiple_files=True,
        key="doc_uploader",
    )

    _gpu = service.detect_gpu_profile()
    if _gpu.tier == "cpu":
        st.caption(
            "No GPU detected — translation will run on CPU (slow). "
            "Scanned-PDF OCR is unavailable."
        )
    else:
        st.caption(
            f"Allocated GPU: **{_gpu.name}** "
            f"({_gpu.vram_mb // 1000} GB). Translation batches adapt to "
            "available memory. OCR runs before translation with translation "
            "weights released; unsupported or failed OCR blocks the affected "
            "deliverable instead of skipping pages."
        )

    has_pdf = any(
        d.name.lower().endswith((".pdf", ".zip")) for d in docs or []
    )
    pdf_output_labels = {
        "markdown": "Markdown (text, tables and figures; best for reading)",
        "pdf": "PDF (same layout as the original)",
    }
    pdf_outputs = ("markdown", "pdf")
    math_ocr = False
    if has_pdf:
        pdf_outputs = tuple(
            st.multiselect(
                "PDF outputs",
                list(pdf_output_labels),
                default=list(pdf_output_labels),
                format_func=pdf_output_labels.__getitem__,
                key="pdf_outputs",
                help="Choose only what you need: each output is extracted "
                "separately. Text shared by both is translated once.",
            )
        )
        if "markdown" in pdf_outputs:
            math_ocr = st.checkbox(
                "Convert equations to LaTeX in the Markdown (slower)",
                key="pdf_math_ocr",
                help="Runs OCR on pages with equations. Otherwise equations "
                "are kept as they appear in the PDF, untranslated.",
            )

    opt_detect, opt_review = st.columns(2)
    with opt_detect:
        detect_source = st.checkbox(
            "Detect each document's language",
            value=True,
            key="doc_detect_source",
            help=f"Uses {src_name} (selected above) when a document's "
            "language cannot be identified with confidence.",
        )
    with opt_review:
        make_review = st.checkbox(
            "Add side-by-side review file",
            value=True,
            key="doc_review",
            help="Source and translation paragraph by paragraph, as HTML "
            "and Word. The easiest way to check a translation.",
        )

    total_size = sum(d.size for d in docs) if docs else 0
    if total_size > 5_000_000:
        st.warning(
            "Large files detected. Translation may take several minutes."
        )

    if src_code == tgt_code and not detect_source:
        st.warning("Source and target languages are the same.")
    elif not opus_mt_supported and not detect_source:
        st.error(
            "OPUS-MT does not support direct translation between "
            f"{src_name} and {tgt_name}."
        )

    run_doc = st.button(
        "Translate document(s)",
        type="primary",
        disabled=(
            not docs
            or (
                not detect_source
                and (src_code == tgt_code or not opus_mt_supported)
            )
            or (has_pdf and not pdf_outputs)
        ),
        key="translate_doc_btn",
    )
    if st.session_state.pop("doc_cancelled", False):
        st.info("Translation cancelled. No files were produced.")

    if run_doc and docs:
        st.session_state["doc_results"] = None
        free_gpu_for(
            gpu_manager.TRANSLATION,
            ollama_model=ollama_model,
        )
        card = st.container(border=True)
        with card:
            title_ph = st.empty()
            stage_ph = st.empty()
            bar = st.progress(0.0)
            info_ph = st.empty()
            # Clicking reruns the page, which stops this run at its next
            # progress update (Streamlit interrupts the running script).
            st.button(
                "Cancel",
                key="cancel_doc_btn",
                on_click=lambda: st.session_state.update(doc_cancelled=True),
            )
        run_started = time.monotonic()

        def _show_progress(update) -> None:
            elapsed = int(time.monotonic() - run_started)
            stage_ph.markdown(
                f"**Stage:** {update.message}  \n"
                f"Elapsed {elapsed // 60}:{elapsed % 60:02d}"
            )
            if update.fraction is not None:
                bar.progress(update.fraction)

        source_label = "auto-detect" if detect_source else src_name
        info_ph.markdown(
            f"**{source_label} → {tgt_name}** · engine: `{backend_label}` · "
            f"glossary: {len(options.glossary)} term(s)"
        )

        unpacked = service.unpack_uploads((up.name, up.read()) for up in docs)
        for entry in unpacked.skipped:
            st.warning(f"Skipped {entry}: unsupported file type inside ZIP.")
        for name in unpacked.invalid:
            st.error(f"{name}: not a valid ZIP archive.")

        if not unpacked.files:
            st.warning("No translatable files found in the upload.")
            st.stop()

        title_ph.markdown(
            f"### Translating {len(unpacked.files)} file"
            f"{'s' if len(unpacked.files) != 1 else ''}"
        )

        # Kept in session state so downloads (and any other rerun) do not
        # wipe the results; replaced by the next translation run.
        st.session_state["doc_results"] = service.translate_documents(
            unpacked.files,
            options,
            service.DocumentOptions(
                pdf_outputs=pdf_outputs,
                math_ocr=math_ocr,
                detect_source=detect_source,
                review=make_review,
            ),
            on_progress=_show_progress,
        )

    if st.session_state.get("doc_results"):
        _render_doc_results(st.session_state["doc_results"])
