"""
Translate page for Text Lab.

Two clearly separated workflows:

* **Text** — DeepL-style split-screen editor with source-language
  auto-detection, character/word counters, glossary/term-lock,
  language-swap, and formality control (LLM backend).
* **Document** — drag-and-drop upload for .md / .txt / .srt / .vtt /
  .pdf / .docx / .xlsx / .pptx files (or ZIP of any). The tool parses,
  translates, and reconstructs the file in its original format with
  structural markup preserved. Supports multi-file batch upload.

All translation is routed through :func:`core.translation.shielded_translate`
so Markdown links, LaTeX equations, inline code, HTML tags, URLs, and
placeholders survive intact. Glossary terms are enforced via the same
sentinel mechanism.

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
import io
import os
import sys
import zipfile

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
favicon_path = os.path.join(src_dir, "assets", "text_lab_logo.png")
favicon = Image.open(favicon_path)

st.set_page_config(page_title="Translate", page_icon=favicon, layout="wide")

if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from auth import check_token
from language_mappings import TRANSLATE_LANGUAGE_MAPPING
from core.translation import (
    FORMALITY_CAPABLE_BACKENDS,
    FORMALITY_CHOICES,
    TRANSLATION_BACKENDS,
    detect_language,
    make_translate_fn,
    shielded_translate,
    translate_docx,
    translate_markdown,
    translate_pdf,
    translate_pptx,
    translate_xlsx,
)


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
TEXT_SOFT_CAP = 5_000  # characters; UI warns above this
GLOSSARY_MAX_ROWS = 50


check_token()
st.title("Translate")
st.caption("Neural machine translation — all inference runs locally on UBELIX.")


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
    "backend_label": next(iter(TRANSLATION_BACKENDS.values())),
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
}
for _k, _v in _STATE_DEFAULTS.items():
    st.session_state.setdefault(_k, _v)


def _swap_languages() -> None:
    """
    Callback for the ⇄ button.

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
        list(TRANSLATION_BACKENDS.values()),
        key="backend_label",
        help=(
            "NLLB-200 covers 200 languages. MADLAD-400 is strong on "
            "low-resource languages. OPUS-MT is small/fast per pair. "
            "The Ollama LLM option is best for dialects (e.g. Swiss German)."
        ),
    )
backend_label = st.session_state["backend_label"]
backend_key = next(k for k, v in TRANSLATION_BACKENDS.items() if v == backend_label)

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
        from core.chat_engine import check_ollama_server, get_gpu_name
        from core.model_config import get_available_models

        if check_ollama_server():
            models = get_available_models(get_gpu_name())
            if models:
                ollama_model = st.selectbox("LLM model (Ollama)", models, index=0)
            else:
                st.warning("No Ollama models are available on this GPU.")
        else:
            st.error("Ollama server is not reachable.")
    except Exception as exc:
        st.error(f"Could not query Ollama: {exc}")

# Formality control — only shown for backends that actually honour it.
formality = "default"
if backend_key in FORMALITY_CAPABLE_BACKENDS:
    formality = st.radio(
        "Formality",
        FORMALITY_CHOICES,
        key="formality",
        horizontal=True,
        help=(
            "Steer the LLM's register. 'Default' lets the model decide "
            "based on the source text."
        ),
    )

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
    "📖 Glossary / term lock  "
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

glossary = _current_glossary()
glossary_case_sensitive = st.session_state["glossary_case_sensitive"]

st.divider()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _mime_for(name: str) -> str:
    lower = name.lower()
    if lower.endswith(".pdf"):
        return "application/pdf"
    if lower.endswith(".docx"):
        return (
            "application/vnd.openxmlformats-officedocument"
            ".wordprocessingml.document"
        )
    if lower.endswith(".pptx"):
        return (
            "application/vnd.openxmlformats-officedocument"
            ".presentationml.presentation"
        )
    if lower.endswith(".xlsx"):
        return (
            "application/vnd.openxmlformats-officedocument"
            ".spreadsheetml.sheet"
        )
    if lower.endswith(".md"):
        return "text/markdown"
    if lower.endswith(".zip"):
        return "application/zip"
    return "text/plain"


def _translate_one(name: str, data: bytes, tfn, progress_stage_cb, tgt_code_):
    """Dispatch by extension; return ``(translated_bytes, output_name)``."""
    base, ext = os.path.splitext(name)
    ext_lower = ext.lower()
    stem = os.path.basename(base)
    out_stem = f"{stem}.{tgt_code_}"

    if ext_lower == ".md":
        progress_stage_cb(0, 1, "parsing markdown")
        text = data.decode("utf-8", errors="replace")
        result = translate_markdown(
            text, tfn, progress_cb=progress_stage_cb, glossary=glossary,
        )
        progress_stage_cb(1, 1, "reconstructing markdown")
        return result.encode("utf-8"), f"{out_stem}.md"

    if ext_lower == ".pdf":
        progress_stage_cb(0, 1, "parsing pdf")
        result = translate_pdf(
            data, tfn, progress_cb=progress_stage_cb, glossary=glossary,
        )
        progress_stage_cb(1, 1, "reconstructing pdf")
        return result, f"{out_stem}.pdf"

    if ext_lower == ".docx":
        progress_stage_cb(0, 1, "parsing docx")
        result = translate_docx(
            data, tfn, progress_cb=progress_stage_cb, glossary=glossary,
        )
        progress_stage_cb(1, 1, "reconstructing docx")
        return result, f"{out_stem}.docx"

    if ext_lower == ".xlsx":
        progress_stage_cb(0, 1, "parsing xlsx")
        result = translate_xlsx(
            data, tfn, progress_cb=progress_stage_cb, glossary=glossary,
        )
        progress_stage_cb(1, 1, "reconstructing xlsx")
        return result, f"{out_stem}.xlsx"

    if ext_lower == ".pptx":
        progress_stage_cb(0, 1, "parsing pptx")
        result = translate_pptx(
            data, tfn, progress_cb=progress_stage_cb, glossary=glossary,
        )
        progress_stage_cb(1, 1, "reconstructing pptx")
        return result, f"{out_stem}.pptx"

    if ext_lower in (".txt", ".srt", ".vtt"):
        progress_stage_cb(0, 1, f"translating {ext_lower}")
        text = data.decode("utf-8", errors="replace")
        result = shielded_translate(
            text, tfn,
            glossary=glossary,
            glossary_case_sensitive=glossary_case_sensitive,
        )
        progress_stage_cb(1, 1, "done")
        return result.encode("utf-8"), f"{out_stem}{ext_lower}"

    raise ValueError(f"Unsupported file type: {ext_lower}")


def _count_words(text: str) -> int:
    return len(text.split()) if text and text.strip() else 0


# ---------------------------------------------------------------------------
# Workflow tabs
# ---------------------------------------------------------------------------
text_tab, doc_tab = st.tabs(["📝 Text", "📄 Document"])


# ===========================================================================
# TEXT WORKFLOW
# ===========================================================================
with text_tab:
    st.markdown(
        "Paste text on the left and press **Translate** to see the result on the right. "
        "Markdown links, inline code, LaTeX, HTML tags, URLs, and placeholders are "
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
        cap_style = "color:#c0392b;font-weight:600;" if over_cap else "color:#666;"
        st.markdown(
            f"<div style='{cap_style}font-size:13px;'>"
            f"{char_count:,} / {TEXT_SOFT_CAP:,} characters &nbsp;·&nbsp; "
            f"{word_count:,} words"
            f"{' &nbsp;·&nbsp; ⚠ exceeds soft limit' if over_cap else ''}"
            "</div>",
            unsafe_allow_html=True,
        )

        # Detect controls.
        det_col1, det_col2 = st.columns([1, 3])
        with det_col1:
            do_detect = st.button(
                "🔍 Detect",
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
                        "#2e7d32" if det.confidence >= 0.85
                        else "#f9a825" if det.confidence >= 0.60
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
                st.session_state["detection"] = detect_language(source_value)
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
                    style="padding:6px 14px;border-radius:6px;border:1px solid #888;
                           background:#f6f6f6;cursor:pointer;font-size:14px;">
                    📋 Copy translation
                </button>
                <span id="tl-copy-msg" style="margin-left:10px;color:#0a0;font-size:13px;"></span>
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

    do_translate = st.button(
        "Translate →",
        type="primary",
        disabled=not st.session_state["source_text"].strip(),
        key="translate_btn",
    )

    # Show a persisted error from the previous translation attempt — must
    # be rendered from session_state because st.rerun() below discards any
    # inline st.error() emitted during the run that raised.
    if st.session_state["translate_error"]:
        st.error(st.session_state["translate_error"])
        if st.session_state["translate_traceback"]:
            with st.expander("Show traceback"):
                st.code(st.session_state["translate_traceback"])

    if do_translate:
        progress = st.progress(0.0, text="Translating…")

        def _cb(done: int, total: int) -> None:
            if total > 0:
                progress.progress(
                    min(done / total, 1.0),
                    text=f"Translating chunk {done}/{total}",
                )

        tfn = make_translate_fn(
            src_lang=src_code, tgt_lang=tgt_code,
            backend=backend_key,
            ollama_model=ollama_model,
            src_lang_name=src_name, tgt_lang_name=tgt_name,
            formality=formality,
            progress_cb=_cb,
        )
        try:
            translated = shielded_translate(
                st.session_state["source_text"], tfn,
                glossary=glossary,
                glossary_case_sensitive=glossary_case_sensitive,
            )
            st.session_state["target_text"] = translated
            # Clear any previous error on a successful run.
            st.session_state["translate_error"] = None
            st.session_state["translate_traceback"] = None
        except Exception as exc:
            import traceback as _tb
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
        "Drop one or more documents below. The translation preserves the "
        "**original file format**: Markdown stays Markdown, DOCX stays DOCX, "
        "PDF stays PDF — with headings, lists, tables, images, links, code, "
        "and math intact."
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

    run_doc = st.button(
        "Translate document(s)",
        type="primary",
        disabled=not docs,
        key="translate_doc_btn",
    )

    if run_doc and docs:
        card = st.container(border=True)
        with card:
            title_ph = st.empty()
            stage_ph = st.empty()
            bar = st.progress(0.0)
            info_ph = st.empty()

        def _stage(stage: str) -> None:
            stage_ph.markdown(f"**Stage:** {stage}")

        def _prog_translate(done: int, total: int) -> None:
            if total > 0:
                bar.progress(min(done / total, 1.0))

        def _prog_stage(done: int, total: int, stage: str) -> None:
            _stage(stage)
            if total > 0:
                bar.progress(min(done / total, 1.0))

        tfn = make_translate_fn(
            src_lang=src_code, tgt_lang=tgt_code,
            backend=backend_key,
            ollama_model=ollama_model,
            src_lang_name=src_name, tgt_lang_name=tgt_name,
            formality=formality,
            progress_cb=_prog_translate,
        )

        info_ph.markdown(
            f"**{src_name} → {tgt_name}** · engine: `{backend_label}` · "
            f"glossary: {len(glossary)} term(s)"
        )

        flat_inputs: list[tuple[str, bytes]] = []
        for up in docs:
            name = up.name
            raw = up.read()
            if name.lower().endswith(".zip"):
                try:
                    with zipfile.ZipFile(io.BytesIO(raw), "r") as zin:
                        for entry in zin.namelist():
                            if entry.endswith("/"):
                                continue
                            flat_inputs.append((entry, zin.read(entry)))
                except zipfile.BadZipFile:
                    st.error(f"{name}: not a valid ZIP archive.")
                    continue
            else:
                flat_inputs.append((name, raw))

        if not flat_inputs:
            st.warning("No translatable files found in the upload.")
            st.stop()

        title_ph.markdown(
            f"### 📄 Translating {len(flat_inputs)} file"
            f"{'s' if len(flat_inputs) != 1 else ''}"
        )

        successes: list[tuple[str, bytes]] = []
        errors: list[tuple[str, str]] = []

        for i, (entry_name, entry_data) in enumerate(flat_inputs, start=1):
            _stage(f"[{i}/{len(flat_inputs)}] {entry_name}")
            bar.progress((i - 1) / len(flat_inputs))
            try:
                tbytes, tname = _translate_one(
                    entry_name, entry_data, tfn, _prog_stage, tgt_code,
                )
                successes.append((tname, tbytes))
            except Exception as exc:
                errors.append((entry_name, str(exc)))

        _stage("done")
        bar.progress(1.0)

        if errors:
            with st.expander(f"⚠ {len(errors)} file(s) failed", expanded=True):
                for name, msg in errors:
                    st.error(f"**{name}** — {msg}")

        if not successes:
            st.error("No files were translated successfully.")
        elif len(successes) == 1:
            tname, tbytes = successes[0]
            st.success(f"Translated → {tname}")
            st.download_button(
                f"⬇ Download {tname}",
                data=tbytes,
                file_name=tname,
                mime=_mime_for(tname),
            )
        else:
            out_zip = io.BytesIO()
            with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zout:
                for tname, tbytes in successes:
                    zout.writestr(tname, tbytes)
                for name, msg in errors:
                    zout.writestr(
                        f"{name}.ERROR.txt",
                        f"Failed to translate: {msg}".encode("utf-8"),
                    )
            st.success(
                f"Batch complete: {len(successes)}/{len(flat_inputs)} translated."
            )
            st.download_button(
                "⬇ Download translated ZIP",
                data=out_zip.getvalue(),
                file_name=f"translated_{tgt_code}.zip",
                mime="application/zip",
            )
