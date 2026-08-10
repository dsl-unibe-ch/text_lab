import os

# FORCE these to 1 to prevent OpenBLAS crashes with Paddle/PyTorch
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

os.environ["VLLM_GPU_MEMORY_UTILIZATION"] = "0.6"
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
os.environ.setdefault("PADDLE_PDX_CACHE_HOME", os.environ.get("PADDLEX_HOME", os.path.expanduser("~/.paddlex")))
os.environ.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "none")

import streamlit as st
import subprocess
import uuid
import pathlib
import shutil
import sys
import json
import base64
import cv2
import ollama
import io
import pandas as pd
from PIL import Image

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
favicon_path = os.path.join(src_dir, "assets", "text_lab_logo.png")

favicon = Image.open(favicon_path)

st.set_page_config(page_title="OCR", page_icon=favicon, layout="wide")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from auth import check_token
from core.ocr_engine import (
    make_json_serializable,
    compact_paddle_prediction,
    render_easyocr_preview,
    render_paddle_preview,
    extract_html_table,
    render_layout_preview,
    LAYOUT_TYPE_COLORS,
)
from core import auto_ocr, doc_ir, form_extract, searchable_pdf, survey_batch, vision_enrich

try:
    from language_mappings import EASYOCR_LANGUAGE_MAPPING, PADDLEOCR_LANGUAGE_MAPPING
except ImportError:
    EASYOCR_LANGUAGE_MAPPING = {"English": "en"}
    PADDLEOCR_LANGUAGE_MAPPING = {"English": "en"}

check_token()

st.title("📄 Document & Image OCR")

# --- 1. Get required paths from environment variables ---
HOST_HOME = os.environ.get("HOME")

if not HOST_HOME:
    st.error("**Configuration Error:** `HOME` environment variable is not set.")
    st.stop()

OCR_JOBS_BASE_DIR = pathlib.Path(HOST_HOME) / "ondemand_text_lab_ocr_jobs"
OLMOCR_GPU_MEMORY_UTILIZATION = os.environ.get("OLMOCR_GPU_MEMORY_UTILIZATION", "0.6")

AUTO_INPUT_TYPES = ["pdf", "png", "jpg", "jpeg", "bmp", "tiff", "tif"]

# Hides the survey/form controls and Responses tab while the extractor is being
# validated; the backend stays reachable via process_document(extract_survey=True).
SURVEY_EXTRACTION_UI_ENABLED = False


@st.cache_resource(show_spinner=False)
def get_easyocr_reader(lang_code="en"):
    import easyocr
    return easyocr.Reader([lang_code], gpu=True)

def run_paddleocr_backend(image_paths, lang_code="en"):
    backend_python = os.environ.get("PADDLE_BACKEND_PYTHON", "/opt/conda/envs/paddle_backend/bin/python")
    worker_path = pathlib.Path(src_dir) / "core" / "paddle_ocr_worker.py"
    env = os.environ.copy()
    env["PATH"] = f"{pathlib.Path(backend_python).parent}:{env.get('PATH', '')}"
    env["LD_LIBRARY_PATH"] = f"/opt/conda/envs/paddle_backend/lib:{env.get('LD_LIBRARY_PATH', '')}"
    env.setdefault("DISABLE_MODEL_SOURCE_CHECK", "True")
    env.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    cmd = [backend_python, str(worker_path), "--lang", lang_code, *[str(p) for p in image_paths]]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", env=env)
    if result.returncode != 0:
        raise RuntimeError(
            "PaddleOCR backend failed.\n"
            f"stdout:\n{result.stdout[-4000:]}\n\nstderr:\n{result.stderr[-4000:]}"
        )
    marker = "TEXTLAB_PADDLEOCR_RESULT_JSON="
    for line in reversed(result.stdout.splitlines()):
        if line.startswith(marker):
            return json.loads(line[len(marker):]).get("pages", [])
    raise RuntimeError(
        "PaddleOCR backend did not return JSON.\n"
        f"stdout:\n{result.stdout[-4000:]}\n\nstderr:\n{result.stderr[-4000:]}"
    )

def decode_png_b64(value):
    if not value:
        return None
    try:
        return base64.b64decode(value)
    except Exception:
        return None

def clear_results(reset_running=False):
    keys_to_clear = [
        # legacy engines
        "ocr_complete", "extracted_text", "json_content", "txt_name",
        "json_name", "ocr_error", "ocr_error_details", "ocr_preview_images",
        "ocr_preview_page", "ocr_preview_engine", "ocr_zip_bytes",
        "batch_ocr_complete", "batch_ocr_zip_bytes",
        # automatic pipeline
        "auto_complete", "auto_document", "auto_summary", "auto_downloads",
        "auto_error", "batch_auto_complete", "batch_auto_zip",
        # questionnaire batch: kept so the detected form can be reviewed and
        # the tables rebuilt without parsing everything again
        "survey_template", "survey_readings",
    ]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
    # Interactive survey-review widgets are keyed "rev_<group>_<row>..."; drop them
    # so a new document does not inherit the previous document's selections.
    for key in [k for k in st.session_state if isinstance(k, str) and k.startswith("rev_")]:
        del st.session_state[key]
    if reset_running:
        st.session_state.ocr_running = False

def _cleanup_job_dir(job_dir):
    """Aggressive, self-cleaning removal of a job workspace (privacy)."""
    if not job_dir.exists():
        return
    import time
    import stat
    time.sleep(1)

    def handle_remove_readonly(func, path, exc):
        try:
            os.chmod(path, stat.S_IWRITE)
            func(path)
        except Exception:
            pass

    try:
        shutil.rmtree(job_dir, onexc=handle_remove_readonly)
    except Exception:
        subprocess.run(["rm", "-rf", str(job_dir)], check=False)


if "ocr_running" not in st.session_state:
    st.session_state.ocr_running = False


# ==========================================
#      AUTOMATIC PIPELINE — RUNNERS
# ==========================================

def run_auto_single(
    uploaded_file,
    native_fast_lane=True,
    *,
    describe_images=False,
    extract_survey=False,
    searchable_pdf=False,
    ocr_lang="eng",
):
    clear_results(reset_running=False)
    st.session_state.ocr_running = True

    JOB_ID = str(uuid.uuid4())
    JOB_DIR = OCR_JOBS_BASE_DIR / JOB_ID
    INPUT_DIR = JOB_DIR / "input"
    WORKSPACE_DIR = JOB_DIR / "workspace"
    progress_bar = st.progress(0.0, text="Parsing document...")

    try:
        INPUT_DIR.mkdir(parents=True, exist_ok=True)
        WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)

        input_file_path = INPUT_DIR / uploaded_file.name
        with open(input_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        def _progress(frac, text):
            progress_bar.progress(frac, text=text)

        document = auto_ocr.process_document(
            input_file_path,
            WORKSPACE_DIR,
            native_fast_lane=native_fast_lane,
            progress=_progress,
            source_name=uploaded_file.name,
            describe_images=describe_images,
            extract_survey=extract_survey,
            searchable_pdf=searchable_pdf,
            ocr_lang=ocr_lang,
        )

        stem = pathlib.Path(uploaded_file.name).stem or "document"
        st.session_state.auto_document = document
        st.session_state.auto_summary = auto_ocr.document_summary(document)
        st.session_state.auto_downloads = {
            "stem": stem,
            "markdown_zip": doc_ir.build_markdown_zip(document, stem),
            "text": doc_ir.to_text(document).encode("utf-8"),
            "docx": doc_ir.build_docx(document, stem),
            "searchable_pdf": document.searchable_pdf,
            "json": doc_ir.to_json(document).encode("utf-8"),
            "tables_zip": doc_ir.build_tables_csv_zip(document),
            "responses_csv": doc_ir.build_form_responses_csv(document),
            "full": doc_ir.build_full_bundle(document, stem),
        }
        st.session_state.auto_complete = True

    except Exception as e:
        st.session_state.auto_error = f"Automatic OCR failed: {e}"
        st.exception(e)
    finally:
        progress_bar.empty()
        st.session_state.ocr_running = False
        _cleanup_job_dir(JOB_DIR)


def run_auto_batch(
    batch_zip,
    native_fast_lane=True,
    *,
    describe_images=False,
    extract_survey=False,
    same_template=False,
    survey_batch_mode=False,
    searchable_pdf=False,
    ocr_lang="eng",
):
    import zipfile

    clear_results(reset_running=False)
    st.session_state.ocr_running = True

    JOB_ID = str(uuid.uuid4())
    JOB_DIR = OCR_JOBS_BASE_DIR / JOB_ID
    INPUT_DIR = JOB_DIR / "input"
    WORKSPACE_DIR = JOB_DIR / "workspace"
    RESULTS_DIR = WORKSPACE_DIR / "results"

    shared_vision_client = (
        vision_enrich.OllamaVisionClient()
        if describe_images or extract_survey
        else None
    )
    same_layout_template = (
        form_extract.SameLayoutTemplate()
        if extract_survey and same_template
        else None
    )

    try:
        INPUT_DIR.mkdir(parents=True, exist_ok=True)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(batch_zip, "r") as z:
            z.extractall(INPUT_DIR)

        valid_exts = {"." + e for e in AUTO_INPUT_TYPES}
        valid_files = []
        for root, dirs, files in os.walk(INPUT_DIR):
            for file in files:
                if file.startswith("._"):
                    continue
                file_path = pathlib.Path(root) / file
                if file_path.suffix.lower() in valid_exts:
                    valid_files.append(file_path)
        valid_files.sort(key=lambda path: str(path.relative_to(INPUT_DIR)).casefold())

        if not valid_files:
            raise RuntimeError("No valid documents or images found in the ZIP.")

        progress_bar = st.progress(0.0)
        status_text = st.empty()

        # One questionnaire template for the whole batch, learned before any
        # file is parsed: the blank is the median of the copies, so it needs
        # them all up front.
        template, blanks, readings = None, None, []
        if survey_batch_mode:
            def _template_progress(fraction, text):
                progress_bar.progress(min(0.99, max(0.0, fraction)))
                status_text.markdown(f"**Questionnaire layout** — {text}")

            _template_progress(0.0, f"reading {len(valid_files)} file(s)...")
            template, blanks = survey_batch.prepare_template(
                valid_files, label=True, progress=_template_progress
            )
            status_text.markdown(
                f"**Questionnaire layout** — {template.control_count} response "
                f"controls in {len(template.rules)} answers"
            )
            small_batch = template.provenance.get("small_batch_warning")
            if small_batch:
                st.warning(small_batch)

        n_files = len(valid_files)
        batch_provenance = []
        for idx, file_path in enumerate(valid_files):
            rel_path = file_path.relative_to(INPUT_DIR)

            def _file_progress(frac, text, _idx=idx, _rel=rel_path):
                # Each file's own page progress, folded into the batch bar.
                progress_bar.progress(
                    min(1.0, (_idx + max(0.0, min(1.0, frac))) / n_files)
                )
                status_text.markdown(
                    f"**File {_idx + 1} of {n_files}** · `{_rel}` — {text}"
                )

            _file_progress(0.0, "starting...")

            file_output_dir = RESULTS_DIR / rel_path.parent / file_path.stem
            file_output_dir.mkdir(parents=True, exist_ok=True)
            per_file_ws = WORKSPACE_DIR / "tmp" / f"job_{idx}"

            document = auto_ocr.process_document(
                file_path, per_file_ws,
                native_fast_lane=native_fast_lane,
                progress=_file_progress,
                source_name=file_path.name,
                describe_images=describe_images,
                extract_survey=extract_survey,
                searchable_pdf=searchable_pdf,
                ocr_lang=ocr_lang,
                vision_client=shared_vision_client,
                same_layout_template=same_layout_template,
            )

            # Same writer as the single-document downloads; the provenance
            # summary is collected instead, and written once at the root.
            doc_ir.write_document_outputs(
                document, file_output_dir, "document", provenance=False
            )
            batch_provenance.append(doc_ir.model_provenance(document))

            if template is not None:
                # Same file, second pass: the text extraction above is
                # unchanged, this adds the questionnaire answers.
                reading = survey_batch.read_document(file_path, template)
                readings.append(reading)
                survey_batch.answers_for_document(reading, template).to_csv(
                    file_output_dir / "survey_answers.csv", index=False
                )

            shutil.rmtree(per_file_ws, ignore_errors=True)
            progress_bar.progress((idx + 1) / n_files)

        if template is not None:
            status_text.markdown("**Collecting questionnaire responses...**")
            survey_batch.write_batch_outputs(
                readings, template, RESULTS_DIR / "survey"
            )

        status_text.markdown(f"**{n_files} file(s) parsed** — zipping results...")

        # One summary at the root, the union over files: a batch can mix lanes.
        merged_provenance = doc_ir.provenance_to_text(
            doc_ir.merge_provenance(batch_provenance)
        )
        if merged_provenance:
            (RESULTS_DIR / "models_used.txt").write_text(
                merged_provenance + "\n", encoding="utf-8"
            )

        out_zip_buffer = io.BytesIO()
        with zipfile.ZipFile(out_zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, dirs, files in os.walk(RESULTS_DIR):
                for file in files:
                    fp = pathlib.Path(root) / file
                    zf.write(fp, fp.relative_to(RESULTS_DIR))

        st.session_state.batch_auto_zip = out_zip_buffer.getvalue()
        st.session_state.batch_auto_complete = True
        if template is not None:
            st.session_state.survey_template = template
            st.session_state.survey_readings = readings

    except Exception as e:
        st.session_state.auto_error = f"Batch automatic OCR failed: {e}"
        st.exception(e)
    finally:
        if shared_vision_client is not None:
            # Not evicted: it expires after keep_alive, and the next OCR worker
            # frees the card itself. See vision_enrich.free_gpu.
            shared_vision_client.close()
        st.session_state.ocr_running = False
        _cleanup_job_dir(JOB_DIR)


def _rebuild_survey_zip(zip_bytes, template, readings):
    """Swap the survey/ folder in an existing result ZIP for a fresh one.

    Dropping a control changes only how answers are grouped and exported, not
    what was read off the page, so the questionnaires do not have to be parsed
    again.
    """
    import tempfile
    import zipfile

    with tempfile.TemporaryDirectory() as tmp:
        out = pathlib.Path(tmp) / "survey"
        summary = survey_batch.write_batch_outputs(readings, template, out)
        replacements = {
            f"survey/{path.name}": path.read_bytes() for path in out.iterdir()
        }
        per_file = survey_batch.to_checkbox_table(readings, template)

    buffer = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as source:
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as target:
            for item in source.namelist():
                if item.startswith("survey/"):
                    continue
                if item.endswith("/survey_answers.csv"):
                    # Folder names are the file stems, so match on the stem
                    # rather than a prefix: "split_1_2" must not claim
                    # "split_1_20".
                    folder = pathlib.PurePosixPath(item).parent.name
                    rows = per_file[
                        per_file["document"].map(
                            lambda name: pathlib.PurePosixPath(str(name)).stem == folder
                        )
                    ]
                    target.writestr(item, rows.to_csv(index=False))
                    continue
                target.writestr(item, source.read(item))
            for name, data in replacements.items():
                target.writestr(name, data)
    return buffer.getvalue(), summary


def render_survey_review():
    """Let the user check the detected form and drop anything spurious."""
    template = st.session_state.get("survey_template")
    readings = st.session_state.get("survey_readings")
    if not template or not readings:
        return

    st.markdown("### 📋 The questionnaire TextLab detected")
    st.caption(
        f"{template.control_count} response controls in {len(template.rules)} "
        f"answers, learned from the batch itself. Check the outlines below: "
        "printed text can occasionally be mistaken for an empty checkbox."
    )
    overlays = survey_batch.template_overlays(template)
    if overlays:
        tabs = st.tabs([f"Page {i + 1}" for i in range(len(overlays))])
        for tab, (name, data) in zip(tabs, sorted(overlays.items())):
            with tab:
                st.image(data, caption=name, use_container_width=True)

    overview = survey_batch.answer_overview(readings, template)
    dead = overview[overview["never_marked"]]
    if len(dead):
        st.warning(
            f"{len(dead)} answer(s) nobody in this batch marked (shown as "
            "`never_marked` below). That is either a question they all skipped, "
            "or printed text mistaken for a control — the image above settles "
            "which."
        )
    st.dataframe(
        overview.drop(columns=["control_ids"]),
        use_container_width=True, hide_index=True,
    )

    labels = {row["answer"]: row["control_ids"] for _, row in overview.iterrows()}
    chosen = st.multiselect(
        "Remove answers that are not really on the form",
        options=list(labels),
        # Deliberately not pre-selected: an answer nobody marked is just as
        # likely a question the whole batch skipped as a false positive, and
        # only the image above settles it.
        key="survey_drop",
        help=(
            "Removes them from the response tables and renumbers the rest. "
            "The questionnaires are not parsed again."
        ),
    )
    if chosen and st.button("♻️ Rebuild the response tables", key="survey_rebuild"):
        ids = [i for name in chosen for i in labels[name].split(",") if i]
        removed = survey_batch.drop_controls(template, ids)
        zip_bytes, summary = _rebuild_survey_zip(
            st.session_state.batch_auto_zip, template, readings
        )
        st.session_state.batch_auto_zip = zip_bytes
        st.session_state.survey_template = template
        st.success(
            f"Removed {removed} control(s); {summary['controls']} remain in "
            f"{summary['answer_groups']} answers. Download again for the "
            "updated tables."
        )
        st.rerun()


# ==========================================
#      AUTOMATIC PIPELINE — RESULT TABS
# ==========================================

def _b64_bytes(b64):
    if not b64:
        return None
    try:
        return base64.b64decode(b64)
    except Exception:
        return None


def render_document_tab(document):
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
                html = region.content.get("html", "").strip()
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
                    st.caption(f"AI description: {region.visual_description.description}")
            elif rtype == doc_ir.CHECKBOX:
                state = (region.markup or {}).get("state", "uncertain")
                icon = {"checked": "☑", "unchecked": "☐", "uncertain": "❓"}.get(state, "❓")
                st.markdown(f"{icon} {region.text.strip()}".rstrip())
            else:
                text = region.text.strip()
                if text:
                    st.markdown(text)
        if len(document.pages) > 1:
            st.divider()


def render_tables_tab(document):
    tables = doc_ir.tables_to_dataframes(document)
    if not tables:
        st.info("No tables were detected in this document.")
        return
    for entry in tables:
        st.markdown(f"**Table — page {entry['page']}** (`{entry['region_id']}`)")
        st.dataframe(entry["dataframe"], use_container_width=True)
        st.download_button(
            "📥 Download this table (CSV)",
            entry["dataframe"].to_csv(index=False).encode("utf-8"),
            file_name=f"table_{entry['region_id']}.csv",
            mime="text/csv",
            key=f"tbl_csv_{entry['region_id']}",
        )
        st.divider()


def render_figures_tab(document):
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
                st.image(data, use_container_width=True,
                         caption=f"Page {page.page_number} · {region.type}")
            caption = region.text.strip()
            if caption:
                st.caption(f"Printed caption/text: {caption[:300]}")
            generated = region.visual_description
            if generated:
                st.markdown(generated.description)
                if generated.visible_text:
                    st.caption(f"Visible text: {generated.visible_text[:500]}")
                st.caption(f"AI description · {generated.source} · {generated.model}")


STATE_BADGE = {
    "checked": "✅ checked",
    "unchecked": "⬜ unchecked",
    "uncertain": "⚠️ uncertain",
}


def _group_is_multiselect(group):
    """True when a question's rows allow several answers (checkboxes vs one radio)."""
    return group.question_type == "multiple" or group.selection_rule == "zero_or_more"


REVIEW_REASON_LABELS = {
    "answer_geometry_disagreement": "answer conflicts with geometric ink evidence",
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
    """Render a whole matrix as one table and return row -> selected position."""
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
        for row, answer in zip(group.rows, edited["Answer"].tolist())
    }


def _render_form_review(document):
    """Summary-first response review with compact question-level editors."""
    form_groups = [(page, g) for page in document.pages for g in page.form_groups]
    if not form_groups:
        return
    n_review = sum(g.status == "needs_review" for _, g in form_groups)
    if n_review:
        st.warning(
            f"⚠️ {n_review} of {len(form_groups)} question(s) were flagged for review. "
            "Flagged questions are open below; accepted questions stay collapsed. "
            "Saving updates the downloads and retains the original model answer in JSON."
        )
    else:
        st.success(
            f"Extracted {len(form_groups)} question(s). Expand any row below to make a correction."
        )

    summary_records = [
        {
            "Question": group.question_text or group.id,
            "Answer": _group_answer_summary(group),
            "Status": "Needs review" if group.status == "needs_review" else "Accepted",
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
            flag = "⚠️ " if group.status == "needs_review" else ""
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
                layout = st.columns([3, 2]) if needs_review else [st.container()]
                with layout[0]:
                    st.caption(
                        f"`{group.question_type}` · `{group.selection_rule}` · "
                        f"page {page.page_number}"
                    )
                    if group.condition_text:
                        st.caption(f"↳ conditional: {group.condition_text}")
                    if group.review_reasons:
                        reasons = [
                            REVIEW_REASON_LABELS.get(reason, reason.replace("_", " "))
                            for reason in group.review_reasons
                        ]
                        st.caption(f"Review because: {'; '.join(reasons)}")
                    for warning in group.warnings:
                        st.caption(f"⚠️ {warning}")
                    if not group.rows or not any(row.options for row in group.rows):
                        st.info(
                            "Options for this question could not be extracted automatically. "
                            "This item remains flagged after saving because there is no safe "
                            "correction control."
                        )
                    elif _matrix_uses_shared_single_choice_options(group):
                        matrix_edits[group.id] = _render_matrix_editor(group)
                    else:
                        for row in group.rows:
                            key_base = f"rev_{group.id}_{row.id}"
                            labels = _option_labels(row)
                            if not row.options:
                                st.caption(f"_{row.label or 'row'}: no options detected_")
                                continue
                            if row.review_reasons:
                                reasons = [
                                    REVIEW_REASON_LABELS.get(
                                        reason, reason.replace("_", " ")
                                    )
                                    for reason in row.review_reasons
                                ]
                                st.caption(
                                    f"⚠️ {row.label or 'Answer'}: {'; '.join(reasons)}"
                                )
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
                                        for index, option in enumerate(row.options)
                                        if option.state == "selected"
                                    ),
                                    0,
                                )
                                st.radio(
                                    row.label or "Answer",
                                    options=list(range(len(row.options) + 1)),
                                    index=selected_index,
                                    format_func=lambda value, _labels=labels: (
                                        "— none —" if value == 0 else _labels[value - 1]
                                    ),
                                    key=key_base,
                                    horizontal=(len(row.options) <= 6),
                                )
                            for index, option in enumerate(row.options):
                                if option.associated_text:
                                    st.text_input(
                                        f"✍️ handwriting near “{labels[index]}”",
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
                                "The source crop is unavailable; this item cannot be "
                                "verified visually."
                            )
        submitted = st.form_submit_button("💾 Save corrections", type="primary")

    if submitted:
        _apply_form_corrections(document, matrix_edits)
        downloads = st.session_state.setdefault("auto_downloads", {})
        stem = downloads.get("stem", "document")
        downloads["json"] = doc_ir.to_json(document).encode("utf-8")
        downloads["responses_csv"] = doc_ir.build_form_responses_csv(document)
        downloads["full"] = doc_ir.build_full_bundle(document, stem)
        st.session_state.auto_summary = auto_ocr.document_summary(document)
        st.success("✅ Corrections saved. The downloads above now reflect your edits.")


def _apply_form_corrections(document, matrix_edits=None):
    """Write reviewer selections back into the document, preserving model originals."""
    matrix_edits = matrix_edits or {}
    for page in document.pages:
        for group in page.form_groups:
            multi = _group_is_multiselect(group)
            if "model_answer" not in group.provenance:
                group.provenance["model_answer"] = [
                    {
                        "row": row.label or row.id,
                        "selected": [o.label for o in row.options if o.state == "selected"],
                        "states": {o.id: o.state for o in row.options},
                    }
                    for row in group.rows
                ]
                group.provenance["pre_review_reasons"] = {
                    "group": list(group.review_reasons),
                    "rows": {
                        row.id: list(row.review_reasons)
                        for row in group.rows
                    },
                }
            reviewed_rows = 0
            unresolved_rows = 0
            for row in group.rows:
                key_base = f"rev_{group.id}_{row.id}"
                if not row.options:
                    unresolved_rows += 1
                    continue
                previous_states = {option.id: option.state for option in row.options}
                if group.id in matrix_edits:
                    selected_position = matrix_edits[group.id].get(row.id, 0)
                    for index, option in enumerate(row.options):
                        option.state = (
                            "selected" if selected_position == index + 1 else "unselected"
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
        1 for _, r in checkbox_marks if (r.markup or {}).get("state") == "uncertain"
    ) + sum((r.markup or {}).get("n_uncertain", 0) for _, r in glyph_regions)
    with st.expander("Geometric mark detector (legacy) — details in the JSON / Layout preview"):
        st.caption(
            f"{n_total} geometric mark(s) across "
            f"{len(glyph_regions) + len(checkbox_marks)} region(s)"
            + (f"; {n_uncertain} uncertain" if n_uncertain else "")
            + "."
        )


def render_markup_tab(document):
    form_groups = [
        (page, group)
        for page in document.pages
        for group in page.form_groups
    ]
    checkbox_marks = [
        (page, region)
        for page, region in document.all_regions()
        if region.type == doc_ir.CHECKBOX and region.markup
    ]
    glyph_regions = [
        (page, region)
        for page, region in document.all_regions()
        if region.type != doc_ir.CHECKBOX and (region.markup or {}).get("kind") == "glyph-marks"
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


def render_layout_tab(document):
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
        st.info("No page image is available for this page (native text-only page).")
        return

    regions = [{"bbox": r.bbox, "type": r.type} for r in page.regions if r.bbox]
    preview = render_layout_preview(img_bytes, regions)
    if preview:
        st.image(preview, use_container_width=True)
    else:
        st.image(img_bytes, use_container_width=True)

    present_types = sorted({r.type for r in page.regions})
    if present_types:
        legend = "  ".join(
            f"<span style='color:rgb({LAYOUT_TYPE_COLORS.get(t, (90,90,90))[2]},"
            f"{LAYOUT_TYPE_COLORS.get(t, (90,90,90))[1]},"
            f"{LAYOUT_TYPE_COLORS.get(t, (90,90,90))[0]})'>&#9632; {t}</span>"
            for t in present_types
        )
        st.markdown(f"**Legend:** {legend}", unsafe_allow_html=True)


def render_auto_results():
    document = st.session_state.get("auto_document")
    if document is None:
        return
    summary = st.session_state.get("auto_summary", {})
    downloads = st.session_state.get("auto_downloads", {})

    n_pages = summary.get("n_pages", 0)
    counts = summary.get("region_counts", {})
    chips = " · ".join(f"{k}: {v}" for k, v in counts.items()) or "no regions"
    st.success(
        f"🎉 Parsed {n_pages} page(s) · "
        f"{', '.join(summary.get('routes', [])) or 'no route'} · {chips}"
    )

    # One collapsed block, so notices cannot push the downloads off-screen.
    notices = []
    if summary.get("n_form_groups"):
        notices.append(
            f"Extracted {summary['n_form_groups']} survey/form response group(s) — "
            "see the **Responses** tab."
        )
    if summary.get("n_markup_disagreements"):
        notices.append(
            f"{summary['n_markup_disagreements']} OCR/geometric mark disagreement(s) "
            "were left unchanged and flagged for review."
        )
    if summary.get("n_uncertain_marks"):
        notices.append(
            f"{summary['n_uncertain_marks']} checkbox/mark(s) flagged uncertain — "
            "see the **Responses** tab."
        )
    if notices:
        with st.expander(f"⚠️ {len(notices)} thing(s) to check", expanded=False):
            for notice in notices:
                st.markdown(f"- {notice}")

    # --- Downloads ---
    stem = downloads.get("stem", "document")
    # Grouped by purpose; every slot always renders, so a missing output greys
    # out in place instead of reflowing the grid.
    searchable = downloads.get("searchable_pdf")
    text_bytes = downloads.get("text")
    docx_bytes = downloads.get("docx")
    md_zip = downloads.get("markdown_zip")
    tables_zip = downloads.get("tables_zip")

    st.caption("**Read and edit**")
    d1, d2, d3, d4 = st.columns(4)
    with d1:
        st.download_button(
            "⬇️ Plain text", text_bytes or b"", file_name=f"{stem}.txt",
            mime="text/plain", disabled=not text_bytes, use_container_width=True,
        )
    with d2:
        st.download_button(
            "⬇️ Word", docx_bytes or b"", file_name=f"{stem}.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            disabled=not docx_bytes, use_container_width=True,
            help=None if docx_bytes else "Unavailable: python-docx is not installed.",
        )
    with d3:
        st.download_button(
            "⬇️ Markdown", md_zip or b"", file_name=f"{stem}_markdown.zip",
            mime="application/zip", disabled=not md_zip, use_container_width=True,
            help="Markdown plus an `assets/` folder of cropped figures.",
        )
    with d4:
        st.download_button(
            "⬇️ Searchable PDF", searchable or b"", file_name=f"{stem}_searchable.pdf",
            mime="application/pdf", disabled=not searchable, use_container_width=True,
            help=(
                "The original pages with an invisible, selectable text layer."
                if searchable else
                "Tick **Searchable PDF** before parsing to produce this."
            ),
        )

    st.caption("**Analyse**")
    d5, d6, d7 = st.columns(3)
    with d5:
        st.download_button(
            "⬇️ JSON", downloads.get("json") or b"{}", file_name=f"{stem}.json",
            mime="application/json", use_container_width=True,
            help="Regions, bounding boxes, confidence and markup states.",
        )
    with d6:
        st.download_button(
            "⬇️ Tables (CSV)", tables_zip or b"", file_name=f"{stem}_tables.zip",
            mime="application/zip", disabled=not tables_zip, use_container_width=True,
            help=None if tables_zip else "No tables were detected in this document.",
        )
    with d7:
        st.download_button(
            "⬇️ Everything", downloads.get("full") or b"", file_name=f"{stem}_bundle.zip",
            mime="application/zip", disabled=not downloads.get("full"), use_container_width=True,
            type="primary", help="Every format above in one ZIP.",
        )

    responses_csv = downloads.get("responses_csv")
    if responses_csv:
        st.download_button(
            "⬇️ Form responses (CSV)",
            responses_csv,
            file_name=f"{stem}_form_responses.csv",
            mime="text/csv",
        )

    # Read off the document, so the citation cannot drift from what ran.
    provenance = doc_ir.model_provenance(document)
    if provenance:
        with st.expander("🔬 Models used (for citation)", expanded=False):
            labels = {
                "text_recognition": "**Text recognition**",
                "figure_descriptions": "**Figure descriptions**",
                "text_layer": "**Searchable-PDF word geometry**",
            }
            for key, value in provenance.items():
                joined = ", ".join(value) if isinstance(value, list) else str(value)
                st.markdown(f"- {labels.get(key, key)}: {joined}")
            st.caption(
                "Printed text is transcribed by the recognition model; figure "
                "descriptions are *generated* by a vision-language model and are "
                "not part of the document. The same summary ships as "
                "`models_used.txt` in the bundle and under `models` in the JSON."
            )

    # Without survey extraction the Responses tab would always be empty.
    labels = ["📄 Document", "📊 Tables", "🖼️ Figures"]
    if SURVEY_EXTRACTION_UI_ENABLED:
        labels.append("☑️ Responses")
    labels.append("🗺️ Layout preview")

    tabs = dict(zip(labels, st.tabs(labels)))
    with tabs["📄 Document"]:
        render_document_tab(document)
    with tabs["📊 Tables"]:
        render_tables_tab(document)
    with tabs["🖼️ Figures"]:
        render_figures_tab(document)
    if SURVEY_EXTRACTION_UI_ENABLED:
        with tabs["☑️ Responses"]:
            render_markup_tab(document)
    with tabs["🗺️ Layout preview"]:
        render_layout_tab(document)


# ==========================================
#      AUTOMATIC PIPELINE — UI
# ==========================================

def _searchable_pdf_language(key, enabled):
    """Tesseract language for the word-positioning pass.

    Only positions come from Tesseract, but the language still matters: on a
    scanned German questionnaire `deu` placed 87.6% of tokens on their exact box
    against 82.1% for `eng`. Mismatched tokens just highlight a wider span.
    """
    auto = "Detect automatically"
    names = [auto, *searchable_pdf.TESSERACT_LANGUAGES]
    # Greyed out rather than hidden, so ticking the box shifts nothing below it.
    choice = st.selectbox(
        "Document language",
        names,
        index=0,
        key=key,
        disabled=not enabled,
        help=(
            "Detection reads the text Text Lab already extracted, per page, so "
            "mixed-language documents are handled page by page; pages with too "
            "little text keep the English default. Set this explicitly if a "
            "document is misdetected. Either way it only affects how precisely "
            "each word is located — the text itself always comes from Text Lab's OCR."
        ),
    )
    if not enabled:
        return searchable_pdf.DEFAULT_TESSERACT_LANG
    if choice == auto:
        return "auto"
    return searchable_pdf.TESSERACT_LANGUAGES[choice]


#: Rough per-option cost, shown before a job that may run for minutes.
_OPTION_COSTS = {
    "highest_quality": "every page through the vision model",
    "searchable_pdf": "a word-positioning pass per page",
    "describe_images": "one vision-model call per figure",
    "extract_survey": "one vision-model call per question",
    "survey_batch_mode": "one pass to learn the form, then a second read per file",
}


def _parse_options(prefix, *, batch=False):
    """Shared parse-option panel for single and batch. Returns the settings."""
    options = {}
    with st.container(border=True):
        if batch:
            # A choice for batch, where the cost multiplies by the file count.
            c1, c2 = st.columns([1, 1])
            with c1:
                options["searchable_pdf"] = st.checkbox(
                    "🔍 Searchable PDF for each file",
                    value=True,
                    key=f"{prefix}_searchable_pdf",
                    help=(
                        "Adds `document_searchable.pdf` per file: the original "
                        "pages with an invisible, selectable text layer."
                    ),
                )
        else:
            # Always on for a single document: seconds a page, and the most useful output.
            options["searchable_pdf"] = True
            c2 = st.container()
        with c2:
            options["ocr_lang"] = _searchable_pdf_language(
                f"{prefix}_pdf_lang", options["searchable_pdf"]
            )

        st.caption("**Quality and AI analysis**")
        c3, c4 = st.columns([1, 1])
        with c3:
            options["highest_quality"] = st.checkbox(
                "🔬 Highest quality",
                value=False,
                key=f"{prefix}_hq",
                help=(
                    "Sends every page through PaddleOCR-VL, even pages that "
                    "already have a digital text layer. Best for equations, "
                    "forms and complex layouts. Pages with detected math are "
                    "routed to the vision model automatically either way."
                ),
            )
        with c4:
            options["describe_images"] = st.checkbox(
                "🖼️ Describe figures",
                value=False,
                key=f"{prefix}_describe_images",
                help=(
                    "Adds a generated description and visible-text "
                    "transcription to each detected figure."
                ),
            )

        options["survey_batch_mode"] = False
        if batch:
            options["survey_batch_mode"] = st.checkbox(
                "📋 Extract questionnaire responses",
                value=False,
                key=f"{prefix}_survey_batch",
                help=(
                    "For a batch of the **same** paper questionnaire filled in by "
                    "different people. TextLab learns the blank form from the "
                    "batch itself, then reads every respondent against it and "
                    "adds a survey/ folder: one row per respondent, TRUE/FALSE "
                    "per checkbox, and a certainty beside every answer. The "
                    "normal text extraction still runs for each file."
                ),
            )

        options["extract_survey"] = False
        options["same_template"] = False
        if SURVEY_EXTRACTION_UI_ENABLED:
            options["extract_survey"] = st.checkbox(
                "🧪 Extract survey/form responses (experimental)",
                value=False,
                key=f"{prefix}_extract_survey",
                help=(
                    "Question-level response extraction at 300 DPI. Original OCR "
                    "text is preserved and every answer is flagged for review."
                ),
            )
            if batch and options["extract_survey"]:
                options["same_template"] = st.checkbox(
                    "📐 All files use the same questionnaire layout",
                    value=False,
                    key=f"{prefix}_same_template",
                )

        # Only what was opted into: the searchable PDF is the baseline for a
        # single document, not a cost.
        chargeable = dict(_OPTION_COSTS)
        if not batch:
            chargeable.pop("searchable_pdf", None)
        enabled = [chargeable[name] for name in chargeable if options.get(name)]
        if enabled:
            st.caption(f"⏱️ Adds {'; '.join(enabled)}.")
        else:
            st.caption("⏱️ Fastest settings — scanned pages only go to the vision model.")
    return options


def auto_single_ui():
    st.markdown(
        "Upload a **PDF** or **image** and press **Parse document**. TextLab "
        "automatically detects layout, tables, figures and formulas. Optional "
        "AI analysis can describe detected figures and images."
    )
    uploaded_file = st.file_uploader(
        "Choose a PDF or image file",
        type=AUTO_INPUT_TYPES,
        on_change=clear_results,
        args=(True,),
        key="auto_single_upload",
    )
    if uploaded_file is not None:
        opts = _parse_options("auto_single")
        if st.session_state.ocr_running:
            st.warning("⏳ A job is currently running. The button is disabled until completion.")
        if st.button("📑 Parse document", type="primary",
                     disabled=st.session_state.ocr_running, key="auto_single_btn"):
            run_auto_single(
                uploaded_file,
                native_fast_lane=not opts["highest_quality"],
                describe_images=opts["describe_images"],
                extract_survey=opts["extract_survey"],
                searchable_pdf=opts["searchable_pdf"],
                ocr_lang=opts["ocr_lang"],
            )

    if st.session_state.get("auto_complete"):
        render_auto_results()
    elif st.session_state.get("auto_error"):
        st.error(st.session_state.auto_error)


def auto_batch_ui():
    st.markdown(
        "Upload a **ZIP archive** of PDFs or images. Each file is parsed with the "
        "automatic pipeline; the result ZIP mirrors your folder structure with a "
        "`document.md`, `document.json`, `tables/` and `assets/` per file. For a "
        "batch of the same filled-in questionnaire, tick **Extract questionnaire "
        "responses** to also get a `survey/` folder with one row per respondent."
    )
    batch_zip = st.file_uploader(
        "Upload ZIP file",
        type=["zip"],
        on_change=clear_results,
        args=(True,),
        key="auto_batch_upload",
    )
    if batch_zip is not None:
        opts = _parse_options("auto_batch", batch=True)
        if st.session_state.ocr_running:
            st.warning("⏳ A job is currently running. The button is disabled until completion.")
        if st.button("📦 Parse batch", type="primary",
                     disabled=st.session_state.ocr_running, key="auto_batch_btn"):
            run_auto_batch(
                batch_zip,
                native_fast_lane=not opts["highest_quality"],
                describe_images=opts["describe_images"],
                extract_survey=opts["extract_survey"],
                same_template=opts["same_template"],
                survey_batch_mode=opts["survey_batch_mode"],
                searchable_pdf=opts["searchable_pdf"],
                ocr_lang=opts["ocr_lang"],
            )

    if st.session_state.get("batch_auto_complete"):
        st.success("✅ Batch parsing completed successfully!")
        st.download_button(
            "📥 Download all results (ZIP)",
            st.session_state.batch_auto_zip,
            file_name="batch_auto_ocr_results.zip",
            mime="application/zip",
            use_container_width=True,
            type="primary",
        )
        render_survey_review()
    elif st.session_state.get("auto_error"):
        st.error(st.session_state.auto_error)


# ==========================================
#      LEGACY ENGINES (unchanged behavior)
# ==========================================

def legacy_single_flow(ocr_engine, ocr_language, glm_mode):
    st.markdown("Upload a **PDF** or **Image** to extract its text content and preview the results.")

    uploaded_file = st.file_uploader(
        "Choose a PDF or Image file",
        type=["pdf", "png", "jpg", "jpeg", "bmp", "tiff"],
        on_change=clear_results,
        args=(True,),
        key="legacy_single_upload",
    )

    if uploaded_file is not None:
        if st.session_state.ocr_running:
            st.warning("⏳ OCR is currently running. The button is disabled until completion.")

        if st.button("Run OCR", disabled=st.session_state.ocr_running, key="legacy_single_btn"):
            if st.session_state.ocr_running:
                st.stop()

            clear_results(reset_running=False)
            st.session_state.ocr_running = True
            run_notice = st.empty()
            run_notice.info("⏳ OCR has started.")

            # Check and Pull GLM-OCR if needed
            if ocr_engine == "GLM-OCR":
                model_name = "glm-ocr:latest"
                try:
                    models_dict = ollama.list()
                    models_list = []
                    if isinstance(models_dict, dict):
                        models_list = models_dict.get("models") or []
                    else:
                        models_list = getattr(models_dict, 'models', [])
                    local_models = [str(getattr(m, 'model', getattr(m, 'name', m.get('name', '')))) for m in models_list]
                    is_present = any(model_name in name or name in model_name for name in local_models)

                    if not is_present:
                        with st.spinner(f"📥 Pulling model '{model_name}'..."):
                            ollama.pull(model_name)
                        st.success(f"Model {model_name} ready.")
                except Exception as e:
                    try:
                        ollama.pull(model_name)
                    except Exception as pull_error:
                        st.error(f"Failed to pull GLM-OCR model: {pull_error}")
                        st.session_state.ocr_running = False
                        st.stop()

            JOB_ID = str(uuid.uuid4())
            JOB_DIR = OCR_JOBS_BASE_DIR / JOB_ID
            INPUT_DIR = JOB_DIR / "input"
            WORKSPACE_DIR = JOB_DIR / "workspace"

            try:
                INPUT_DIR.mkdir(parents=True, exist_ok=True)
                WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
                preview_images = []

                # Save uploaded file
                input_file_path = INPUT_DIR / uploaded_file.name
                with open(input_file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Detect if input is PDF or Image
                is_pdf = input_file_path.suffix.lower() == ".pdf"

                results_dir = WORKSPACE_DIR / "results"
                results_dir.mkdir(parents=True, exist_ok=True)

                # --- OLMOCR PATH ---
                if ocr_engine == "OlmOCR":
                    CONT_INPUT_FILE = str(input_file_path)

                    if not is_pdf:
                        try:
                            img = Image.open(input_file_path).convert("RGB")
                            pdf_path = input_file_path.with_suffix(".pdf")
                            img.save(pdf_path, "PDF", resolution=100.0)
                            CONT_INPUT_FILE = str(pdf_path)
                        except Exception as e:
                            raise RuntimeError(f"Failed to convert image to PDF for OlmOCR: {e}")

                    CONT_WORKSPACE_DIR = str(WORKSPACE_DIR)
                    # Point explicitly to the isolated OlmOCR conda environment
                    # Explicitly inject the Conda Environment variables into the subprocess
                    olmocr_env = os.environ.copy()
                    olmocr_env["PATH"] = f"/opt/conda/envs/olmocr_backend/bin:{olmocr_env.get('PATH', '')}"
                    olmocr_env["LD_LIBRARY_PATH"] = f"/opt/conda/envs/olmocr_backend/lib:{olmocr_env.get('LD_LIBRARY_PATH', '')}"

                    cmd = [
                        "/opt/conda/envs/olmocr_backend/bin/python", "-m", "olmocr.pipeline",
                        CONT_WORKSPACE_DIR,
                        "--markdown",
                        "--pdfs", CONT_INPUT_FILE,
                        "--gpu-memory-utilization", OLMOCR_GPU_MEMORY_UTILIZATION,
                    ]

                    with st.spinner("Running OlmOCR..."):
                        # Pass the custom env dictionary here!
                        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', env=olmocr_env)

                    if result.returncode != 0:
                        st.session_state.ocr_error = f"OCR process failed. Code: {result.returncode}"
                        st.session_state.ocr_error_details = (result.stdout, result.stderr)
                        raise RuntimeError(st.session_state.ocr_error)

                    jsonl_files = list(results_dir.glob("*.jsonl"))
                    if not jsonl_files:
                        raise RuntimeError("No .jsonl output found.")

                    output_file_path = jsonl_files[0]
                    with open(output_file_path, 'r', encoding='utf-8') as f:
                        first_line = f.readline()
                        data = json.loads(first_line)

                    st.session_state.extracted_text = data.get("text")
                    st.session_state.json_content = first_line
                    st.session_state.txt_name = input_file_path.with_suffix(".txt").name
                    st.session_state.json_name = output_file_path.name
                    st.session_state.ocr_preview_engine = "OlmOCR"
                    st.session_state.ocr_preview_images = []

                # --- IMAGE-BASED PATH (EasyOCR, Paddle, GLM-OCR) ---
                else:
                    tmp_img_dir = WORKSPACE_DIR / "images"
                    tmp_img_dir.mkdir(parents=True, exist_ok=True)
                    image_paths = []

                    if is_pdf:
                        prefix = tmp_img_dir / "page"
                        subprocess.run(
                            ["pdftoppm", "-png", str(input_file_path), str(prefix)],
                            check=True, capture_output=True, text=True, encoding="utf-8",
                        )
                        image_paths = sorted(tmp_img_dir.glob("page-*.png"))
                        if not image_paths:
                            raise RuntimeError("No images generated from PDF.")
                    else:
                        dest_path = tmp_img_dir / input_file_path.name
                        shutil.copy(input_file_path, dest_path)
                        image_paths = [dest_path]

                    ocr_results = []
                    progress_bar = st.progress(0.0, text=f"Running {ocr_engine}...")

                    # --- 1. EasyOCR ---
                    if ocr_engine == "EasyOCR":
                        reader = get_easyocr_reader(ocr_language)
                        for idx, img_path in enumerate(image_paths, start=1):
                            page_res = reader.readtext(str(img_path), detail=1, paragraph=True)
                            page_text = "\n".join([r[1] for r in page_res])
                            ocr_results.append({"page": idx, "text": page_text, "raw": page_res})

                            pl, pr = render_easyocr_preview(img_path, page_res)
                            if pl and pr: preview_images.append((pl, pr))
                            progress_bar.progress(idx / len(image_paths), text=f"Running EasyOCR... page {idx}/{len(image_paths)}")

                    # --- 2. PaddleOCR ---
                    elif ocr_engine == "PaddleOCR":
                        paddle_pages = run_paddleocr_backend(image_paths, ocr_language)
                        for idx, page in enumerate(paddle_pages, start=1):
                            img_path = pathlib.Path(page.get("image") or image_paths[idx - 1])
                            compact_preds = page.get("raw", [])
                            page_text = page.get("text", "")
                            ocr_results.append({"page": idx, "text": page_text, "raw": compact_preds})

                            rendered_png = decode_png_b64(page.get("rendered_png_b64"))
                            if rendered_png:
                                preview_images.append(rendered_png)
                            else:
                                pl, _ = render_paddle_preview(img_path, compact_preds)
                                if pl:
                                    preview_images.append(pl)
                            progress_bar.progress(idx / len(image_paths), text=f"Running PaddleOCR... page {idx}/{len(image_paths)}")

                    # --- 3. GLM-OCR ---
                    elif ocr_engine == "GLM-OCR":
                        for idx, img_path in enumerate(image_paths, start=1):
                            img = cv2.imread(str(img_path))

                            max_dim = 2048
                            h, w = img.shape[:2]
                            if h > max_dim or w > max_dim:
                                scale = max_dim / max(h, w)
                                new_w = int(w * scale)
                                new_h = int(h * scale)
                                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

                            success, encoded_img = cv2.imencode('.png', img)
                            if not success:
                                ocr_results.append({"page": idx, "text": "[Error encoding image]", "raw": {}})
                                continue

                            img_bytes = encoded_img.tobytes()

                            try:
                                response = ollama.chat(
                                    model='glm-ocr:latest',
                                    messages=[{
                                        'role': 'user',
                                        'content': glm_mode,
                                        'images': [img_bytes]
                                    }],
                                    options={
                                        'temperature': 0,
                                        'num_ctx': 8192
                                    }
                                )
                                page_text = response.get('message', {}).get('content', '')
                            except Exception as e:
                                page_text = f"[Error processing page {idx}: {str(e)}]"

                            ocr_results.append({"page": idx, "text": page_text, "raw": {"content": page_text}})
                            preview_images.append(img_bytes)
                            progress_bar.progress(idx / len(image_paths), text=f"Running GLM-OCR... page {idx}/{len(image_paths)}")

                    progress_bar.empty()

                    # --- Write Outputs ---
                    all_text = []
                    for item in ocr_results:
                        page = item["page"]
                        page_text = item["text"]
                        all_text.append(page_text)
                        (results_dir / f"page_{page:04d}.txt").write_text(page_text, encoding="utf-8")
                        (results_dir / f"page_{page:04d}.json").write_text(
                            json.dumps(make_json_serializable(item), ensure_ascii=False), encoding="utf-8"
                        )

                    combined_text = "\n\n".join(all_text)
                    st.session_state.extracted_text = combined_text
                    st.session_state.json_content = json.dumps(make_json_serializable(ocr_results), ensure_ascii=False)
                    st.session_state.txt_name = input_file_path.with_suffix(".txt").name
                    st.session_state.json_name = input_file_path.with_suffix(".json").name
                    st.session_state.ocr_preview_engine = ocr_engine
                    st.session_state.ocr_preview_images = preview_images
                    st.session_state.ocr_preview_page = 0

                # Zip results
                zip_path = shutil.make_archive(str(WORKSPACE_DIR / "ocr_results"), "zip", results_dir)
                st.session_state.ocr_zip_bytes = pathlib.Path(zip_path).read_bytes()
                st.session_state.ocr_complete = True

            except Exception as e:
                st.session_state.ocr_error = f"An unexpected error occurred: {e}"
                st.exception(e)

            finally:
                run_notice.empty()
                st.session_state.ocr_running = False
                _cleanup_job_dir(JOB_DIR)

    # --- SINGLE FILE RESULTS DISPLAY ---
    if "ocr_complete" in st.session_state:
        st.success("🎉 OCR complete!")

        # 1. HTML Table Detection & CSV Conversion
        df_table = extract_html_table(st.session_state.extracted_text)
        if df_table is not None:
            st.markdown("### 📊 Detected Table")
            st.dataframe(df_table, use_container_width=True)
            csv_data = df_table.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Table as CSV",
                data=csv_data,
                file_name=f"{st.session_state.txt_name.replace('.txt', '')}_table.csv",
                mime="text/csv",
                type="primary"
            )
        elif "<table>" in st.session_state.extracted_text or '<table class="' in st.session_state.extracted_text:
            # Fallback if pandas parsing failed but HTML table exists
            st.markdown("### 📊 Detected Table")
            st.markdown(st.session_state.extracted_text, unsafe_allow_html=True)

        # 2. Raw Text Output
        st.markdown("### Extracted Text / Code")
        st.text_area("Result", st.session_state.extracted_text, height=400, key="md_result")

        # 3. Downloads
        c1, c2, c3 = st.columns(3)
        with c1: st.download_button("Download as .txt", st.session_state.extracted_text, st.session_state.txt_name, "text/plain")
        with c2: st.download_button("Download as .jsonl", st.session_state.json_content, st.session_state.json_name, "application/json")
        with c3: st.download_button("Download all outputs (.zip)", st.session_state.ocr_zip_bytes, "ocr_outputs.zip", "application/zip")

        # 4. Preview Section
        preview_images = st.session_state.get("ocr_preview_images", [])
        if preview_images:
            st.markdown("---")
            st.markdown("### 👁️ Document Preview")
            preview_engine = st.session_state.get("ocr_preview_engine", "")

            current_page = st.session_state.get("ocr_preview_page", 0)
            current_page = max(0, min(current_page, len(preview_images) - 1))
            st.session_state.ocr_preview_page = current_page

            c_prev, c_info, c_next = st.columns([1, 2, 1])
            if c_prev.button("⬅ Previous", disabled=current_page <= 0, key="legacy_prev"):
                st.session_state.ocr_preview_page -= 1
                st.rerun()
            with c_info:
                st.caption(f"Page {current_page + 1} of {len(preview_images)}")
            if c_next.button("Next ➡", disabled=current_page >= len(preview_images) - 1, key="legacy_next"):
                st.session_state.ocr_preview_page += 1
                st.rerun()

            current_preview = preview_images[current_page]

            if preview_engine == "EasyOCR" and isinstance(current_preview, (list, tuple)) and len(current_preview) >= 2:
                left_img, right_img = current_preview[0], current_preview[1]
                cl, cr = st.columns(2)
                with cl:
                    st.caption("Detected boxes")
                    st.image(left_img, use_container_width=True)
                with cr:
                    st.caption("OCR Layout")
                    st.image(right_img, use_container_width=True)
            else:
                left_img = current_preview[0] if isinstance(current_preview, (list, tuple)) else current_preview
                st.image(left_img, caption="Original Document", use_container_width=True)

    elif "ocr_error" in st.session_state:
        st.error(st.session_state.ocr_error)
        if "ocr_error_details" in st.session_state:
            st.text_area("Error Details", str(st.session_state.ocr_error_details), height=150)
    elif st.session_state.get("ocr_running"):
        st.info("OCR is running. Please wait...")


def legacy_batch_flow(ocr_engine, ocr_language, glm_mode):
    st.markdown("Upload a **ZIP archive** containing multiple PDFs or Images. They will be processed and returned as a single organized ZIP.")

    batch_zip = st.file_uploader(
        "Upload ZIP file",
        type=["zip"],
        on_change=clear_results,
        args=(True,),
        key="legacy_batch_upload",
    )

    if batch_zip is not None:
        if st.session_state.ocr_running:
            st.warning("⏳ OCR is currently running. The button is disabled until completion.")

        if st.button("Run Batch OCR", disabled=st.session_state.ocr_running, key="legacy_batch_btn"):
            if st.session_state.ocr_running:
                st.stop()

            clear_results(reset_running=False)
            st.session_state.ocr_running = True
            run_notice = st.empty()
            run_notice.info("⏳ Batch OCR has started. This may take a while depending on the number of files.")

            # Check and Pull GLM-OCR if needed
            if ocr_engine == "GLM-OCR":
                model_name = "glm-ocr:latest"
                try:
                    models_dict = ollama.list()
                    models_list = models_dict.get("models") if isinstance(models_dict, dict) else getattr(models_dict, 'models', [])
                    local_models = [str(getattr(m, 'model', getattr(m, 'name', m.get('name', '')))) for m in models_list]
                    if not any(model_name in name or name in model_name for name in local_models):
                        with st.spinner(f"📥 Pulling model '{model_name}'..."):
                            ollama.pull(model_name)
                except Exception as e:
                    try:
                        ollama.pull(model_name)
                    except Exception as pull_error:
                        st.error(f"Failed to pull GLM-OCR model: {pull_error}")
                        st.session_state.ocr_running = False
                        st.stop()

            # Workspace Setup
            import zipfile
            JOB_ID = str(uuid.uuid4())
            JOB_DIR = OCR_JOBS_BASE_DIR / JOB_ID
            INPUT_DIR = JOB_DIR / "input"
            WORKSPACE_DIR = JOB_DIR / "workspace"
            RESULTS_DIR = WORKSPACE_DIR / "results"

            try:
                INPUT_DIR.mkdir(parents=True, exist_ok=True)
                RESULTS_DIR.mkdir(parents=True, exist_ok=True)

                # Extract the uploaded ZIP securely
                with zipfile.ZipFile(batch_zip, "r") as z:
                    z.extractall(INPUT_DIR)

                valid_exts = {".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
                valid_files = []
                for root, dirs, files in os.walk(INPUT_DIR):
                    for file in files:
                        if file.startswith("._"): continue # Skip macOS metadata files
                        file_path = pathlib.Path(root) / file
                        if file_path.suffix.lower() in valid_exts:
                            valid_files.append(file_path)

                if not valid_files:
                    raise RuntimeError("No valid documents or images found in the ZIP.")

                # Pre-load Models for Image Engines (Saves massive time)
                reader = None
                if ocr_engine == "EasyOCR":
                    reader = get_easyocr_reader(ocr_language)

                progress_bar = st.progress(0.0)
                status_text = st.empty()

                # Processing Loop
                for idx, file_path in enumerate(valid_files):
                    base_name = file_path.stem
                    rel_path = file_path.relative_to(INPUT_DIR)
                    status_text.text(f"Processing ({idx+1}/{len(valid_files)}): {rel_path}")

                    # Create dedicated output folder replicating zip structure
                    file_output_dir = RESULTS_DIR / rel_path.parent / base_name
                    file_output_dir.mkdir(parents=True, exist_ok=True)

                    is_pdf = file_path.suffix.lower() == ".pdf"

                    if ocr_engine == "OlmOCR":
                        CONT_INPUT_FILE = str(file_path)
                        if not is_pdf:
                            img = Image.open(file_path).convert("RGB")
                            pdf_path = file_path.with_suffix(".pdf")
                            img.save(pdf_path, "PDF", resolution=100.0)
                            CONT_INPUT_FILE = str(pdf_path)

                        olmocr_env = os.environ.copy()
                        olmocr_env["PATH"] = f"/opt/conda/envs/olmocr_backend/bin:{olmocr_env.get('PATH', '')}"
                        olmocr_env["LD_LIBRARY_PATH"] = f"/opt/conda/envs/olmocr_backend/lib:{olmocr_env.get('LD_LIBRARY_PATH', '')}"

                        cmd = [
                            "/opt/conda/envs/olmocr_backend/bin/python", "-m", "olmocr.pipeline",
                            str(file_output_dir), "--markdown", "--pdfs", CONT_INPUT_FILE,
                            "--gpu-memory-utilization", OLMOCR_GPU_MEMORY_UTILIZATION
                        ]

                        # Pass the custom env dictionary here!
                        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', env=olmocr_env)

                        if result.returncode == 0:
                            jsonl_files = list(file_output_dir.glob("*.jsonl"))
                            if jsonl_files:
                                with open(jsonl_files[0], 'r', encoding='utf-8') as f:
                                    data = json.loads(f.readline())
                                (file_output_dir / f"{base_name}.txt").write_text(data.get("text", ""), encoding="utf-8")
                                shutil.move(str(jsonl_files[0]), str(file_output_dir / f"{base_name}.jsonl"))

                    else:
                        # Image-based Engines (EasyOCR, PaddleOCR, GLM-OCR)
                        tmp_img_dir = file_output_dir / "images_tmp"
                        tmp_img_dir.mkdir(parents=True, exist_ok=True)
                        image_paths = []

                        if is_pdf:
                            prefix = tmp_img_dir / "page"
                            subprocess.run(["pdftoppm", "-png", str(file_path), str(prefix)], check=True, capture_output=True)
                            image_paths = sorted(tmp_img_dir.glob("page-*.png"))
                        else:
                            dest_path = tmp_img_dir / file_path.name
                            shutil.copy(file_path, dest_path)
                            image_paths = [dest_path]

                        ocr_results = []
                        if ocr_engine == "PaddleOCR":
                            ocr_results = [
                                {"page": page.get("page", p_idx), "text": page.get("text", ""), "raw": page.get("raw", [])}
                                for p_idx, page in enumerate(run_paddleocr_backend(image_paths, ocr_language), start=1)
                            ]
                        else:
                            for p_idx, img_path in enumerate(image_paths, start=1):
                                if ocr_engine == "EasyOCR":
                                    page_res = reader.readtext(str(img_path), detail=1, paragraph=True)
                                    page_text = "\n".join([r[1] for r in page_res])
                                    ocr_results.append({"page": p_idx, "text": page_text, "raw": page_res})

                                elif ocr_engine == "GLM-OCR":
                                    img = cv2.imread(str(img_path))
                                    max_dim = 2048
                                    h, w = img.shape[:2]
                                    if h > max_dim or w > max_dim:
                                        scale = max_dim / max(h, w)
                                        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

                                    success, encoded_img = cv2.imencode('.png', img)
                                    if success:
                                        try:
                                            response = ollama.chat(
                                                model='glm-ocr:latest',
                                                messages=[{'role': 'user', 'content': glm_mode, 'images': [encoded_img.tobytes()]}],
                                                options={'temperature': 0, 'num_ctx': 8192}
                                            )
                                            page_text = response.get('message', {}).get('content', '')
                                        except Exception as e:
                                            page_text = f"[Error processing page {p_idx}: {str(e)}]"
                                    else:
                                        page_text = "[Error encoding image]"

                                    ocr_results.append({"page": p_idx, "text": page_text, "raw": {"content": page_text}})

                        # Write Results
                        all_text = []
                        for item in ocr_results:
                            all_text.append(item["text"])
                            (file_output_dir / f"page_{item['page']:04d}.txt").write_text(item["text"], encoding="utf-8")

                        (file_output_dir / f"{base_name}.txt").write_text("\n\n".join(all_text), encoding="utf-8")
                        (file_output_dir / f"{base_name}.json").write_text(json.dumps(make_json_serializable(ocr_results), ensure_ascii=False), encoding="utf-8")

                        shutil.rmtree(tmp_img_dir, ignore_errors=True)

                    progress_bar.progress((idx + 1) / len(valid_files))

                status_text.text("Batch OCR complete! Zipping results...")

                # Zip RESULTS_DIR directly to memory
                out_zip_buffer = io.BytesIO()
                with zipfile.ZipFile(out_zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                    for root, dirs, files in os.walk(RESULTS_DIR):
                        for file in files:
                            file_path = pathlib.Path(root) / file
                            arcname = file_path.relative_to(RESULTS_DIR)
                            zf.write(file_path, arcname)

                st.session_state.batch_ocr_zip_bytes = out_zip_buffer.getvalue()
                st.session_state.batch_ocr_complete = True

            except Exception as e:
                st.session_state.ocr_error = f"Batch processing failed: {e}"
                st.exception(e)

            finally:
                run_notice.empty()
                st.session_state.ocr_running = False
                _cleanup_job_dir(JOB_DIR)

    # --- BATCH RESULTS DISPLAY ---
    if st.session_state.get("batch_ocr_complete"):
        st.success("✅ Batch OCR completed successfully!")
        st.download_button(
            "📥 Download All OCR Results (ZIP)",
            st.session_state.batch_ocr_zip_bytes,
            file_name="batch_ocr_results.zip",
            mime="application/zip",
            use_container_width=True,
            type="primary"
        )
    elif "ocr_error" in st.session_state:
        st.error(st.session_state.ocr_error)


def legacy_engines_expander(workflow_mode):
    with st.expander("⚙️ Advanced: legacy engines (EasyOCR / PaddleOCR / OlmOCR / GLM-OCR)"):
        st.caption(
            "The classic engine-picker workflow. Each engine returns plain text; "
            "the automatic pipeline above is recommended for structured documents."
        )

        col_eng, col_mode = st.columns([1, 1])
        with col_eng:
            ocr_engine = st.selectbox(
                "OCR engine",
                ["EasyOCR", "PaddleOCR", "OlmOCR", "GLM-OCR"],
                index=0,
                on_change=clear_results,
                args=(True,),
                key="legacy_engine_select",
                help="Select the OCR backend. GLM-OCR is best for complex layouts and tables.",
            )

        glm_mode = "Text Recognition"
        ocr_language = "en"
        if ocr_engine == "GLM-OCR":
            with col_mode:
                glm_mode = st.selectbox(
                    "GLM-OCR Mode",
                    ["Text Recognition", "Table Recognition", "Figure Recognition"],
                    key="legacy_glm_mode",
                    help="Choose what specific aspect of the document you want to extract."
                )
        elif ocr_engine == "EasyOCR":
            easyocr_language_labels = list(EASYOCR_LANGUAGE_MAPPING.keys())
            easyocr_default_index = (
                easyocr_language_labels.index("English")
                if "English" in easyocr_language_labels
                else 0
            )
            with col_mode:
                easyocr_lang_label = st.selectbox(
                    "Document Language",
                    easyocr_language_labels,
                    index=easyocr_default_index,
                    on_change=clear_results,
                    args=(True,),
                    key="easyocr_language_select",
                    help="Select the text language for EasyOCR.",
                )
            ocr_language = EASYOCR_LANGUAGE_MAPPING[easyocr_lang_label]
        elif ocr_engine == "PaddleOCR":
            with col_mode:
                paddle_lang_label = st.selectbox(
                    "Document Language",
                    list(PADDLEOCR_LANGUAGE_MAPPING.keys()),
                    index=0,
                    on_change=clear_results,
                    args=(True,),
                    key="paddle_language_select",
                    help="Select the text language for PaddleOCR.",
                )
            ocr_language = PADDLEOCR_LANGUAGE_MAPPING[paddle_lang_label]

        if workflow_mode == "Single Document OCR":
            legacy_single_flow(ocr_engine, ocr_language, glm_mode)
        else:
            legacy_batch_flow(ocr_engine, ocr_language, glm_mode)


# ==========================================
#                 PAGE LAYOUT
# ==========================================

workflow_mode = st.radio(
    "Workflow",
    ["Single Document OCR", "Batch OCR (ZIP)"],
    index=0,
    horizontal=True,
    help="Choose to process a single file or batch process a ZIP archive",
    on_change=clear_results,
    args=(True,),
)

st.divider()

if workflow_mode == "Single Document OCR":
    auto_single_ui()
else:
    auto_batch_ui()

st.divider()
legacy_engines_expander(workflow_mode)
