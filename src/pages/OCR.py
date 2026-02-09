import os
os.environ["VLLM_GPU_MEMORY_UTILIZATION"] = "0.6"
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
os.environ.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "none")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import streamlit as st
import subprocess
import uuid
import pathlib
import shutil
import sys 
import json
import numpy as np

st.set_page_config(page_title="OCR", layout="wide")


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from auth import check_token

check_token()



st.title("📄 Document OCR")
st.markdown("Upload a PDF and choose an OCR engine to extract its text content.")

# --- 1. Get required paths from environment variables ---
# Get the user's home directory, which is persistent and accessible
HOST_HOME = os.environ.get("HOME")

if not HOST_HOME:
    st.error("**Configuration Error:** `HOME` environment variable is not set.")
    st.stop()

# Define a base directory for all OCR jobs
OCR_JOBS_BASE_DIR = pathlib.Path(HOST_HOME) / "ondemand_text_lab_ocr_jobs"
OLMOCR_GPU_MEMORY_UTILIZATION = os.environ.get("OLMOCR_GPU_MEMORY_UTILIZATION", "0.6")


def make_json_serializable(value):
    """Recursively convert numpy types to standard Python JSON-safe types."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: make_json_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [make_json_serializable(v) for v in value]
    if hasattr(value, "tolist"):
        try:
            return make_json_serializable(value.tolist())
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        try:
            return {k: make_json_serializable(v) for k, v in vars(value).items()}
        except Exception:
            pass
    return str(value)


def extract_texts_from_ocr_payload(payload):
    """Recursively extract OCR text strings from nested dict/list payloads."""
    texts = []

    if isinstance(payload, str):
        # Only collect strings when they are attached to explicit text keys.
        return texts

    if isinstance(payload, dict):
        for key, val in payload.items():
            key_l = str(key).lower()
            if key_l in {"rec_texts", "texts"}:
                if isinstance(val, list):
                    texts.extend([str(x).strip() for x in val if str(x).strip()])
                elif isinstance(val, str) and val.strip():
                    texts.append(val.strip())
                else:
                    texts.extend(extract_texts_from_ocr_payload(val))
            elif key_l == "text":
                if isinstance(val, str) and val.strip():
                    texts.append(val.strip())
                else:
                    texts.extend(extract_texts_from_ocr_payload(val))
            else:
                # Traverse nested containers but ignore plain strings from metadata fields.
                texts.extend(extract_texts_from_ocr_payload(val))
        return texts

    if isinstance(payload, (list, tuple)):
        for item in payload:
            texts.extend(extract_texts_from_ocr_payload(item))
        return texts

    return texts


def compact_paddle_prediction(pred):
    """
    Convert PaddleOCR prediction to a compact JSON-safe dict.
    Keeps core text-related fields and avoids serializing heavyweight objects.
    """
    if hasattr(pred, "json"):
        raw = pred.json
    elif hasattr(pred, "to_dict"):
        raw = pred.to_dict()
    else:
        raw = pred

    raw = make_json_serializable(raw)
    compact = {}
    if isinstance(raw, dict):
        for key in ("input_path", "page_index", "model_settings", "rec_texts", "rec_scores", "rec_polys", "dt_polys"):
            if key in raw:
                compact[key] = raw[key]
    else:
        compact["raw"] = raw

    rec_texts = extract_texts_from_ocr_payload(compact if compact else raw)
    # de-duplicate while preserving order
    compact["rec_texts"] = list(dict.fromkeys(rec_texts))
    return compact


@st.cache_resource(show_spinner=False)
def get_easyocr_reader():
    import easyocr
    return easyocr.Reader(["en"], gpu=True)


@st.cache_resource(show_spinner=False)
def get_paddleocr_engine():
    from paddleocr import PaddleOCR
    return PaddleOCR(use_textline_orientation=True, lang="en")

# --- NEW: Function to clear results when a new file is uploaded ---
def clear_results(reset_running=False):
    """Clears all OCR results from the session state."""
    for key in [
        "ocr_complete",
        "extracted_text",
        "json_content",
        "txt_name",
        "json_name",
        "ocr_error",
        "ocr_error_details",
    ]:
        if key in st.session_state:
            del st.session_state[key]
    # Only unlock explicitly on config/file changes (not when a run starts).
    if reset_running:
        st.session_state.ocr_running = False

# --- 2. Create File Uploader ---

uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type=["pdf"],
    on_change=clear_results,  # <-- This clears old results on new upload
    args=(True,),
)

if "ocr_running" not in st.session_state:
    st.session_state.ocr_running = False

ocr_engine = st.selectbox(
    "OCR engine",
    ["easyocr", "paddleocr", "olmocr"],
    index=0,
    on_change=clear_results,
    args=(True,),
    help="easyocr is the default. olmocr uses the local pipeline inside this container.",
)

# --- 3. Show Button and Run Process ---
# Only show the button and warning AFTER a file has been uploaded
if uploaded_file is not None:
    
    # st.info(f"File selected: **{uploaded_file.name}**")
    
    # Only show the warning if processing isn't already complete
    # if "ocr_complete" not in st.session_state:
    #     st.warning("⚠️ **Heads up:** Processing can take 4-5 minutes per page. Please keep this tab open until the process is complete.")

    if st.session_state.ocr_running:
        st.warning("⏳ OCR is currently running. The button is disabled until completion.")

    # Create the button. The logic below will only run if it's clicked.
    if st.button("Run OCR", disabled=st.session_state.ocr_running):
        if st.session_state.ocr_running:
            st.warning("OCR is already running. Please wait for the current job to finish.")
            st.stop()
        # Clear any previous state before starting
        clear_results(reset_running=False)
        st.session_state.ocr_running = True
        run_notice = st.empty()
        run_notice.info("⏳ OCR has started.")
    
        # --- 3. Set up temporary job directories ---
        JOB_ID = str(uuid.uuid4())
        JOB_DIR = OCR_JOBS_BASE_DIR / JOB_ID

        # These directories will be created on the host filesystem
        # They will be bind-mounted into the olmocr container
        INPUT_DIR = JOB_DIR / "input"
        WORKSPACE_DIR = JOB_DIR / "workspace"  # olmocr will write results here

        try:
            INPUT_DIR.mkdir(parents=True, exist_ok=True)
            WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)

            # Save the uploaded file to the input directory
            input_file_path = INPUT_DIR / uploaded_file.name
            with open(input_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # --- 4. Run OCR depending on engine ---
            results_dir = WORKSPACE_DIR / "results"
            results_dir.mkdir(parents=True, exist_ok=True)

            if ocr_engine == "olmocr":
                CONT_INPUT_FILE = str(input_file_path)
                CONT_WORKSPACE_DIR = str(WORKSPACE_DIR)
                cmd = [
                    sys.executable, "-m", "olmocr.pipeline",
                    CONT_WORKSPACE_DIR,
                    "--markdown",
                    "--pdfs", CONT_INPUT_FILE,
                    "--gpu-memory-utilization", OLMOCR_GPU_MEMORY_UTILIZATION,
                ]

                with st.spinner(
                    f"Running OlmOCR..."
                ):
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        encoding='utf-8'
                    )

                if result.returncode != 0:
                    st.session_state.ocr_error = f"OCR process failed. Return code: {result.returncode}"
                    st.session_state.ocr_error_details = (result.stdout, result.stderr)
                    raise RuntimeError(st.session_state.ocr_error)

                jsonl_files = list(results_dir.glob("*.jsonl"))
                if not jsonl_files:
                    st.session_state.ocr_error = f"Process ran, but no .jsonl output file was found in {results_dir}"
                    st.session_state.ocr_error_details = (result.stdout, result.stderr)
                    raise RuntimeError(st.session_state.ocr_error)

                output_file_path = jsonl_files[0]
                json_file_name = output_file_path.name
                with open(output_file_path, 'r', encoding='utf-8') as f:
                    first_line = f.readline()
                    data = json.loads(first_line)

                st.session_state.extracted_text = data.get("text")
                st.session_state.json_content = first_line
                st.session_state.txt_name = input_file_path.with_suffix(".txt").name
                st.session_state.json_name = json_file_name

            else:
                # easyocr / paddleocr
                tmp_img_dir = WORKSPACE_DIR / "images"
                tmp_img_dir.mkdir(parents=True, exist_ok=True)

                # Convert PDF to PNG images using pdftoppm (poppler-utils)
                prefix = tmp_img_dir / "page"
                subprocess.run(
                    ["pdftoppm", "-png", str(input_file_path), str(prefix)],
                    check=True,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                )

                image_paths = sorted(tmp_img_dir.glob("page-*.png"))
                if not image_paths:
                    raise RuntimeError("No images were generated from the PDF.")

                if ocr_engine == "easyocr":
                    with st.spinner("Loading model..."):
                        reader = get_easyocr_reader()
                    progress = st.progress(0.0, text="Running EasyOCR...")
                    ocr_results = []
                    for idx, img_path in enumerate(image_paths, start=1):
                        page_res = reader.readtext(str(img_path), detail=1, paragraph=True)
                        page_text = "\n".join([r[1] for r in page_res])
                        ocr_results.append({"page": idx, "text": page_text, "raw": page_res})
                        progress.progress(idx / len(image_paths), text=f"Running EasyOCR... page {idx}/{len(image_paths)}")
                    progress.empty()

                else:
                    with st.spinner("Loading model..."):
                        ocr = get_paddleocr_engine()
                    progress = st.progress(0.0, text="Running PaddleOCR...")
                    ocr_results = []
                    for idx, img_path in enumerate(image_paths, start=1):
                        page_preds = ocr.predict(str(img_path))
                        compact_preds = [compact_paddle_prediction(pred) for pred in page_preds]
                        page_text_lines = []
                        for pred in compact_preds:
                            page_text_lines.extend(pred.get("rec_texts", []))
                        page_text = "\n".join([line for line in page_text_lines if line])
                        ocr_results.append({"page": idx, "text": page_text, "raw": compact_preds})
                        progress.progress(idx / len(image_paths), text=f"Running PaddleOCR... page {idx}/{len(image_paths)}")
                    progress.empty()

                # Write per-page outputs + combined text
                all_text = []
                write_progress = st.progress(0.0, text="Writing OCR outputs...")
                for item in ocr_results:
                    page = item["page"]
                    page_text = item["text"]
                    all_text.append(page_text)
                    (results_dir / f"page_{page:04d}.txt").write_text(page_text, encoding="utf-8")
                    (results_dir / f"page_{page:04d}.json").write_text(
                        json.dumps(make_json_serializable(item), ensure_ascii=False), encoding="utf-8"
                    )
                    write_progress.progress(
                        page / len(ocr_results),
                        text=f"Writing OCR outputs... page {page}/{len(ocr_results)}"
                    )
                write_progress.empty()

                combined_text = "\n\n".join(all_text)
                st.session_state.extracted_text = combined_text
                st.session_state.json_content = json.dumps(make_json_serializable(ocr_results), ensure_ascii=False)
                st.session_state.txt_name = input_file_path.with_suffix(".txt").name
                st.session_state.json_name = input_file_path.with_suffix(".json").name

            # Create ZIP of all outputs
            # st.info("Packaging outputs...")
            zip_path = shutil.make_archive(str(WORKSPACE_DIR / "ocr_results"), "zip", results_dir)
            st.session_state.ocr_zip_bytes = pathlib.Path(zip_path).read_bytes()
            st.session_state.ocr_complete = True

        except Exception as e:
            st.session_state.ocr_error = f"An unexpected error occurred: {e}"
            st.exception(e)
        
        finally:
            run_notice.empty()
            st.session_state.ocr_running = False
            # --- 7. Clean up the temporary job directory ---
            if JOB_DIR.exists():
                try:
                    subprocess.Popen(
                        ["rm", "-rf", str(JOB_DIR)],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                except Exception as e:
                    st.warning(f"Could not clean up {JOB_DIR}. Error: {e}")

# --- 7. DISPLAY RESULTS (Moved outside the button click) ---
# This block now runs on *every* rerun, checking the session state.

if "ocr_complete" in st.session_state:
    st.success("🎉 OCR complete!")
    
    st.markdown("### Extracted Text")
    st.text_area(
        "Result", 
        st.session_state.extracted_text, 
        height=400,
        key="md_result"
    )
    
    # --- Create columns for download buttons ---
    col1, col2 = st.columns(2)
    
    with col1:
        st.download_button(
            label="Download as .txt",
            data=st.session_state.extracted_text,
            file_name=st.session_state.txt_name,
            mime="text/plain",
        )
    
    with col2:
        st.download_button(
            label="Download as .jsonl",
            data=st.session_state.json_content,
            file_name=st.session_state.json_name,
            mime="application/json",
        )

    st.download_button(
        label="Download all outputs (.zip)",
        data=st.session_state.ocr_zip_bytes,
        file_name="ocr_outputs.zip",
        mime="application/zip",
    )

# Display errors if they were saved to session state
elif "ocr_error" in st.session_state:
    st.error(st.session_state.ocr_error)
    if "ocr_error_details" in st.session_state:
        details = st.session_state.ocr_error_details
        if isinstance(details, tuple) and len(details) == 2:
            st.subheader("Process STDOUT:")
            st.text_area("stdout", details[0], height=150, key="stdout_err")
            st.subheader("Process STDERR:")
            st.text_area("stderr", details[1], height=150, key="stderr_err")
        else:
            st.text_area("Error Details", str(details), height=100)
elif st.session_state.get("ocr_running"):
    st.info("OCR is running. Please wait...")
