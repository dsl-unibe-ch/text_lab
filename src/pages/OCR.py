import streamlit as st
import subprocess
import os
import uuid
import pathlib
import shutil
import sys 
import json 

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

# --- NEW: Function to clear results when a new file is uploaded ---
def clear_results():
    """Clears all OCR results from the session state."""
    for key in ["ocr_complete", "extracted_text", "json_content", "txt_name", "json_name", "ocr_error", "ocr_error_details"]:
        if key in st.session_state:
            del st.session_state[key]

# --- 2. Create File Uploader ---

uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type=["pdf"],
    on_change=clear_results  # <-- This clears old results on new upload
)

ocr_engine = st.selectbox(
    "OCR engine",
    ["easyocr", "paddleocr", "olmocr"],
    index=0,
    help="easyocr is the default. olmocr uses the local pipeline inside this container.",
)

# --- 3. Show Button and Run Process ---
# Only show the button and warning AFTER a file has been uploaded
if uploaded_file is not None:
    
    st.info(f"File selected: **{uploaded_file.name}**")
    
    # Only show the warning if processing isn't already complete
    # if "ocr_complete" not in st.session_state:
    #     st.warning("⚠️ **Heads up:** Processing can take 4-5 minutes per page. Please keep this tab open until the process is complete.")

    # Create the button. The logic below will only run if it's clicked.
    if st.button("🚀 Run OCR"):
    
        # Clear any previous state before starting
        clear_results()
    
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
                    "--pdfs", CONT_INPUT_FILE
                ]

                with st.spinner("Running OlmOCR... This may take several minutes. Please wait."):
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
                    import easyocr
                    reader = easyocr.Reader(["en"], gpu=True)
                    ocr_results = []
                    for idx, img_path in enumerate(image_paths, start=1):
                        page_res = reader.readtext(str(img_path), detail=1, paragraph=True)
                        page_text = "\n".join([r[1] for r in page_res])
                        ocr_results.append({"page": idx, "text": page_text, "raw": page_res})

                else:
                    from paddleocr import PaddleOCR
                    ocr = PaddleOCR(use_angle_cls=True, lang="en")
                    ocr_results = []
                    for idx, img_path in enumerate(image_paths, start=1):
                        page_res = ocr.ocr(str(img_path), cls=True)
                        page_text = "\n".join([line[1][0] for line in page_res[0]])
                        ocr_results.append({"page": idx, "text": page_text, "raw": page_res})

                # Write per-page outputs + combined text
                all_text = []
                for item in ocr_results:
                    page = item["page"]
                    page_text = item["text"]
                    all_text.append(page_text)
                    (results_dir / f"page_{page:04d}.txt").write_text(page_text, encoding="utf-8")
                    (results_dir / f"page_{page:04d}.json").write_text(
                        json.dumps(item, ensure_ascii=False), encoding="utf-8"
                    )

                combined_text = "\n\n".join(all_text)
                st.session_state.extracted_text = combined_text
                st.session_state.json_content = json.dumps(ocr_results, ensure_ascii=False)
                st.session_state.txt_name = input_file_path.with_suffix(".txt").name
                st.session_state.json_name = input_file_path.with_suffix(".json").name

            # Create ZIP of all outputs
            zip_path = shutil.make_archive(str(WORKSPACE_DIR / "ocr_results"), "zip", results_dir)
            st.session_state.ocr_zip_bytes = pathlib.Path(zip_path).read_bytes()
            st.session_state.ocr_complete = True

        except Exception as e:
            st.session_state.ocr_error = f"An unexpected error occurred: {e}"
            st.exception(e)
        
        finally:
            # --- 7. Clean up the temporary job directory ---
            if JOB_DIR.exists():
                try:
                    shutil.rmtree(JOB_DIR)
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
