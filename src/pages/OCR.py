import streamlit as st
import subprocess
import os
import uuid
import pathlib
import shutil
import sys 
import json 

st.set_page_config(page_title="OLM OCR", layout="wide")


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from auth import check_token

check_token()



st.title("📄 Document OCR (using olmocr)")
st.markdown("Upload a PDF extract its text content.")

# --- 1. Get required paths from environment variables ---

# Get the path to the OCR container SIF file, set in script.sh.erb
OCR_SIF_PATH = os.environ.get("OCR_CONTAINER")

# Get the user's home directory, which is persistent and accessible
HOST_HOME = os.environ.get("HOME")

if not OCR_SIF_PATH:
    st.error(
        "**Configuration Error:** `OCR_CONTAINER` environment variable is not set."
    )
    st.stop()

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
    "Choose a PDF or image file", 
    type=["pdf"],
    on_change=clear_results  # <-- This clears old results on new upload
)

# --- 3. Show Button and Run Process ---
# Only show the button and warning AFTER a file has been uploaded
if uploaded_file is not None:
    
    st.info(f"File selected: **{uploaded_file.name}**")
    
    # Only show the warning if processing isn't already complete
    if "ocr_complete" not in st.session_state:
        st.warning("⚠️ **Heads up:** Processing can take 4-5 minutes per page. Please keep this tab open until the process is complete.")

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

            # --- 4. Construct the Apptainer command ---
            
            # Define container-internal paths
            CONT_INPUT_DIR = "/inputs"
            CONT_WORKSPACE_DIR = "/workspace"
            CONT_INPUT_FILE = f"{CONT_INPUT_DIR}/{uploaded_file.name}"
            SERVER_HOST = "localhost"

            cmd = [
                "apptainer", "exec", "--nv",
                "--bind", f"{INPUT_DIR}:{CONT_INPUT_DIR}",
                "--bind", f"{WORKSPACE_DIR}:{CONT_WORKSPACE_DIR}",
                "--env", "HF_HOME=/workspace/hf_cache",
                "--env", "TRANSFORMERS_CACHE=/workspace/hf_cache",
                OCR_SIF_PATH,
                "python", "-m", "olmocr.pipeline",
                CONT_WORKSPACE_DIR,
                "--markdown",
                "--server", f"http://{SERVER_HOST}:8000/v1",
                "--pdfs", CONT_INPUT_FILE
            ]

            # --- 5. Run the OCR process ---
            with st.spinner("Running OCR... This may take several minutes. Please wait."):
                
                result = subprocess.run(
                    cmd, 
                    capture_output=True, 
                    text=True, 
                    encoding='utf-8'
                )

            # --- 6. Process the result (Saving to Session State) ---
            
            results_dir = WORKSPACE_DIR / "results"
            
            if result.returncode == 0 and results_dir.exists():
                jsonl_files = list(results_dir.glob("*.jsonl"))
                
                if jsonl_files:
                    try:
                        output_file_path = jsonl_files[0]
                        json_file_name = output_file_path.name
                        
                        with open(output_file_path, 'r', encoding='utf-8') as f:
                            first_line = f.readline()
                            data = json.loads(first_line)
                        
                        # --- SAVE TO SESSION STATE ---
                        st.session_state.extracted_text = data.get("text")
                        st.session_state.json_content = first_line
                        st.session_state.txt_name = input_file_path.with_suffix(".txt").name
                        st.session_state.json_name = json_file_name
                        st.session_state.ocr_complete = True  # <-- Set success flag
                        
                    except Exception as e:
                        st.session_state.ocr_error = f"Error parsing JSONL output: {e}"
                        if 'first_line' in locals():
                            st.session_state.ocr_error_details = first_line
                else:
                    st.session_state.ocr_error = f"Process ran, but no .jsonl output file was found in {results_dir}"
                    st.session_state.ocr_error_details = (result.stdout, result.stderr)

            else:
                # Show error details if it failed
                st.session_state.ocr_error = f"OCR process failed. Return code: {result.returncode}"
                st.session_state.ocr_error_details = (result.stdout, result.stderr)

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