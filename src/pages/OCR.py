import st_autorefresh
import streamlit as st
import subprocess
import os
import uuid
import pathlib
import shutil
import sys 
import json
import time
import signal

st.set_page_config(page_title="OLM OCR", layout="wide")


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from auth import check_token

check_token()

MAX_OCR_RUNTIME = 3600  # seconds (1 hour)

def start_ocr_job(input_file_path, ocr_sif_path, input_dir, workspace_dir, cont_input_dir, cont_workspace_dir, server_url=None):
    """
    Start olmocr.pipeline in the container using subprocess.Popen (non-blocking)
    and store process + paths in session_state.
    """
    cont_input_file = f"{cont_input_dir}/{input_file_path.name}"

    cmd = [
        "apptainer", "exec", "--nv",
        "--bind", f"{input_dir}:{cont_input_dir}",
        "--bind", f"{workspace_dir}:{cont_workspace_dir}",
        "--env", "HF_HOME=/workspace/hf_cache",
        "--env", "TRANSFORMERS_CACHE=/workspace/hf_cache",
        ocr_sif_path,
        "python",
        "-m", "olmocr.pipeline",
        cont_workspace_dir,
        "--markdown",
    ]

    # If you are using a persistent vLLM server:
    if server_url:
        cmd.extend(["--server", server_url])

    cmd.extend(["--pdfs", cont_input_file])

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    st.session_state.ocr_job = {
        "pid": proc.pid,
        "start_time": time.time(),
        "job_dir": str(workspace_dir.parent),  # JOB_DIR
        "input_dir": str(input_dir),
        "workspace_dir": str(workspace_dir),
        "input_file": str(input_file_path),
        "cmd": cmd,
    }
    st.session_state.ocr_proc = proc  # keep the Popen object itself


def check_ocr_job():
    """
    Check if the OCR job has produced results or exceeded timeout.
    If results exist, load them into session_state and clean up.
    """
    job = st.session_state.get("ocr_job")
    proc = st.session_state.get("ocr_proc")

    if not job or proc is None:
        return  # nothing to do

    workspace_dir = pathlib.Path(job["workspace_dir"])
    job_dir = pathlib.Path(job["job_dir"])
    input_file_path = pathlib.Path(job["input_file"])

    results_dir = workspace_dir / "results"
    jsonl_files = list(results_dir.glob("*.jsonl"))

    # If results exist, we consider the job successful regardless of proc state
    if jsonl_files:
        try:
            output_file_path = jsonl_files[0]
            json_file_name = output_file_path.name

            with open(output_file_path, "r", encoding="utf-8") as f:
                first_line = f.readline()
                data = json.loads(first_line)

            st.session_state.extracted_text = data.get("text")
            st.session_state.json_content = first_line
            st.session_state.txt_name = input_file_path.with_suffix(".txt").name
            st.session_state.json_name = json_file_name
            st.session_state.ocr_complete = True

        except Exception as e:
            st.session_state.ocr_error = f"Error parsing JSONL output: {e}"
            if "first_line" in locals():
                st.session_state.ocr_error_details = first_line

        # Regardless of success/failure parsing, stop the process & cleanup
        try:
            if proc.poll() is None:  # still running
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
        except Exception:
            pass

        # Optional: capture final stdout/stderr for debugging
        try:
            out, err = proc.communicate(timeout=1)
            st.session_state.ocr_debug = {
                "cmd": " ".join(job["cmd"]),
                "stdout": out,
                "stderr": err,
            }
        except Exception:
            pass

        # Cleanup job dir
        try:
            if job_dir.exists():
                shutil.rmtree(job_dir)
        except Exception as e:
            st.warning(f"Could not clean up {job_dir}. Error: {e}")

        # Clear job state
        st.session_state.pop("ocr_job", None)
        st.session_state.pop("ocr_proc", None)

        return

    # No JSONL yet -> check timeout
    elapsed = time.time() - job["start_time"]
    if elapsed > MAX_OCR_RUNTIME:
        st.session_state.ocr_error = (
            f"OCR job exceeded maximum runtime of {MAX_OCR_RUNTIME} seconds "
            "and was aborted."
        )
        # Try to kill the process
        try:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
        except Exception:
            pass

        # Capture whatever logs we can
        try:
            out, err = proc.communicate(timeout=1)
        except Exception:
            out, err = "", ""

        st.session_state.ocr_error_details = (out, err)

        # Cleanup
        try:
            if job_dir.exists():
                shutil.rmtree(job_dir)
        except Exception as e:
            st.warning(f"Could not clean up {job_dir}. Error: {e}")

        st.session_state.pop("ocr_job", None)
        st.session_state.pop("ocr_proc", None)


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
            OLM_OCR_PORT = 8003
            SERVER_URL = f"http://{SERVER_HOST}:{OLM_OCR_PORT}/v1"

            # --- 5. Run the OCR process. Non blocking ---

            start_ocr_job(
                input_file_path=input_file_path,
                ocr_sif_path=OCR_SIF_PATH,
                input_dir=INPUT_DIR,
                workspace_dir=WORKSPACE_DIR,
                cont_input_dir=CONT_INPUT_DIR,
                cont_workspace_dir=CONT_WORKSPACE_DIR,
                server_url=SERVER_URL,
            )

            st.info("OCR job started. Results will appear once ready.")

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

if "ocr_job" in st.session_state:
    if st.button("🔄 Check OCR Status"):
        check_ocr_job()

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