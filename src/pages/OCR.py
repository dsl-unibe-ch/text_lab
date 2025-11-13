import streamlit as st
import subprocess
import os
import uuid
import pathlib
import shutil
import sys 
import json 

st.set_page_config(page_title="OLM OCR", layout="wide")

# --- Authentication ---
# Add the parent directory to the path to find the auth module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
try:
    from auth import check_token
    check_token()
except ImportError:
    st.error("Could not import authentication module.")
    st.stop()
# -------------------------------------

st.title("📄 Document OCR (using olmocr)")
st.markdown("Upload a PDF or image file to extract its text content.")

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


# --- 2. Create File Uploader ---

uploaded_file = st.file_uploader(
    "Choose a PDF or image file", type=["pdf", "png", "jpg", "jpeg"]
)

# --- 3. Show Button and Run Process ---
# Only show the button and warning AFTER a file has been uploaded
if uploaded_file is not None:
    
    st.info(f"File selected: **{uploaded_file.name}**")
    st.warning("⚠️ **Heads up:** Processing can take 4-5 minutes per page. Please keep this tab open until the process is complete.")

    # Create the button. The logic below will only run if it's clicked.
    if st.button("🚀 Run OCR"):
    
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

            cmd = [
                "apptainer", "exec", "--nv",
                
                # Bind the host directories to the container paths
                "--bind", f"{INPUT_DIR}:{CONT_INPUT_DIR}",
                "--bind", f"{WORKSPACE_DIR}:{CONT_WORKSPACE_DIR}",
                
                # Override BOTH cache variables to point to our writable workspace
                "--env", "HF_HOME=/workspace/hf_cache",
                "--env", "TRANSFORMERS_CACHE=/workspace/hf_cache",
                
                OCR_SIF_PATH,
                
                # The command to run inside the olmocr container
                "python", "-m", "olmocr.pipeline",
                CONT_WORKSPACE_DIR,  # The output workspace path
                "--markdown", # <-- We leave this, as it does no harm
                "--pdfs", CONT_INPUT_FILE # The input file path
            ]

            # --- 5. Run the OCR process ---
            with st.spinner("Running OCR... This may take several minutes. Please wait."):
                
                # st.code(" ".join(cmd), language="bash") # <-- HIDDEN as requested
                
                result = subprocess.run(
                    cmd, 
                    capture_output=True, 
                    text=True, 
                    encoding='utf-8'
                )

            # --- 6. Process the result (MODIFIED) ---
            
            results_dir = WORKSPACE_DIR / "results"
            extracted_text = None
            jsonl_files = []

            if result.returncode == 0 and results_dir.exists():
                # Find any .jsonl file in the results directory
                jsonl_files = list(results_dir.glob("*.jsonl"))
                
                if jsonl_files:
                    try:
                        output_file_path = jsonl_files[0]
                        # Read the first line of the .jsonl file
                        with open(output_file_path, 'r', encoding='utf-8') as f:
                            first_line = f.readline()
                            data = json.loads(first_line)
                            extracted_text = data.get("text")
                    except Exception as e:
                        st.error(f"Error parsing JSONL output: {e}")
                        st.text_area("JSONL Content", first_line, height=100)
                
            if extracted_text:
                st.success("🎉 OCR complete!")
                
                st.markdown("### Extracted Text")
                st.text_area(
                    "Result", 
                    extracted_text, 
                    height=400,
                    key="md_result"
                )
                
                # Create a name for the download file
                download_file_name = input_file_path.with_suffix(".txt").name
                st.download_button(
                    label="Download as .txt",
                    data=extracted_text,
                    file_name=download_file_name,
                    mime="text/plain",
                )

            else:
                # Show error details if it failed
                st.error(f"OCR process failed. Return code: {result.returncode}")
                if not results_dir.exists():
                    st.warning(f"Output directory was not found at: {results_dir}")
                elif not jsonl_files:
                    st.warning(f"Process ran, but no .jsonl output file was found in {results_dir}")
                
                st.subheader("Process STDOUT:")
                st.text_area("stdout", result.stdout, height=150, key="stdout_err")
                
                st.subheader("Process STDERR:")
                st.text_area("stderr", result.stderr, height=150, key="stderr_err")

        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            st.exception(e)
        
        finally:
            # --- 7. Clean up the temporary job directory ---
            if JOB_DIR.exists():
                try:
                    shutil.rmtree(JOB_DIR)
                    # st.info(f"Cleaned up temporary job directory: {JOB_DIR}") # <-- HIDDEN from user
                except Exception as e:
                    st.warning(f"Could not clean up {JOB_DIR}. Error: {e}")