import shutil

import streamlit as st
import tempfile
import json
import sys
import os
import subprocess

st.set_page_config(page_title="OLM OCR", layout="wide")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from auth import check_token

check_token()

def run_ocr():
    st.title("OLM OCR")

    # File uploader
    document_file = st.file_uploader(
        "Upload an image file",
        type=["pdf","png","jpg","jpeg"]
    )

    # Initialize storage for the last OCR result
    if "ocr_result" not in st.session_state:
        st.session_state["ocr_result"] = None

    # Read button
    if st.button("Read"):
        if document_file is None:
            st.warning("Please upload a document file first.")
        else:
            with st.spinner("Loading Whisper model and performing OCR. This might take a while. Please don't close or reload this page."):
                # Write the uploaded file to a temp file for OLM OCR
                document_extension = os.path.splitext(document_file.name)[1]
                bytes = document_file.read()

                # Perfoming OCR
                with tempfile.NamedTemporaryFile(suffix=document_extension, delete=False) as tmp:
                    tmp.write(bytes)
                    tmp.flush()

                    try:
                        # Execute OCR using apptainer
                        olmocr_container = os.getenv("OCR_CONTAINER")

                        if not olmocr_container:
                            st.error("OCR_CONTAINER environment variable is not set")
                            return
                        else:
                            st.info(f"Using OCR_CONTAINER: {olmocr_container}")

                        cmd = ["apptainer", "exec", "--nv", olmocr_container, "python3", "-m", "olmocr.pipeline",
                              tempfile.gettempdir(), "--markdown", "--pdfs", tmp.name]

                        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                        print(result.stdout)

                        md_filename = os.path.splitext(tmp.name)[0] + ".md"
                        if not os.path.exists(md_filename):
                            raise FileNotFoundError(f"OCR result file not found: {md_filename}")

                    except subprocess.CalledProcessError as e:
                        st.error(f"OCR process failed: {e.stderr}")
                        return
                    except json.JSONDecodeError:
                        st.error("Failed to parse OCR output")
                        return

            st.success("OCR complete!")
            st.session_state["ocr_result"] = md_filename

    # If we have a stored result, show download options
    if st.session_state["ocr_result"] is not None:
        result_file = st.session_state["ocr_result"]
        with open(result_file, "r") as f:
            text = f.read()

        st.subheader("Download your text")
        st.download_button(
            label=f"Download .md",
            data=text,
            file_name=f"result.md",
            mime="text/markdown",
        )

def main():
    run_ocr()

if __name__ == "__main__":
    main()
