import io
import zipfile
import streamlit as st
import tempfile
import json
import sys
import os
import subprocess
from datetime import datetime

st.set_page_config(page_title="OLM OCR", layout="wide")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from auth import check_token
check_token()

def _run_apptainer_ocr_batch(workdir: str, rel_pdf_paths: list[str], container: str):
    """
    Runs a single Apptainer call to process multiple PDFs.
    Each rel path is relative to workdir and will be mounted under /localworkspace.
    Produces .md files next to each PDF.
    """
    if not rel_pdf_paths:
        return

    cmd = [
        "apptainer", "exec", "--nv", "--writable-tmpfs", "--no-home",
        "--pwd", "root",
        "--bind", f"{workdir}:/localworkspace",
        container,
        "python", "-m", "olmocr.pipeline", "/localworkspace",
        "--model", "/opt/models/olmocr-7b",
        "--markdown", "--pdfs",
    ] + [f"/localworkspace/{p}" for p in rel_pdf_paths]

    # One batch run for all PDFs
    subprocess.run(cmd, check=True, capture_output=True)


def run_ocr():
    st.title("OLM OCR (single PDF or ZIP)")

    st.write(
        "Upload **exactly one** file:\n"
        "• **PDF** → returns one `.md`\n"
        "• **ZIP** → preserves the full folder tree and files; OCRs **all PDFs in one go**, "
        "adding `.md` files next to each PDF, then returns the entire tree as a ZIP"
    )

    upload = st.file_uploader(
        "Upload one PDF or one ZIP",
        type=["pdf", "zip"],
        accept_multiple_files=False,
    )

    # Output placeholders
    if "single_md_bytes" not in st.session_state:
        st.session_state["single_md_bytes"] = None
        st.session_state["single_md_name"] = None
    if "tree_zip_bytes" not in st.session_state:
        st.session_state["tree_zip_bytes"] = None
        st.session_state["tree_zip_name"] = None

    if st.button("Run OCR"):
        if not upload:
            st.warning("Please upload a single PDF or a single ZIP.")
            return

        suffix = os.path.splitext(upload.name)[1].lower()
        if suffix not in (".pdf", ".zip"):
            st.error("Unsupported file type. Please upload one .pdf or one .zip.")
            return

        container = os.getenv("OCR_CONTAINER")
        if not container:
            st.error("OCR_CONTAINER environment variable is not set.")
            return

        if suffix == ".pdf":
            # ---- Single PDF branch
            with st.spinner("Performing OCR on the PDF..."):
                with tempfile.TemporaryDirectory() as workdir:
                    pdf_name = upload.name
                    pdf_path = os.path.join(workdir, pdf_name)
                    with open(pdf_path, "wb") as f:
                        f.write(upload.read())

                    rel_pdf = os.path.relpath(pdf_path, workdir)

                    try:
                        _run_apptainer_ocr_batch(workdir, [rel_pdf], container)
                    except subprocess.CalledProcessError as e:
                        st.error(
                            f"OCR failed for {pdf_name}:\n"
                            f"Return code: {e.returncode}\n"
                            f"stderr:\n{e.stderr.decode(errors='ignore')}"
                        )
                        return
                    except json.JSONDecodeError:
                        st.error("Failed to parse OCR output.")
                        return

                    md_path = os.path.splitext(pdf_path)[0] + ".md"
                    if not os.path.exists(md_path):
                        st.error(f"OCR result not found: expected {os.path.basename(md_path)}")
                        return

                    with open(md_path, "rb") as f:
                        md_bytes = f.read()

                    base = os.path.splitext(os.path.basename(pdf_name))[0]
                    md_name = f"{base}.md"

                    st.session_state["single_md_bytes"] = md_bytes
                    st.session_state["single_md_name"] = md_name

            st.success("OCR complete. Your Markdown file is ready below.")

        else:
            # ---- ZIP branch (batch all PDFs in one command)
            with st.spinner("Unpacking ZIP and performing OCR on PDF..."):
                with tempfile.TemporaryDirectory() as workdir:
                    # 1) Extract the uploaded ZIP into workdir
                    zbytes = upload.read()
                    try:
                        with zipfile.ZipFile(io.BytesIO(zbytes)) as zf:
                            zf.extractall(workdir)
                    except zipfile.BadZipFile:
                        st.error("The uploaded file is not a valid ZIP.")
                        return

                    # 2) Gather all PDFs recursively (abs + rel paths)
                    pdf_abs_paths = []
                    pdf_rel_paths = []
                    for root, _, files in os.walk(workdir):
                        for f in files:
                            if f.lower().endswith(".pdf"):
                                abs_p = os.path.join(root, f)
                                rel_p = os.path.relpath(abs_p, workdir)
                                pdf_abs_paths.append(abs_p)
                                pdf_rel_paths.append(rel_p)

                    if not pdf_rel_paths:
                        st.warning("No PDFs found inside the ZIP.")
                        return

                    # 3) Single batch OCR call
                    try:
                        _run_apptainer_ocr_batch(workdir, sorted(pdf_rel_paths), container)
                    except subprocess.CalledProcessError as e:
                        st.error(
                            "OCR failed during the batch run.\n"
                            f"Return code: {e.returncode}\n"
                            f"stderr:\n{e.stderr.decode(errors='ignore')}"
                        )
                        return
                    except json.JSONDecodeError:
                        st.error("Failed to parse OCR output.")
                        return

                    # 4) Verify every PDF got a sibling .md
                    missing = []
                    for abs_pdf in pdf_abs_paths:
                        md_abs = os.path.splitext(abs_pdf)[0] + ".md"
                        if not os.path.exists(md_abs):
                            missing.append(os.path.relpath(abs_pdf, workdir))
                    if missing:
                        st.error("Some markdown outputs were not created:\n" + "\n".join(missing))
                        return

                    # 5) Re-zip the entire workdir (original files + new .md)
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    upload_name = os.path.splitext(os.path.basename(upload.name))[0]
                    out_zip_name = f"{upload_name}_{ts}.zip"
                    buf = io.BytesIO()
                    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                        for root, _, files in os.walk(workdir):
                            for f in files:
                                abs_path = os.path.join(root, f)
                                arcname = os.path.relpath(abs_path, workdir)
                                zf.write(abs_path, arcname=arcname)
                    buf.seek(0)

                    st.session_state["tree_zip_bytes"] = buf.getvalue()
                    st.session_state["tree_zip_name"] = out_zip_name

            st.success("Done! Your full folder tree (with added .md files) is ready below.")

    # ---- Download areas
    if st.session_state.get("single_md_bytes"):
        st.subheader("Download Markdown")
        st.download_button(
            label=f"Download {st.session_state['single_md_name']}",
            data=st.session_state["single_md_bytes"],
            file_name={st.session_state['single_md_name']},
            mime="text/markdown",
        )

    if st.session_state.get("tree_zip_bytes"):
        st.subheader("Download ZIP of Entire Tree")
        st.download_button(
            label=f"Download ZIP ({st.session_state['tree_zip_name']})",
            data=st.session_state["tree_zip_bytes"],
            file_name=st.session_state["tree_zip_name"],
            mime="application/zip",
        )


def main():
    run_ocr()

if __name__ == "__main__":
    main()
