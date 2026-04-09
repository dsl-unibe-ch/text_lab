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
import cv2
import ollama
import io
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
    extract_html_table
)

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

@st.cache_resource(show_spinner=False)
def get_easyocr_reader(lang_code="en"):
    import easyocr
    return easyocr.Reader([lang_code], gpu=True)

@st.cache_resource(show_spinner=False)
def get_paddleocr_engine(lang_code="en"):
    from paddleocr import PaddleOCR
    return PaddleOCR(use_textline_orientation=True, lang=lang_code)

def clear_results(reset_running=False):
    keys_to_clear = [
        "ocr_complete", "extracted_text", "json_content", "txt_name", 
        "json_name", "ocr_error", "ocr_error_details", "ocr_preview_images", 
        "ocr_preview_page", "ocr_preview_engine", "ocr_zip_bytes",
        "batch_ocr_complete", "batch_ocr_zip_bytes"
    ]
    for key in keys_to_clear:
        if key in st.session_state: 
            del st.session_state[key]
    if reset_running: 
        st.session_state.ocr_running = False

# ==========================================
#              UI LOGIC
# ==========================================

workflow_mode = st.radio(
    "Workflow",
    ["Single Document OCR", "Batch OCR (ZIP)"],
    index=0,
    horizontal=True,
    help="Choose to process a single file or batch process a ZIP archive",
    on_change=clear_results,
    args=(True,)
)

st.divider()

if "ocr_running" not in st.session_state:
    st.session_state.ocr_running = False

# --- Engine Selection UI ---
col_eng, col_mode = st.columns([1, 1])
with col_eng:
    ocr_engine = st.selectbox(
        "OCR engine",
        ["EasyOCR", "PaddleOCR", "OlmOCR", "GLM-OCR"],
        index=0,
        on_change=clear_results,
        args=(True,),
        help="Select the OCR backend. GLM-OCR is best for complex layouts and tables.",
    )

# Show GLM mode selector or OCR language selector in the second column
glm_mode = "Text Recognition"
ocr_language = "en"
if ocr_engine == "GLM-OCR":
    with col_mode:
        glm_mode = st.selectbox(
            "GLM-OCR Mode",
            ["Text Recognition", "Table Recognition", "Figure Recognition"],
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


# ==========================================
#         SINGLE DOCUMENT MODE
# ==========================================
if workflow_mode == "Single Document OCR":
    st.markdown("Upload a **PDF** or **Image** to extract its text content and preview the results.")
    
    uploaded_file = st.file_uploader(
        "Choose a PDF or Image file",
        type=["pdf", "png", "jpg", "jpeg", "bmp", "tiff"],
        on_change=clear_results, 
        args=(True,),
    )

    if uploaded_file is not None:
        if st.session_state.ocr_running:
            st.warning("⏳ OCR is currently running. The button is disabled until completion.")

        if st.button("Run OCR", disabled=st.session_state.ocr_running):
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
                        ocr = get_paddleocr_engine(ocr_language)
                        for idx, img_path in enumerate(image_paths, start=1):
                            page_preds = ocr.predict(str(img_path))
                            compact_preds = [compact_paddle_prediction(pred) for pred in page_preds]
                            
                            page_text_lines = []
                            for pred in compact_preds:
                                page_text_lines.extend(pred.get("rec_texts", []))
                            page_text = "\n".join([line for line in page_text_lines if line])
                            
                            ocr_results.append({"page": idx, "text": page_text, "raw": compact_preds})
                            
                            pl, pr = render_paddle_preview(img_path, page_preds)
                            if pl and pr: preview_images.append((pl, pr))
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
                
                # Robust, aggressive cleanup
                if JOB_DIR.exists():
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
                        shutil.rmtree(JOB_DIR, onexc=handle_remove_readonly)
                    except Exception as e:
                        subprocess.run(["rm", "-rf", str(JOB_DIR)], check=False)

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
            if c_prev.button("⬅ Previous", disabled=current_page <= 0):
                st.session_state.ocr_preview_page -= 1
                st.rerun()
            with c_info:
                st.caption(f"Page {current_page + 1} of {len(preview_images)}")
            if c_next.button("Next ➡", disabled=current_page >= len(preview_images) - 1):
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


# ==========================================
#         BATCH DOCUMENT MODE (ZIP)
# ==========================================
elif workflow_mode == "Batch OCR (ZIP)":
    st.markdown("Upload a **ZIP archive** containing multiple PDFs or Images. They will be processed and returned as a single organized ZIP.")
    
    batch_zip = st.file_uploader(
        "Upload ZIP file",
        type=["zip"],
        on_change=clear_results, 
        args=(True,),
        key="batch_zip_upload"
    )

    if batch_zip is not None:
        if st.session_state.ocr_running:
            st.warning("⏳ OCR is currently running. The button is disabled until completion.")

        if st.button("Run Batch OCR", disabled=st.session_state.ocr_running):
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
                reader, paddle_ocr = None, None
                if ocr_engine == "EasyOCR":
                    reader = get_easyocr_reader(ocr_language)
                elif ocr_engine == "PaddleOCR":
                    paddle_ocr = get_paddleocr_engine(ocr_language)
                
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
                        for p_idx, img_path in enumerate(image_paths, start=1):
                            if ocr_engine == "EasyOCR":
                                page_res = reader.readtext(str(img_path), detail=1, paragraph=True)
                                page_text = "\n".join([r[1] for r in page_res])
                                ocr_results.append({"page": p_idx, "text": page_text, "raw": page_res})
                            
                            elif ocr_engine == "PaddleOCR":
                                page_preds = paddle_ocr.predict(str(img_path))
                                compact_preds = [compact_paddle_prediction(pred) for pred in page_preds]
                                page_text_lines = []
                                for pred in compact_preds:
                                    page_text_lines.extend(pred.get("rec_texts", []))
                                page_text = "\n".join([line for line in page_text_lines if line])
                                ocr_results.append({"page": p_idx, "text": page_text, "raw": compact_preds})
                            
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
                
                # Robust cleanup
                if JOB_DIR.exists():
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
                        shutil.rmtree(JOB_DIR, onexc=handle_remove_readonly)
                    except Exception as e:
                        subprocess.run(["rm", "-rf", str(JOB_DIR)], check=False)

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