import os

# FORCE these to 1 to prevent OpenBLAS crashes with Paddle/PyTorch
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

os.environ["VLLM_GPU_MEMORY_UTILIZATION"] = "0.6"
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
# PaddleX uses PADDLE_PDX_CACHE_HOME and otherwise falls back to ~/.paddlex.
os.environ.setdefault("PADDLE_PDX_CACHE_HOME", os.environ.get("PADDLEX_HOME", os.path.expanduser("~/.paddlex")))
os.environ.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "none")

import streamlit as st
import subprocess
import uuid
import pathlib
import shutil
import sys 
import json
import io
import numpy as np
import cv2
import ollama
import pandas as pd  # Added for table parsing

st.set_page_config(page_title="OCR", layout="wide")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from auth import check_token
check_token()

st.title("📄 Document OCR")
st.markdown("Upload a PDF and choose an OCR engine to extract its text content.")

# --- 1. Get required paths from environment variables ---
HOST_HOME = os.environ.get("HOME")

if not HOST_HOME:
    st.error("**Configuration Error:** `HOME` environment variable is not set.")
    st.stop()

OCR_JOBS_BASE_DIR = pathlib.Path(HOST_HOME) / "ondemand_text_lab_ocr_jobs"
OLMOCR_GPU_MEMORY_UTILIZATION = os.environ.get("OLMOCR_GPU_MEMORY_UTILIZATION", "0.6")

# --- Helper Functions ---

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
    return str(value)

def extract_texts_from_ocr_payload(payload):
    texts = []
    if isinstance(payload, str):
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
                texts.extend(extract_texts_from_ocr_payload(val))
        return texts
    if isinstance(payload, (list, tuple)):
        for item in payload:
            texts.extend(extract_texts_from_ocr_payload(item))
        return texts
    return texts

def compact_paddle_prediction(pred):
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
    compact["rec_texts"] = list(dict.fromkeys(rec_texts))
    return compact

def _poly_to_int_points(poly):
    arr = np.array(poly)
    if arr.size == 0: return None
    arr = arr.astype(np.float32).reshape(-1)
    if arr.size == 4:
        x1, y1, x2, y2 = arr.tolist()
        arr = np.array([x1, y1, x2, y1, x2, y2, x1, y2], dtype=np.float32)
    if arr.size < 8: return None
    if arr.size % 2 != 0: arr = arr[:-1]
    pts = arr.reshape(-1, 2).astype(np.int32)
    if pts.shape[0] < 3: return None
    return pts

def _encode_png(image):
    ok, encoded = cv2.imencode(".png", image)
    return encoded.tobytes() if ok else None

def _to_png_bytes(image_like):
    if image_like is None: return None
    if isinstance(image_like, (bytes, bytearray)): return bytes(image_like)
    if isinstance(image_like, np.ndarray): return _encode_png(image_like)
    if hasattr(image_like, "save"):
        buf = io.BytesIO()
        image_like.save(buf, format="PNG")
        return buf.getvalue()
    return None

def _pick_paddle_vis_image(pred):
    if not hasattr(pred, "img"): return None
    payload = pred.img
    if isinstance(payload, dict):
        for key in ("ocr_res_img", "overall_ocr_res", "layout_det_res"):
            if key in payload:
                out = _to_png_bytes(payload[key])
                if out is not None: return out
        for val in payload.values():
            out = _to_png_bytes(val)
            if out is not None: return out
        return None
    return _to_png_bytes(payload)

def _extract_paddle_core(raw):
    if isinstance(raw, dict) and isinstance(raw.get("res"), dict):
        return raw["res"]
    return raw if isinstance(raw, dict) else {}

def _draw_text_canvas(image_shape, items):
    h, w = image_shape[:2]
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX

    def _wrap_line_to_width(text, max_width, font_scale, thickness):
        words = text.split(" ")
        if not words: return [""]
        lines = []
        current = words[0]
        for word in words[1:]:
            candidate = f"{current} {word}"
            candidate_w = cv2.getTextSize(candidate, font, font_scale, thickness)[0][0]
            if candidate_w <= max_width:
                current = candidate
            else:
                lines.append(current)
                current = word
        lines.append(current)
        return lines

    def _fit_text_to_box(text, bw, bh):
        text = str(text).replace("\r\n", "\n").replace("\r", "\n")
        if not text.strip(): return [], 0.4, 1, 12
        for font_scale in (0.8, 0.7, 0.6, 0.5, 0.45, 0.4, 0.35):
            thickness = 1
            line_h = cv2.getTextSize("Ag", font, font_scale, thickness)[0][1] + 4
            max_width = max(8, bw - 4)
            max_lines = max(1, bh // max(1, line_h))
            wrapped = []
            for raw_line in text.split("\n"):
                raw_line = raw_line.strip()
                if not raw_line:
                    wrapped.append("")
                    continue
                wrapped.extend(_wrap_line_to_width(raw_line, max_width, font_scale, thickness))
            if len(wrapped) <= max_lines:
                return wrapped, font_scale, thickness, line_h
        font_scale = 0.35
        thickness = 1
        line_h = cv2.getTextSize("Ag", font, font_scale, thickness)[0][1] + 4
        max_width = max(8, bw - 4)
        max_lines = max(1, bh // max(1, line_h))
        wrapped = []
        for raw_line in text.split("\n"):
            raw_line = raw_line.strip()
            if not raw_line:
                wrapped.append("")
                continue
            wrapped.extend(_wrap_line_to_width(raw_line, max_width, font_scale, thickness))
        wrapped = wrapped[:max_lines]
        if wrapped:
            last = wrapped[-1]
            while last and cv2.getTextSize(last + "...", font, font_scale, thickness)[0][0] > max_width:
                last = last[:-1]
            wrapped[-1] = (last + "...") if last else "..."
        return wrapped, font_scale, thickness, line_h

    for pts, text in items:
        if pts is None: continue
        cv2.polylines(canvas, [pts], isClosed=True, color=(210, 210, 210), thickness=1)
        if not text: continue
        x, y, bw, bh = cv2.boundingRect(pts)
        lines, font_scale, thickness, line_h = _fit_text_to_box(text, bw, bh)
        if not lines: continue
        y_cursor = max(12, y + line_h)
        y_limit = min(h - 2, y + bh - 2)
        for line in lines:
            if y_cursor > y_limit: break
            cv2.putText(canvas, line, (max(0, x + 2), min(h - 4, y_cursor)), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
            y_cursor += line_h
    return canvas

def _is_number_list(seq):
    if not isinstance(seq, (list, tuple)) or not seq: return False
    return all(isinstance(v, (int, float, np.integer, np.floating)) for v in seq)

def _extract_polys_from_ocr_payload(payload):
    polys = []
    poly_keys = {"rec_polys", "dt_polys", "polys", "boxes", "rec_boxes"}
    if isinstance(payload, dict):
        for key, val in payload.items():
            key_l = str(key).lower()
            if key_l in poly_keys:
                polys.extend(_extract_polys_from_ocr_payload(val))
                continue
            polys.extend(_extract_polys_from_ocr_payload(val))
        return polys
    if isinstance(payload, (list, tuple)):
        if _is_number_list(payload):
            pts = _poly_to_int_points(payload)
            if pts is not None: polys.append(pts)
            return polys
        if payload and isinstance(payload[0], (list, tuple)):
            if all(isinstance(p, (list, tuple)) and len(p) >= 2 for p in payload):
                pts = _poly_to_int_points(payload)
                if pts is not None: polys.append(pts)
                return polys
        for item in payload:
            polys.extend(_extract_polys_from_ocr_payload(item))
        return polys
    return polys

def render_easyocr_preview(image_path, page_res):
    image = cv2.imread(str(image_path))
    if image is None: return None, None
    text_items = []
    for item in page_res:
        if not isinstance(item, (list, tuple)) or len(item) < 2: continue
        pts = _poly_to_int_points(item[0])
        if pts is None: continue
        text = str(item[1]).strip()
        text_items.append((pts, text))
        cv2.polylines(image, [pts], isClosed=True, color=(0, 200, 0), thickness=2)
    text_canvas = _draw_text_canvas(image.shape, text_items)
    return _encode_png(image), _encode_png(text_canvas)

def render_paddle_preview(image_path, page_preds):
    image = cv2.imread(str(image_path))
    if image is None: return None, None
    left_png = None
    text_items = []
    for pred in page_preds:
        if left_png is None: left_png = _pick_paddle_vis_image(pred)
        if hasattr(pred, "json"): raw = pred.json
        elif hasattr(pred, "to_dict"): raw = pred.to_dict()
        else: raw = pred
        raw = make_json_serializable(raw)
        core = _extract_paddle_core(raw)
        texts = core.get("rec_texts")
        if not isinstance(texts, list): texts = extract_texts_from_ocr_payload(core)
        polys_src = core.get("rec_polys") or core.get("dt_polys") or core.get("rec_boxes") or core.get("boxes")
        if polys_src is None: polys = _extract_polys_from_ocr_payload(core)
        else: polys = _extract_polys_from_ocr_payload(polys_src)
        for i, poly in enumerate(polys):
            pts = _poly_to_int_points(poly)
            if pts is None: continue
            text = str(texts[i]).strip() if i < len(texts) else ""
            text_items.append((pts, text))
    text_canvas = _draw_text_canvas(image.shape, text_items)
    if left_png is None:
        for pts, _ in text_items:
            cv2.polylines(image, [pts], isClosed=True, color=(0, 200, 0), thickness=2)
        left_png = _encode_png(image)
    return left_png, _encode_png(text_canvas)

@st.cache_resource(show_spinner=False)
def get_easyocr_reader():
    import easyocr
    return easyocr.Reader(["en"], gpu=True)

@st.cache_resource(show_spinner=False)
def get_paddleocr_engine():
    from paddleocr import PaddleOCR
    return PaddleOCR(use_textline_orientation=True, lang="en")

def clear_results(reset_running=False):
    for key in ["ocr_complete", "extracted_text", "json_content", "txt_name", "json_name", "ocr_error", "ocr_error_details", "ocr_preview_images", "ocr_preview_page", "ocr_preview_engine", "ocr_zip_bytes"]:
        if key in st.session_state: del st.session_state[key]
    if reset_running: st.session_state.ocr_running = False

# --- UI START ---

uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type=["pdf"],
    on_change=clear_results, 
    args=(True,),
)

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

# Show Mode selector only for GLM-OCR
glm_mode = "Text Recognition"
if ocr_engine == "GLM-OCR":
    with col_mode:
        glm_mode = st.selectbox(
            "GLM-OCR Mode",
            ["Text Recognition", "Table Recognition", "Figure Recognition"],
            help="Choose what specific aspect of the document you want to extract."
        )

# --- Button Logic ---
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
                
                # Robustly handle API response structure
                models_list = []
                if isinstance(models_dict, dict):
                    models_list = models_dict.get("models") or []
                
                local_models = [str(m.get('name')) for m in models_list if m.get('name')]
                
                is_present = any(model_name in m or m in model_name for m in local_models)
                
                if not is_present:
                    with st.spinner(f"📥 Pulling model '{model_name}'. This happens once and may take a few minutes..."):
                        ollama.pull(model_name)
                    st.success(f"Model {model_name} ready.")
            except Exception as e:
                # If checking fails, attempt pulling anyway
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

            input_file_path = INPUT_DIR / uploaded_file.name
            with open(input_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            results_dir = WORKSPACE_DIR / "results"
            results_dir.mkdir(parents=True, exist_ok=True)

            # --- OLMOCR PATH ---
            if ocr_engine == "OlmOCR":
                CONT_INPUT_FILE = str(input_file_path)
                CONT_WORKSPACE_DIR = str(WORKSPACE_DIR)
                cmd = [
                    sys.executable, "-m", "olmocr.pipeline",
                    CONT_WORKSPACE_DIR,
                    "--markdown",
                    "--pdfs", CONT_INPUT_FILE,
                    "--gpu-memory-utilization", OLMOCR_GPU_MEMORY_UTILIZATION,
                ]
                with st.spinner("Running OlmOCR..."):
                    result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')

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

                # PDF -> Images
                prefix = tmp_img_dir / "page"
                subprocess.run(
                    ["pdftoppm", "-png", str(input_file_path), str(prefix)],
                    check=True, capture_output=True, text=True, encoding="utf-8",
                )
                image_paths = sorted(tmp_img_dir.glob("page-*.png"))
                if not image_paths:
                    raise RuntimeError("No images generated from PDF.")

                ocr_results = []
                progress_bar = st.progress(0.0, text=f"Running {ocr_engine}...")

                # --- 1. EasyOCR ---
                if ocr_engine == "EasyOCR":
                    reader = get_easyocr_reader()
                    for idx, img_path in enumerate(image_paths, start=1):
                        page_res = reader.readtext(str(img_path), detail=1, paragraph=True)
                        page_text = "\n".join([r[1] for r in page_res])
                        ocr_results.append({"page": idx, "text": page_text, "raw": page_res})
                        
                        pl, pr = render_easyocr_preview(img_path, page_res)
                        if pl and pr: preview_images.append((pl, pr))
                        progress_bar.progress(idx / len(image_paths))

                # --- 2. PaddleOCR ---
                elif ocr_engine == "PaddleOCR":
                    ocr = get_paddleocr_engine()
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
                        progress_bar.progress(idx / len(image_paths))

                # --- 3. GLM-OCR (Updated with fixes) ---
                elif ocr_engine == "GLM-OCR":
                    for idx, img_path in enumerate(image_paths, start=1):
                        # 1. Load image using OpenCV
                        img = cv2.imread(str(img_path))
                        
                        # 2. Resize if too large (Prevents Memory/Tensor Overflow on H200)
                        max_dim = 2048
                        h, w = img.shape[:2]
                        if h > max_dim or w > max_dim:
                            scale = max_dim / max(h, w)
                            new_w = int(w * scale)
                            new_h = int(h * scale)
                            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                        
                        # 3. Encode back to bytes for Ollama and UI
                        success, encoded_img = cv2.imencode('.png', img)
                        if not success:
                            ocr_results.append({"page": idx, "text": "[Error encoding image]", "raw": {}})
                            continue
                        
                        img_bytes = encoded_img.tobytes()
                        
                        # 4. Call Ollama with limits
                        try:
                            response = ollama.chat(
                                model='glm-ocr:latest',
                                messages=[{
                                    'role': 'user',
                                    'content': glm_mode, 
                                    'images': [img_bytes]
                                }],
                                # Limit context to 8k to prevent massive VRAM allocation crash
                                options={
                                    'temperature': 0,
                                    'num_ctx': 8192 
                                }
                            )
                            page_text = response.get('message', {}).get('content', '')
                        except Exception as e:
                            page_text = f"[Error processing page {idx}: {str(e)}]"

                        ocr_results.append({"page": idx, "text": page_text, "raw": {"content": page_text}})
                        
                        # Fix: Store pure bytes for Streamlit display (not numpy array)
                        preview_images.append(img_bytes) 
                        
                        progress_bar.progress(idx / len(image_paths))

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
            if JOB_DIR.exists():
                subprocess.Popen(["rm", "-rf", str(JOB_DIR)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# --- RESULTS DISPLAY ---
if "ocr_complete" in st.session_state:
    st.success("🎉 OCR complete!")
    
    # 1. HTML Table Detection & CSV Conversion
    is_html_table = "<table>" in st.session_state.extracted_text or '<table class="' in st.session_state.extracted_text
    
    if is_html_table:
        st.markdown("### 📊 Detected Table")
        try:
            # We use io.StringIO to wrap the string as a file-like object for pandas
            dfs = pd.read_html(io.StringIO(st.session_state.extracted_text))
            
            if dfs:
                df = dfs[0]  # Get the first table found
                st.dataframe(df, use_container_width=True)
                
                # Add specific download button for the table data
                csv_data = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Table as CSV",
                    data=csv_data,
                    file_name=f"{st.session_state.txt_name.replace('.txt', '')}_table.csv",
                    mime="text/csv",
                    type="primary"
                )
        except Exception as e:
            st.warning(f"Could not convert to CSV automatically: {e}")
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
        
        # Pagination
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

        # Logic for GLM (Single Image) vs Easy/Paddle (Side-by-Side)
        if isinstance(current_preview, (list, tuple)) and len(current_preview) >= 2:
            left_img, right_img = current_preview[0], current_preview[1]
            cl, cr = st.columns(2)
            with cl:
                st.caption("Detected boxes")
                st.image(left_img, use_container_width=True)
            with cr:
                st.caption("OCR Layout")
                st.image(right_img, use_container_width=True)
        else:
            # Single image path (GLM-OCR)
            st.image(current_preview, caption="Original Document", use_container_width=True)

elif "ocr_error" in st.session_state:
    st.error(st.session_state.ocr_error)
    if "ocr_error_details" in st.session_state:
        st.text_area("Error Details", str(st.session_state.ocr_error_details), height=150)
elif st.session_state.get("ocr_running"):
    st.info("OCR is running. Please wait...")