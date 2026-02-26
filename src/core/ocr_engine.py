import os
import io
import json
import numpy as np
import cv2
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

# ==========================================
#        DATA PARSING HELPERS
# ==========================================

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

def _is_number_list(seq):
    if not isinstance(seq, (list, tuple)) or not seq: return False
    return all(isinstance(v, (int, float, np.integer, np.floating)) for v in seq)

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

def _extract_paddle_core(raw):
    if isinstance(raw, dict) and isinstance(raw.get("res"), dict):
        return raw["res"]
    return raw if isinstance(raw, dict) else {}

# ==========================================
#        IMAGE/DRAWING HELPERS
# ==========================================

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

def _draw_text_canvas(image_shape, items):
    h, w = image_shape[:2]
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)
    
    pil_img = None
    pil_draw = None
    pil_font_cache = {}
    pil_font_paths = []
    pil_font_candidates = [
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for fp in pil_font_candidates:
        if os.path.exists(fp):
            pil_font_paths.append(fp)

    def _has_non_ascii(text):
        return any(ord(ch) > 127 for ch in str(text))

    def _get_pil_font(px_size):
        px_size = max(12, int(px_size))
        if px_size in pil_font_cache:
            return pil_font_cache[px_size]
        for path in pil_font_paths:
            try:
                fnt = ImageFont.truetype(path, px_size)
                pil_font_cache[px_size] = fnt
                return fnt
            except Exception:
                pass
        fnt = ImageFont.load_default()
        pil_font_cache[px_size] = fnt
        return fnt

    def _wrap_line_to_width(text, max_width, font_scale, thickness):
        words = text.split(" ")
        if not words: return [""]
        lines = []
        current = words[0]
        for word in words[1:]:
            candidate = f"{current} {word}"
            candidate_w = cv2.getTextSize(candidate, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0][0]
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
            line_h = cv2.getTextSize("Ag", cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0][1] + 4
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
        line_h = cv2.getTextSize("Ag", cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0][1] + 4
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
            while last and cv2.getTextSize(last + "...", cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0][0] > max_width:
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
            px = max(0, x + 2)
            py = min(h - 4, y_cursor)
            if _has_non_ascii(line):
                if pil_img is None:
                    pil_img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
                    pil_draw = ImageDraw.Draw(pil_img)
                pil_font = _get_pil_font(26 * font_scale)
                pil_draw.text((px, max(0, py - line_h + 4)), line, fill=(0, 0, 0), font=pil_font)
            else:
                cv2.putText(canvas, line, (px, py), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
            y_cursor += line_h
    if pil_img is not None:
        canvas = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return canvas

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
        if polys_src is None: 
            polys = _extract_polys_from_ocr_payload(core)
        else: 
            polys = _extract_polys_from_ocr_payload(polys_src)
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

# ==========================================
#        DATA PARSING (TABLES)
# ==========================================

def extract_html_table(html_content):
    """Checks if string contains HTML table, returns pd.DataFrame if successful."""
    if "<table>" in html_content or '<table class="' in html_content:
        try:
            dfs = pd.read_html(io.StringIO(html_content))
            if dfs:
                return dfs[0]
        except Exception:
            return None
    return None