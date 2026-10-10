import argparse
import base64
import io
import json
import os
from pathlib import Path

import numpy as np


os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("DISABLE_MODEL_SOURCE_CHECK", "True")
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")


def make_json_serializable(value):
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
        for key in (
            "input_path",
            "page_index",
            "model_settings",
            "rec_texts",
            "rec_scores",
            "rec_polys",
            "dt_polys",
            "rec_boxes",
            "boxes",
        ):
            if key in raw:
                compact[key] = raw[key]
        if isinstance(raw.get("res"), dict):
            for key in ("rec_texts", "rec_scores", "rec_polys", "dt_polys", "rec_boxes", "boxes"):
                if key in raw["res"] and key not in compact:
                    compact[key] = raw["res"][key]
    else:
        compact["raw"] = raw
    rec_texts = extract_texts_from_ocr_payload(compact if compact else raw)
    compact["rec_texts"] = list(dict.fromkeys(rec_texts))
    return compact


def image_to_png_b64(image_like):
    if image_like is None:
        return None
    try:
        import cv2
        from PIL import Image

        if isinstance(image_like, (bytes, bytearray)):
            return base64.b64encode(bytes(image_like)).decode("ascii")
        if isinstance(image_like, np.ndarray):
            ok, encoded = cv2.imencode(".png", image_like)
            if ok:
                return base64.b64encode(encoded.tobytes()).decode("ascii")
            return None
        if isinstance(image_like, Image.Image) or hasattr(image_like, "save"):
            buf = io.BytesIO()
            image_like.save(buf, format="PNG")
            return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        return None
    return None


def pick_paddle_rendered_png_b64(preds):
    for pred in preds:
        payload = getattr(pred, "img", None)
        if payload is None:
            continue
        if isinstance(payload, dict):
            for key in ("ocr_res_img", "overall_ocr_res", "layout_det_res"):
                if key in payload:
                    out = image_to_png_b64(payload[key])
                    if out:
                        return out
            for val in payload.values():
                out = image_to_png_b64(val)
                if out:
                    return out
        else:
            out = image_to_png_b64(payload)
            if out:
                return out
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", default="en")
    parser.add_argument("images", nargs="+")
    args = parser.parse_args()

    import paddle
    from paddleocr import PaddleOCR

    if paddle.device.is_compiled_with_cuda():
        paddle.set_device("gpu")

    try:
        ocr = PaddleOCR(use_textline_orientation=True, lang=args.lang, device="gpu")
    except TypeError:
        ocr = PaddleOCR(use_textline_orientation=True, lang=args.lang)

    pages = []
    for idx, image in enumerate(args.images, start=1):
        image_path = str(Path(image))
        preds = ocr.predict(image_path) if hasattr(ocr, "predict") else ocr.ocr(image_path)
        compact_preds = [compact_paddle_prediction(pred) for pred in preds]
        lines = []
        for pred in compact_preds:
            lines.extend(pred.get("rec_texts", []))
        pages.append(
            {
                "page": idx,
                "image": image_path,
                "text": "\n".join(line for line in lines if line),
                "raw": compact_preds,
                "rendered_png_b64": pick_paddle_rendered_png_b64(preds),
            }
        )

    print("TEXTLAB_PADDLEOCR_RESULT_JSON=" + json.dumps({"pages": pages}, ensure_ascii=False))


if __name__ == "__main__":
    main()
