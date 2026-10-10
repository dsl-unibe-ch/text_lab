"""PaddleOCR-VL subprocess worker (runs in the isolated ``paddle_backend`` env).

Mirrors ``paddle_ocr_worker.py``: the parent process invokes this script with a
list of single-page image paths, the script runs the ``PaddleOCRVL`` doc-parser
pipeline (paddleocr >= 3.6, ``[doc-parser]`` extra) fully offline, and prints a
single marker line::

    TEXTLAB_PADDLEVL_RESULT_JSON=<json>

The JSON is ``{"pages": [ ... ]}`` where each page carries the typed layout
blocks, per-region layout confidence, the pipeline markdown, and base64 PNG
crops for non-text regions (figures, charts, seals, checkboxes). The parent
turns these into the typed IR via ``doc_ir.from_paddle_vl``.

With ``--serve`` the script instead stays alive and reads one JSON request per
line from stdin, answering each with the same result line. Loading the weights
costs ~17 s and the first prediction another ~7 s of warm-up, which a batch
would otherwise pay once per file.
"""

import argparse
import base64
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np


# ---- offline hygiene: identical policy to paddle_ocr_worker.py --------------
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("DISABLE_MODEL_SOURCE_CHECK", "True")
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

RESULT_MARKER = "TEXTLAB_PADDLEVL_RESULT_JSON="

#: Emitted on stdout as each page finishes, so the parent can report progress
#: through a stage that runs ~25-50 s per page.
PROGRESS_MARKER = "TEXTLAB_PADDLEVL_PROGRESS="

#: Emitted once in ``--serve`` mode when the weights are loaded and the worker
#: is waiting for requests.
READY_MARKER = "TEXTLAB_PADDLEVL_READY="

# Layout labels whose regions we cut out as image crops (superset of the IR's
# asset types so the parent can classify checkboxes and render figures).
DEFAULT_ASSET_LABELS = {
    "image",
    "figure",
    "chart",
    "seal",
    "stamp",
    "checkbox",
    "check_box",
    "header_image",
    "footer_image",
}


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


def _res_dict(result):
    """Return the inner result dict from a pipeline result object."""
    if hasattr(result, "json"):
        raw = result.json
    elif hasattr(result, "to_dict"):
        raw = result.to_dict()
    elif isinstance(result, dict):
        raw = result
    else:
        raw = {}
    raw = make_json_serializable(raw)
    if isinstance(raw, dict) and isinstance(raw.get("res"), dict):
        return raw["res"]
    return raw if isinstance(raw, dict) else {}


def _markdown_text(result):
    md = getattr(result, "markdown", None)
    if md is None:
        return None
    if isinstance(md, str):
        return md
    if isinstance(md, dict):
        return md.get("markdown_texts") or md.get("markdown") or md.get("text")
    return getattr(md, "markdown_texts", None)


def _bbox4(bbox):
    """Normalise any bbox/polygon representation to [x1, y1, x2, y2]."""
    if not bbox:
        return None
    flat = []
    for v in bbox:
        if isinstance(v, (list, tuple)):
            flat.extend(v)
        else:
            flat.append(v)
    try:
        nums = [float(x) for x in flat]
    except (TypeError, ValueError):
        return None
    if len(nums) == 4:
        x1, y1, x2, y2 = nums
    elif len(nums) >= 8:
        xs, ys = nums[0::2], nums[1::2]
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    else:
        return None
    return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]


def _normalize_blocks(res):
    out = []
    for block in res.get("parsing_res_list") or []:
        if not isinstance(block, dict):
            continue
        out.append(
            {
                "block_label": block.get("block_label") or block.get("label"),
                "block_content": block.get("block_content", block.get("content", "")),
                "block_bbox": _bbox4(block.get("block_bbox") or block.get("bbox")),
                "block_order": block.get("block_order"),
                "block_score": block.get("block_score") or block.get("score"),
            }
        )
    return out


def _normalize_layout(res):
    layout = res.get("layout_det_res")
    raw_boxes = []
    if isinstance(layout, dict):
        raw_boxes = layout.get("boxes") or []
    elif isinstance(layout, list):
        raw_boxes = layout
    boxes = []
    for box in raw_boxes:
        if not isinstance(box, dict):
            continue
        boxes.append(
            {
                "label": box.get("label"),
                "score": box.get("score"),
                "coordinate": _bbox4(box.get("coordinate") or box.get("bbox")),
                "cls_id": box.get("cls_id"),
            }
        )
    return boxes


def _crop_b64(image, bbox, margin_frac):
    import cv2

    if image is None or not bbox:
        return None
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox
    mw = max(4, int((x2 - x1) * margin_frac))
    mh = max(4, int((y2 - y1) * margin_frac))
    x1 = max(0, int(x1) - mw)
    y1 = max(0, int(y1) - mh)
    x2 = min(w, int(x2) + mw)
    y2 = min(h, int(y2) + mh)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = image[y1:y2, x1:x2]
    ok, encoded = cv2.imencode(".png", crop)
    if not ok:
        return None
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def _build_pipeline():
    try:
        import paddle

        if paddle.device.is_compiled_with_cuda():
            paddle.set_device("gpu")
    except Exception:
        pass

    from paddleocr import PaddleOCRVL

    return PaddleOCRVL()


def process_image(pipeline, image_path, asset_labels, margin_frac, page_number):
    import cv2

    src = cv2.imread(str(image_path))
    height, width = (src.shape[0], src.shape[1]) if src is not None else (None, None)

    result = None
    for res in pipeline.predict(str(image_path)):
        result = res
        break

    res_dict = _res_dict(result) if result is not None else {}
    blocks = _normalize_blocks(res_dict)
    layout = _normalize_layout(res_dict)

    assets = {}
    for idx, block in enumerate(blocks):
        label = (block.get("block_label") or "").strip().lower()
        if label in asset_labels:
            crop_b64 = _crop_b64(src, block.get("block_bbox"), margin_frac)
            if crop_b64:
                assets[str(idx)] = {"b64": crop_b64, "ext": "png"}

    return {
        "page_number": page_number,
        "image": str(image_path),
        "width": width,
        "height": height,
        "parsing_res_list": blocks,
        "layout_det_res": layout,
        "markdown": _markdown_text(result),
        "assets": assets,
    }


def _asset_labels(extra: str) -> set:
    labels = set(DEFAULT_ASSET_LABELS)
    for lbl in (extra or "").split(","):
        lbl = lbl.strip().lower()
        if lbl:
            labels.add(lbl)
    return labels


def _run_request(pipeline, images, asset_labels, crop_margin) -> dict:
    """Recognise one request's pages, reporting each as it lands."""
    total = len(images)
    pages = []
    for idx, image in enumerate(images, start=1):
        pages.append(process_image(pipeline, Path(image), asset_labels, crop_margin, idx))
        print(f"{PROGRESS_MARKER}{idx}/{total}", flush=True)
    return {"pages": pages}


def serve(default_margin: float):
    """Answer one JSON request per stdin line until stdin closes.

    A request is ``{"images": [...], "extra_labels": "...", "crop_margin": f}``.
    Every request is answered with exactly one result line, an error included,
    so one bad document cannot leave the parent waiting or desynchronise the
    stream.
    """
    pipeline = _build_pipeline()
    print(f"{READY_MARKER}1", flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            images = [str(p) for p in request.get("images") or []]
            print(f"{PROGRESS_MARKER}0/{len(images)}", flush=True)
            payload = _run_request(
                pipeline,
                images,
                _asset_labels(request.get("extra_labels", "")),
                float(request.get("crop_margin", default_margin)),
            )
        except Exception as exc:  # one document's failure, not the session's
            traceback.print_exc()
            payload = {"pages": [], "error": f"{type(exc).__name__}: {exc}"}
        print(RESULT_MARKER + json.dumps(payload, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description="PaddleOCR-VL doc-parser worker")
    parser.add_argument("images", nargs="*", help="single-page image paths")
    parser.add_argument(
        "--extra-labels",
        default="",
        help="comma-separated extra layout labels to crop as assets",
    )
    parser.add_argument("--crop-margin", type=float, default=0.06, help="crop margin as fraction of box")
    parser.add_argument(
        "--serve", action="store_true",
        help="stay alive and answer one JSON request per stdin line",
    )
    args = parser.parse_args()

    if args.serve:
        serve(args.crop_margin)
        return
    if not args.images:
        parser.error("one or more image paths are required without --serve")

    total = len(args.images)
    print(f"{PROGRESS_MARKER}0/{total}", flush=True)
    pipeline = _build_pipeline()
    payload = _run_request(
        pipeline, args.images, _asset_labels(args.extra_labels), args.crop_margin
    )
    print(RESULT_MARKER + json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
