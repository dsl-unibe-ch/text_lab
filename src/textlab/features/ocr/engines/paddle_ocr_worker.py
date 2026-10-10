r"""PaddleOCR 2 worker: recognize page images, print the result as JSON.

Runs in the image's PaddleOCR environment (``common.container.PADDLE_ENV``),
started by :func:`.paddle_ocr.run_worker`::

    python -m textlab.features.ocr.engines.paddle_ocr_worker \
        --lang en page-1.png page-2.png

The result is printed as one line starting with
:data:`.paddle_ocr.RESULT_MARKER`. Only the standard library, this
environment's packages and :mod:`.payloads` may be imported.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path

from textlab.features.ocr.engines.payloads import (
    compact_paddle_prediction,
    rendered_png,
)

# Paddle crashes in OpenBLAS with more than one thread, and must not try to
# reach the internet for model checks.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("DISABLE_MODEL_SOURCE_CHECK", "True")
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

#: Prefix of the result line; the same as ``paddle_ocr.RESULT_MARKER``,
#: which this worker cannot import.
RESULT_MARKER = "TEXTLAB_PADDLEOCR_RESULT_JSON="


def rendered_png_b64(preds) -> str | None:
    """Return the first result image of a page's predictions, as base64."""
    for pred in preds:
        png = rendered_png(pred)
        if png:
            return base64.b64encode(png).decode("ascii")
    return None


def main() -> None:
    """Recognize the images named on the command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", default="en")
    parser.add_argument("images", nargs="+")
    args = parser.parse_args()

    import paddle
    from paddleocr import PaddleOCR

    if paddle.device.is_compiled_with_cuda():
        paddle.set_device("gpu")

    try:
        ocr = PaddleOCR(
            use_textline_orientation=True, lang=args.lang, device="gpu"
        )
    except TypeError:
        ocr = PaddleOCR(use_textline_orientation=True, lang=args.lang)

    pages = []
    for idx, image in enumerate(args.images, start=1):
        image_path = str(Path(image))
        preds = (
            ocr.predict(image_path)
            if hasattr(ocr, "predict")
            else ocr.ocr(image_path)
        )
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
                "rendered_png_b64": rendered_png_b64(preds),
            }
        )

    print(RESULT_MARKER + json.dumps({"pages": pages}, ensure_ascii=False))


if __name__ == "__main__":
    main()
