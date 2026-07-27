"""Run the automatic OCR pipeline on a real document and report markup results.

Intended to run inside the container on a GPU node::

    apptainer exec --nv --bind /storage:/storage <sif> \
        /opt/conda/envs/text_lab_main/bin/python tests/validate_real_doc.py <pdf> [outdir]

Prints a per-region markup report (glyphs, geometry status, overrides) and
writes markdown/JSON plus evidence crops to *outdir* for visual inspection.
"""

import conftest_path  # noqa: F401

import base64
import json
import pathlib
import sys
import tempfile

from core import auto_ocr, doc_ir


def main():
    pdf = pathlib.Path(sys.argv[1])
    outdir = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else pdf.parent / f"{pdf.stem}_validation"
    outdir.mkdir(parents=True, exist_ok=True)
    workspace = pathlib.Path(tempfile.mkdtemp(prefix="textlab_validate_"))

    def progress(frac, text):
        print(f"[{frac:5.0%}] {text}", flush=True)

    document = auto_ocr.process_document(
        pdf, workspace, progress=progress, source_name=pdf.name, debug_dir=outdir
    )

    print("\n================ MARKUP REPORT ================")
    summary = auto_ocr.document_summary(document)
    print(json.dumps({k: v for k, v in summary.items() if k != "region_counts"}, indent=2))
    print("region counts:", summary["region_counts"])

    n_crops = 0
    for page in document.pages:
        for region in page.ordered_regions():
            markup = region.markup or {}
            if not markup:
                continue
            if markup.get("kind") == "glyph-marks":
                items = markup.get("items", [])
                states = "".join(
                    {"checked": "X", "unchecked": "o", "uncertain": "?"}.get(i["state"], "?")
                    + ("*" if i.get("method") == "geometric-override" else "")
                    for i in items
                )
                print(
                    f"p{page.page_number} {region.id:>10} {region.type:<7} "
                    f"status={markup.get('status'):<18} marks=[{states}] "
                    f"({markup.get('n_checked')}✔/{markup.get('n_unchecked')}·/{markup.get('n_uncertain')}?)"
                )
                snippet = region.text.strip().replace("\n", " ")[:90]
                if snippet:
                    print(f"             text: {snippet}")
                for i, item in enumerate(items, start=1):
                    if item.get("crop_b64"):
                        fname = outdir / f"{region.id}_mark{i}_{item['state']}.png"
                        fname.write_bytes(base64.b64decode(item["crop_b64"]))
                        n_crops += 1
            else:
                print(
                    f"p{page.page_number} {region.id:>10} checkbox "
                    f"state={markup.get('state')} method={markup.get('method')} score={markup.get('score')}"
                )

    (outdir / "document.md").write_text(doc_ir.to_markdown(document), encoding="utf-8")
    (outdir / "document.json").write_text(doc_ir.to_json(document), encoding="utf-8")
    for page in document.pages:
        if page.image_b64:
            (outdir / f"page_{page.page_number}.png").write_bytes(base64.b64decode(page.image_b64))
    print(f"\nOutputs written to {outdir} ({n_crops} flagged-mark crops)")
    print("Full audit: page_N_marks.png overlays every located mark on the scan "
          "(green=checked, blue=unchecked, orange=uncertain, OVR=corrected).")


if __name__ == "__main__":
    main()
