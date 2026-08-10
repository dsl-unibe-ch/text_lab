#!/usr/bin/env python
"""Batch survey extraction from the command line.

    python src/tools/survey_cli.py run --input scans/ --out results/

``run`` does the whole job: synthesize the blank from the batch, locate the
response controls, read every questionnaire against them, and write the
per-respondent table. ``build-template`` and ``read`` split the same work when
the template needs a human pass in between.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from core import survey_batch, survey_template  # noqa: E402

SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def _documents(folder) -> list:
    paths = sorted(
        path for path in pathlib.Path(folder).rglob("*")
        if path.suffix.lower() in SUFFIXES and not path.name.startswith("._")
    )
    if not paths:
        raise SystemExit(f"No questionnaires found in {folder}")
    return paths


def _write_overlays(template, blanks, out_dir) -> None:
    import cv2

    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for page, blank in zip(template.pages, blanks):
        vis = survey_template.overlay(
            blank.image,
            [{"bbox": c.pixel_bbox(page.width, page.height), "shape": c.shape}
             for c in page.controls],
        )
        cv2.imwrite(str(out_dir / f"template_page{page.page_index + 1}.png"), vis)


def _label(template, blanks) -> None:
    """Name the controls by running layout+OCR over the synthesized blanks.

    Kept optional because it needs the PaddleOCR-VL backend environment, while
    everything else in this tool is OpenCV only.
    """
    import tempfile

    import cv2

    from core import auto_ocr, survey_label

    print("Labelling controls from the blank form...")
    with tempfile.TemporaryDirectory() as tmp:
        images = []
        for page, blank in zip(template.pages, blanks):
            path = pathlib.Path(tmp) / f"blank_page{page.page_index + 1}.png"
            cv2.imwrite(str(path), blank.image)
            images.append(path)
        page_jsons = auto_ocr.run_vl_worker(images)
    labelled = survey_label.label_template(template, page_jsons)
    survey_template.infer_structure(template)
    named = survey_label.name_answer_rows(template)
    questions = {
        control.question_id
        for page in template.pages for control in page.controls if control.question_id
    }
    print(f"  {labelled}/{template.control_count} controls labelled "
          f"in {len(questions)} question groups; {named} answer rows named")


def _build(args) -> None:
    paths = _documents(args.input)
    print(f"Synthesizing the blank form from {len(paths)} document(s)...")
    template, blanks = survey_template.build_template(paths, dpi=args.dpi)
    for page, blank in zip(template.pages, blanks):
        failed = ", ".join(name for name, _ in blank.failures) or "none"
        print(f"  page {page.page_index + 1}: {len(page.controls)} controls "
              f"from {len(blank.contributors)} copies (failed: {failed})")
    if getattr(args, "labels", False):
        _label(template, blanks)

    rules = survey_template.infer_structure(template)
    singles = sum(1 for rule in rules.values() if rule == "single")
    print(f"  {len(rules)} answer groups: {singles} single-choice, "
          f"{len(rules) - singles} multi-select")

    template.save(args.template)
    print(f"Template -> {args.template} ({template.control_count} controls)")
    if args.overlay:
        _write_overlays(template, blanks, args.overlay)
        print(f"Audit overlays -> {args.overlay}")


def _read(args) -> None:
    template = survey_template.SurveyTemplate.load(args.template)
    paths = _documents(args.input)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    results = survey_batch.read_batch(
        paths, template,
        debug_dir=(out / "overlays") if args.overlays else None,
        progress=lambda _frac, text: print(f"  {text}"),
    )

    survey_batch.to_checkbox_table(results, template).to_csv(
        out / "responses_checkboxes.csv", index=False
    )
    survey_batch.to_wide(results, template).to_csv(out / "responses_matrix.csv", index=False)
    survey_batch.to_long(results, template).to_csv(out / "responses_long.csv", index=False)
    queue = survey_batch.review_queue(results, template)
    queue.to_csv(out / "review_queue.csv", index=False)
    unused = survey_batch.unused_controls(results, template)
    unused.to_csv(out / "unused_controls.csv", index=False)

    summary = survey_batch.summarize(results)
    print(f"\n{summary['documents']} documents x {summary['controls_per_document']} controls")
    print(f"  marked        : {summary['checked']}")
    print(f"  needs a look  : {summary['uncertain']} ({summary['uncertain_rate'] * 100:.2f}%)")
    print(f"  worst page registration: {summary['worst_registration']}")
    if len(unused):
        rows = int(unused["whole_row_unused"].sum())
        print(f"  {len(unused)} control(s) nobody marked"
              + (f", {rows} in answer rows nobody touched (likely false positives)" if rows else "")
              + " - see unused_controls.csv")
    for result in results:
        for warning in result.warnings:
            print(f"  ! {result.document}: {warning}")
    print(f"\nWrote responses_checkboxes.csv, responses_matrix.csv, "
          f"responses_long.csv, review_queue.csv, unused_controls.csv to {out}")


def _run(args) -> None:
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    args.template = args.template or str(out / "survey_template.json")
    args.overlay = args.overlay or str(out / "template")
    _build(args)
    print()
    _read(args)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build-template", help="synthesize the blank and find controls")
    build.add_argument("--input", required=True, help="folder of questionnaires")
    build.add_argument("--template", required=True, help="template JSON to write")
    build.add_argument("--overlay", help="folder for the audit overlay PNGs")
    build.add_argument("--dpi", type=int, default=survey_template.DEFAULT_DPI)
    build.add_argument("--labels", action="store_true",
                       help="name the controls with PaddleOCR-VL (needs the VL backend)")
    build.set_defaults(func=_build)

    read = sub.add_parser("read", help="read a batch against an existing template")
    read.add_argument("--input", required=True, help="folder of questionnaires")
    read.add_argument("--template", required=True, help="template JSON to read")
    read.add_argument("--out", required=True, help="folder for the CSVs")
    read.add_argument("--overlays", action="store_true", help="write per-document overlays")
    read.set_defaults(func=_read)

    run = sub.add_parser("run", help="build the template and read the batch")
    run.add_argument("--input", required=True, help="folder of questionnaires")
    run.add_argument("--out", required=True, help="folder for template and CSVs")
    run.add_argument("--template", help="template JSON (default: <out>/survey_template.json)")
    run.add_argument("--overlay", help="folder for template overlays (default: <out>/template)")
    run.add_argument("--overlays", action="store_true", help="write per-document overlays")
    run.add_argument("--dpi", type=int, default=survey_template.DEFAULT_DPI)
    run.add_argument("--labels", action="store_true",
                     help="name the controls with PaddleOCR-VL (needs the VL backend)")
    run.set_defaults(func=_run)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
