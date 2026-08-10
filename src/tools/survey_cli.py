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
    survey_label.disambiguate_labels(template)
    survey_label.assign_sheet_pages(template)
    options = survey_label.name_options(template)
    survey_label.disambiguate_labels(template)
    named = survey_label.name_answer_rows(template)
    questions = {
        control.question_id
        for page in template.pages for control in page.controls if control.question_id
    }
    print(f"  {labelled}/{template.control_count} controls labelled "
          f"in {len(questions)} question groups; {named} answer rows and "
          f"{options} options named from the blank")


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


LABEL_INSTRUCTIONS = """# How to label these questionnaires

Fill in the **answer** column of `answer_sheet.csv`. One line per answer, so a
questionnaire is around 35 lines rather than 184 checkboxes.

**Work from the original PDFs, and do not open the pipeline's output first.**
The point is to measure what the pipeline gets wrong; if you start from its
answers you will measure whether you notice its mistakes instead.

## Finding the question on the paper

The scans are two-up: one PDF page holds two questionnaire pages side by side,
so the PDF page number is not much help. Use these instead:

- `sheet_page` -- the questionnaire's own printed page number, as it appears in
  the footer ("1/4", "2/4", ...). PDF page 1 holds sheets 4/4 and 1/4; PDF
  page 2 holds sheets 2/4 and 3/4.
- `question` -- the number printed on the form ("Q10" is the question printed
  as "10)").
- `row` -- for a question with several answers: the printed row name, or
  "row 2 of 3 (top to bottom)" when the row has no name of its own.

The lines are in reading order: down each questionnaire page in turn.

## What to write

| Situation | Write |
|---|---|
| One option marked | that option's text, copied from the `options` column |
| Nothing marked | leave the cell empty |
| Several marked (`type` = multiple) | the options separated by `;` |
| Several marked where only one was allowed | the options separated by `;` |
| You cannot tell what the respondent meant | `?` |

Rows marked `?` are excluded from scoring -- there is no ground truth to score
against, and that is a fair answer for a genuinely ambiguous mark.

`option 1`, `option 2`, ... appear where the form's own wording could not be
read off the page reliably. Count in printed order: left-to-right for a row of
choices, top-to-bottom for a vertical list.

Spelling and capitalisation do not matter; order within a `;` list does not
matter either.
"""


def _prune(args) -> None:
    """Delete controls from a template -- the template review pass.

    unused_controls.csv flags answer rows nobody ever marked; those are almost
    always detection false positives, and removing them keeps them out of the
    export and out of the row numbering on the answer sheet.
    """
    template = survey_template.SurveyTemplate.load(args.template)
    drop = {c.strip() for c in args.controls.split(",") if c.strip()}
    before = template.control_count
    for page in template.pages:
        page.controls = [c for c in page.controls if c.id not in drop]
    removed = before - template.control_count

    survey_template.infer_structure(template)
    from core import survey_label

    survey_label.disambiguate_labels(template)
    template.save(args.template)
    print(f"Removed {removed} control(s); {template.control_count} remain "
          f"in {len(template.rules)} answer groups")
    if removed != len(drop):
        print(f"  note: {len(drop) - removed} id(s) were not in the template")


def _sheet(args) -> None:
    template = survey_template.SurveyTemplate.load(args.template)
    names = (
        [n.strip() for n in args.documents.split(",") if n.strip()]
        if args.documents else [p.name for p in _documents(args.input)]
    )
    sheet = survey_batch.answer_sheet(template, names)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.to_csv(out, index=False)
    (out.parent / "HOW_TO_LABEL.md").write_text(LABEL_INSTRUCTIONS, encoding="utf-8")
    print(f"Blank answer sheet -> {out}")
    print(f"Instructions       -> {out.parent / 'HOW_TO_LABEL.md'}")
    print(f"  {len(names)} document(s) x {len(sheet) // max(1, len(names))} answers each "
          f"= {len(sheet)} lines to fill in")


def _score(args) -> None:
    import pandas as pd

    template = survey_template.SurveyTemplate.load(args.template)
    sheet = pd.read_csv(args.sheet).fillna("")
    names = sorted({str(d) for d in sheet["document"]})
    paths = [p for p in _documents(args.input) if p.name in names]
    missing = set(names) - {p.name for p in paths}
    if missing:
        raise SystemExit(f"Documents named in the sheet but not found: {sorted(missing)}")

    results = survey_batch.read_batch(paths, template)
    per_answer, summary = survey_batch.score_sheet(sheet, results, template)

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    per_answer.to_csv(out / "score_per_answer.csv", index=False)
    wrong = per_answer[(~per_answer["correct"]) & (~per_answer["flagged"])
                       & (~per_answer["human_unsure"])]
    wrong.to_csv(out / "score_disagreements.csv", index=False)

    print()
    for key, value in summary.items():
        print(f"  {key:34s} {value}")
    if len(wrong):
        print(f"\n  {len(wrong)} silent error(s) -> score_disagreements.csv")
        print(wrong[["document", "answer_id", "row", "truth",
                     "predicted", "certainty"]].to_string(index=False))
    print(f"\nWrote score_per_answer.csv, score_disagreements.csv to {out}")


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

    sheet = sub.add_parser("answer-sheet", help="blank sheet for hand-labelling ground truth")
    sheet.add_argument("--template", required=True, help="template JSON to read")
    sheet.add_argument("--input", help="folder of questionnaires (for the document names)")
    sheet.add_argument("--documents", help="comma-separated filenames to label instead")
    sheet.add_argument("--out", required=True, help="CSV to write")
    sheet.set_defaults(func=_sheet)

    score = sub.add_parser("score", help="score a filled answer sheet against the pipeline")
    score.add_argument("--template", required=True, help="template JSON to read")
    score.add_argument("--sheet", required=True, help="the filled-in answer sheet CSV")
    score.add_argument("--input", required=True, help="folder holding the questionnaires")
    score.add_argument("--out", required=True, help="folder for the score report")
    score.set_defaults(func=_score)

    prune = sub.add_parser("prune-template", help="delete controls from a template")
    prune.add_argument("--template", required=True, help="template JSON to edit in place")
    prune.add_argument("--controls", required=True, help="comma-separated control ids")
    prune.set_defaults(func=_prune)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
