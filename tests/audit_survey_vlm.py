"""Save every survey VLM request and response for visual review.

Run through the container launcher on a GPU node::

    bash tests/run_survey_vlm_audit.sh input.pdf output_dir \
        --model gemma4:26b --contract schema-free-v2

The output intentionally retains document images. It is never enabled by the
normal OCR UI and must be deleted manually when the review is complete.
"""

import conftest_path  # noqa: F401

import argparse
import json
import pathlib
import shutil
import tempfile

from core import auto_ocr, doc_ir, form_extract, vision_enrich


def write_review_index(output: pathlib.Path):
    """Create a browsable Markdown contact sheet from an existing audit."""
    calls_dir = output / "vlm_calls"
    lines = [
        f"# VLM audit: {output.name}",
        "",
        "Each image below is the exact complete question section supplied to the "
        "model. The answer is the exact raw `message.content`, before TextLab "
        "reconciliation.",
        "",
    ]
    call_dirs = sorted(path for path in calls_dir.iterdir() if path.is_dir())
    for call_dir in call_dirs:
        relative = call_dir.relative_to(output)
        raw_path = call_dir / "raw_content.txt"
        error_path = call_dir / "error.txt"
        raw = raw_path.read_text(encoding="utf-8") if raw_path.exists() else "(no content)"
        lines.extend(
            [
                f"## {call_dir.name}",
                "",
                f"![exact model input]({relative.as_posix()}/input.png)",
                "",
                f"[prompt]({relative.as_posix()}/prompt.txt) · "
                f"[response schema]({relative.as_posix()}/response_schema.json) · "
                f"[raw Ollama envelope]({relative.as_posix()}/raw_response.json)",
                "",
                "```json",
                raw,
                "```",
                "",
            ]
        )
        if error_path.exists():
            lines.extend(["**Error:**", "", error_path.read_text(encoding="utf-8"), ""])
    (output / "REVIEW.md").write_text("\n".join(lines), encoding="utf-8")
    labels_path = output / "boundary-review.template.json"
    if not labels_path.exists():
        labels_path.write_text(
            json.dumps(
                {
                    "instructions": (
                        "Copy this file to boundary-review.json and label every section. "
                        "A release dataset must report boundary precision/recall and "
                        "choice-clipping rate; null means not reviewed."
                    ),
                    "sections": [
                        {
                            "call": call_dir.name,
                            "is_correct_question_boundary": None,
                            "choice_clipped": None,
                            "includes_adjacent_question": None,
                            "notes": "",
                        }
                        for call_dir in call_dirs
                    ],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=pathlib.Path)
    parser.add_argument("output", type=pathlib.Path)
    parser.add_argument("--model", default="gemma4:26b")
    parser.add_argument("--base-url", default=None)
    parser.add_argument(
        "--contract",
        choices=("schema-free-v1", "schema-free-v2", "paddle-id-v1"),
        default="schema-free-v2",
        help="Use the universal contract or the repaired Paddle-ID A/B baseline",
    )
    args = parser.parse_args()
    requested_contract = args.contract
    args.contract = form_extract._survey_contract(args.contract)
    if requested_contract != args.contract:
        print(
            f"Contract {requested_contract!r} is superseded; "
            f"running {args.contract!r}."
        )

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    calls_dir = output / "vlm_calls"
    calls_dir.mkdir(parents=True, exist_ok=True)
    if any(calls_dir.iterdir()):
        raise SystemExit(
            f"Audit calls already exist in {calls_dir}; use a new output directory "
            "so separate runs are not mixed"
        )
    workspace = pathlib.Path(tempfile.mkdtemp(prefix="textlab_survey_audit_"))
    client = vision_enrich.OllamaVisionClient(
        model=args.model,
        base_url=args.base_url,
        audit_dir=calls_dir,
    )

    try:
        document = auto_ocr.process_document(
            args.input,
            workspace,
            source_name=args.input.name,
            extract_survey=True,
            vision_client=client,
            debug_dir=output / "geometry",
            survey_contract=args.contract,
        )
        summary = auto_ocr.document_summary(document)
        (output / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (output / "document.json").write_text(doc_ir.to_json(document), encoding="utf-8")
        responses = doc_ir.build_form_responses_csv(document)
        if responses:
            (output / "form_responses.csv").write_bytes(responses)
        (output / "full_bundle.zip").write_bytes(
            doc_ir.build_full_bundle(document, args.input.stem)
        )
        index = {
            "input": str(args.input.resolve()),
            "provider": client.provider,
            "model": client.model,
            "contract": args.contract,
            "requested_contract": requested_contract,
            "n_vlm_calls": client._audit_count,
            "calls_directory": str(calls_dir),
            "contents": {
                "input.png": "exact complete question section sent to the model",
                "prompt.txt": "exact versioned text prompt",
                "response_schema.json": "Ollama structured-output schema",
                "raw_response.json": "complete Ollama response envelope",
                "raw_content.txt": "exact model answer before JSON parsing",
                "parsed_response.json": "parsed answer when valid",
                "error.txt": "request/parse error when a call failed",
                "boundary-review.template.json": (
                    "manual labels for boundary precision/recall and choice clipping"
                ),
            },
        }
        (output / "README.json").write_text(
            json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        write_review_index(output)
        print(json.dumps({"output": str(output), "summary": summary}, indent=2))
    finally:
        client.close(unload_model=True)
        shutil.rmtree(workspace, ignore_errors=True)


if __name__ == "__main__":
    main()
