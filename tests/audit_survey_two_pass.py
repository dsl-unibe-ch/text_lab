"""Compare direct lightweight extraction with a two-pass survey pipeline.

Path A sends each saved crop to Qwen with a small final JSON contract. Path B
asks Qwen for a non-JSON visual evidence ledger, then asks a text-only model to
normalize that ledger into the same JSON contract. Every intermediate response
is retained for separate perception/structuring review; credentials are not.
"""

from __future__ import annotations

import argparse
import base64
import json
import pathlib
import shutil
import time
import urllib.error
import urllib.request


VALID_STATES = ["selected", "cancelled", "ambiguous"]
VALID_MARKS = ["x", "tick", "filled", "scribbled", "other", "uncertain"]

LIGHTWEIGHT_SCHEMA = {
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "properties": {
                    "question_text": {"type": "string"},
                    "condition_text": {"type": "string"},
                    "choices": {
                        "type": "array",
                        "minItems": 1,
                        "items": {"type": "string"},
                    },
                    "rows": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "properties": {
                                "row_text": {"type": "string"},
                                "marks": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "choice_text": {"type": "string"},
                                            "state": {
                                                "type": "string",
                                                "enum": VALID_STATES,
                                            },
                                            "visual_mark": {
                                                "type": "string",
                                                "enum": VALID_MARKS,
                                            },
                                            "associated_text": {"type": "string"},
                                            "evidence_text": {"type": "string"},
                                        },
                                        "required": [
                                            "choice_text",
                                            "state",
                                            "visual_mark",
                                            "associated_text",
                                            "evidence_text",
                                        ],
                                        "additionalProperties": False,
                                    },
                                },
                            },
                            "required": ["row_text", "marks"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": [
                    "question_text",
                    "condition_text",
                    "choices",
                    "rows",
                ],
                "additionalProperties": False,
            },
        },
        "ignored_text": {"type": "array", "items": {"type": "string"}},
        "uncertainties": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["questions", "ignored_text", "uncertainties"],
    "additionalProperties": False,
}

FORMAT_EXAMPLE = (
    '{"questions":[{"question_text":"...","condition_text":"",'
    '"choices":["..."],"rows":[{"row_text":"","marks":['
    '{"choice_text":"...","state":"selected","visual_mark":"x",'
    '"associated_text":"","evidence_text":"..."}]}]}],'
    '"ignored_text":[],"uncertainties":[]}'
)


DIRECT_PROMPT = """Read this cropped paper survey section from visible pixels only.
Return the small supplied JSON contract. Transcribe each actual question and every
printed response label exactly once. Different choice sets are separate questions.
For a simple question use one row with empty row_text. For a matrix use shared
choices and one row per statement. condition_text contains a printed conditional
phrase such as 'Falls ja'; otherwise it is empty. Do not create response types,
selection rules, IDs, page-number questions, prize/legal-text questions, or N/A
choices that are not printed response controls.

marks contains every control with respondent-added ink and omits clean printed
outlines. Use selected for an ordinary X/tick/fill, cancelled for a visibly
crossed-out prior response, and ambiguous only when intent is unclear. choice_text
must exactly match one item in choices. associated_text is only handwriting visibly
linked to that marked choice, including conditional phone/detail fields. Briefly
describe the visible ink in evidence_text. Put non-question prose/page numbers in
ignored_text and genuine visual doubts in uncertainties. Do not infer answers.
Return only JSON matching the supplied schema."""


LEDGER_PROMPT = """Act only as a visual evidence transcriber for this cropped paper
survey section. Do not output JSON and do not design a database schema. Read visible
pixels, not prior OCR. Produce a concise Markdown evidence ledger using this form for
each real question or conditional subquestion:

## QUESTION
TEXT: exact printed question
CONDITION: exact printed condition, or NONE
CHOICES:
- one exact printed response label per line
ROWS:
- For a simple question, output exactly one ROW: SIMPLE. Never turn scale endpoint
  annotations such as 'Sehr schlecht/gut' or individual choices into rows.
- For a matrix, output one ROW per exact printed matrix statement.
- Under a row, output one MARK line per control with respondent-added ink:
  MARK: exact printed choice label | INK: visible X/tick/fill/overwrite description |
  INTENT: selected/cancelled/ambiguous | ASSOCIATED: exact linked handwriting or NONE
- If a row has no respondent ink, output MARK: NONE only.

List every printed choice and matrix row, including unanswered ones. List every
respondent-added mark, but never report a clean printed circle/box as a mark. A
clean control must not get a MARK line and the ledger must never use `unselected`.
A scale endpoint annotation is ignored explanatory text, not a choice or row. A
free-text-only field is not a question; attach its handwriting to the conditioned
marked choice. Do not create N/A choices, response types, rules, IDs, or questions
from examples, prizes, legal prose, or page numbers. End with:

## IGNORED PRINTED TEXT
- concise item, or NONE
## VISUAL UNCERTAINTY
- concise item, or NONE

Never infer an answer. Keep the ledger below 1,200 words."""


def direct_prompt() -> str:
    return DIRECT_PROMPT.replace(
        "Return only JSON matching the supplied schema.",
        "Return only one JSON object in exactly this shape (with as many questions, "
        "rows, choices and marks as the evidence requires):\n" + FORMAT_EXAMPLE,
    )


def structure_prompt(ledger: str) -> str:
    return """Convert the visual evidence ledger below into the supplied lightweight
JSON schema. This is lossless normalization, not visual interpretation. Use only
facts explicitly present in the ledger. Never invent choices, marks, questions,
N/A, response types, selection rules, or IDs. Preserve every printed choice and
matrix row. A SIMPLE row has empty row_text. Each mark's choice_text must exactly
match an item in that question's choices. Copy the corresponding ledger MARK line
into evidence_text. Map the stated intent to selected/cancelled/ambiguous and the
ink description to x/tick/filled/scribbled/other/uncertain. Put ignored prose and
uncertainties in their top-level arrays. `MARK: NONE` and any accidental
`INTENT: unselected` mean that there is no mark and must be omitted. If the ledger is inconsistent, preserve
the evidence and describe the conflict in uncertainties. Return only JSON matching
the required format. Return only one JSON object in exactly this shape (with as
many questions, rows, choices and marks as the ledger requires):
""" + FORMAT_EXAMPLE + """

VISUAL EVIDENCE LEDGER:
""" + ledger


def request_completion(base_url: str, api_key: str, payload: dict, timeout: int):
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:2000]
        raise RuntimeError(f"GPUStack HTTP {exc.code}: {body}") from exc


def response_content(response: dict) -> str:
    choices = response.get("choices") or []
    if not choices:
        raise ValueError("response has no choices")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if not isinstance(content, str) or not content:
        raise ValueError("response has no text content")
    return content


def image_message(prompt: str, image_b64: str):
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64," + image_b64},
                },
            ],
        }
    ]


def json_payload(model: str, messages: list, max_tokens: int):
    return {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "seed": 42,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }


def plain_payload(model: str, messages: list, max_tokens: int):
    return {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "seed": 42,
        "max_tokens": max_tokens,
    }


def validate_lightweight(result: object):
    if not isinstance(result, dict):
        raise ValueError("JSON result is not an object")
    if set(result) != {"questions", "ignored_text", "uncertainties"}:
        raise ValueError("JSON result has incorrect top-level fields")
    if not isinstance(result["questions"], list) or not result["questions"]:
        raise ValueError("questions must be a non-empty array")
    if not all(isinstance(item, str) for item in result["ignored_text"]):
        raise ValueError("ignored_text must contain strings")
    if not all(isinstance(item, str) for item in result["uncertainties"]):
        raise ValueError("uncertainties must contain strings")
    for question in result["questions"]:
        if not isinstance(question, dict) or set(question) != {
            "question_text", "condition_text", "choices", "rows"
        }:
            raise ValueError("question has incorrect fields")
        if not isinstance(question["question_text"], str):
            raise ValueError("question_text must be a string")
        if not isinstance(question["condition_text"], str):
            raise ValueError("condition_text must be a string")
        if not isinstance(question["choices"], list) or not question["choices"]:
            raise ValueError("choices must be a non-empty string array")
        if not all(isinstance(choice, str) for choice in question["choices"]):
            raise ValueError("choice labels must be strings")
        if not isinstance(question["rows"], list) or not question["rows"]:
            raise ValueError("rows must be a non-empty array")
        for row in question["rows"]:
            if not isinstance(row, dict) or set(row) != {"row_text", "marks"}:
                raise ValueError("row has incorrect fields")
            if not isinstance(row["row_text"], str) or not isinstance(row["marks"], list):
                raise ValueError("row fields have incorrect types")
            for mark in row["marks"]:
                if not isinstance(mark, dict) or set(mark) != {
                    "choice_text", "state", "visual_mark", "associated_text",
                    "evidence_text"
                }:
                    raise ValueError("mark has incorrect fields")
                if mark["state"] not in VALID_STATES:
                    raise ValueError("mark has invalid state")
                if mark["visual_mark"] not in VALID_MARKS:
                    raise ValueError("mark has invalid visual_mark")
                if not all(
                    isinstance(mark[field], str)
                    for field in ("choice_text", "associated_text", "evidence_text")
                ):
                    raise ValueError("mark text fields must be strings")
                if mark["choice_text"] not in question["choices"]:
                    raise ValueError("marked choice is absent from choices")


def save_call(target: pathlib.Path, response: dict, *, expect_json: bool):
    target.mkdir(parents=True, exist_ok=True)
    (target / "raw_response.json").write_text(
        json.dumps(response, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    content = response_content(response)
    (target / "raw_content.txt").write_text(content, encoding="utf-8")
    valid = True
    error = ""
    if expect_json:
        try:
            parsed = json.loads(content)
            validate_lightweight(parsed)
            (target / "parsed_response.json").write_text(
                json.dumps(parsed, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception as exc:
            valid = False
            error = repr(exc)
            (target / "error.txt").write_text(error, encoding="utf-8")
    return content, valid, error


def usage_record(response: dict):
    usage = response.get("usage") or {}
    choice = (response.get("choices") or [{}])[0]
    return {
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "finish_reason": choice.get("finish_reason"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=pathlib.Path)
    parser.add_argument("output", type=pathlib.Path)
    parser.add_argument("--key-file", type=pathlib.Path, required=True)
    parser.add_argument("--vision-model", default="qwen3-vl-30b-a3b-instruct")
    parser.add_argument("--structure-model", default="qwen3-vl-30b-a3b-instruct")
    parser.add_argument("--base-url", default="https://gpustack.unibe.ch/v1")
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--call-pattern", default="call_*")
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()

    api_key = args.key_file.read_text(encoding="utf-8").strip()
    if not api_key:
        raise SystemExit("key file is empty")
    source_calls = args.source.resolve() / "vlm_calls"
    calls = sorted(path for path in source_calls.glob(args.call_pattern) if path.is_dir())
    if not calls:
        raise SystemExit("no source calls matched")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise SystemExit(f"output is not empty: {output}")

    (output / "lightweight_schema.json").write_text(
        json.dumps(LIGHTWEIGHT_SCHEMA, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "direct_prompt.txt").write_text(direct_prompt(), encoding="utf-8")
    (output / "ledger_prompt.txt").write_text(LEDGER_PROMPT, encoding="utf-8")
    records = []
    for repetition in range(1, args.repetitions + 1):
        run_dir = output / f"run_{repetition:02d}"
        run_dir.mkdir()
        for index, source_call in enumerate(calls, start=1):
            call_dir = run_dir / source_call.name.rstrip(".")
            call_dir.mkdir()
            shutil.copy2(source_call / "input.png", call_dir / "input.png")
            image_b64 = base64.b64encode((call_dir / "input.png").read_bytes()).decode(
                "ascii"
            )

            direct_dir = call_dir / "direct"
            direct_dir.mkdir()
            current_direct_prompt = direct_prompt()
            (direct_dir / "prompt.txt").write_text(
                current_direct_prompt, encoding="utf-8"
            )
            started = time.monotonic()
            try:
                direct_response = request_completion(
                    args.base_url,
                    api_key,
                    json_payload(
                        args.vision_model,
                        image_message(current_direct_prompt, image_b64),
                        1800,
                    ),
                    args.timeout,
                )
                direct_content, direct_valid, direct_error = save_call(
                    direct_dir, direct_response, expect_json=True
                )
                direct_usage = usage_record(direct_response)
            except Exception as exc:
                direct_content, direct_valid, direct_error = "", False, repr(exc)
                direct_usage = {}
                (direct_dir / "error.txt").write_text(
                    direct_error, encoding="utf-8"
                )
            direct_seconds = round(time.monotonic() - started, 3)

            ledger_dir = call_dir / "two_pass" / "visual"
            ledger_dir.mkdir(parents=True)
            (ledger_dir / "prompt.txt").write_text(LEDGER_PROMPT, encoding="utf-8")
            started = time.monotonic()
            try:
                ledger_response = request_completion(
                    args.base_url,
                    api_key,
                    plain_payload(
                        args.vision_model,
                        image_message(LEDGER_PROMPT, image_b64),
                        1800,
                    ),
                    args.timeout,
                )
                ledger, ledger_valid, ledger_error = save_call(
                    ledger_dir, ledger_response, expect_json=False
                )
                ledger_usage = usage_record(ledger_response)
                ledger_valid = bool(ledger.strip())
            except Exception as exc:
                ledger, ledger_valid, ledger_error = "", False, repr(exc)
                ledger_usage = {}
                (ledger_dir / "error.txt").write_text(
                    ledger_error, encoding="utf-8"
                )
            ledger_seconds = round(time.monotonic() - started, 3)

            structure_dir = call_dir / "two_pass" / "structured"
            structure_dir.mkdir()
            struct_prompt = structure_prompt(ledger) if ledger_valid else ""
            (structure_dir / "prompt.txt").write_text(
                struct_prompt, encoding="utf-8"
            )
            structured_valid = False
            structured_error = "visual ledger failed"
            structured_usage = {}
            structure_seconds = 0.0
            if ledger_valid:
                started = time.monotonic()
                try:
                    structured_response = request_completion(
                        args.base_url,
                        api_key,
                        json_payload(
                            args.structure_model,
                            [{"role": "user", "content": struct_prompt}],
                            1800,
                        ),
                        args.timeout,
                    )
                    _, structured_valid, structured_error = save_call(
                        structure_dir, structured_response, expect_json=True
                    )
                    structured_usage = usage_record(structured_response)
                except Exception as exc:
                    structured_error = repr(exc)
                    (structure_dir / "error.txt").write_text(
                        structured_error, encoding="utf-8"
                    )
                structure_seconds = round(time.monotonic() - started, 3)

            record = {
                "repetition": repetition,
                "call": source_call.name,
                "direct": {
                    "valid": direct_valid,
                    "seconds": direct_seconds,
                    "error": direct_error,
                    **direct_usage,
                },
                "two_pass_visual": {
                    "valid": ledger_valid,
                    "seconds": ledger_seconds,
                    "error": ledger_error,
                    **ledger_usage,
                },
                "two_pass_structured": {
                    "valid": structured_valid,
                    "seconds": structure_seconds,
                    "error": structured_error,
                    **structured_usage,
                },
            }
            records.append(record)
            print(
                json.dumps(
                    {
                        "run": repetition,
                        "call": index,
                        "name": source_call.name,
                        "direct": direct_valid,
                        "ledger": ledger_valid,
                        "structured": structured_valid,
                    }
                ),
                flush=True,
            )

    summary = {
        "source": str(args.source.resolve()),
        "vision_model": args.vision_model,
        "structure_model": args.structure_model,
        "repetitions": args.repetitions,
        "calls_per_run": len(calls),
        "direct_valid": sum(record["direct"]["valid"] for record in records),
        "ledger_valid": sum(
            record["two_pass_visual"]["valid"] for record in records
        ),
        "structured_valid": sum(
            record["two_pass_structured"]["valid"] for record in records
        ),
        "total": len(records),
        "records": records,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
