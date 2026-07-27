"""Replay saved survey crops through an OpenAI-compatible vision endpoint.

The input audit supplies the exact PNG, prompt, and response schema used for a
previous provider. Credentials are read from a file and are never copied into
the output. This makes provider and repeatability comparisons independent of a
fresh Paddle/OCR run.
"""

from __future__ import annotations

import argparse
import base64
import json
import pathlib
import re
import shutil
import time
import urllib.error
import urllib.request


def _completion_content(response: dict) -> str:
    choices = response.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        raise ValueError("response has no completion choice")
    message = choices[0].get("message") or {}
    content = message.get("content") if isinstance(message, dict) else None
    if isinstance(content, str) and content:
        return content
    if isinstance(content, list):
        text = "".join(
            str(item.get("text") or "")
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        )
        if text:
            return text
    raise ValueError("response has no textual completion content")


def _request(endpoint: str, api_key: str, payload: dict, timeout: int) -> dict:
    request = urllib.request.Request(
        endpoint.rstrip("/") + "/chat/completions",
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


def _usage(response: dict) -> dict:
    usage = response.get("usage")
    return usage if isinstance(usage, dict) else {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=pathlib.Path)
    parser.add_argument("output", type=pathlib.Path)
    parser.add_argument("--key-file", type=pathlib.Path, required=True)
    parser.add_argument("--model", default="qwen3-vl-30b-a3b-instruct")
    parser.add_argument("--base-url", default="https://gpustack.unibe.ch/v1")
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--max-tokens", type=int, default=3000)
    args = parser.parse_args()

    if args.repetitions < 1:
        raise SystemExit("--repetitions must be positive")
    source_calls = args.source.resolve() / "vlm_calls"
    call_dirs = sorted(path for path in source_calls.glob("call_*") if path.is_dir())
    if not call_dirs:
        raise SystemExit(f"No saved calls found in {source_calls}")
    api_key = args.key_file.read_text(encoding="utf-8").strip()
    if not api_key:
        raise SystemExit("GPUStack key file is empty")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise SystemExit(f"Output directory is not empty: {output}")

    records = []
    review = [
        f"# Qwen survey audit: {output.name}",
        "",
        f"Model: `{args.model}`. Repetitions: {args.repetitions}.",
        "",
        "Every image, prompt, and schema is copied from the source Gemma audit. "
        "Raw content is shown without TextLab repair.",
        "",
    ]
    for repetition in range(1, args.repetitions + 1):
        run_dir = output / f"run_{repetition:02d}"
        run_dir.mkdir()
        review.extend([f"## Run {repetition}", ""])
        for call_index, source_call in enumerate(call_dirs, start=1):
            target = run_dir / source_call.name.rstrip(".")
            target.mkdir()
            for filename in ("input.png", "prompt.txt", "response_schema.json"):
                shutil.copy2(source_call / filename, target / filename)

            prompt = (target / "prompt.txt").read_text(encoding="utf-8")
            schema = json.loads(
                (target / "response_schema.json").read_text(encoding="utf-8")
            )
            image_b64 = base64.b64encode((target / "input.png").read_bytes()).decode(
                "ascii"
            )
            payload = {
                "model": args.model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": "data:image/png;base64," + image_b64
                                },
                            },
                        ],
                    }
                ],
                "temperature": 0,
                "seed": 42,
                "max_tokens": args.max_tokens,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "textlab_survey_response",
                        "strict": True,
                        "schema": schema,
                    },
                },
            }
            contract_match = re.search(
                r"Contract version:\s*([A-Za-z0-9_.-]+)", prompt
            )
            (target / "request_metadata.json").write_text(
                json.dumps(
                    {
                        "provider": "gpustack-university",
                        "model": args.model,
                        "base_url": args.base_url,
                        "temperature": 0,
                        "seed": 42,
                        "max_tokens": args.max_tokens,
                        "contract_version": (
                            contract_match.group(1) if contract_match else ""
                        ),
                        "source_call": source_call.name,
                        "repetition": repetition,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            started = time.monotonic()
            valid_json = False
            error = ""
            content = ""
            usage = {}
            try:
                response = _request(
                    args.base_url, api_key, payload, timeout=args.timeout
                )
                elapsed = time.monotonic() - started
                (target / "raw_response.json").write_text(
                    json.dumps(response, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                content = _completion_content(response)
                (target / "raw_content.txt").write_text(content, encoding="utf-8")
                usage = _usage(response)
                parsed = json.loads(content)
                if not isinstance(parsed, dict):
                    raise ValueError("completion JSON is not an object")
                (target / "parsed_response.json").write_text(
                    json.dumps(parsed, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                valid_json = True
            except Exception as exc:
                elapsed = time.monotonic() - started
                error = repr(exc)
                (target / "error.txt").write_text(error, encoding="utf-8")

            record = {
                "repetition": repetition,
                "call": source_call.name,
                "valid_json": valid_json,
                "elapsed_seconds": round(elapsed, 3),
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "error": error,
            }
            records.append(record)
            relative = target.relative_to(output).as_posix()
            review.extend(
                [
                    f"### {source_call.name}",
                    "",
                    f"![exact model input]({relative}/input.png)",
                    "",
                    f"Valid JSON: **{valid_json}** · elapsed: {elapsed:.3f}s",
                    "",
                    "```json",
                    content or error,
                    "```",
                    "",
                ]
            )
            print(
                json.dumps(
                    {
                        "run": repetition,
                        "call": call_index,
                        "name": source_call.name,
                        "valid": valid_json,
                        "seconds": round(elapsed, 3),
                    }
                ),
                flush=True,
            )

    summary = {
        "source": str(args.source.resolve()),
        "provider": "gpustack-university",
        "model": args.model,
        "repetitions": args.repetitions,
        "calls_per_run": len(call_dirs),
        "valid_json": sum(record["valid_json"] for record in records),
        "total_calls": len(records),
        "completion_tokens": sum(
            int(record["completion_tokens"] or 0) for record in records
        ),
        "elapsed_seconds": round(
            sum(float(record["elapsed_seconds"]) for record in records), 3
        ),
        "records": records,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "REVIEW.md").write_text("\n".join(review), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
