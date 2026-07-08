"""
Client-side "magic prompt" expansion: turn a free-form user prompt into the
structured JSON caption that the Ideogram-4 model requires.

Ideogram 4 was trained exclusively on structured JSON captions; feeding it raw
plain text trips its baked-in safety filter (the gray "Image blocked by safety
filter" placeholder). So every prompt must be expanded into a JSON caption first.

This module runs in the Streamlit ``text_lab_main`` env and drives the local
Ollama server (no API key, fully local) with the already-loaded chat model. The
``ideogram4`` package is NOT importable here, so the captioner system prompt is
vendored alongside this file as ``magic_prompt_v1.txt`` (keep it in sync with the
fork's ``src/ideogram4/magic_prompt_system_prompts/v1.txt``).

The exact Ollama config below was validated empirically — all three knobs matter:
  * ``think=False``       — otherwise thinking-mode models run away in loops.
  * ``num_ctx`` >= ~8192  — the captioner system prompt is ~6.8k tokens; the
                            default 4096 silently truncates it.
  * ``format=<schema>``   — constrained decoding guarantees schema-shaped JSON,
                            eliminating the "flattened schema" slips small models
                            make otherwise.
With these, ministral-3:14b and gemma4:26b both produced 5/5 valid captions.
"""

from __future__ import annotations

import functools
import json
import math
import os
import pathlib
import re

# NOTE: `ollama` is imported lazily inside `_call_ollama` (not at module top) so
# the pure helpers below (schema, parser, normalizer, validator) can be reused
# from the `imagegen_backend` env, which does not ship the ollama client.

_PROMPT_FILE = pathlib.Path(__file__).with_name("magic_prompt_v1.txt")

# Total context window for the captioner call. The system prompt is ~6.8k tokens,
# so this must stay comfortably above 4096 to avoid truncation and leave room for
# a verbose (multi-element) caption. Tested working at 8192; overridable.
CAPTION_NUM_CTX: int = int(os.environ.get("TEXT_LAB_MAGIC_PROMPT_NUM_CTX", "8192"))

# JSON schema handed to Ollama's structured-output (`format`) so the model is
# constrained to emit exactly the Ideogram caption shape.
CAPTION_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "aspect_ratio": {"type": "string"},
        "high_level_description": {"type": "string"},
        "compositional_deconstruction": {
            "type": "object",
            "properties": {
                "background": {"type": "string"},
                "elements": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string"},
                            "desc": {"type": "string"},
                            "text": {"type": "string"},
                        },
                        "required": ["type", "desc"],
                    },
                },
            },
            "required": ["background", "elements"],
        },
    },
    "required": ["aspect_ratio", "high_level_description", "compositional_deconstruction"],
}


@functools.lru_cache(maxsize=1)
def _load_sections() -> dict[str, str]:
    """Parse the vendored prompt file into its ``[SECTION]`` blocks.

    Mirrors ``ideogram4.magic_prompt._load_sections``: ``[NAME]`` markers alone on
    a line delimit sections; returns lower-cased section name -> stripped body.
    """
    raw = _PROMPT_FILE.read_text(encoding="utf-8")
    sections: dict[str, str] = {}
    current: str | None = None
    lines: list[str] = []
    for line in raw.splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]") and " " not in stripped:
            if current is not None:
                sections[current] = "\n".join(lines).strip()
            current = stripped[1:-1].strip().lower()
            lines = []
        else:
            lines.append(line)
    if current is not None:
        sections[current] = "\n".join(lines).strip()
    if "system" not in sections:
        raise RuntimeError(f"{_PROMPT_FILE} has no [SYSTEM] section")
    return sections


def aspect_ratio_from_size(width: int, height: int) -> str:
    """Reduce a pixel ``width``x``height`` to a ``"W:H"`` aspect-ratio string."""
    w = max(1, int(width))
    h = max(1, int(height))
    divisor = math.gcd(w, h) or 1
    return f"{w // divisor}:{h // divisor}"


def _build_messages(prompt: str, aspect_ratio: str) -> list[dict]:
    sections = _load_sections()
    template = sections.get("user") or "TARGET IMAGE ASPECT RATIO: {{aspect_ratio}} (width:height)."
    user = template.replace("{{aspect_ratio}}", aspect_ratio)
    if "{{original_prompt}}" in user:
        user = user.replace("{{original_prompt}}", prompt)
    else:
        user = f"{user}\n\n{prompt}"
    return [
        {"role": "system", "content": sections["system"]},
        {"role": "user", "content": user},
    ]


def _extract_json(text: str) -> dict | None:
    """Parse a JSON object from model output, tolerating fences/prose around it."""
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[index:])
        except Exception:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def _clean_prose(value):
    """Strip stray markdown bold that some models sprinkle into prose fields."""
    if isinstance(value, str):
        return re.sub(r"\*\*(.*?)\*\*", r"\1", value)
    return value


def _normalize(obj: dict | None, aspect_ratio: str) -> dict | None:
    """Reshape common near-miss shapes into the exact Ideogram caption schema.

    Handles: flattened ``background``/``elements`` at top level, ``description``
    used instead of ``desc``, ``type`` values other than obj/text, empty ``text``
    on object elements, and missing ``aspect_ratio``.
    """
    if not isinstance(obj, dict):
        return None
    hld = _clean_prose(obj.get("high_level_description") or obj.get("description") or "")
    cd = obj.get("compositional_deconstruction")
    if not isinstance(cd, dict):
        cd = {"background": obj.get("background", ""), "elements": obj.get("elements", [])}
    background = _clean_prose(cd.get("background") or "")
    raw_elements = cd.get("elements") if isinstance(cd.get("elements"), list) else []
    elements: list[dict] = []
    for el in raw_elements:
        if not isinstance(el, dict):
            continue
        desc = _clean_prose(el.get("desc") or el.get("description") or el.get("content") or "")
        text = el.get("text")
        if isinstance(text, str) and text.strip() and el.get("type") != "obj":
            elements.append({"type": "text", "text": text, "desc": desc})
        else:
            elements.append({"type": "obj", "desc": desc})
    if not elements:
        elements = [{"type": "obj", "desc": hld or background or "the subject"}]
    return {
        "aspect_ratio": obj.get("aspect_ratio") or aspect_ratio,
        "high_level_description": hld,
        "compositional_deconstruction": {
            "background": background or "plain neutral background",
            "elements": elements,
        },
    }


def _is_valid(obj: dict | None) -> bool:
    if not isinstance(obj, dict):
        return False
    if not isinstance(obj.get("high_level_description"), str) or not obj["high_level_description"].strip():
        return False
    cd = obj.get("compositional_deconstruction")
    if not isinstance(cd, dict) or not isinstance(cd.get("background"), str):
        return False
    elements = cd.get("elements")
    if not isinstance(elements, list) or not elements:
        return False
    for el in elements:
        if not isinstance(el, dict) or el.get("type") not in ("obj", "text"):
            return False
        if not isinstance(el.get("desc"), str):
            return False
    return True


def _minimal_caption(prompt: str, aspect_ratio: str) -> dict:
    """Last-resort caption so we never fall back to sending raw text to Ideogram."""
    return {
        "aspect_ratio": aspect_ratio,
        "high_level_description": prompt.strip() or "an image",
        "compositional_deconstruction": {
            "background": "plain neutral background",
            "elements": [{"type": "obj", "desc": prompt.strip() or "the subject"}],
        },
    }


def _call_ollama(model: str, messages: list[dict]) -> str:
    import ollama

    resp = ollama.chat(
        model=model,
        messages=messages,
        stream=False,
        think=False,
        format=CAPTION_SCHEMA,
        options={"temperature": 0.2, "num_ctx": CAPTION_NUM_CTX},
    )
    message = resp["message"] if isinstance(resp, dict) else resp.message
    return (message.get("content") if isinstance(message, dict) else message.content) or ""


def expand_prompt_ollama(
    prompt: str,
    model: str,
    aspect_ratio: str = "1:1",
    attempts: int = 2,
) -> str:
    """Expand a free-form prompt into an Ideogram caption JSON **string**.

    Drives the local Ollama ``model`` with constrained JSON output. Retries once
    on a malformed/invalid response, then falls back to a minimal hand-built
    caption. The return value is a minified JSON string ready to hand to the image
    MCP tool (which passes already-structured caption JSON straight through).

    Never raises for model-quality reasons and never returns raw text — the worst
    case is a plainer caption, not a gray safety block.
    """
    messages = _build_messages(prompt, aspect_ratio)
    last: dict | None = None
    for _ in range(max(1, attempts)):
        try:
            caption = _normalize(_extract_json(_call_ollama(model, messages)), aspect_ratio)
        except Exception:
            caption = None
        if _is_valid(caption):
            return json.dumps(caption, ensure_ascii=False, separators=(",", ":"))
        last = caption
    caption = last if _is_valid(last) else _minimal_caption(prompt, aspect_ratio)
    return json.dumps(caption, ensure_ascii=False, separators=(",", ":"))
