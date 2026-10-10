"""Research topics for each paper, extracted from its abstract by an LLM.

The model is reached through an OpenAI-compatible API: the session's
Ollama server, or GPUStack (setting ``gpustack_url``) with the user's API
key. The model is asked for 3 to 8 topics, each a specific label within a
broad category; categories are what connect papers in the graphs.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from textlab.common.config import get_settings
from textlab.common.ollama import server_address
from textlab.common.progress import Progress, ProgressCallback, no_progress
from textlab.features.knowledge_graph.corpus import TABLE_JSONL, TOPICS_JSONL
from textlab.features.knowledge_graph.models import TopicSummary

LOGGER = logging.getLogger(__name__)

#: The model offered on GPUStack.
GPUSTACK_MODEL = "gpt-oss-120b"
#: At most this many topics are kept per paper.
MAX_TOPICS = 8

# Model input: kept exactly as written. Trailing spaces are written as
# \x20 so that editors do not strip them.
SYSTEM_PROMPT = """
You are an expert research curator.

Your task: read the provided scientific text (title, abstract or excerpt)\x20
and extract high-quality, hierarchical research topics.

Output requirements:
- Return ONLY valid JSON (no explanation, no intro, no markdown).
- Use this exact schema:

{
  "topics": [
    {\x20
      "category": "string",
      "label": "string",\x20
      "confidence": 0.0,\x20
      "rationale": "string"\x20
    }
  ]
}

Rules:
- Provide 3–8 topics.
- Each topic has TWO levels:
  * "category": A BROAD research area (e.g., "Machine Learning", "Climate Science", "Medical Imaging", "Neuroscience")
  * "label": A SPECIFIC topic within that category (2–6 words)
- Categories should be general enough that multiple papers could share them.
- Labels must be specific to the scientific content of this paper.
- Avoid generic words ("methods", "results", "datasets", "analysis", "study", "paper", "experiment").
- Avoid meta-topics ("limitations", "future work", "introduction section").
- Do not mention humans or animals unless they are central to the research question.
- Confidence should be between 0 and 1.
- Rationales must be one sentence each.

Example output:
{
  "topics": [
    { "category": "Computer Vision", "label": "Object Detection Networks", "confidence": 0.95, "rationale": "Paper focuses on improving YOLO architecture." },
    { "category": "Deep Learning", "label": "Transformer Architectures", "confidence": 0.85, "rationale": "Uses attention mechanisms extensively." }
  ]
}
"""  # noqa: E501


def ollama_client() -> Any:
    """Return an OpenAI-compatible client for the session's Ollama server.

    Returns:
        An ``openai.OpenAI`` client.
    """
    from openai import OpenAI

    host, port = server_address()
    return OpenAI(base_url=f"http://{host}:{port}/v1", api_key="ollama")


def gpustack_client(api_key: str) -> Any:
    """Return a client for GPUStack.

    Args:
        api_key: The user's GPUStack API key.

    Returns:
        An ``openai.OpenAI`` client.

    Raises:
        MissingSettingError: If ``gpustack_url`` is not configured.
    """
    from openai import OpenAI

    return OpenAI(
        base_url=get_settings().require("gpustack_url"), api_key=api_key
    )


def run_llm(
    messages: list[dict[str, str]],
    client: Any,
    model: str = GPUSTACK_MODEL,
    temperature: float = 0.2,
    top_p: float = 0.9,
    max_tokens: int = 16384,
    retries: int = 2,
) -> str:
    """Return a model's answer, retrying failed calls.

    Args:
        messages: The chat messages.
        client: An OpenAI-compatible client.
        model: The model name.
        temperature: Sampling temperature.
        top_p: Nucleus sampling parameter.
        max_tokens: Maximum length of the answer.
        retries: Further attempts after a failed call.

    Returns:
        The answer text.

    Raises:
        RuntimeError: If every attempt fails.
    """
    for attempt in range(retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                messages=messages,
            )
            return response.choices[0].message.content
        except Exception as exc:
            if attempt == retries:
                raise RuntimeError(
                    f"LLM failed after {retries + 1} attempts: {exc}"
                ) from exc
            LOGGER.warning(
                "LLM error, retrying (%d/%d): %s", attempt + 1, retries, exc
            )
    raise AssertionError("unreachable")


def build_messages(title: str, abstract: str) -> list[dict[str, str]]:
    """Return the messages that ask for a paper's topics.

    Args:
        title: The paper's title.
        abstract: Its abstract.

    Returns:
        The system prompt and the paper.
    """
    user_content = (
        f"TITLE: {title.strip()}\n\nABSTRACT_OR_TEXT:\n{abstract.strip()}"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def parse_topics(raw: str) -> list[dict[str, Any]]:
    """Return the topics in a model's answer, cleaned.

    Markdown fences and text around the JSON object are tolerated. Topics
    without a label are dropped, a missing category becomes ``"General"``,
    confidences are clipped to [0, 1], and at most :data:`MAX_TOPICS` are
    kept.

    Args:
        raw: The answer.

    Returns:
        Topics with ``category``, ``label``, ``confidence`` and
        ``rationale``.

    Raises:
        ValueError: If the answer has no JSON object with a ``topics``
            list (``json.JSONDecodeError`` is a ``ValueError``).
    """
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("` \n")
        if raw.lower().startswith("json"):
            raw = raw[4:].lstrip()

    try:
        answer = json.loads(raw)
    except json.JSONDecodeError:
        first, last = raw.find("{"), raw.rfind("}")
        if first == -1 or last <= first:
            raise
        answer = json.loads(raw[first : last + 1])

    if not isinstance(answer, dict) or not isinstance(
        answer.get("topics"), list
    ):
        raise ValueError("Model output missing top-level 'topics' list.")

    topics = []
    for topic in answer["topics"]:
        if not isinstance(topic, dict):
            continue
        label = (topic.get("label") or "").strip()
        if not label:
            continue
        try:
            confidence = float(topic.get("confidence", 0.0))
        except (TypeError, ValueError, OverflowError):
            confidence = 0.0
        topics.append(
            {
                "category": (topic.get("category") or "").strip() or "General",
                "label": label,
                "confidence": max(0.0, min(1.0, confidence)),
                "rationale": (topic.get("rationale") or "").strip(),
            }
        )
    return topics[:MAX_TOPICS]


def extract_topics(
    corpus: str | Path,
    client: Any,
    model: str = GPUSTACK_MODEL,
    on_progress: ProgressCallback = no_progress,
) -> TopicSummary:
    """Add topics to every paper of a corpus.

    Reads ``corpus_table.jsonl`` and writes
    ``corpus_table.with_topics.jsonl``, each record with ``topics`` and
    ``topics_ts`` (UTC time), or with empty ``topics`` and
    ``topics_note: "no_abstract"`` when it has no abstract. The file is
    replaced only when every paper is done.

    Args:
        corpus: The corpus folder.
        client: An OpenAI-compatible client (:func:`ollama_client` or
            :func:`gpustack_client`).
        model: The model name.
        on_progress: Receives a :class:`Progress` update per paper.

    Returns:
        The counts and the path written.

    Raises:
        FileNotFoundError: If the corpus has no table yet.
        RuntimeError: If the model cannot be reached.
        ValueError: If an answer has no topics list.
    """
    corpus = Path(corpus)
    source = corpus / TABLE_JSONL
    if not source.exists():
        raise FileNotFoundError(f"Missing input file: {source}")
    partial = corpus / "corpus_table.with_topics.tmp.jsonl"
    target = corpus / TOPICS_JSONL

    with source.open(encoding="utf-8") as file:
        total = sum(1 for _ in file)
    processed = skipped = 0
    with (
        source.open(encoding="utf-8") as lines,
        partial.open("w", encoding="utf-8") as out,
    ):
        for current, line in enumerate(lines, start=1):
            record = json.loads(line)
            title = record.get("title", "")
            abstract = (record.get("abstract") or "").strip()
            paper_id = record.get("paper_id", "unknown")
            fraction = (current - 1) / total
            if abstract:
                on_progress(
                    Progress(
                        f"Processing {current}/{total}: {paper_id} - "
                        f"{title[:50]}...",
                        fraction,
                    )
                )
                raw = run_llm(build_messages(title, abstract), client, model)
                record["topics"] = parse_topics(raw)
                record["topics_ts"] = datetime.now(UTC).isoformat()
                processed += 1
            else:
                on_progress(
                    Progress(
                        f"Skipping {current}/{total}: {paper_id} "
                        "(no abstract)",
                        fraction,
                    )
                )
                record["topics"] = []
                record["topics_note"] = "no_abstract"
                skipped += 1
            out.write(json.dumps(record, ensure_ascii=False) + "\n")

    partial.replace(target)
    LOGGER.info(
        "Topics extracted: %d with abstract, %d without, written to %s",
        processed,
        skipped,
        target,
    )
    return TopicSummary(processed, skipped, target)
