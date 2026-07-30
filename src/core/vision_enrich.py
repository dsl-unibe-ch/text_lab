"""Shared local vision-model client and optional document enrichments.

The base OCR pipeline never imports or contacts a large VLM unless the caller
explicitly requests an enrichment. The initial provider is the Ollama server
already launched with TextLab; its interface is deliberately small so a local
Qwen provider or the university GPUStack can be benchmarked behind the same
contract later.
"""

from __future__ import annotations

import base64
import fcntl
import json
import os
from pathlib import Path
import re
import time
from typing import Any, Dict, Optional, Protocol
import urllib.error
import urllib.request

try:
    from core import doc_ir
except ImportError:  # pragma: no cover - standalone imports
    import doc_ir  # type: ignore


#: How long to wait for Ollama to actually free a model's VRAM after being asked
#: to unload it. Measured: the request returns instantly with all ~20 GiB still
#: resident, and the runner takes a few seconds to exit.
UNLOAD_WAIT_TIMEOUT = float(os.environ.get("TEXTLAB_UNLOAD_WAIT_TIMEOUT", "60"))


class VisionModelError(RuntimeError):
    """A controlled local/remote vision-model failure."""


class VisionClient(Protocol):
    model: str
    provider: str

    def analyze(self, image_bytes: bytes, prompt: str, schema: dict) -> Dict[str, Any]: ...


def _base_url(value: Optional[str] = None) -> str:
    host = value or os.environ.get("OLLAMA_HOST", "127.0.0.1:11434")
    if not host.startswith(("http://", "https://")):
        host = "http://" + host
    return host.rstrip("/")


def _ollama_request(base_url: str, path: str, payload: Optional[dict] = None,
                    timeout: float = 60.0) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        base_url + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def loaded_models(base_url: Optional[str] = None) -> list:
    """Names of the models Ollama currently holds in VRAM."""
    try:
        response = _ollama_request(_base_url(base_url), "/api/ps")
    except Exception:
        return []
    return [
        str(item.get("model") or item.get("name") or "")
        for item in (response.get("models") or [])
        if isinstance(item, dict)
    ]


def free_gpu(base_url: Optional[str] = None,
             timeout: float = None) -> list:
    """Evict every Ollama model and block until the VRAM is really released.

    Ollama accounts for its own models, so it never needs this. A *non-Ollama*
    GPU consumer does: TextLab's PaddleOCR-VL worker is a separate process that
    allocates ~8.4 GiB, which does not fit beside the ~20 GiB vision model on a
    23 GiB card. ``keep_alive: 0`` only *schedules* an unload -- it returns with
    the whole model still resident and the runner takes seconds to exit -- so
    without waiting here the worker starts against a nearly full card and dies
    part-way through loading its own weights.

    Called at the point of need rather than after each document, so a model that
    is still useful stays warm (see ``OllamaVisionClient.keep_alive``).
    """
    if timeout is None:
        timeout = UNLOAD_WAIT_TIMEOUT
    url = _base_url(base_url)
    resident = loaded_models(url)
    if not resident:
        return []
    for name in resident:
        try:
            _ollama_request(
                url, "/api/generate",
                {"model": name, "prompt": "", "stream": False, "keep_alive": 0},
            )
        except Exception:
            pass
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not loaded_models(url):
            break
        time.sleep(0.25)
    return resident


class OllamaVisionClient:
    """Structured image analysis through TextLab's local Ollama server.

    A per-Ollama-server file lock serialises heavyweight model use inside a
    TextLab job. Before loading the target model, other Ollama models are
    explicitly unloaded so the 23 GiB RTX 4090 cannot retain a chat model next
    to the ~20 GiB Qwen3-VL vision model (the default; override with
    TEXTLAB_VISION_MODEL).
    """

    provider = "ollama-local"

    def __init__(
        self,
        model: Optional[str] = None,
        *,
        base_url: Optional[str] = None,
        timeout: int = 300,
        keep_alive: str = "5m",
        unload_others: bool = True,
        lock_timeout: int = 120,
        audit_dir=None,
    ):
        self.model = model or os.environ.get(
            "TEXTLAB_VISION_MODEL", "qwen3-vl:30b-a3b-instruct"
        )
        self.base_url = _base_url(base_url)
        self.timeout = timeout
        self.keep_alive = keep_alive
        self.unload_others = unload_others
        self.lock_timeout = lock_timeout
        self.audit_dir = Path(audit_dir) if audit_dir is not None else None
        self.num_ctx = int(os.environ.get("TEXTLAB_VISION_NUM_CTX", "8192"))
        # The default model is the non-thinking Instruct build, so this budget
        # is spent entirely on the answer. Ordinary sections need 177-949
        # tokens, but a dense 16-row rating matrix legitimately needs ~6.8k, so
        # the headroom is for that worst case rather than for reasoning. Do not
        # lower this below ~7000 without re-checking the widest matrix in
        # instruct-model-benchmark-results.md.
        self.num_predict = int(os.environ.get("TEXTLAB_VISION_NUM_PREDICT", "8000"))
        self._lock_file = None
        self._prepared = False
        self._audit_count = 0

    def _begin_audit(self, image_bytes: bytes, prompt: str, schema: dict):
        """Persist the exact VLM request when an explicit audit dir is set.

        This is disabled by default because it intentionally retains sensitive
        document crops. The caller owns cleanup and access control for the
        supplied persistent directory.
        """
        if self.audit_dir is None:
            return None
        self._audit_count += 1
        question_match = re.search(r'"question_id":"([^"]+)"', prompt)
        section_match = re.search(r"Audit section reference:\s*([A-Za-z0-9_-]+)", prompt)
        question_id = (
            question_match.group(1)
            if question_match
            else (section_match.group(1) if section_match else "vision")
        )
        contract_match = re.search(r"Contract version:\s*([A-Za-z0-9_.-]+)", prompt)
        safe_id = re.sub(r"[^A-Za-z0-9_.-]", "_", question_id)[:80] or "vision"
        call_dir = self.audit_dir / f"call_{self._audit_count:04d}_{safe_id}"
        while call_dir.exists():
            self._audit_count += 1
            call_dir = self.audit_dir / f"call_{self._audit_count:04d}_{safe_id}"
        call_dir.mkdir(parents=True, exist_ok=False)
        (call_dir / "input.png").write_bytes(image_bytes)
        (call_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
        (call_dir / "response_schema.json").write_text(
            json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (call_dir / "request_metadata.json").write_text(
            json.dumps(
                {
                    "provider": self.provider,
                    "model": self.model,
                    "base_url": self.base_url,
                    "think": False,
                    "temperature": 0,
                    "seed": 42,
                    "num_ctx": self.num_ctx,
                    "num_predict": self.num_predict,
                    "contract_version": contract_match.group(1) if contract_match else "",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return call_dir

    @staticmethod
    def _write_audit_text(call_dir, filename: str, value: str):
        if call_dir is not None:
            (call_dir / filename).write_text(str(value), encoding="utf-8")

    def _request(self, path: str, payload: Optional[dict] = None) -> dict:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.base_url + path,
            data=data,
            headers={"Content-Type": "application/json"},
            method="GET" if payload is None else "POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.load(response)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise VisionModelError(f"Ollama request failed ({path}): {exc}") from exc

    def _acquire_lock(self):
        if self._lock_file is not None:
            return
        endpoint = re.sub(r"[^A-Za-z0-9_.-]", "_", self.base_url)
        lock_path = Path("/tmp") / f"textlab_vision_{os.getuid()}_{endpoint}.lock"
        handle = lock_path.open("a+")
        deadline = time.monotonic() + self.lock_timeout
        while True:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                self._lock_file = handle
                return
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    handle.close()
                    raise VisionModelError(
                        "Timed out waiting for another TextLab GPU analysis job"
                    )
                time.sleep(0.25)

    def _available_models(self) -> set[str]:
        response = self._request("/api/tags")
        names = set()
        for item in response.get("models") or []:
            if isinstance(item, dict):
                name = item.get("model") or item.get("name")
                if name:
                    names.add(str(name))
        return names

    def _unload_other_models(self):
        response = self._request("/api/ps")
        for item in response.get("models") or []:
            if not isinstance(item, dict):
                continue
            name = str(item.get("model") or item.get("name") or "")
            if not name or name == self.model:
                continue
            self._request(
                "/api/generate",
                {"model": name, "prompt": "", "stream": False, "keep_alive": 0},
            )

    def prepare(self):
        if self._prepared:
            return
        self._acquire_lock()
        try:
            available = self._available_models()
            if self.model not in available:
                raise VisionModelError(
                    f"Local vision model '{self.model}' is not staged in Ollama"
                )
            if self.unload_others:
                self._unload_other_models()
            self._prepared = True
        except Exception:
            self.close()
            raise

    def analyze(self, image_bytes: bytes, prompt: str, schema: dict) -> Dict[str, Any]:
        self.prepare()
        audit_call = self._begin_audit(image_bytes, prompt, schema)
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [base64.b64encode(image_bytes).decode("ascii")],
                }
            ],
            "stream": False,
            "think": False,
            "format": schema,
            "options": {
                "temperature": 0,
                "seed": 42,
                "num_ctx": self.num_ctx,
                "num_predict": self.num_predict,
            },
            "keep_alive": self.keep_alive,
        }
        try:
            response = self._request("/api/chat", payload)
        except Exception as exc:
            self._write_audit_text(audit_call, "error.txt", repr(exc))
            raise
        if audit_call is not None:
            (audit_call / "raw_response.json").write_text(
                json.dumps(response, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        message = response.get("message") or {}
        content = message.get("content") if isinstance(message, dict) else None
        if not content:
            self._write_audit_text(audit_call, "error.txt", "no final content")
            raise VisionModelError("Vision model returned no final content")
        self._write_audit_text(audit_call, "raw_content.txt", content)
        try:
            result = json.loads(content)
        except json.JSONDecodeError as exc:
            self._write_audit_text(audit_call, "error.txt", repr(exc))
            raise VisionModelError("Vision model returned invalid JSON") from exc
        if not isinstance(result, dict):
            self._write_audit_text(audit_call, "error.txt", "JSON was not an object")
            raise VisionModelError("Vision model JSON must be an object")
        if audit_call is not None:
            (audit_call / "parsed_response.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        return result

    def close(self, *, unload_model: bool = False, wait_for_unload: bool = True):
        """Release the job lock; optionally evict the model from VRAM.

        Eviction is *not* the default and callers rarely want it: the model
        expires on its own after ``keep_alive``, and whatever needs the card next
        frees it at the point of need (:func:`free_gpu`). Unloading here instead
        would throw away a warm 20 GiB model that the next document may reuse.
        """
        if unload_model and self._prepared:
            try:
                self._request(
                    "/api/generate",
                    {
                        "model": self.model,
                        "prompt": "",
                        "stream": False,
                        "keep_alive": 0,
                    },
                )
                if wait_for_unload:
                    self._await_unload()
            except Exception:
                pass
        self._prepared = False
        if self._lock_file is not None:
            try:
                fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_UN)
                self._lock_file.close()
            finally:
                self._lock_file = None

    def _await_unload(self, timeout: float = UNLOAD_WAIT_TIMEOUT):
        """Block until Ollama has actually released the model's VRAM.

        ``keep_alive: 0`` only *schedules* the unload: the request returns while
        the whole model is still resident, and the runner needs seconds to exit.
        The vision model is ~20 GiB of a 23 GiB card, so whatever runs next --
        in practice the PaddleOCR-VL worker for the user's next document -- then
        starts against a nearly full GPU and dies part-way through loading.
        Waiting here costs a few seconds once and keeps that GPU handover clean.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                loaded = {
                    str(item.get("model") or item.get("name") or "")
                    for item in (self._request("/api/ps").get("models") or [])
                    if isinstance(item, dict)
                }
            except Exception:
                return
            if self.model not in loaded:
                return
            time.sleep(0.25)

    def __enter__(self):
        self.prepare()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


FIGURE_DESCRIPTION_SCHEMA = {
    "type": "object",
    "properties": {
        "description": {"type": "string"},
        "visible_text": {"type": "string"},
    },
    "required": ["description", "visible_text"],
    "additionalProperties": False,
}


def describe_page_figures(page: "doc_ir.Page", client: VisionClient):
    """Attach generated descriptions to figure regions that carry an asset."""
    for region in page.regions:
        if region.type != doc_ir.FIGURE or not region.asset:
            continue
        encoded = region.asset.get("b64")
        if not encoded:
            continue
        try:
            image_bytes = base64.b64decode(encoded)
            printed_context = region.text.strip()
            prompt = (
                "Describe this document figure or image for a reader who cannot see it. "
                "Be factual and concise; do not infer facts not visible in the image. "
                "Transcribe important visible text separately. The printed OCR caption, "
                f"if any, is: {printed_context!r}. Return only the requested JSON."
            )
            result = client.analyze(image_bytes, prompt, FIGURE_DESCRIPTION_SCHEMA)
            description = str(result.get("description") or "").strip()
            visible_text = str(result.get("visible_text") or "").strip()
            if not description:
                raise VisionModelError("Figure description was empty")
            region.visual_description = doc_ir.VisualDescription(
                description=description,
                visible_text=visible_text,
                source=client.provider,
                model=client.model,
            )
        except Exception as exc:
            region.warnings.append(f"figure description failed: {exc}")
