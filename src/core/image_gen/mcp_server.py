"""
Model Context Protocol (MCP) server exposing local Ideogram-4 text-to-image
generation as a single tool.

Launched as a stdio subprocess by ``image_engine``. It runs under the isolated
``imagegen_backend`` conda env (``TEXT_LAB_IMAGEGEN_PYTHON``), which ships the
``ideogram4`` package on transformers 5.x — the version the ideogram-4-nf4 model
(Qwen3-VL text encoder) requires. The pipeline is loaded lazily on the first call
and cached in a module global for the process lifetime, so a persistent server
reuses it across generations.

The weights repo is gated. The shared cache is pre-downloaded and complete, so by
default the server loads OFFLINE — no network, no per-user HF token needed, which
is what makes it work for every user and not just whoever populated the cache. Set
``TEXT_LAB_IMAGEGEN_OFFLINE=0`` to force online loading (e.g. an admin refreshing
the cache), which then needs a resolved token (see ``image_config.get_hf_token``).

Free-form prompts are expanded into Ideogram's structured caption JSON through
the configured magic-prompt provider before local generation. Already-structured
caption JSON is passed through unchanged.
"""

import datetime
import json
import logging
import os
import sys
import traceback
import uuid

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

# The ideogram-4-nf4 repo is GATED: without a token, HEAD requests to it 401
# (not 404), so any user who didn't personally `huggingface-cli login` fails. The
# shared cache is complete (single-file weights; the sharded .index.json is
# legitimately absent and recorded in .no_exist), so offline is the default: it
# honors the .no_exist marker, never touches the network, and needs no token.
# Opt out (TEXT_LAB_IMAGEGEN_OFFLINE=0) only to refresh the cache online.
if os.environ.get("TEXT_LAB_IMAGEGEN_OFFLINE", "1") == "1":
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

# This script is launched as an isolated subprocess by the MCP stdio client, so
# make the 'src' root importable for `core.*` packages.
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from mcp.server.fastmcp import FastMCP

from core.image_gen.image_config import (
    DEFAULT_PRESET,
    MODEL_REPO,
    get_artifacts_dir,
    get_hf_home,
    get_hf_token,
)

# Point the HF libraries at the shared cache that holds the pre-downloaded NF4
# weights (reachable on host and in-container), before any of them is imported.
_HF_HOME = get_hf_home()
os.environ["HF_HOME"] = _HF_HOME
os.environ.setdefault("HF_HUB_CACHE", os.path.join(_HF_HOME, "hub"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(_HF_HOME, "hub"))

# Authenticate the gated repo (no-op in offline mode). Resolved from env/token
# files so no personal token is hard-coded; see image_config.get_hf_token.
_HF_TOKEN = get_hf_token()
if _HF_TOKEN:
    os.environ.setdefault("HF_TOKEN", _HF_TOKEN)
    os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", _HF_TOKEN)

# Keep logging off stdout/stderr so it can't corrupt the JSON-RPC stream.
logging.basicConfig(level=logging.ERROR)
logging.getLogger("mcp").setLevel(logging.ERROR)

# --- File-based debug logging (stdout/stderr must stay clean for JSON-RPC) -----
_DEBUG_LOG = os.environ.get(
    "IMAGE_GEN_DEBUG_LOG",
    os.path.join(os.path.expanduser("~"), "image_gen_debug.log"),
)


def _debug(msg: str) -> None:
    """Append a timestamped line to the image-gen debug log (best effort)."""
    try:
        with open(_DEBUG_LOG, "a", encoding="utf-8") as f:
            f.write(f"[{datetime.datetime.now().isoformat(timespec='seconds')}] {msg}\n")
    except Exception:
        pass


_MODEL_CACHE_DIR = os.path.join(
    os.environ.get("HF_HUB_CACHE", os.path.join(_HF_HOME, "hub")),
    "models--" + MODEL_REPO.replace("/", "--"),
)
_debug("=" * 60)
_debug(f"MCP server startup pid={os.getpid()} exe={sys.executable}")
_debug(f"HF_HOME={os.environ.get('HF_HOME')}")
_debug(f"HF_HUB_CACHE={os.environ.get('HF_HUB_CACHE')}")
_debug(f"HF_HUB_OFFLINE={os.environ.get('HF_HUB_OFFLINE')} "
       f"TRANSFORMERS_OFFLINE={os.environ.get('TRANSFORMERS_OFFLINE')}")
_debug(f"HF_TOKEN set={bool(os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN'))}")
_debug(f"model repo={MODEL_REPO}")
_debug(f"model cache dir={_MODEL_CACHE_DIR} exists={os.path.isdir(_MODEL_CACHE_DIR)}")

mcp = FastMCP("Ideogram Image Generation MCP Server")

_PIPELINE = None  # loaded lazily, cached for the process lifetime
_MAGIC_PROMPT = None


def _looks_like_caption_json(text: str) -> bool:
    try:
        data = json.loads(text)
    except Exception:
        return False
    return (
        isinstance(data, dict)
        and isinstance(data.get("high_level_description"), str)
        and isinstance(data.get("compositional_deconstruction"), dict)
    )


def _aspect_ratio(width: int, height: int) -> str:
    import math

    w = max(1, int(width))
    h = max(1, int(height))
    divisor = math.gcd(w, h)
    return f"{w // divisor}:{h // divisor}"


def _remote_api_key() -> str | None:
    """API key for the remote (GPUStack/OpenRouter) magic-prompt providers, if set."""
    return (
        os.environ.get("GPUSTACK_API_KEY")
        or os.environ.get("MAGIC_PROMPT_API_KEY")
        or os.environ.get("OPENROUTER_API_KEY")
    )


def _get_magic_prompt():
    """Lazily build the configured remote magic-prompt provider (requires a key)."""
    global _MAGIC_PROMPT
    if _MAGIC_PROMPT is None:
        from ideogram4 import MAGIC_PROMPTS

        model_name = os.environ.get("TEXT_LAB_MAGIC_PROMPT_MODEL", "gpustack-minimax-m2.7-v1")
        cls = MAGIC_PROMPTS.get(model_name)
        if cls is None:
            raise RuntimeError(
                f"Unknown magic prompt model {model_name!r}. "
                f"Available: {', '.join(sorted(MAGIC_PROMPTS))}"
            )
        timeout = float(os.environ.get("TEXT_LAB_MAGIC_PROMPT_TIMEOUT", "120"))
        _MAGIC_PROMPT = cls(api_key=_remote_api_key(), timeout=timeout)
        _debug(f"Magic prompt provider initialized: {model_name}")
    return _MAGIC_PROMPT


def _local_ollama_expand(prompt: str, width: int, height: int) -> str | None:
    """Server-side keyless fallback: expand via the local Ollama server over HTTP.

    Uses the same vendored captioner prompt + JSON-schema config as the client
    path. Requires ``TEXT_LAB_MAGIC_PROMPT_OLLAMA_MODEL`` (the active chat model
    isn't known here). Returns a caption JSON string, or None if unavailable.
    """
    model = os.environ.get("TEXT_LAB_MAGIC_PROMPT_OLLAMA_MODEL", "").strip()
    if not model:
        return None
    try:
        import requests

        from core.image_gen.magic_prompt import (
            CAPTION_NUM_CTX,
            CAPTION_SCHEMA,
            _build_messages,
            _extract_json,
            _is_valid,
            _normalize,
            aspect_ratio_from_size,
        )

        aspect = aspect_ratio_from_size(width, height)
        host = (
            os.environ.get("OLLAMA_HOST", "127.0.0.1:11434")
            .replace("http://", "")
            .replace("https://", "")
        )
        body = {
            "model": model,
            "messages": _build_messages(prompt, aspect),
            "stream": False,
            "think": False,
            "format": CAPTION_SCHEMA,
            "options": {"temperature": 0.2, "num_ctx": CAPTION_NUM_CTX},
        }
        resp = requests.post(f"http://{host}/api/chat", json=body, timeout=180)
        resp.raise_for_status()
        content = resp.json().get("message", {}).get("content", "")
        caption = _normalize(_extract_json(content), aspect)
        if _is_valid(caption):
            return json.dumps(caption, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        _debug("Local Ollama expansion failed:\n" + traceback.format_exc())
    return None


def _minimal_caption_json(prompt: str, width: int, height: int) -> str:
    """Last-resort caption so raw text is NEVER sent to the model (it gray-blocks)."""
    aspect = _aspect_ratio(width, height)
    data = {
        "aspect_ratio": aspect,
        "high_level_description": prompt.strip() or "an image",
        "compositional_deconstruction": {
            "background": "plain neutral background",
            "elements": [{"type": "obj", "desc": prompt.strip() or "the subject"}],
        },
    }
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def _expand_prompt(prompt: str, width: int, height: int) -> str:
    """Ensure a JSON caption reaches the model.

    Normal path: the Streamlit client already expanded the prompt, so ``prompt`` is
    caption JSON and is passed through. If a raw prompt arrives directly, expand it
    here — remote provider (only when a key is set), then local Ollama, then a
    minimal hand-built caption. Raw plain text is never forwarded to Ideogram.
    """
    if _looks_like_caption_json(prompt):
        return prompt
    _debug("Raw prompt reached the server; expanding server-side.")
    if _remote_api_key():
        try:
            aspect = _aspect_ratio(width, height)
            caption = _get_magic_prompt().expand(prompt, aspect_ratio=aspect)
            _debug(f"Expanded caption (remote): {caption}")
            return caption
        except Exception:
            _debug("Remote magic prompt failed:\n" + traceback.format_exc())
    local = _local_ollama_expand(prompt, width, height)
    if local:
        _debug(f"Expanded caption (local ollama): {local}")
        return local
    _debug("Falling back to minimal hand-built caption.")
    return _minimal_caption_json(prompt, width, height)


def _get_pipeline():
    """Load (once) and return the Ideogram-4 pipeline."""
    global _PIPELINE
    if _PIPELINE is None:
        import torch
        from ideogram4 import Ideogram4Pipeline, Ideogram4PipelineConfig

        device = "cuda" if torch.cuda.is_available() else "cpu"
        _debug(f"Loading pipeline: device={device} torch={torch.__version__}")
        try:
            _PIPELINE = Ideogram4Pipeline.from_pretrained(
                config=Ideogram4PipelineConfig(
                    weights_repo=MODEL_REPO,
                    tokenizer_subfolder="tokenizer",
                ),
                device=device,
                dtype=torch.bfloat16,
            )
            if not getattr(_PIPELINE.text_tokenizer, "chat_template", None):
                from huggingface_hub import hf_hub_download

                template_path = hf_hub_download(
                    repo_id=MODEL_REPO,
                    filename="tokenizer/chat_template.jinja",
                )
                with open(template_path, encoding="utf-8") as fh:
                    _PIPELINE.text_tokenizer.chat_template = fh.read()
                _debug("Loaded tokenizer chat template from tokenizer/chat_template.jinja.")
            _debug("Pipeline loaded successfully.")
        except Exception:
            _debug("Pipeline load FAILED:\n" + traceback.format_exc())
            raise
    return _PIPELINE


def _looks_blocked(image) -> bool:
    """Heuristically detect Ideogram's gray "Image blocked by safety filter" frame.

    That placeholder is a near-uniform gray fill with a little centered text, so it
    has very low colour variance and roughly-equal channel means. Real renders have
    texture/edges and thus high variance. Thresholds are deliberately conservative
    to avoid flagging genuine (if minimal) images.
    """
    try:
        import numpy as np

        arr = np.asarray(image.convert("RGB"), dtype=np.float32).reshape(-1, 3)
        std = arr.std(axis=0)
        means = arr.mean(axis=0)
        grayish = abs(means[0] - means[1]) < 12 and abs(means[1] - means[2]) < 12
        blocked = bool(grayish and std.max() < 18)
        _debug(f"blocked-check std={std.round(1).tolist()} means={means.round(1).tolist()} -> {blocked}")
        return blocked
    except Exception:
        _debug("blocked-check failed:\n" + traceback.format_exc())
        return False


@mcp.tool()
def generate_image(
    prompt: str,
    width: int = 1024,
    height: int = 1024,
    sampler_preset: str = DEFAULT_PRESET,
    seed: int = 0,
) -> str:
    """
    Generate an image locally from a text prompt using the Ideogram 4 model.

    ``prompt`` is normally a structured JSON caption (produced by the client-side
    magic-prompt step); a raw prompt is expanded server-side via ``_expand_prompt``
    before generation, since Ideogram gray-blocks plain text.

    Returns "<file_path>|||<json metadata>" on success, "BLOCKED|||<json metadata>"
    if the safety filter blocked both the initial render and one auto-retry, or a
    string beginning with "Error" on failure.
    """
    try:
        from ideogram4 import PRESETS

        preset = PRESETS.get(sampler_preset) or PRESETS[DEFAULT_PRESET]
        caption = _expand_prompt(prompt, int(width), int(height))
        pipe = _get_pipeline()

        # Render, and if the safety filter gray-blocks it, retry once with a new
        # seed before giving up (the filter is stochastic and over-triggers).
        image = None
        used_seed = int(seed)
        for attempt, gen_seed in enumerate((int(seed), int(seed) + 1)):
            _debug(f"Generating image attempt={attempt} seed={gen_seed} size={width}x{height} preset={sampler_preset}")
            images = pipe(
                caption,
                height=int(height),
                width=int(width),
                num_steps=preset.num_steps,
                guidance_schedule=preset.guidance_schedule,
                mu=preset.mu,
                std=preset.std,
                seed=gen_seed,
                raise_on_caption_issues=False,
            )
            image = images[0]
            used_seed = gen_seed
            if not _looks_blocked(image):
                break
            _debug(f"Render looked blocked on attempt={attempt}; {'retrying' if attempt == 0 else 'giving up'}.")

        meta = {
            "prompt": prompt,
            "caption": caption,
            "preset": sampler_preset,
            "seed": used_seed,
            "width": int(width),
            "height": int(height),
        }
        if image is None or _looks_blocked(image):
            return f"BLOCKED|||{json.dumps(meta)}"

        artifacts_dir = get_artifacts_dir()
        os.makedirs(artifacts_dir, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ideogram_{stamp}_{uuid.uuid4().hex[:8]}.png"
        out_path = os.path.join(artifacts_dir, filename)
        image.save(out_path)
        return f"{out_path}|||{json.dumps(meta)}"
    except Exception as e:  # noqa: BLE001 - surface any failure back to the caller
        _debug("generate_image FAILED:\n" + traceback.format_exc())
        return f"Error generating image: {e}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
