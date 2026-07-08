"""
Persistent client for the Ideogram-4 image-generation MCP server.

Keeps ONE stdio MCP session alive on a background asyncio loop so the diffusion
model (loaded lazily inside the server process) stays resident across Streamlit
reruns and successive generations — only the first request pays the load cost.

All public functions are synchronous and safe to call from Streamlit worker
threads. A module-level singleton backend survives reruns because the module
stays imported for the app's lifetime.
"""

from __future__ import annotations

import asyncio
import datetime
import json
import os
import sys
import threading
from typing import Optional

from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client

from core.image_gen.image_config import DEFAULT_PRESET, ImageArtifact, get_artifacts_dir


class ImageBlockedError(RuntimeError):
    """Raised when Ideogram's safety filter blocked the render (gray placeholder).

    The MCP server auto-retries once with a new seed before signalling this, so it
    means both attempts were blocked. The UI turns this into a friendly, actionable
    message rather than surfacing the gray image or a generic error.
    """


class _Backend:
    """Owns a background asyncio loop and a long-lived MCP client session."""

    def __init__(self, server_script: str, artifacts_dir: str) -> None:
        self._server_script = server_script
        self._artifacts_dir = artifacts_dir
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._session: Optional[ClientSession] = None
        self._stop: Optional[asyncio.Event] = None
        self._ready = threading.Event()
        self._error: Optional[str] = None
        self._lock = threading.Lock()

    # ---- lifecycle -------------------------------------------------------
    def start(self, timeout: float = 180.0) -> None:
        """Start the backend (idempotent) and block until the session is ready."""
        with self._lock:
            if self._thread and self._thread.is_alive():
                if self._ready.is_set() and self._error:
                    raise RuntimeError(self._error)
                return
            self._ready.clear()
            self._error = None
            self._session = None
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

        if not self._ready.wait(timeout):
            raise TimeoutError("Image-generation backend did not start in time.")
        if self._error:
            raise RuntimeError(self._error)

    def _run(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._serve())
        except Exception as e:  # noqa: BLE001 - report startup failure to callers
            self._error = str(e)
            self._ready.set()
        finally:
            self._session = None

    async def _serve(self) -> None:
        self._stop = asyncio.Event()
        env = os.environ.copy()
        env["IMAGE_GEN_ARTIFACTS_DIR"] = self._artifacts_dir
        # The ideogram-4-nf4 model needs transformers 5.x, so the MCP server runs
        # in the isolated `imagegen_backend` conda env (see text_lab.def), not the
        # Streamlit env. Falls back to this interpreter when the var is unset
        # (e.g. running outside the container in a single-env dev setup).
        python_exe = os.environ.get("TEXT_LAB_IMAGEGEN_PYTHON", sys.executable)
        params = StdioServerParameters(
            command=python_exe,
            args=[self._server_script],
            env=env,
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                self._session = session
                self._ready.set()
                # Hold the subprocess + session open until explicitly stopped.
                await self._stop.wait()

    def stop(self, join_timeout: float = 60.0) -> None:
        """Signal teardown and block until the subprocess is gone (VRAM freed).

        The MCP subprocess holds the ~20 GB diffusion pipeline, so on a low-VRAM
        GPU it MUST fully exit before the chat model can be reloaded. Joining the
        worker thread waits for ``stdio_client`` to terminate that subprocess.
        """
        if self._loop and self._stop and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._stop.set)
        self._session = None
        thread = self._thread
        if thread and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=join_timeout)

    @property
    def alive(self) -> bool:
        return bool(
            self._thread
            and self._thread.is_alive()
            and self._session is not None
            and not self._error
        )

    # ---- generation ------------------------------------------------------
    def generate(
        self,
        prompt: str,
        width: int,
        height: int,
        sampler_preset: str,
        seed: int,
        timeout: float,
    ) -> ImageArtifact:
        if not self.alive:
            self.start()
        assert self._session is not None and self._loop is not None

        coro = self._session.call_tool(
            "generate_image",
            arguments={
                "prompt": prompt,
                "width": int(width),
                "height": int(height),
                "sampler_preset": sampler_preset,
                "seed": int(seed),
            },
            # Align the MCP protocol timeout with ours so a slow first-load or a
            # high-step render is not cut off by the client's default deadline.
            read_timeout_seconds=datetime.timedelta(seconds=timeout),
        )
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        result = future.result(timeout=timeout)

        raw = ""
        if result.content and isinstance(result.content[0], types.TextContent):
            raw = result.content[0].text
        if raw.startswith("BLOCKED"):
            raise ImageBlockedError("Image blocked by the model's safety filter.")
        if not raw or raw.startswith("Error") or "|||" not in raw:
            raise RuntimeError(raw.strip() or "Image generation returned no output.")

        path_part, meta_part = raw.split("|||", 1)
        try:
            meta = json.loads(meta_part)
        except json.JSONDecodeError:
            meta = {}

        return {
            "path": path_part.strip(),
            "prompt": meta.get("prompt", prompt),
            "preset": meta.get("preset", sampler_preset),
            "seed": int(meta.get("seed", seed)),
            "width": int(meta.get("width", width)),
            "height": int(meta.get("height", height)),
        }


_BACKEND: Optional[_Backend] = None
_BACKEND_LOCK = threading.Lock()


def _get_backend(server_script: str) -> _Backend:
    global _BACKEND
    with _BACKEND_LOCK:
        if _BACKEND is None:
            _BACKEND = _Backend(server_script, get_artifacts_dir())
        return _BACKEND


def generate_image(
    prompt: str,
    server_script: str,
    width: int = 1024,
    height: int = 1024,
    sampler_preset: str = DEFAULT_PRESET,
    seed: int = 0,
    timeout: float = 900.0,
) -> ImageArtifact:
    """Synchronously generate one image via the persistent MCP backend."""
    backend = _get_backend(server_script)
    return backend.generate(prompt, width, height, sampler_preset, seed, timeout)


def shutdown_backend() -> None:
    """Tear down the backend subprocess (e.g. to free VRAM). Safe to call anytime."""
    global _BACKEND
    with _BACKEND_LOCK:
        if _BACKEND is not None:
            _BACKEND.stop()
            _BACKEND = None
