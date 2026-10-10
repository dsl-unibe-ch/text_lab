"""Site settings for Text Lab, read from the environment in one place.

Everything that differs between clusters or deployments (where models live,
which Grobid image to run, which remote LLM endpoint to offer) comes from
environment variables. The launch script exports them from the site
configuration, ``deploy/site.env``, so no feature hardcodes a path.

Settings are read lazily: a missing setting only fails the feature that
needs it, with an error naming the variable to set, instead of stopping the
whole app at start-up. Use :func:`get_settings` for the current values and
:meth:`Settings.require` where a feature cannot work without one.

Runtime values that the launch script computes per job (ports, the Ollama
host) are not site settings and are still read where they are used.
"""

from __future__ import annotations

import dataclasses
import functools
import os
from collections.abc import Mapping
from pathlib import Path

#: Kind of value a setting holds: ``"path"`` or ``"text"``.
_PATH = "path"
_TEXT = "text"


@dataclasses.dataclass(frozen=True)
class SettingSpec:
    """Description of one setting.

    Attributes:
        name: Attribute name on :class:`Settings`.
        env: Environment variable the value is read from.
        kind: ``"path"`` (converted to :class:`~pathlib.Path`) or ``"text"``.
        purpose: What the setting is for, shown in error messages.
    """

    name: str
    env: str
    kind: str
    purpose: str


#: Every site setting, in the order they appear in ``deploy/site.env``.
SETTINGS: tuple[SettingSpec, ...] = (
    SettingSpec(
        "workdir",
        "TEXT_LAB_WORKDIR",
        _PATH,
        "the private per-job folder for temporary user files",
    ),
    SettingSpec(
        "hf_token_file",
        "TEXT_LAB_HF_TOKEN_FILE",
        _PATH,
        "the Hugging Face token used for speaker diarization",
    ),
    SettingSpec(
        "custom_whisper_dir",
        "TEXT_LAB_CUSTOM_WHISPER_DIR",
        _PATH,
        "the folder holding the Swiss German Whisper models",
    ),
    SettingSpec(
        "grobid_container",
        "TEXT_LAB_GROBID_CONTAINER",
        _PATH,
        "the Grobid image used by the Knowledge Graph",
    ),
    SettingSpec(
        "gpustack_url",
        "TEXT_LAB_GPUSTACK_URL",
        _TEXT,
        "the GPUStack endpoint offered by the Knowledge Graph",
    ),
)

_SPECS_BY_NAME = {spec.name: spec for spec in SETTINGS}


class MissingSettingError(RuntimeError):
    """A feature needs a setting that the site configuration does not set.

    Attributes:
        spec: The setting that is missing.
    """

    def __init__(self, spec: SettingSpec):
        """Build the error message from the missing setting.

        Args:
            spec: The setting that is missing.
        """
        self.spec = spec
        super().__init__(
            f"Text Lab is not configured with {spec.purpose}. Set "
            f"{spec.env} in the site configuration (deploy/site.env)."
        )


@dataclasses.dataclass(frozen=True)
class Settings:
    """Site settings; a value is ``None`` when it is not configured.

    Attributes:
        workdir: Private per-job folder for temporary user files.
        hf_token_file: File holding the Hugging Face token for diarization.
        custom_whisper_dir: Folder with the Swiss German Whisper models.
        grobid_container: Grobid Apptainer image for the Knowledge Graph.
        gpustack_url: GPUStack endpoint (OpenAI-compatible) for the
            Knowledge Graph.
    """

    workdir: Path | None = None
    hf_token_file: Path | None = None
    custom_whisper_dir: Path | None = None
    grobid_container: Path | None = None
    gpustack_url: str | None = None

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> Settings:
        """Read all settings from environment variables.

        Args:
            environ: Mapping to read from; defaults to ``os.environ``.

        Returns:
            The settings. Unset or empty variables become ``None``.
        """
        environ = os.environ if environ is None else environ
        values = {}
        for spec in SETTINGS:
            raw = environ.get(spec.env, "").strip()
            if not raw:
                values[spec.name] = None
            elif spec.kind == _PATH:
                values[spec.name] = Path(raw)
            else:
                values[spec.name] = raw
        return cls(**values)

    def require(self, name: str) -> Path | str:
        """Return a setting that the caller cannot work without.

        Args:
            name: Attribute name of the setting, e.g. ``"grobid_container"``.

        Returns:
            The configured value.

        Raises:
            MissingSettingError: If the setting is not configured.
            KeyError: If ``name`` is not a known setting.
        """
        spec = _SPECS_BY_NAME[name]
        value = getattr(self, name)
        if value is None:
            raise MissingSettingError(spec)
        return value


@functools.cache
def get_settings() -> Settings:
    """Return the settings of this process, read once from the environment.

    Returns:
        The cached settings. Tests that change the environment call
        ``get_settings.cache_clear()`` first.
    """
    return Settings.from_env()
