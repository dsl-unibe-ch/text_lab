"""The web UI's settings, fixed when the server starts."""

from __future__ import annotations

import dataclasses
import os
import secrets
from collections.abc import Mapping

#: Largest request body by default: 10 GB, the Streamlit UI's limit.
DEFAULT_UPLOAD_LIMIT = 10_000 * 1000 * 1000

#: Values of ``TEXT_LAB_WEB_DEV`` that turn the developer pages on.
TRUE_VALUES = {"1", "true", "yes", "on"}


def normalize_base_path(path: str) -> str:
    """Return a base path with a leading and without a final slash.

    Args:
        path: The path the proxy serves the app under, such as
            ``/node/gnode21/31234/``.

    Returns:
        For example ``/node/gnode21/31234``; ``""`` for the root.
    """
    path = path.strip().strip("/")
    return f"/{path}" if path else ""


@dataclasses.dataclass(frozen=True)
class WebSettings:
    """Settings of one web UI server.

    Attributes:
        token: The session password, compared with the one a login sends.
        base_path: The path the proxy serves the app under, without a final
            slash, such as ``/node/gnode21.hpc.unibe.ch/31234``; empty to
            serve it at the root.
        owner: The user the session belongs to. Requests the proxy marks as
            another user's (``X-Forwarded-User``) are refused; empty turns
            the check off.
        public_host: The portal's host name as browsers see it. A login
            posted from a page of another origin is refused; empty turns
            the check off.
        upload_limit: Largest request body, in bytes.
        dev: Whether the developer pages under ``/dev`` are served.
        session_end: When the Slurm job of the session ends (seconds since
            the epoch), or ``None`` if not known. Results are lost then, so
            the status bar shows it.
        secret_key: Signs the session cookies. It is new at every start, so
            restarting the server signs everyone out.
    """

    token: str = dataclasses.field(repr=False)
    base_path: str = ""
    owner: str = ""
    public_host: str = ""
    upload_limit: int = DEFAULT_UPLOAD_LIMIT
    dev: bool = False
    session_end: float | None = None
    secret_key: bytes = dataclasses.field(
        default_factory=lambda: secrets.token_bytes(32), repr=False
    )

    def __post_init__(self) -> None:
        """Check the password and normalize the base path.

        Raises:
            ValueError: If the password is empty.
        """
        if not self.token:
            raise ValueError(
                "The session password is empty. The launch script's "
                "before.sh sets it as TOKEN."
            )
        object.__setattr__(
            self, "base_path", normalize_base_path(self.base_path)
        )

    @classmethod
    def from_env(
        cls,
        *,
        base_path: str = "",
        public_host: str = "",
        upload_limit: int = DEFAULT_UPLOAD_LIMIT,
        environ: Mapping[str, str] | None = None,
    ) -> WebSettings:
        """Build the settings from the launcher's options and the environment.

        Reads ``TOKEN`` (the session password), ``USER`` (the owner),
        ``TEXT_LAB_WEB_DEV`` (the developer pages) and Slurm's
        ``SLURM_JOB_END_TIME`` (the session's end).

        Args:
            base_path: The path the proxy serves the app under.
            public_host: The portal's host name as browsers see it.
            upload_limit: Largest request body, in bytes.
            environ: Mapping to read from; defaults to ``os.environ``.

        Returns:
            The settings, with a new secret key.

        Raises:
            ValueError: If ``TOKEN`` is not set.
        """
        environ = os.environ if environ is None else environ
        return cls(
            token=environ.get("TOKEN", ""),
            base_path=base_path,
            owner=environ.get("USER", ""),
            public_host=public_host.strip(),
            upload_limit=upload_limit,
            dev=environ.get("TEXT_LAB_WEB_DEV", "").strip().lower()
            in TRUE_VALUES,
            session_end=_timestamp(environ.get("SLURM_JOB_END_TIME", "")),
        )


def _timestamp(value: str) -> float | None:
    """Return a positive Unix time from Slurm, or ``None`` if there is none."""
    try:
        timestamp = float(value)
    except ValueError:
        return None
    return timestamp if timestamp > 0 else None
