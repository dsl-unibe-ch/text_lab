"""Serving the web UI with uvicorn; the launcher calls :func:`serve`."""

from __future__ import annotations

import logging

import uvicorn

from textlab.ui.web.app import create_app
from textlab.ui.web.settings import WebSettings

#: Seconds an idle connection stays open, so the proxy can reuse it.
KEEP_ALIVE_SECONDS = 30

#: Seconds open event streams get to end when the session stops.
SHUTDOWN_GRACE_SECONDS = 5


def serve(
    *, port: int, base_path: str, public_host: str, upload_limit: int
) -> None:
    """Serve the web UI until the process is stopped.

    One process serves everything: the logins, jobs and results live in its
    memory, and each Open OnDemand session has one user. It listens on all
    of the node's addresses, so the proxy reaches it; the password protects
    it (see :mod:`~textlab.ui.web.auth`).

    Args:
        port: Port to listen on.
        base_path: The path the proxy serves the app under.
        public_host: The portal's host name as browsers see it.
        upload_limit: Largest request body, in bytes.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    settings = WebSettings.from_env(
        base_path=base_path, public_host=public_host, upload_limit=upload_limit
    )
    uvicorn.run(
        create_app(settings),
        host="0.0.0.0",
        port=port,
        log_config=None,
        # The proxy's own host and scheme are never used: links are built
        # from the base path.
        proxy_headers=False,
        server_header=False,
        timeout_keep_alive=KEEP_ALIVE_SECONDS,
        timeout_graceful_shutdown=SHUTDOWN_GRACE_SECONDS,
    )
