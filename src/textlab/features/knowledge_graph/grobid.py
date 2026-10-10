"""The Grobid server that parses the papers' PDFs into TEI XML.

Grobid runs from its own Apptainer image (setting ``grobid_container``),
started from inside the Text Lab container, so it needs ``apptainer`` or
``singularity`` there. It is started once per session, detached so it
survives page reloads, and listens on ``GROBID_PORT`` (8070 by default; the
admin port is the next one). Grobid keeps the PDFs it is processing in its
tmp folder, which is bound to the ``grobid`` area of the job workspace.
"""

from __future__ import annotations

import logging
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import requests

from textlab.common.config import MissingSettingError, get_settings
from textlab.common.storage import get_workspace

LOGGER = logging.getLogger(__name__)

GROBID_HOST = "127.0.0.1"
GROBID_PORT = int(os.environ.get("GROBID_PORT", 8070))
GROBID_URL = f"http://{GROBID_HOST}:{GROBID_PORT}/api/processFulltextDocument"

#: Seconds to wait for a starting server to accept connections.
START_TIMEOUT_SECONDS = 60
#: Seconds Grobid may take for one PDF.
REQUEST_TIMEOUT_SECONDS = 60

NO_APPTAINER_MESSAGE = (
    "Knowledge Graph feature unavailable: Grobid requires nested container "
    "support (apptainer/singularity not available). This is a known "
    "limitation when running TextLab inside an apptainer container. The "
    "Knowledge Graph feature requires infrastructure changes to support "
    "container nesting or shared network namespaces. Please use Document "
    "OCR or Chat features instead."
)


class GrobidError(Exception):
    """The Grobid server cannot be started or cannot parse a PDF."""


def ensure_grobid_server() -> None:
    """Start the Grobid server unless it is already running.

    Raises:
        GrobidError: If the image or Apptainer is missing, or the server
            does not accept connections within ``START_TIMEOUT_SECONDS``.
    """
    if _port_open(GROBID_HOST, GROBID_PORT):
        LOGGER.info("Grobid is already running on port %s", GROBID_PORT)
        return

    try:
        container = str(get_settings().require("grobid_container"))
    except MissingSettingError as exc:
        raise GrobidError(str(exc)) from exc
    tmp_dir = get_workspace().dir("grobid")

    apptainer = next(
        (cmd for cmd in ("apptainer", "singularity") if shutil.which(cmd)),
        None,
    )
    if apptainer is None:
        raise GrobidError(NO_APPTAINER_MESSAGE)
    if not os.path.exists(container):
        raise GrobidError(f"Grobid container SIF not found at: {container}")

    command = server_command(apptainer, container, tmp_dir)
    LOGGER.info("Starting Grobid: %s", " ".join(command))
    try:
        subprocess.Popen(
            command,
            stdout=sys.stdout,
            stderr=sys.stderr,
            # Detached, so the server survives page reloads.
            start_new_session=True,
        )
    except Exception as exc:
        raise GrobidError(f"Failed to start Grobid container: {exc}") from exc

    LOGGER.info("Waiting for Grobid to accept connections")
    for _ in range(START_TIMEOUT_SECONDS):
        if _port_open(GROBID_HOST, GROBID_PORT):
            LOGGER.info("Grobid started")
            return
        time.sleep(1)
    raise GrobidError(
        f"Grobid server started but port {GROBID_PORT} did not open within "
        f"{START_TIMEOUT_SECONDS} seconds."
    )


def server_command(apptainer: str, container: str, tmp_dir: Path) -> list[str]:
    """Return the command that runs the Grobid server on ``GROBID_PORT``.

    Grobid's configuration fixes its ports, so the command copies it into
    the tmp folder and rewrites the ports there before starting the server.

    Args:
        apptainer: ``"apptainer"`` or ``"singularity"``.
        container: Path of the Grobid image.
        tmp_dir: Folder bound as Grobid's tmp folder.

    Returns:
        The command, as a list for :class:`subprocess.Popen`.
    """
    config = "grobid-home/tmp/custom_grobid.yaml"
    script = (
        "cd /opt/grobid && "
        f"cp grobid-home/config/grobid.yaml {config} && "
        f"sed -i 's/port: 8070/port: {GROBID_PORT}/g' {config} && "
        f"sed -i 's/port: 8071/port: {GROBID_PORT + 1}/g' {config} && "
        f"./grobid-service/bin/grobid-service server {config}"
    )
    return [
        apptainer,
        "exec",
        "-B",
        f"{tmp_dir}:/opt/grobid/grobid-home/tmp",
        "--env",
        "GROBID_HOME=/opt/grobid/grobid-home",
        container,
        "bash",
        "-c",
        script,
    ]


def process_pdf(pdf_path: str | Path) -> str:
    """Parse a PDF with Grobid, starting the server if needed.

    Args:
        pdf_path: The PDF.

    Returns:
        The TEI XML of the full text.

    Raises:
        GrobidError: If the server cannot be started or rejects the PDF.
    """
    ensure_grobid_server()
    with open(pdf_path, "rb") as file:
        response = requests.post(
            GROBID_URL, files={"input": file}, timeout=REQUEST_TIMEOUT_SECONDS
        )
    if response.status_code != 200:
        raise GrobidError(
            f"Grobid returned status {response.status_code}: {response.text}"
        )
    return response.text


def _port_open(host: str, port: int) -> bool:
    """Return True if something accepts TCP connections on host:port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as connection:
        return connection.connect_ex((host, port)) == 0
