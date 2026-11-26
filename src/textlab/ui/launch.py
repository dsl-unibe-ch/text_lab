r"""Start Text Lab's user interface: the one ``TEXT_LAB_UI`` names.

A session's launch script (``template/script.sh.erb``) runs, inside the
image::

    python -u -m textlab.ui.launch --port PORT \\
        --base-path /node/HOST/PORT/ --public-host PORTAL --log-dir JOB_FOLDER

``TEXT_LAB_UI`` comes from the site configuration (``deploy/site.env``) or a
sandbox app's ``template/dev.env``:

``streamlit`` (the default)
    The Streamlit app, started with the options the launch script gave it
    before the launcher existed.
``web``
    The web UI (:mod:`textlab.ui.web`), served by uvicorn.

The interface writes its output to its log in the job folder,
``streamlit.log`` or ``textlab.log``. The launcher's own messages before
that, such as an error about ``TEXT_LAB_UI``, go to Open OnDemand's
``output.log``.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

#: The interfaces ``TEXT_LAB_UI`` can name.
STREAMLIT = "streamlit"
WEB = "web"
INTERFACES = (STREAMLIT, WEB)

#: The interface started when ``TEXT_LAB_UI`` is not set.
DEFAULT_INTERFACE = STREAMLIT

#: Largest upload in either interface, in megabytes.
UPLOAD_LIMIT_MB = 10_000

#: Each interface's log file, in the folder given with ``--log-dir``.
LOG_FILES = {STREAMLIT: "streamlit.log", WEB: "textlab.log"}

#: The Streamlit app's entry page.
STREAMLIT_HOME = Path(__file__).resolve().parent / "streamlit" / "Home.py"


class LaunchError(RuntimeError):
    """The interface to start cannot be determined."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Read the command line.

    Args:
        argv: The arguments after the program name; defaults to
            ``sys.argv[1:]``.

    Returns:
        The options: ``port``, ``base_path``, ``public_host`` and
        ``log_dir`` (``None`` when not given).
    """
    parser = argparse.ArgumentParser(
        prog="python -m textlab.ui.launch",
        description=(
            "Start the Text Lab interface that TEXT_LAB_UI names "
            f"({' or '.join(INTERFACES)}; default {DEFAULT_INTERFACE})."
        ),
    )
    parser.add_argument(
        "--port",
        type=int,
        required=True,
        help="port to listen on, on every address of the node",
    )
    parser.add_argument(
        "--base-path",
        default="",
        help="path the proxy serves the app under, e.g. /node/HOST/PORT/",
    )
    parser.add_argument(
        "--public-host",
        default="",
        help="host name of the Open OnDemand portal, as browsers see it",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        help="folder for the interface's log; without it, output stays here",
    )
    return parser.parse_args(argv)


def chosen_interface(environ: Mapping[str, str] | None = None) -> str:
    """Return the interface ``TEXT_LAB_UI`` names.

    Args:
        environ: Mapping to read from; defaults to ``os.environ``.

    Returns:
        One of :data:`INTERFACES`; :data:`DEFAULT_INTERFACE` when the
        variable is unset or empty. Case and surrounding spaces are
        ignored.

    Raises:
        LaunchError: If the variable names no interface.
    """
    environ = os.environ if environ is None else environ
    value = environ.get("TEXT_LAB_UI", "").strip().lower()
    if not value:
        return DEFAULT_INTERFACE
    if value not in INTERFACES:
        raise LaunchError(
            f"TEXT_LAB_UI={value!r} names no interface; set it to "
            f"{' or '.join(INTERFACES)} in the site configuration."
        )
    return value


def streamlit_command(
    args: argparse.Namespace, python: str = sys.executable
) -> list[str]:
    """Return the command that starts the Streamlit app.

    Args:
        args: The options from :func:`parse_args`.
        python: The interpreter to run Streamlit with.

    Returns:
        The command, with the options the launch script used before the
        launcher existed.
    """
    command = [
        python, "-u", "-m", "streamlit", "run", str(STREAMLIT_HOME),
        "--server.address", "0.0.0.0",
        "--server.port", str(args.port),
        "--server.baseUrlPath", args.base_path,
        "--server.headless", "true",
        "--server.maxUploadSize", str(UPLOAD_LIMIT_MB),
    ]  # fmt: skip
    if args.public_host:
        command += ["--browser.serverAddress", args.public_host]
    command += [
        "--browser.gatherUsageStats", "false",
        "--logger.level", "info",
        "--client.showErrorDetails", "true",
    ]  # fmt: skip
    return command


def redirect_output(log_file: Path) -> None:
    """Send this process's standard output and error to ``log_file``.

    The file descriptors themselves are redirected, so the redirection also
    holds for a program this process becomes with :func:`os.execv`.

    Args:
        log_file: The log; created readable by the owner only, or appended
            to if it exists.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    descriptor = os.open(
        log_file, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600
    )
    os.dup2(descriptor, sys.stdout.fileno())
    os.dup2(descriptor, sys.stderr.fileno())
    os.close(descriptor)


def main(argv: Sequence[str] | None = None) -> int:
    """Start the interface; return only for the web UI or on errors.

    Args:
        argv: The command-line arguments; see :func:`parse_args`.

    Returns:
        The exit status: 0 after the web UI stopped, 2 on a configuration
        error. Streamlit replaces this process and does not return.
    """
    args = parse_args(argv)
    try:
        interface = chosen_interface()
    except LaunchError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2

    log_file = args.log_dir / LOG_FILES[interface] if args.log_dir else None
    where = f"; log: {log_file}" if log_file else ""
    print(f"Starting the {interface} interface on port {args.port}{where}")
    if log_file is not None:
        redirect_output(log_file)

    if interface == STREAMLIT:
        command = streamlit_command(args)
        os.execv(command[0], command)

    # Imported here: only the web UI needs FastAPI and uvicorn.
    from textlab.ui.web.server import serve

    serve(
        port=args.port,
        base_path=args.base_path,
        public_host=args.public_host,
        upload_limit=UPLOAD_LIMIT_MB * 1000 * 1000,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
