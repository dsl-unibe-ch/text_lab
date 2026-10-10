"""Where a session writes the files it generates for one user.

Chat and Visualize Data both build artifacts as they run -- the tabular file a
user uploaded, the charts an analysis produced, the images the model generated.
These used to live in ``src/mcp_artifacts``, inside the deployed source tree,
which is wrong twice over:

* The tree is shared by every user of the OnDemand app, but the directory is
  created ``0700`` because uploaded data is private. Whoever opened the page
  first owned it, and everybody else got ``PermissionError`` from the
  ``os.makedirs`` at import time -- before the page could even render.
* Runtime data does not belong in a deployed source tree at all. A redeploy
  that mirrors the repo would delete it, and the tree may be read-only.

So each user gets their own directory instead, off the shared tree entirely,
and it stays ``0700``. No change to the job script is needed: this location is
already bound into the container.
"""

from __future__ import annotations

import os
import pathlib

#: Set this to override the location outright -- a deployment that wants the
#: artifacts elsewhere (network scratch, say) sets it in the job script, and
#: everything here follows without a code change.
ENV_OVERRIDE = "TEXT_LAB_ARTIFACTS_DIR"

#: The standard per-user cache, which is where this lands unless a deployment
#: says otherwise. Chosen over network scratch deliberately: scratch would need
#: ``/rs_scratch`` bound into the container -- ``/scratch/network`` is only a
#: symlink to it -- and that means an IT redeploy of the OnDemand app. The
#: workload does not justify one: a month of real use produced 16 files and
#: 548 KB, spread across each user's own home, against a quota measured in
#: millions of files. ``$HOME`` is already bound and already holds this app's
#: HuggingFace cache, and it is network storage, so a session and the MCP
#: subprocesses it spawns see the same path from any node.
_CACHE_SUBPATH = ("text_lab", "mcp_artifacts")


def artifacts_root() -> pathlib.Path:
    """The base directory for this user's artifacts. Not created here."""
    override = os.environ.get(ENV_OVERRIDE)
    if override:
        return pathlib.Path(override)

    cache = os.environ.get("XDG_CACHE_HOME") or (pathlib.Path.home() / ".cache")
    return pathlib.Path(cache).joinpath(*_CACHE_SUBPATH)


def ensure_artifacts_dir(*parts: str) -> str:
    """Create and return this user's artifacts directory, or a path under it.

    ``0700`` throughout: uploaded data and generated charts are the user's own.
    The mode is set explicitly rather than left to the umask, and re-applied on
    a directory that already exists, so a directory created by an earlier, more
    permissive version is tightened rather than trusted.
    """
    path = artifacts_root().joinpath(*parts)
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        os.chmod(path, 0o700)
    except OSError:
        # Someone else owns it. Nothing is lost -- the caller can still use it
        # if it is usable, and will fail loudly on write if it is not.
        pass
    return str(path)
