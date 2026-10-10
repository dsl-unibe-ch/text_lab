"""The site configuration provides every setting the app reads.

``deploy/site.env`` is the reference site configuration. Every app setting
in ``textlab.common.config`` must be exported there, so a new setting cannot
be added to the code without documenting its value for the site.
"""

import re
from pathlib import Path

from textlab.common import config

SITE_ENV = Path(__file__).resolve().parents[1] / "deploy" / "site.env"

#: Settings the launch script computes per job instead of the site file.
COMPUTED_BY_LAUNCH_SCRIPT = {"TEXT_LAB_WORKDIR"}


def exported_names():
    """Names exported by ``export NAME=...`` lines in site.env."""
    text = SITE_ENV.read_text(encoding="utf-8")
    return set(re.findall(r"^export\s+([A-Z_][A-Z0-9_]*)=", text, re.M))


def test_every_app_setting_is_exported_by_the_site_file():
    expected = {spec.env for spec in config.SETTINGS}
    missing = expected - COMPUTED_BY_LAUNCH_SCRIPT - exported_names()
    assert not missing, f"Add to deploy/site.env: {sorted(missing)}"


def test_every_assignment_can_be_overridden():
    # dev.env is sourced first, so site.env must not overwrite its values.
    text = SITE_ENV.read_text(encoding="utf-8")
    pattern = r"^(?:export\s+)?([A-Z_][A-Z0-9_]*)=(.*)$"
    for name, value in re.findall(pattern, text, re.M):
        assert value.startswith(f'"${{{name}:-'), (
            f'{name} must use the form {name}="${{{name}:-default}}"'
        )
