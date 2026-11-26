"""Rules for the web UI's code, templates and static files.

- Pages import from the features only their ``service`` modules, and only
  the names those export in ``__all__``.
- Templates hold no scripts or styles of their own (the Content Security
  Policy would block them) and no emojis.
- Every icon and static file a template names exists.
- The dark colours are the same for the system setting and the switch.
- The vendored files are the ones checked against the registry.
"""

import ast
import hashlib
import re
from pathlib import Path

import pytest

from textlab.ui.web import pages
from textlab.ui.web.render import ICONS_DIR, STATIC_DIR, TEMPLATES_DIR

WEB = Path(pages.__file__).resolve().parents[1]
PAGE_MODULES = sorted((WEB / "pages").glob("*.py"))
TEMPLATES = sorted(TEMPLATES_DIR.rglob("*.html"))
TEMPLATE_IDS = [str(path.relative_to(TEMPLATES_DIR)) for path in TEMPLATES]

#: Ranges of emoji and pictographs.
EMOJI = re.compile("[\U0001f000-\U0001faff☀-➿️]")


def feature_imports(path):
    """Return (module, names) of the feature imports of a page module."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
            "textlab.features"
        ):
            found.append((node.module, [alias.name for alias in node.names]))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("textlab.features"):
                    found.append((alias.name, []))
    return found


@pytest.mark.parametrize("path", PAGE_MODULES, ids=lambda path: path.name)
def test_pages_use_only_the_services_exports(path):
    import importlib

    for module, names in feature_imports(path):
        if module.endswith(".service"):
            service = importlib.import_module(module)
            assert set(names) <= set(service.__all__), (module, names)
        else:
            # "from textlab.features.x import service" is the other way.
            assert names == ["service"], f"{path.name} imports {module}"


@pytest.mark.parametrize("path", TEMPLATES, ids=TEMPLATE_IDS)
def test_templates_hold_no_scripts_styles_or_emojis(path):
    text = path.read_text(encoding="utf-8")
    for tag in re.findall(r"<script\b[^>]*>", text):
        assert "src=" in tag, f"inline script in {path.name}"
    assert not re.search(r"\son[a-z]+\s*=", text), "inline event handler"
    assert not re.search(r"\sstyle\s*=", text), "inline style"
    assert "<style" not in text
    assert "javascript:" not in text
    assert not EMOJI.search(text), "emoji"


def test_every_icon_named_exists_and_no_other_is_vendored():
    names = set()
    for path in TEMPLATES:
        names |= set(re.findall(r'icon\("([a-z0-9-]+)"', path.read_text()))
    vendored = {path.stem for path in ICONS_DIR.glob("*.svg")}
    assert names == vendored


def test_every_static_file_named_exists():
    for path in TEMPLATES:
        for name in re.findall(r"static\('([^']+)'\)", path.read_text()):
            assert (STATIC_DIR / name).is_file(), (path.name, name)


def test_every_file_a_style_sheet_names_exists():
    for sheet in (STATIC_DIR / "css").glob("*.css"):
        for name in re.findall(r'url\("([^"]+)"\)', sheet.read_text()):
            assert (sheet.parent / name).resolve().is_file(), (
                sheet.name,
                name,
            )


def test_preloaded_fonts_are_the_ones_the_style_sheet_uses():
    head = (TEMPLATES_DIR / "partials" / "head.html").read_text()
    fonts = (STATIC_DIR / "css" / "fonts.css").read_text()
    preloaded = re.findall(
        r'href="\{\{ root \}\}/static/([^"]+\.woff2)"', head
    )
    assert preloaded
    for name in preloaded:
        assert f'url("../{name}")' in fonts


def test_static_files_hold_no_emojis():
    for path in STATIC_DIR.rglob("*"):
        if (
            path.suffix in {".css", ".js", ".svg"}
            and "vendor" not in path.parts
        ):
            assert not EMOJI.search(path.read_text(encoding="utf-8")), path


def test_dark_colours_are_the_same_for_the_system_and_the_switch():
    css = (STATIC_DIR / "css" / "tokens.css").read_text()

    def block(selector):
        start = css.index("{", css.index(selector)) + 1
        return sorted(
            line.strip()
            for line in css[start : css.index("}", start)].splitlines()
            if line.strip()
        )

    assert block(':root[data-theme="dark"]') == block(
        ':root:not([data-theme="light"])'
    )


def test_vendored_htmx_is_the_checked_release():
    content = (STATIC_DIR / "vendor" / "htmx-2.0.11.min.js").read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    assert digest in (STATIC_DIR / "vendor" / "README.md").read_text()


def test_vendored_icons_are_unchanged_lucide_files():
    for path in ICONS_DIR.glob("*.svg"):
        text = path.read_text()
        assert text.startswith("<!-- @license lucide-static v1.71.0 - ISC -->")
        assert "<script" not in text
