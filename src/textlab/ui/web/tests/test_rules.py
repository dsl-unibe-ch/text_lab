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
    """Return (module, [(name, local name)]) of a page's feature imports."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
            "textlab.features"
        ):
            found.append(
                (
                    node.module,
                    [(a.name, a.asname or a.name) for a in node.names],
                )
            )
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("textlab.features"):
                    found.append((alias.name, []))
    return found


def attributes_read(path, local_name):
    """Return the names a page module reads as ``local_name.<name>``."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == local_name
    }


@pytest.mark.parametrize("path", PAGE_MODULES, ids=lambda path: path.name)
def test_pages_use_only_the_services_exports(path):
    import importlib

    for module, names in feature_imports(path):
        if module.endswith(".service"):
            service = importlib.import_module(module)
            imported = {name for name, _ in names}
            assert imported <= set(service.__all__), (module, imported)
            continue
        # "from textlab.features.x import service [as y]" is the other way;
        # then only the exported names are read from it.
        assert [name for name, _ in names] == ["service"], (
            f"{path.name} imports {module}"
        )
        service = importlib.import_module(f"{module}.service")
        used = attributes_read(path, names[0][1])
        assert used <= set(service.__all__), used - set(service.__all__)


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


def test_vendored_plotly_is_the_checked_release():
    folder = STATIC_DIR / "vendor" / "plotly-3.7.0"
    content = (folder / "plotly-strict.min.js").read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    assert digest in (STATIC_DIR / "vendor" / "README.md").read_text()


def test_plotly_style_sheet_is_read_from_its_bundle():
    import subprocess
    import sys

    folder = STATIC_DIR / "vendor" / "plotly-3.7.0"
    script = WEB.parents[3] / "scripts" / "plotly_css.py"
    written = subprocess.run(
        [sys.executable, str(script), str(folder / "plotly-strict.min.js")],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert written == (folder / "plotly.css").read_text()
    assert ".js-plotly-plot .plotly .modebar" in written


def test_vendored_icons_are_unchanged_lucide_files():
    for path in ICONS_DIR.glob("*.svg"):
        text = path.read_text()
        assert text.startswith("<!-- @license lucide-static v1.71.0 - ISC -->")
        assert "<script" not in text


def top_level_rules(css):
    """Return ``(selectors, body)`` of the rules outside ``@media`` blocks."""
    outside, depth, start = [], 0, 0
    for match in re.finditer(r"@media[^{]*\{|\{|\}", css):
        if match.group().startswith("@media"):
            if depth == 0:
                outside.append(css[start : match.start()])
            depth += 1
        elif match.group() == "{" and depth:
            depth += 1
        elif match.group() == "}" and depth:
            depth -= 1
            if depth == 0:
                start = match.end()
    outside.append(css[start:])
    text = re.sub(r"/\*.*?\*/", "", "".join(outside), flags=re.S)
    return [
        ({part.strip() for part in head.split(",")}, body)
        for head, body in re.findall(r"([^{}]+)\{([^{}]*)\}", text)
    ]


def class_roots(css):
    """Return the classes that begin the rules of a style sheet."""
    text = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    return set(re.findall(r"^\s*\.([a-zA-Z][\w-]*)", text, flags=re.M))


def test_chat_styles_have_class_names_of_their_own():
    # The chat's thread once shared the class "transcript" with the audio
    # player's list of passages, and so its height limit and its scrolling.
    css = (STATIC_DIR / "css" / "components.css").read_text()
    start = css.index("/* ---- Chat ----")
    others = css[:start] + (STATIC_DIR / "css" / "layout.css").read_text()
    shared = class_roots(css[start:]) & class_roots(others)
    # .main:has(> .chat) widens the page's frame for Chat only.
    assert shared == {"main"}, shared


def test_the_menu_button_is_hidden_on_wide_screens():
    # The index is always shown on wide screens; a Menu button there would
    # do nothing. The general button rule must not show it again.
    css = (STATIC_DIR / "css" / "layout.css").read_text()
    display = [
        re.search(r"display:\s*([a-z-]+)", body).group(1)
        for selectors, body in top_level_rules(css)
        if selectors & {".statusbar__menu", ".statusbar__button"}
        and "display:" in body
    ]
    assert display[-1] == "none"
