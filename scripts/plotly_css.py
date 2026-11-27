r"""Write plotly.js's own style rules as a style sheet.

plotly.js adds its style rules (the mode bar, hover labels, notifiers) to
the page in a ``<style>`` element, which the web UI's Content Security
Policy blocks. It skips that when the page already holds an element with
the id ``plotly.js-style-global`` and the class ``no-inline-styles``; the
web UI then links this style sheet under that id instead.

The rules are read from the bundle itself, so they match its version:

    python3 scripts/plotly_css.py \
        src/textlab/ui/web/static/vendor/plotly-3.7.0/plotly-strict.min.js \
        > src/textlab/ui/web/static/vendor/plotly-3.7.0/plotly.css

plotly.js inserts each rule before the ones it added already, so the rules
are written in reverse order, which gives the same cascade.
"""

from __future__ import annotations

import re
import sys

#: How the bundle names the plot and the notifier in its rules' selectors.
MARKER = (
    '.replace(/X/g,".js-plotly-plot .plotly").replace(/Y/g,".plotly-notifier")'
)

STRING = r'"((?:[^"\\]|\\.)*)"|\'((?:[^\'\\]|\\.)*)\''
ENTRY = re.compile(
    rf"(?:{STRING}|([A-Za-z_$][\w$]*))\s*:\s*(?:{STRING})\s*(?:,|$)"
)


def rules_literal(bundle: str) -> str:
    """Return the object literal that maps selectors to declarations."""
    marker = bundle.index(MARKER)
    end = bundle.rindex("};for(", 0, marker) + 1
    start = bundle.rindex("={", 0, end) + 1
    return bundle[start:end]


def rules(literal: str) -> list[tuple[str, str]]:
    """Return the ``(selector, declarations)`` pairs of the literal."""
    body = literal.strip()[1:-1]
    found = []
    position = 0
    while position < len(body):
        match = ENTRY.match(body, position)
        if match is None:
            raise ValueError(
                f"Cannot read the rules at: {body[position:][:80]}"
            )
        key = next(group for group in match.groups()[:3] if group is not None)
        value = next(
            group for group in match.groups()[3:] if group is not None
        )
        selector = (
            re.sub(r"^,", " ,", key)
            .replace("X", ".js-plotly-plot .plotly")
            .replace("Y", ".plotly-notifier")
        )
        found.append((selector, value.replace('\\"', '"')))
        position = match.end()
    return found


def main() -> None:
    """Print the style sheet of the bundle named on the command line."""
    with open(sys.argv[1], encoding="utf-8") as file:
        bundle = file.read()
    version = re.search(r"plotly\.js \(strict - minified\) v([\d.]+)", bundle)
    print(
        f"/* plotly.js {version.group(1) if version else ''} style rules, "
        "read from its bundle by scripts/plotly_css.py. */"
    )
    for selector, declarations in reversed(rules(rules_literal(bundle))):
        print(f"{selector}{{{declarations}}}")


if __name__ == "__main__":
    main()
