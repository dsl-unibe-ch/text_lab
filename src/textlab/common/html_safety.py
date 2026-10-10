"""Small allowlist sanitizer for OCR-produced table HTML."""

from html import escape
from html.parser import HTMLParser


TABLE_TAGS = {
    "table", "thead", "tbody", "tfoot", "tr", "th", "td", "caption",
    "colgroup", "col",
}
VOID_TAGS = {"col"}
TABLE_ATTRIBUTES = {"colspan", "rowspan", "scope"}
DISCARDED_CONTENT_TAGS = {"script", "style", "iframe", "object", "svg"}


class _TableSanitizer(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.discard_depth = 0

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        if tag in DISCARDED_CONTENT_TAGS:
            self.discard_depth += 1
            return
        if self.discard_depth or tag not in TABLE_TAGS:
            return
        safe_attrs = []
        for name, value in attrs:
            name = name.lower()
            if name not in TABLE_ATTRIBUTES or value is None:
                continue
            safe_attrs.append(f' {name}="{escape(value, quote=True)}"')
        self.parts.append(f"<{tag}{''.join(safe_attrs)}>")

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag in DISCARDED_CONTENT_TAGS:
            self.discard_depth = max(0, self.discard_depth - 1)
            return
        if not self.discard_depth and tag in TABLE_TAGS and tag not in VOID_TAGS:
            self.parts.append(f"</{tag}>")

    def handle_data(self, data):
        if not self.discard_depth:
            self.parts.append(escape(data))


def sanitize_table_html(value: str) -> str:
    """Keep table structure while removing executable or styling markup."""
    parser = _TableSanitizer()
    parser.feed(value or "")
    parser.close()
    return "".join(parser.parts)