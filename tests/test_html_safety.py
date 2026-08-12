import conftest_path  # noqa: F401

from core.html_safety import sanitize_table_html


def test_table_sanitizer_keeps_structure_and_safe_spans():
    source = '<table class="x"><tr><th scope="col">Name</th><td colspan="2">A &amp; B</td></tr></table>'
    assert sanitize_table_html(source) == (
        '<table><tr><th scope="col">Name</th><td colspan="2">A &amp; B</td></tr></table>'
    )


def test_table_sanitizer_removes_active_content_and_attributes():
    source = (
        '<table onclick="alert(1)"><tr><td style="background:url(x)">safe'
        '<script>alert(1)</script><img src=x onerror=alert(2)></td></tr></table>'
    )
    cleaned = sanitize_table_html(source)
    assert cleaned == "<table><tr><td>safe</td></tr></table>"
    assert "script" not in cleaned and "onclick" not in cleaned and "img" not in cleaned