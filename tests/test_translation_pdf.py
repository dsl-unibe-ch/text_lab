"""PDF coverage and overflow checks with synthetic, local-only documents."""

import conftest_path  # noqa: F401

import io
from types import SimpleNamespace

import fitz
from PIL import Image
import pytest

from core.translation import format as formats
from core.translation.pdf_checks import (
    PDFIntegrityError,
    inspect_pdf,
    require_native_coverage,
    validate_document_pages,
)


def make_pdf(*kinds):
    document = fitz.open()
    for kind in kinds:
        page = document.new_page(width=400, height=500)
        if kind in {"native", "mixed"}:
            page.insert_text((40, 40), "This is a complete example paragraph.")
        if kind == "short":
            page.insert_text((40, 40), "Cover")
        if kind in {"scan", "mixed"}:
            image = Image.new("RGB", (50, 50), "gray")
            image_bytes = io.BytesIO()
            image.save(image_bytes, format="PNG")
            page.insert_image(page.rect, stream=image_bytes.getvalue())
    data = document.tobytes()
    document.close()
    return data


def test_mixed_document_cannot_hide_scanned_pages_in_total_text():
    source = make_pdf("native", "scan", "blank", "short", "mixed")
    plans = inspect_pdf(source)
    assert [plan.route for plan in plans] == [
        "native", "ocr", "blank", "native", "ocr",
    ]
    with pytest.raises(PDFIntegrityError) as error:
        require_native_coverage(plans)
    assert error.value.pages == (2, 5)


def test_native_pdf_rejects_mixed_scan_before_translating():
    calls = []
    with pytest.raises(PDFIntegrityError) as error:
        formats.translate_pdf(
            make_pdf("native", "scan"),
            lambda text: calls.append(text) or text,
        )
    assert error.value.pages == (2,)
    assert not calls


def test_native_translation_keeps_blank_and_short_pages():
    source = make_pdf("native", "blank", "short")
    result = formats.translate_pdf(source, lambda text: text)
    with fitz.open(stream=result, filetype="pdf") as document:
        assert len(document) == 3
        assert "complete example" in document[0].get_text()
        assert not document[1].get_text().strip()
        assert "Cover" in document[2].get_text()


def test_overflow_is_scaled_to_fit_instead_of_truncated():
    source = make_pdf("native")
    warnings = []
    result = formats.translate_pdf(
        source, lambda text: "expanded words " * 300, warnings=warnings,
    )
    with fitz.open(stream=result, filetype="pdf") as document:
        text = document[0].get_text()
    assert "complete example" not in text
    assert text.count("expanded") == 300  # nothing shortened or dropped
    assert "…" not in text
    assert any("small font" in warning and "1" in warning
               for warning in warnings)


def test_unsupported_font_glyphs_block_pdf(monkeypatch):
    monkeypatch.setattr(formats, "_unicode_font_path", lambda: None)
    with pytest.raises(PDFIntegrityError, match="font"):
        formats.translate_pdf(make_pdf("native"), lambda text: "中文翻译")


def test_empty_translation_is_not_a_successful_redaction():
    with pytest.raises(ValueError):
        formats.translate_pdf(make_pdf("native"), lambda text: "")


def test_split_by_ratio_does_not_lose_characters():
    text = "abcdefghijklmnopqrstuvwxyz"
    pieces = formats._split_by_ratio(text, [0.5, 0.5])
    assert "".join(pieces) == text


@pytest.mark.parametrize("numbers", [[1], [1, 1], [1, 3]])
def test_ocr_page_coverage_rejects_missing_duplicate_or_wrong_pages(numbers):
    plans = inspect_pdf(make_pdf("native", "scan"))
    document = SimpleNamespace(pages=[
        SimpleNamespace(page_number=number, regions=[
            SimpleNamespace(text="Recognized content"),
        ]) for number in numbers
    ])
    with pytest.raises(PDFIntegrityError, match="pages"):
        validate_document_pages(document, plans)


def test_nonblank_page_with_empty_ocr_result_is_blocked():
    plans = inspect_pdf(make_pdf("scan"))
    document = SimpleNamespace(pages=[
        SimpleNamespace(page_number=1, regions=[]),
    ])
    with pytest.raises(PDFIntegrityError, match="empty") as error:
        validate_document_pages(document, plans)
    assert error.value.pages == (1,)
