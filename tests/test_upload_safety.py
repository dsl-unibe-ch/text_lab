import conftest_path  # noqa: F401

import io
import pathlib
import zipfile

import pytest

from core import upload_safety


def _archive(entries):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in entries:
            archive.writestr(name, data)
    buffer.seek(0)
    return zipfile.ZipFile(buffer)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("report.pdf", "report.pdf"),
        ("../../report.pdf", "report.pdf"),
        (r"C:\\Users\\name\\report.pdf", "report.pdf"),
        ("", "upload"),
    ],
)
def test_upload_names_are_reduced_to_a_basename(name, expected):
    assert upload_safety.safe_upload_name(name) == expected


def test_safe_zip_extraction_rejects_parent_traversal(tmp_path):
    with _archive([("../outside.pdf", b"pdf")]) as archive:
        with pytest.raises(ValueError, match="escapes"):
            upload_safety.extract_zip_safely(archive, tmp_path)
    assert not (tmp_path.parent / "outside.pdf").exists()


def test_safe_zip_extraction_only_expands_allowed_files(tmp_path):
    with _archive([("docs/a.pdf", b"pdf"), ("payload.bin", b"ignored")]) as archive:
        paths = upload_safety.extract_zip_safely(
            archive, tmp_path, allowed_extensions={".pdf"}
        )
    assert [path.relative_to(tmp_path) for path in paths] == [pathlib.Path("docs/a.pdf")]
    assert not (tmp_path / "payload.bin").exists()


def test_safe_zip_extraction_rejects_excessive_expansion(tmp_path, monkeypatch):


    def test_safe_zip_members_bounds_archives_read_directly(monkeypatch):
        monkeypatch.setattr(upload_safety, "MAX_ARCHIVE_MEMBER_BYTES", 3)
        with _archive([("large.txt", b"1234")]) as archive:
            with pytest.raises(ValueError, match="per-file limit"):
                upload_safety.safe_zip_members(archive, allowed_extensions={".txt"})
    monkeypatch.setattr(upload_safety, "MAX_ARCHIVE_TOTAL_BYTES", 3)
    with _archive([("a.pdf", b"1234")]) as archive:
        with pytest.raises(ValueError, match="total size"):
            upload_safety.extract_zip_safely(archive, tmp_path)


def test_output_directories_only_change_when_stems_collide():
    used = set()
    assert upload_safety.unique_output_directory("a/report.pdf", used) == pathlib.Path("a/report")
    assert upload_safety.unique_output_directory("a/report.png", used) == pathlib.Path("a/report_png")
    assert upload_safety.unique_output_directory("b/report.pdf", used) == pathlib.Path("b/report")