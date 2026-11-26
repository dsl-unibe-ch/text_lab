import io

import pytest
from starlette.datastructures import FormData, UploadFile

from textlab.ui.web import uploads


def upload(name, content=b"data"):
    return UploadFile(io.BytesIO(content), filename=name)


def test_a_saved_upload_keeps_only_its_own_name(tmp_path):
    path = uploads.save(upload("../../secret/notes.txt", b"hello"), tmp_path)
    assert path == tmp_path / "notes.txt"
    assert path.read_bytes() == b"hello"


def test_names_already_used_get_a_number(tmp_path):
    names = [
        uploads.save(upload(name), tmp_path).name
        for name in ("talk.mp3", "TALK.mp3", "talk.mp3")
    ]
    assert names == ["talk.mp3", "TALK_2.mp3", "talk_3.mp3"]


def test_an_empty_name_gets_a_fallback(tmp_path):
    assert uploads.save(upload(".."), tmp_path).name == "upload"


def test_empty_file_inputs_send_no_file():
    form = FormData([("file", upload("")), ("file", upload("a.wav"))])
    assert [item.filename for item in uploads.files(form, "file")] == ["a.wav"]


def test_one_file_is_required_and_checked():
    with pytest.raises(uploads.UploadError, match="Choose a file"):
        uploads.single_file(FormData([]), "file")
    form = FormData([("file", upload("talk.MP3"))])
    assert uploads.single_file(form, "file", suffixes=[".mp3"]).filename
    with pytest.raises(uploads.UploadError, match="Accepted: mp3, wav"):
        uploads.single_file(
            FormData([("file", upload("notes.docx"))]),
            "file",
            suffixes=[".wav", ".mp3"],
        )
