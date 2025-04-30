import json
import zipfile
import io
from pages.Transcribe import (
    seconds_to_srt_time,
    identity,
    make_srt,
    make_csv,
    make_json,
    create_all_formats_zip
)

# Sample segment data for testing
sample_segments = [
    {"start": 0.0, "end": 2.345, "text": "Hello world."},
    {"start": 2.345, "end": 5.0, "text": "This is a test."}
]

sample_result = {
    "text": "Hello world. This is a test.",
    "segments": sample_segments
}

def test_seconds_to_srt_time():
    assert seconds_to_srt_time(3661.567) == "01:01:01,567"

def test_identity():
    assert identity("some text") == "some text"

def test_make_srt():
    srt = make_srt(sample_segments)
    assert "00:00:00,000 --> 00:00:02,345" in srt
    assert "Hello world." in srt
    assert "00:00:02,345 --> 00:00:05,000" in srt
    assert "This is a test." in srt

def test_make_csv():
    csv = make_csv(sample_segments)
    assert "start,end,text" in csv
    assert '0.0,2.345,"Hello world."' in csv
    assert '2.345,5.0,"This is a test."' in csv

def test_make_json():
    json_output = make_json(sample_result)
    assert json.loads(json_output)["text"] == "Hello world. This is a test."

def test_create_all_formats_zip():
    zip_bytes = create_all_formats_zip(
        text=sample_result["text"],
        segments=sample_segments,
        result_dict=sample_result
    )

    with zipfile.ZipFile(io.BytesIO(zip_bytes), 'r') as zf:
        assert set(zf.namelist()) == {
            "transcription.txt",
            "transcription.srt",
            "transcription.csv",
            "transcription.json"
        }

        # Validate file content exists
        with zf.open("transcription.txt") as f:
            assert "Hello world. This is a test." in f.read().decode()

        with zf.open("transcription.json") as f:
            assert json.loads(f.read().decode())["text"] == "Hello world. This is a test."
