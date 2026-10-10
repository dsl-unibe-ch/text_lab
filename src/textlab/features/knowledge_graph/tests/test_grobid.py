"""Starting Grobid and sending it PDFs, without a Grobid server."""

from types import SimpleNamespace

import pytest

from textlab.common.config import Settings
from textlab.common.storage import Workspace
from textlab.features.knowledge_graph import grobid


@pytest.fixture
def closed_port(monkeypatch, tmp_path):
    monkeypatch.setattr(grobid, "_port_open", lambda host, port: False)
    workspace = Workspace(tmp_path / "workspace")
    monkeypatch.setattr(grobid, "get_workspace", lambda: workspace)


def test_a_running_server_is_used(monkeypatch):
    monkeypatch.setattr(grobid, "_port_open", lambda host, port: True)

    def no_start(*args, **kwargs):
        raise AssertionError("must not start a server")

    monkeypatch.setattr(grobid.subprocess, "Popen", no_start)
    grobid.ensure_grobid_server()


def test_the_image_must_be_configured(monkeypatch, closed_port):
    monkeypatch.setattr(grobid, "get_settings", lambda: Settings())
    with pytest.raises(grobid.GrobidError, match="TEXT_LAB_GROBID"):
        grobid.ensure_grobid_server()


def test_apptainer_is_needed(monkeypatch, closed_port, tmp_path):
    settings = Settings(grobid_container=tmp_path / "grobid.sif")
    monkeypatch.setattr(grobid, "get_settings", lambda: settings)
    monkeypatch.setattr(grobid.shutil, "which", lambda name: None)
    with pytest.raises(grobid.GrobidError, match="nested container"):
        grobid.ensure_grobid_server()


def test_the_server_is_started_detached(monkeypatch, closed_port, tmp_path):
    image = tmp_path / "grobid.sif"
    image.write_bytes(b"")
    settings = Settings(grobid_container=image)
    monkeypatch.setattr(grobid, "get_settings", lambda: settings)
    monkeypatch.setattr(grobid.shutil, "which", lambda name: f"/bin/{name}")
    started = []
    monkeypatch.setattr(
        grobid.subprocess,
        "Popen",
        lambda command, **kwargs: started.append((command, kwargs)),
    )
    answers = iter([False, True])
    monkeypatch.setattr(grobid, "_port_open", lambda h, p: next(answers))
    monkeypatch.setattr(grobid.time, "sleep", lambda seconds: None)

    grobid.ensure_grobid_server()
    ((command, kwargs),) = started
    assert command[:2] == ["apptainer", "exec"]
    assert str(image) in command
    assert kwargs["start_new_session"] is True


def test_the_command_moves_grobid_to_its_ports(tmp_path, monkeypatch):
    monkeypatch.setattr(grobid, "GROBID_PORT", 9000)
    command = grobid.server_command("apptainer", "/img.sif", tmp_path)
    assert f"{tmp_path}:/opt/grobid/grobid-home/tmp" in command
    script = command[-1]
    assert "s/port: 8070/port: 9000/g" in script
    assert "s/port: 8071/port: 9001/g" in script


def test_a_rejected_pdf_raises(monkeypatch, tmp_path):
    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(b"%PDF")
    monkeypatch.setattr(grobid, "ensure_grobid_server", lambda: None)
    answers = iter(
        [
            SimpleNamespace(status_code=200, text="<TEI/>"),
            SimpleNamespace(status_code=500, text="broken"),
        ]
    )
    monkeypatch.setattr(
        grobid.requests, "post", lambda url, files, timeout: next(answers)
    )
    assert grobid.process_pdf(pdf) == "<TEI/>"
    with pytest.raises(grobid.GrobidError, match="status 500: broken"):
        grobid.process_pdf(pdf)
