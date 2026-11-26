"""Jobs through the web: submitting, live updates, cancelling, downloads.

The developer page's demo job stands in for a feature's job.
"""

import re

from textlab.ui.web.jobs import JobState
from textlab.ui.web.tests.conftest import BASE, wait_for


def htmx(client):
    return {"X-CSRF-Token": client.csrf, "HX-Request": "true"}


def submit_demo(client, seconds=1, **fields):
    files = fields.pop("files", None)
    response = client.post(
        f"{BASE}/dev/demo",
        data={"seconds": str(seconds), "outcome": "done", **fields},
        files=files,
        headers=htmx(client),
    )
    assert response.status_code == 200, response.text[:300]
    return re.search(r'id="job-([0-9a-f]+)"', response.text).group(1)


def read_events(client, page="demo"):
    """Read the page's event stream until it says idle."""
    events, name = [], None
    with client.stream("GET", f"{BASE}/events?page={page}") as response:
        assert response.headers["content-type"].startswith("text/event-stream")
        data = []
        for line in response.iter_lines():
            if line.startswith("event:"):
                name = line.partition(":")[2].strip()
            elif line.startswith("data:"):
                data.append(line[5:].removeprefix(" "))
            elif line == "" and name:
                events.append((name, "\n".join(data)))
                if name == "idle":
                    break
                name, data = None, []
    return events


def test_a_job_runs_from_submission_to_download(client, runner):
    job_id = submit_demo(client, seconds=1)
    events = read_events(client)

    assert events[-1][0] == "idle"
    updates = "".join(data for name, data in events if name == "update")
    assert f'id="job-{job_id}"' in updates
    assert 'hx-swap-oob="true"' in updates
    assert 'id="job-count"' in updates
    assert "record--done" in updates

    job = runner.get(job_id)
    assert job.state is JobState.DONE
    report = client.get(f"{BASE}/jobs/{job_id}/files/demo-report.txt")
    assert report.status_code == 200
    assert report.text.startswith("The demo job ran for 1 s.")
    umlauts = client.get(
        f"{BASE}/jobs/{job_id}/files/Gr%C3%BC%C3%9Fe%20aus%20Bern.txt"
    )
    assert umlauts.text == "Grüße aus Bern\n"
    assert (
        "filename*=UTF-8''Gr%C3%BC%C3%9Fe%20aus%20Bern.txt"
        in (umlauts.headers["content-disposition"])
    )


def test_the_card_offers_the_downloads_and_removal(client, runner):
    job_id = submit_demo(client)
    wait_for(lambda: runner.get(job_id).state is JobState.DONE)
    card = client.get(f"{BASE}/jobs/{job_id}").text
    assert f'href="{BASE}/jobs/{job_id}/files/demo-report.txt"' in card
    assert f'hx-post="{BASE}/jobs/{job_id}/remove"' in card
    assert "/cancel" not in card


def test_an_upload_reaches_the_job_and_is_deleted_after(client, runner):
    job_id = submit_demo(
        client, files={"file": ("../../etc/notes.txt", b"hello")}
    )
    job = runner.get(job_id)
    wait_for(lambda: job.state is JobState.DONE)
    assert "Upload: notes.txt, 5 bytes" in job.result["lines"][1]
    assert not job.inputs.exists()


def test_a_running_job_can_be_cancelled(client, runner):
    job_id = submit_demo(client, seconds=60)
    job = runner.get(job_id)
    wait_for(lambda: job.state is JobState.RUNNING)
    response = client.post(
        f"{BASE}/jobs/{job_id}/cancel", headers=htmx(client)
    )
    assert response.status_code == 200
    assert "Stopping" in response.text
    wait_for(lambda: job.state is JobState.CANCELLED)
    assert "Cancelled" in client.get(f"{BASE}/jobs/{job_id}").text


def test_a_waiting_job_shows_its_place(client, runner):
    first = submit_demo(client, seconds=60)
    second = submit_demo(client, seconds=60)
    wait_for(lambda: runner.get(first).state is JobState.RUNNING)
    card = client.get(f"{BASE}/jobs/{second}").text
    assert "Waits for 1 job ahead of it." in card
    for job_id in (second, first):
        client.post(f"{BASE}/jobs/{job_id}/cancel", headers=htmx(client))


def test_a_finished_job_can_be_removed(client, runner):
    job_id = submit_demo(client)
    wait_for(lambda: runner.get(job_id).state is JobState.DONE)
    response = client.post(
        f"{BASE}/jobs/{job_id}/remove", headers=htmx(client)
    )
    assert response.status_code == 200
    assert 'id="job-count"' in response.text
    assert runner.get(job_id) is None
    gone = client.get(f"{BASE}/jobs/{job_id}/files/demo-report.txt")
    assert gone.status_code == 404


def test_an_active_job_cannot_be_removed(client, runner):
    job_id = submit_demo(client, seconds=60)
    response = client.post(
        f"{BASE}/jobs/{job_id}/remove", headers=htmx(client)
    )
    assert response.status_code == 409
    client.post(f"{BASE}/jobs/{job_id}/cancel", headers=htmx(client))


def test_a_failing_job_shows_its_error(client, runner):
    job_id = submit_demo(client, outcome="failed")
    wait_for(lambda: runner.get(job_id).state is JobState.FAILED)
    card = client.get(f"{BASE}/jobs/{job_id}").text
    assert "The demo job failed on purpose." in card
    assert "Details for a problem report" in card


def test_the_stream_of_a_quiet_session_ends_at_once(client):
    events = read_events(client)
    assert [name for name, _ in events] == ["update", "idle"]


def test_reloading_the_page_shows_its_jobs(client, runner):
    job_id = submit_demo(client)
    wait_for(lambda: runner.get(job_id).state is JobState.DONE)
    page = client.get(f"{BASE}/dev/demo").text
    assert f'id="job-{job_id}"' in page
    assert f'href="{BASE}/dev/demo#job-{job_id}"' in page


def test_invalid_values_are_explained(client):
    response = client.post(
        f"{BASE}/dev/demo", data={"seconds": "0"}, headers=htmx(client)
    )
    assert response.status_code == 422
    assert "Choose between 1 and" in response.text
