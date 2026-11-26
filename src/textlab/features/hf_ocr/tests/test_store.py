"""The user's folder of downloaded models."""

from pathlib import Path

from textlab.features.hf_ocr import store


def cached_model(folder, repo_id, size):
    """Lay out a model like a Hugging Face cache: a blob and a link."""
    root = folder / ("models--" + repo_id.replace("/", "--"))
    (root / "blobs").mkdir(parents=True)
    (root / "snapshots" / "abc").mkdir(parents=True)
    blob = root / "blobs" / "1234"
    blob.write_bytes(b"x" * size)
    (root / "snapshots" / "abc" / "model.safetensors").symlink_to(blob)
    return root


def test_the_folder_is_in_the_home_unless_configured(monkeypatch, tmp_path):
    monkeypatch.delenv(store.DIR_VARIABLE, raising=False)
    assert store.models_dir() == (
        Path.home() / ".cache" / "textlab" / "hf_ocr_models"
    )
    monkeypatch.setenv(store.DIR_VARIABLE, str(tmp_path))
    assert store.models_dir() == tmp_path


def test_models_are_listed_with_their_size_counted_once(tmp_path):
    cached_model(tmp_path, "org/b-model", 300)
    cached_model(tmp_path, "org/a-model", 100)
    (tmp_path / "xet").mkdir()  # the download's chunk cache is no model
    assert store.stored_models(tmp_path) == [
        store.StoredModel("org/a-model", 100),
        store.StoredModel("org/b-model", 300),
    ]
    assert store.stored_models(tmp_path / "missing") == []


def test_a_model_is_deleted_and_nothing_else(tmp_path):
    cached_model(tmp_path, "org/a-model", 10)
    keep = cached_model(tmp_path, "org/b-model", 10)
    assert store.delete_model("org/a-model", tmp_path)
    assert not store.delete_model("org/a-model", tmp_path)
    assert not store.delete_model("../outside", tmp_path)
    assert [m.repo_id for m in store.stored_models(tmp_path)] == [
        "org/b-model"
    ]
    assert keep.is_dir()


def test_sizes_are_shown_in_mb_or_gb():
    assert store.format_size(5 * 1024**2) == "5 MB"
    assert store.format_size(int(2.5 * 1024**3)) == "2.5 GB"
