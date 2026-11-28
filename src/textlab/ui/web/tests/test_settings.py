import pytest

from textlab.ui.web.settings import WebSettings, normalize_base_path


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("/node/gnode21/31234/", "/node/gnode21/31234"),
        ("/node/gnode21/31234", "/node/gnode21/31234"),
        ("node/gnode21/31234", "/node/gnode21/31234"),
        ("/", ""),
        ("", ""),
        ("  /x/  ", "/x"),
    ],
)
def test_base_path_is_normalized(given, expected):
    assert normalize_base_path(given) == expected
    assert WebSettings(token="t", base_path=given).base_path == expected


def test_settings_come_from_the_launcher_and_the_environment():
    settings = WebSettings.from_env(
        base_path="/node/n/1/",
        public_host=" portal.example.org ",
        upload_limit=123,
        environ={"TOKEN": "secret", "USER": "alice"},
    )
    assert settings.token == "secret"
    assert settings.owner == "alice"
    assert settings.base_path == "/node/n/1"
    assert settings.public_host == "portal.example.org"
    assert settings.upload_limit == 123


def test_a_missing_password_stops_the_server():
    with pytest.raises(ValueError, match="TOKEN"):
        WebSettings.from_env(environ={"USER": "alice"})


def test_secrets_stay_out_of_the_representation():
    settings = WebSettings(token="secret-password")
    assert "secret-password" not in repr(settings)
    assert repr(settings.secret_key) not in repr(settings)


def test_each_server_signs_with_a_new_key():
    assert (
        WebSettings(token="t").secret_key != WebSettings(token="t").secret_key
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [("1795694800", 1795694800.0), ("", None), ("0", None), ("x", None)],
)
def test_the_session_end_comes_from_slurm(value, expected):
    environ = {"TOKEN": "secret", "SLURM_JOB_END_TIME": value}
    assert WebSettings.from_env(environ=environ).session_end == expected
