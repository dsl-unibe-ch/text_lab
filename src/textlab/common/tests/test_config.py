"""Site settings come from the environment, and say what is missing."""

import dataclasses
from pathlib import Path

import pytest

from textlab.common import config


def test_every_setting_has_a_field_and_the_reverse():
    fields = {field.name for field in dataclasses.fields(config.Settings)}
    assert fields == {spec.name for spec in config.SETTINGS}


def test_variables_are_read_and_typed():
    settings = config.Settings.from_env(
        {
            "TEXT_LAB_WORKDIR": "/scratch/job",
            "TEXT_LAB_GPUSTACK_URL": "https://llm.example.org/v1",
        }
    )
    assert settings.workdir == Path("/scratch/job")
    assert settings.gpustack_url == "https://llm.example.org/v1"


def test_unset_and_blank_variables_are_none():
    settings = config.Settings.from_env({"TEXT_LAB_HF_TOKEN_FILE": "  "})
    assert settings.hf_token_file is None
    assert settings.grobid_container is None


def test_require_names_the_variable_to_set():
    settings = config.Settings.from_env({})
    with pytest.raises(config.MissingSettingError) as error:
        settings.require("grobid_container")
    assert "TEXT_LAB_GROBID_CONTAINER" in str(error.value)
    assert "deploy/site.env" in str(error.value)


def test_require_returns_configured_values():
    settings = config.Settings.from_env(
        {"TEXT_LAB_GROBID_CONTAINER": "/images/grobid.sif"}
    )
    assert settings.require("grobid_container") == Path("/images/grobid.sif")


def test_require_rejects_unknown_names():
    with pytest.raises(KeyError):
        config.Settings().require("no_such_setting")
