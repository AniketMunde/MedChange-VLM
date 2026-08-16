import pytest

from medchange.models.vision import (
    BiomedCLIPConfig,
)


def test_default_biomedclip_config():
    config = (
        BiomedCLIPConfig()
    )

    assert (
        "BiomedCLIP"
        in config.model_name
    )

    assert config.device == "cuda"

    assert config.precision == "fp16"


def test_invalid_device():
    with pytest.raises(
        ValueError
    ):
        BiomedCLIPConfig(
            device="tpu"
        )