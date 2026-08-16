import pytest

from medchange.models.vlm.config import (
    VLMConfig,
)


def test_default_vlm_config():
    config = VLMConfig()

    assert config.load_in_4bit is True

    assert (
        config.min_visual_tokens
        == 256
    )

    assert (
        config.max_visual_tokens
        == 512
    )


def test_invalid_visual_tokens():
    with pytest.raises(
        ValueError
    ):
        VLMConfig(
            min_visual_tokens=512,
            max_visual_tokens=256,
        )


def test_invalid_generation_length():
    with pytest.raises(
        ValueError
    ):
        VLMConfig(
            max_new_tokens=0
        )