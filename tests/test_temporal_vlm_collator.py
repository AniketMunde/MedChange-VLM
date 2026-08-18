from pathlib import Path

import pytest
from PIL import Image

from medchange.training.temporal_vlm_collator import (
    TemporalVLMCollator,
)


class DummyProcessor:
    pass


def test_load_two_images(
    tmp_path: Path,
):
    prior = (
        tmp_path
        / "prior.png"
    )

    current = (
        tmp_path
        / "current.png"
    )

    Image.new(
        "RGB",
        (224, 224),
    ).save(
        prior
    )

    Image.new(
        "RGB",
        (224, 224),
    ).save(
        current
    )

    collator = (
        TemporalVLMCollator(
            processor=DummyProcessor(),
        )
    )

    images = (
        collator._load_images(
            [
                str(prior),
                str(current),
            ]
        )
    )

    assert len(
        images
    ) == 2

    assert (
        images[0].mode
        == "RGB"
    )


def test_requires_two_images(
    tmp_path: Path,
):
    image = (
        tmp_path
        / "image.png"
    )

    Image.new(
        "RGB",
        (224, 224),
    ).save(
        image
    )

    collator = (
        TemporalVLMCollator(
            processor=DummyProcessor(),
        )
    )

    with pytest.raises(
        ValueError,
        match="exactly two images",
    ):
        collator._load_images(
            [
                str(image)
            ]
        )