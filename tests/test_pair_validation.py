from pathlib import Path

import pytest
from PIL import Image

from medchange.safety.validation import (
    validate_longitudinal_pair,
)


def test_valid_pair(
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
        "L",
        (64, 64),
        50,
    ).save(
        prior
    )

    Image.new(
        "L",
        (64, 64),
        100,
    ).save(
        current
    )

    resolved_prior, resolved_current = (
        validate_longitudinal_pair(
            prior,
            current,
        )
    )

    assert resolved_prior == prior
    assert resolved_current == current


def test_same_path_rejected(
    tmp_path: Path,
):
    image = (
        tmp_path
        / "same.png"
    )

    Image.new(
        "L",
        (64, 64),
        50,
    ).save(
        image
    )

    with pytest.raises(
        ValueError,
        match="same file",
    ):
        validate_longitudinal_pair(
            image,
            image,
        )


def test_identical_content_rejected(
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

    image = Image.new(
        "L",
        (64, 64),
        80,
    )

    image.save(
        prior
    )

    image.save(
        current
    )

    with pytest.raises(
        ValueError,
        match="identical file content",
    ):
        validate_longitudinal_pair(
            prior,
            current,
        )


def test_corrupt_image_rejected(
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

    prior.write_text(
        "not an image",
        encoding="utf-8",
    )

    Image.new(
        "L",
        (64, 64),
    ).save(
        current
    )

    with pytest.raises(
        ValueError,
        match="corrupted",
    ):
        validate_longitudinal_pair(
            prior,
            current,
        )