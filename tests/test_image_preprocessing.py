from pathlib import Path

import pytest
import torch
from PIL import Image

from medchange.preprocessing import (
    ImageValidationError,
    MedicalImagePreprocessor,
    validate_image_path,
)


@pytest.fixture
def grayscale_image(
    tmp_path: Path,
) -> Path:
    image_path = tmp_path / "test_xray.png"

    image = Image.new(
        mode="L",
        size=(800, 1000),
        color=128,
    )

    image.save(image_path)

    return image_path


def test_validate_image_path(
    grayscale_image: Path,
):
    path = validate_image_path(
        grayscale_image
    )

    assert path.exists()


def test_missing_image():
    with pytest.raises(
        FileNotFoundError
    ):
        validate_image_path(
            "missing_image.png"
        )


def test_unsupported_extension(
    tmp_path: Path,
):
    text_file = tmp_path / "invalid.txt"

    text_file.write_text(
        "not an image",
        encoding="utf-8",
    )

    with pytest.raises(
        ImageValidationError
    ):
        validate_image_path(
            text_file
        )


def test_preprocessing_shape(
    grayscale_image: Path,
):
    preprocessor = (
        MedicalImagePreprocessor(
            image_size=448
        )
    )

    result = preprocessor(
        grayscale_image
    )

    assert result.pixel_values.shape == (
        3,
        448,
        448,
    )


def test_preprocessing_dtype(
    grayscale_image: Path,
):
    preprocessor = (
        MedicalImagePreprocessor(
            image_size=448
        )
    )

    result = preprocessor(
        grayscale_image
    )

    assert (
        result.pixel_values.dtype
        == torch.float32
    )


def test_pixel_range(
    grayscale_image: Path,
):
    preprocessor = (
        MedicalImagePreprocessor(
            image_size=448
        )
    )

    result = preprocessor(
        grayscale_image
    )

    assert (
        result.pixel_values.min()
        >= 0.0
    )

    assert (
        result.pixel_values.max()
        <= 1.0
    )


def test_metadata_preserved(
    grayscale_image: Path,
):
    preprocessor = (
        MedicalImagePreprocessor(
            image_size=448
        )
    )

    result = preprocessor(
        grayscale_image
    )

    assert (
        result.metadata.original_width
        == 800
    )

    assert (
        result.metadata.original_height
        == 1000
    )

    assert (
        result.metadata.original_mode
        == "L"
    )


def test_rgb_conversion(
    grayscale_image: Path,
):
    preprocessor = (
        MedicalImagePreprocessor(
            image_size=448
        )
    )

    result = preprocessor(
        grayscale_image
    )

    channel_0 = (
        result.pixel_values[0]
    )

    channel_1 = (
        result.pixel_values[1]
    )

    channel_2 = (
        result.pixel_values[2]
    )

    assert torch.allclose(
        channel_0,
        channel_1,
    )

    assert torch.allclose(
        channel_1,
        channel_2,
    )