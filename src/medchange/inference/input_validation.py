from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, UnidentifiedImageError


SUPPORTED_FORMATS = {
    "PNG",
    "JPEG",
    "JPG",
}

MIN_IMAGE_SIZE = 128
MAX_ASPECT_RATIO = 2.0


@dataclass(frozen=True)
class ImageValidationResult:
    valid: bool
    reason: str
    width: int | None = None
    height: int | None = None
    image_format: str | None = None


@dataclass(frozen=True)
class PairValidationResult:
    valid: bool
    reason: str
    prior: ImageValidationResult
    current: ImageValidationResult


def validate_image(
    image_path: str | Path,
) -> ImageValidationResult:

    path = Path(image_path)

    if not path.exists():
        return ImageValidationResult(
            valid=False,
            reason=f"Image does not exist: {path}",
        )

    if not path.is_file():
        return ImageValidationResult(
            valid=False,
            reason=f"Path is not a file: {path}",
        )

    try:
        with Image.open(path) as image:
            image.verify()

        # Reopen after verify()
        with Image.open(path) as image:
            width, height = image.size
            image_format = (
                image.format.upper()
                if image.format
                else "UNKNOWN"
            )

    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
    ):
        return ImageValidationResult(
            valid=False,
            reason="File is not a readable image.",
        )

    if image_format not in SUPPORTED_FORMATS:
        return ImageValidationResult(
            valid=False,
            reason=(
                f"Unsupported image format: "
                f"{image_format}"
            ),
            width=width,
            height=height,
            image_format=image_format,
        )

    if (
        width < MIN_IMAGE_SIZE
        or height < MIN_IMAGE_SIZE
    ):
        return ImageValidationResult(
            valid=False,
            reason=(
                "Image resolution is too small for "
                "MedChange analysis."
            ),
            width=width,
            height=height,
            image_format=image_format,
        )

    aspect_ratio = max(
        width / height,
        height / width,
    )

    if aspect_ratio > MAX_ASPECT_RATIO:
        return ImageValidationResult(
            valid=False,
            reason=(
                "Image dimensions are inconsistent "
                "with the expected chest X-ray input."
            ),
            width=width,
            height=height,
            image_format=image_format,
        )

    return ImageValidationResult(
        valid=True,
        reason="Image passed structural validation.",
        width=width,
        height=height,
        image_format=image_format,
    )


def validate_image_pair(
    prior_path: str | Path,
    current_path: str | Path,
) -> PairValidationResult:

    prior = validate_image(
        prior_path
    )

    current = validate_image(
        current_path
    )

    if not prior.valid:
        return PairValidationResult(
            valid=False,
            reason=(
                f"Prior image rejected: "
                f"{prior.reason}"
            ),
            prior=prior,
            current=current,
        )

    if not current.valid:
        return PairValidationResult(
            valid=False,
            reason=(
                f"Current image rejected: "
                f"{current.reason}"
            ),
            prior=prior,
            current=current,
        )

    return PairValidationResult(
        valid=True,
        reason=(
            "Both images passed structural "
            "input validation."
        ),
        prior=prior,
        current=current,
    )