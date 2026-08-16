from __future__ import annotations

from pathlib import Path

from PIL import Image, UnidentifiedImageError


SUPPORTED_IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
}


class ImageValidationError(ValueError):
    """
    Raised when an input medical image fails validation.
    """


def validate_image_path(
    image_path: str | Path,
) -> Path:
    """
    Validate that an image path exists and uses a supported format.

    Parameters
    ----------
    image_path:
        Path to an image file.

    Returns
    -------
    Path
        Resolved image path.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.

    ImageValidationError
        If the path is not a file or the extension is unsupported.
    """

    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Image does not exist: {path}"
        )

    if not path.is_file():
        raise ImageValidationError(
            f"Expected an image file but received: {path}"
        )

    suffix = path.suffix.lower()

    if suffix not in SUPPORTED_IMAGE_EXTENSIONS:
        raise ImageValidationError(
            f"Unsupported image extension '{suffix}'. "
            f"Supported extensions: "
            f"{sorted(SUPPORTED_IMAGE_EXTENSIONS)}"
        )

    return path


def verify_image_file(
    image_path: str | Path,
) -> Path:
    """
    Verify that the file can actually be decoded as an image.

    Pillow's verify() checks image integrity without fully decoding
    all pixel data.

    Returns
    -------
    Path
        Validated path.
    """

    path = validate_image_path(image_path)

    try:
        with Image.open(path) as image:
            image.verify()

    except (
        UnidentifiedImageError,
        OSError,
        SyntaxError,
    ) as exc:
        raise ImageValidationError(
            f"Image file appears invalid or corrupted: {path}"
        ) from exc

    return path