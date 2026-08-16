from __future__ import annotations

import hashlib
from pathlib import Path

from PIL import Image


SUPPORTED_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
}


def validate_image_file(
    path: str | Path,
) -> Path:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Image does not exist: {path}"
        )

    if not path.is_file():
        raise ValueError(
            f"Path is not a file: {path}"
        )

    if path.suffix.lower() not in (
        SUPPORTED_SUFFIXES
    ):
        raise ValueError(
            "Unsupported image format: "
            f"{path.suffix}"
        )

    try:
        with Image.open(path) as image:
            image.verify()

    except Exception as exc:
        raise ValueError(
            f"Image appears corrupted: {path}"
        ) from exc

    return path


def image_sha256(
    path: str | Path,
) -> str:
    digest = hashlib.sha256()

    with Path(path).open("rb") as file:
        while True:
            chunk = file.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def validate_longitudinal_pair(
    prior_path: str | Path,
    current_path: str | Path,
) -> tuple[
    Path,
    Path,
]:
    prior = validate_image_file(
        prior_path
    )

    current = validate_image_file(
        current_path
    )

    if prior.resolve() == current.resolve():
        raise ValueError(
            "Prior and current image paths "
            "refer to the same file."
        )

    if (
        image_sha256(prior)
        == image_sha256(current)
    ):
        raise ValueError(
            "Prior and current images have "
            "identical file content."
        )

    return (
        prior,
        current,
    )