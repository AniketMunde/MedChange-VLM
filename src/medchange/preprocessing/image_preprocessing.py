from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import torch
from PIL import Image, ImageOps
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode

from medchange.preprocessing.validation import (
    verify_image_file,
)


@dataclass(frozen=True)
class ImageMetadata:
    """
    Metadata describing an image before preprocessing.
    """

    path: Path
    original_width: int
    original_height: int
    original_mode: str


@dataclass(frozen=True)
class PreprocessedImage:
    """
    Output container for processed medical images.
    """

    pixel_values: torch.Tensor
    metadata: ImageMetadata


class MedicalImagePreprocessor:
    """
    Deterministic preprocessing pipeline for medical images.

    Responsibilities
    ----------------
    1. Validate the image.
    2. Load using Pillow.
    3. Correct orientation metadata.
    4. Convert to RGB.
    5. Resize with optional aspect-ratio preservation.
    6. Pad to a square image.
    7. Convert to a float tensor in [0, 1].

    Model-specific normalization is intentionally NOT applied here.
    """

    def __init__(
        self,
        image_size: int = 448,
        preserve_aspect_ratio: bool = True,
    ) -> None:
        if image_size <= 0:
            raise ValueError(
                "image_size must be greater than zero."
            )

        self.image_size = image_size
        self.preserve_aspect_ratio = (
            preserve_aspect_ratio
        )

    def load_image(
        self,
        image_path: str | Path,
    ) -> Tuple[Image.Image, ImageMetadata]:
        """
        Validate and load an image as RGB.
        """

        path = verify_image_file(image_path)

        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image)

            metadata = ImageMetadata(
                path=path,
                original_width=image.width,
                original_height=image.height,
                original_mode=image.mode,
            )

            image = image.convert("RGB")

            return image.copy(), metadata

    def _resize_with_padding(
        self,
        image: Image.Image,
    ) -> Image.Image:
        """
        Resize while preserving aspect ratio and pad to square.
        """

        width, height = image.size

        scale = min(
            self.image_size / width,
            self.image_size / height,
        )

        new_width = max(
            1,
            round(width * scale),
        )

        new_height = max(
            1,
            round(height * scale),
        )

        resized = TF.resize(
            image,
            [new_height, new_width],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )

        pad_width = self.image_size - new_width
        pad_height = self.image_size - new_height

        left = pad_width // 2
        right = pad_width - left

        top = pad_height // 2
        bottom = pad_height - top

        padded = TF.pad(
            resized,
            padding=[
                left,
                top,
                right,
                bottom,
            ],
            fill=0,
        )

        return padded

    def _resize_direct(
        self,
        image: Image.Image,
    ) -> Image.Image:
        """
        Resize directly to a square image.
        """

        return TF.resize(
            image,
            [self.image_size, self.image_size],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )

    def preprocess(
        self,
        image_path: str | Path,
    ) -> PreprocessedImage:
        """
        Run complete image preprocessing.

        Returns
        -------
        PreprocessedImage
            Tensor shape is [3, H, W].
            Pixel range is [0, 1].
        """

        image, metadata = self.load_image(
            image_path
        )

        if self.preserve_aspect_ratio:
            image = self._resize_with_padding(
                image
            )
        else:
            image = self._resize_direct(
                image
            )

        pixel_values = TF.to_tensor(image)

        if pixel_values.shape != (
            3,
            self.image_size,
            self.image_size,
        ):
            raise RuntimeError(
                "Unexpected processed image shape: "
                f"{tuple(pixel_values.shape)}"
            )

        return PreprocessedImage(
            pixel_values=pixel_values,
            metadata=metadata,
        )

    def __call__(
        self,
        image_path: str | Path,
    ) -> PreprocessedImage:
        return self.preprocess(image_path)