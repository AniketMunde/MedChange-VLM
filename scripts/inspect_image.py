import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(SRC_DIR),
    )

from medchange.preprocessing import (
    MedicalImagePreprocessor,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Inspect preprocessing of a "
            "medical image."
        )
    )

    parser.add_argument(
        "image",
        type=str,
        help="Path to the input image.",
    )

    parser.add_argument(
        "--size",
        type=int,
        default=448,
        help=(
            "Target image size. "
            "Default: 448."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    preprocessor = (
        MedicalImagePreprocessor(
            image_size=args.size
        )
    )

    result = preprocessor(
        args.image
    )

    tensor = result.pixel_values
    metadata = result.metadata

    print("=" * 60)
    print(
        "MedChange-VLM Image Inspection"
    )
    print("=" * 60)

    print(
        f"Image path       : "
        f"{metadata.path}"
    )

    print(
        f"Original size    : "
        f"{metadata.original_width}"
        f"x"
        f"{metadata.original_height}"
    )

    print(
        f"Original mode    : "
        f"{metadata.original_mode}"
    )

    print(
        f"Tensor shape     : "
        f"{tuple(tensor.shape)}"
    )

    print(
        f"Tensor dtype     : "
        f"{tensor.dtype}"
    )

    print(
        f"Pixel minimum    : "
        f"{tensor.min().item():.4f}"
    )

    print(
        f"Pixel maximum    : "
        f"{tensor.max().item():.4f}"
    )

    print(
        f"Pixel mean       : "
        f"{tensor.mean().item():.4f}"
    )

    print("=" * 60)
    print(
        "Image preprocessing successful."
    )
    print("=" * 60)


if __name__ == "__main__":
    main()