from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

SRC_DIR = (
    PROJECT_ROOT
    / "src"
)

if str(SRC_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(SRC_DIR),
    )

from medchange.inference import (
    MedChangePipeline,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run MedChange-VLM "
            "single-image inference."
        )
    )

    parser.add_argument(
        "image",
        type=str,
        help="Path to chest X-ray.",
    )

    parser.add_argument(
        "--study-id",
        type=str,
        default="demo-study",
    )

    parser.add_argument(
        "--question",
        type=str,
        default=None,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    pipeline = (
        MedChangePipeline()
    )

    prediction,metrics = (
        pipeline.analyze(
            image_path=args.image,
            study_id=args.study_id,
            question=args.question,
        )
    )

    print()
    print("Inference metrics")
    print("-" * 70)

    print(
        f"Elapsed time     : "
        f"{metrics.elapsed_seconds:.2f} s"
    )

    if metrics.gpu_allocated_gb is not None:
        print(
            f"GPU allocated    : "
            f"{metrics.gpu_allocated_gb:.2f} GB"
        )

        print(
            f"GPU reserved     : "
            f"{metrics.gpu_reserved_gb:.2f} GB"
        )

        print(
            f"GPU peak memory  : "
            f"{metrics.gpu_peak_allocated_gb:.2f} GB"
        )


if __name__ == "__main__":
    main()