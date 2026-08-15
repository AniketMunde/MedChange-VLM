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

    prediction = (
        pipeline.analyze(
            image_path=args.image,
            study_id=args.study_id,
            question=args.question,
        )
    )

    print()
    print("=" * 70)
    print(
        "MedChange-VLM Prediction"
    )
    print("=" * 70)

    print(
        json.dumps(
            prediction.model_dump(
                mode="json"
            ),
            indent=2,
        )
    )

    print("=" * 70)

    print(
        "Research use only — "
        "not for clinical diagnosis."
    )


if __name__ == "__main__":
    main()