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

SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(SRC_DIR),
    )

from medchange.models.vision import (
    BiomedCLIP,
    CHEST_XRAY_FINDINGS,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run BiomedCLIP zero-shot "
            "chest-X-ray analysis."
        )
    )

    parser.add_argument(
        "image",
        type=str,
        help="Path to chest X-ray.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    model = BiomedCLIP()

    scores = model.score_findings(
        image_path=args.image,
        findings=CHEST_XRAY_FINDINGS,
    )

    sorted_scores = dict(
        sorted(
            scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )
    )

    print()
    print("=" * 70)
    print(
        "BiomedCLIP Zero-Shot Scores"
    )
    print("=" * 70)

    print(
        json.dumps(
            sorted_scores,
            indent=2,
        )
    )

    print("=" * 70)
    print(
        "Research baseline only — "
        "not for clinical diagnosis."
    )


if __name__ == "__main__":
    main()