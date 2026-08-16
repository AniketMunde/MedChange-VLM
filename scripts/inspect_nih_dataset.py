from __future__ import annotations

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

from medchange.data.nih import (
    get_nih_split_names,
    iter_nih_examples,
)


def main():
    splits = (
        get_nih_split_names()
    )

    print("=" * 70)
    print(
        "NIH ChestXray14 Hugging Face Inspection"
    )
    print("=" * 70)

    print(
        f"Available splits: {splits}"
    )

    split = (
        "train"
        if "train" in splits
        else splits[0]
    )

    print(
        f"Inspecting split : {split}"
    )

    print()

    examples = iter_nih_examples(
        split=split,
        max_samples=3,
        shuffle=False,
    )

    for index, example in enumerate(
        examples,
        start=1,
    ):
        print(
            f"Example {index}"
        )

        print(
            f"  Image size : "
            f"{example.image.size}"
        )

        print(
            f"  Patient ID : "
            f"{example.patient_id}"
        )

        print(
            f"  View       : "
            f"{example.view_position}"
        )

        print(
            f"  Age        : "
            f"{example.patient_age}"
        )

        print(
            f"  Gender     : "
            f"{example.patient_gender}"
        )

        print(
            f"  Raw labels : "
            f"{example.raw_labels}"
        )

        print(
            f"  Labels     : "
            f"{example.labels}"
        )

        print()

    print("=" * 70)


if __name__ == "__main__":
    main()