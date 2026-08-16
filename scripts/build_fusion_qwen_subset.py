from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


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


from medchange.data.nih.fusion_subset import (
    build_fusion_qwen_subset,
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default=(
            "data/nih/"
            "temporal_pairs_same_view.csv"
        ),
    )

    parser.add_argument(
        "--output",
        default=(
            "data/nih/"
            "fusion_qwen_subset_200.csv"
        ),
    )

    parser.add_argument(
        "--num-pairs",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    dataframe = pd.read_csv(
        args.input
    )

    subset = build_fusion_qwen_subset(
        dataframe=dataframe,
        num_pairs=args.num_pairs,
        seed=args.seed,
    )

    output = Path(args.output)

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    subset.to_csv(
        output,
        index=False,
    )

    counts = (
        subset["fusion_category"]
        .value_counts()
        .to_dict()
    )

    metadata = {
        "num_pairs": len(subset),
        "seed": args.seed,
        "categories": counts,
        "unique_patients": int(
            subset["patient_id"].nunique()
        ),
        "duplicate_pair_ids": int(
            subset["pair_id"]
            .duplicated()
            .sum()
        ),
    }

    metadata_path = output.with_suffix(
        ".json"
    )

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 80)
    print("M4.5.1 QWEN EVIDENCE COHORT")
    print("=" * 80)
    print(f"Pairs           : {len(subset)}")
    print(
        "Unique patients : "
        f"{subset['patient_id'].nunique()}"
    )

    for category, count in counts.items():
        print(
            f"{category:<16}: {count}"
        )

    print(f"Manifest        : {output}")
    print(f"Metadata        : {metadata_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()