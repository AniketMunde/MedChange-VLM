from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


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


from medchange.evaluation.temporal_ablation import (
    run_temporal_ablation,
)


FINDINGS = [
    "atelectasis",
    "cardiomegaly",
    "consolidation",
    "edema",
    "pleural_effusion",
    "pneumonia",
    "pneumothorax",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run M4.3.1 BiomedCLIP "
            "temporal feature ablation."
        )
    )

    parser.add_argument(
        "--pairs",
        type=str,
        default=(
            "data/nih/"
            "temporal_eval_subset.csv"
        ),
    )

    parser.add_argument(
        "--dataset-root",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=(
            "experiments/"
            "temporal_ablation"
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=123,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    dataframe = pd.read_csv(
        args.pairs
    )

    results = (
        run_temporal_ablation(
            dataframe=dataframe,

            dataset_root=(
                args.dataset_root
            ),

            findings=FINDINGS,

            output_dir=(
                args.output_dir
            ),

            seed=args.seed,
        )
    )

    print()
    print("=" * 100)
    print(
        "M4.3.1 TEMPORAL FEATURE ABLATION"
    )
    print("=" * 100)

    print(
        f"{'Finding':<22}"
        f"{'Best feature':<25}"
        f"{'Macro F1':>12}"
    )

    print("-" * 100)

    for finding in FINDINGS:
        result = results[
            finding
        ]

        print(
            f"{finding:<22}"
            f"{result['best_feature_set']:<25}"
            f"{result['best_macro_f1']:>12.4f}"
        )

    print("=" * 100)


if __name__ == "__main__":
    main()