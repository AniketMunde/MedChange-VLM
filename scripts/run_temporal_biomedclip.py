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


import pandas as pd

from medchange.evaluation.temporal_runner import (
    run_temporal_experiment,
)


DEFAULT_FINDINGS = [
    "pleural_effusion",
    "atelectasis",
    "pneumothorax",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run frozen-BiomedCLIP "
            "longitudinal temporal baseline."
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
            "biomedclip_temporal"
        ),
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
        args.pairs
    )

    results = (
        run_temporal_experiment(
            dataframe=dataframe,

            dataset_root=(
                args.dataset_root
            ),

            output_dir=(
                args.output_dir
            ),

            findings=(
                DEFAULT_FINDINGS
            ),

            seed=args.seed,
        )
    )

    print()
    print("=" * 90)
    print(
        "M4.3 TEMPORAL RESULTS"
    )
    print("=" * 90)

    for (
        finding,
        result,
    ) in results.items():

        print()
        print(
            finding
        )

        print(
            "Current-only Macro F1 : "
            f"{result['current_only']['macro_f1']:.4f}"
        )

        print(
            "Longitudinal Macro F1 : "
            f"{result['longitudinal']['macro_f1']:.4f}"
        )

        print(
            "Improvement           : "
            f"{result['delta_macro_f1']:+.4f}"
        )

    print("=" * 90)


if __name__ == "__main__":
    main()