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

SRC_DIR = (
    PROJECT_ROOT
    / "src"
)

if str(
    SRC_DIR
) not in sys.path:
    sys.path.insert(
        0,
        str(
            SRC_DIR
        ),
    )


from medchange.evaluation.qwen_temporal_runner import (
    QwenTemporalBenchmarkRunner,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Qwen2.5-VL on "
            "longitudinal NIH chest X-ray pairs."
        )
    )

    parser.add_argument(
        "--pairs",
        default=(
            "data/nih/"
            "qwen_temporal_benchmark_30.csv"
        ),
    )

    parser.add_argument(
        "--dataset-root",
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "experiments/"
            "qwen_temporal_m44"
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    dataframe = pd.read_csv(
        args.pairs
    )

    runner = (
        QwenTemporalBenchmarkRunner(
            dataset_root=(
                args.dataset_root
            )
        )
    )

    metrics = runner.run(
        dataframe=dataframe,
        output_dir=args.output_dir,
    )

    print()
    print("=" * 100)
    print(
        "M4.4 QWEN TEMPORAL BENCHMARK"
    )
    print("=" * 100)

    print(
        json.dumps(
            metrics,
            indent=2,
        )
    )

    print("=" * 100)


if __name__ == "__main__":
    main()