from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from medchange.data.nih.qwen_temporal_sampling import (
    sample_qwen_temporal_benchmark,
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
            "qwen_temporal_benchmark_30.csv"
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
        args.input
    )

    benchmark = (
        sample_qwen_temporal_benchmark(
            dataframe=dataframe,
            unchanged_pairs=10,
            single_change_pairs=10,
            multi_change_pairs=10,
            seed=args.seed,
        )
    )

    output = Path(
        args.output
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    benchmark.to_csv(
        output,
        index=False,
    )

    print()
    print("=" * 80)
    print(
        "Qwen Temporal Benchmark"
    )
    print("=" * 80)

    print(
        f"Pairs: {len(benchmark)}"
    )

    print(
        benchmark[
            "benchmark_category"
        ]
        .value_counts()
        .to_string()
    )

    print(
        f"Output: {output}"
    )


if __name__ == "__main__":
    main()